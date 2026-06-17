# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""End-to-end triplet dataset generation (Module 1).

Pipeline:

1. Embed every chunk with the chosen backend.
2. For each chunk, generate anchor queries.
3. For each query, mine negative chunks (semantic-opposite / random / hard).
4. Assemble ``(query, positive, negative)`` triplets.
5. Split into train/val at the chunk level.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from tqdm.auto import tqdm

from ..embeddings.base import EmbeddingModel
from ..knowledge_base.base import KnowledgeBase
from .negative_generation import NegativeGenerator
from .negatives import NegativeSampler
from .query_generation import QueryGenerator
from .schema import Triplet, TripletDataset
from .splitter import split_triplets

# Minimum pool of LLM hard negatives generated per chunk, so queries of the same
# chunk can draw distinct negatives even when ``negatives_per_query`` is small.
_NEG_POOL_MIN = 4


@dataclass(slots=True)
class DatasetConfig:
    """Configuration for :class:`DatasetGenerator`.

    Attributes:
        queries_per_chunk: Number of anchor queries generated per chunk.
        negatives_per_query: Number of triplets (negatives) per query.
        strategy: Negative strategy (``random``/``semantic_opposite``/``hard``/``mixed``).
        pool_size: Candidate pool size for opposite/hard mining.
        mix: Strategy weights when ``strategy == "mixed"``.
        val_fraction: Fraction of groups held out for validation.
        split_strategy: How to split train/val: ``"chunk"`` holds out whole
            chunks (generalize to unseen corpus); ``"query"`` holds out queries
            while keeping the corpus visible (generalize to new queries over a
            fixed corpus - the realistic setting for a query-side adapter).
        seed: Global seed for reproducibility.
        max_workers: Number of chunks whose queries are generated concurrently.
            ``1`` keeps generation sequential; higher values issue several
            query-generator calls in parallel, which dramatically speeds up the
            LLM backend (one network round-trip per chunk). Output stays
            deterministic because triplets are assembled in chunk order.
    """

    queries_per_chunk: int = 3
    negatives_per_query: int = 1
    strategy: str = "semantic_opposite"
    pool_size: int = 10
    mix: dict[str, float] | None = None
    val_fraction: float = 0.2
    split_strategy: str = "chunk"
    seed: int = 0
    max_workers: int = 1

    def __post_init__(self) -> None:
        if self.queries_per_chunk <= 0:
            raise ValueError("queries_per_chunk must be positive.")
        if self.negatives_per_query <= 0:
            raise ValueError("negatives_per_query must be positive.")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive.")


@dataclass(slots=True)
class DatasetGenerator:
    """Generate a triplet dataset from a knowledge base.

    Args:
        knowledge_base: The corpus of chunks.
        embedder: Embedding backend used for negative mining.
        query_generator: Backend that produces anchor queries.
        config: Generation hyper-parameters.
    """

    knowledge_base: KnowledgeBase
    embedder: EmbeddingModel
    query_generator: QueryGenerator
    config: DatasetConfig = field(default_factory=DatasetConfig)
    negative_generator: NegativeGenerator | None = None

    def generate(self, *, show_progress: bool = True) -> TripletDataset:
        kb = self.knowledge_base
        cfg = self.config
        chunks = list(kb)

        queries_by_position = self._generate_queries(chunks, show_progress=show_progress)

        if self.negative_generator is not None:
            triplets = self._assemble_llm_negatives(chunks, queries_by_position, show_progress)
            embedding_dim = self.embedder.dimension
            negative_backend = "llm"
        else:
            embeddings = self.embedder.embed(kb.texts)
            sampler = NegativeSampler(
                embeddings=embeddings,
                strategy=cfg.strategy,
                pool_size=cfg.pool_size,
                mix=cfg.mix,
                seed=cfg.seed,
            )
            triplets = self._assemble_mined_negatives(chunks, queries_by_position, sampler)
            embedding_dim = int(embeddings.shape[1])
            negative_backend = cfg.strategy

        train, val = split_triplets(
            triplets,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
            strategy=cfg.split_strategy,
        )
        metadata = {
            "n_chunks": len(kb),
            "n_triplets": len(triplets),
            "n_train": len(train),
            "n_val": len(val),
            "queries_per_chunk": cfg.queries_per_chunk,
            "negatives_per_query": cfg.negatives_per_query,
            "strategy": cfg.strategy,
            "mix": cfg.mix,
            "negative_backend": negative_backend,
            "val_fraction": cfg.val_fraction,
            "split_strategy": cfg.split_strategy,
            "seed": cfg.seed,
            "embedding_dim": embedding_dim,
        }
        return TripletDataset(train=train, val=val, metadata=metadata)

    # -- assembly ----------------------------------------------------------
    def _assemble_mined_negatives(
        self, chunks: list, queries_by_position: list[list[str]], sampler: NegativeSampler
    ) -> list[Triplet]:
        cfg = self.config
        triplets: list[Triplet] = []
        for position, chunk in enumerate(chunks):
            for query in queries_by_position[position]:
                if not query.strip():
                    continue
                for neg_index, strategy in sampler.sample(position, cfg.negatives_per_query):
                    triplets.append(
                        Triplet(
                            query=query.strip(),
                            positive_id=chunk.id,
                            negative_id=self.knowledge_base[neg_index].id,
                            strategy=strategy,
                        )
                    )
        return triplets

    def _assemble_llm_negatives(
        self, chunks: list, queries_by_position: list[list[str]], show_progress: bool
    ) -> list[Triplet]:
        cfg = self.config
        negatives_by_position = self._generate_negatives(chunks, show_progress=show_progress)
        rng = random.Random(cfg.seed)
        triplets: list[Triplet] = []
        for position, chunk in enumerate(chunks):
            pool = [neg for neg in negatives_by_position[position] if neg.strip()]
            if not pool:
                continue
            for query in queries_by_position[position]:
                if not query.strip():
                    continue
                for index, neg_text in enumerate(_choose(pool, cfg.negatives_per_query, rng)):
                    triplets.append(
                        Triplet(
                            query=query.strip(),
                            positive_id=chunk.id,
                            negative_id=f"{chunk.id}::llm-neg::{index}",
                            negative_text=neg_text.strip(),
                            strategy="llm_hard",
                        )
                    )
        return triplets

    def _generate_queries(self, chunks: list, *, show_progress: bool) -> list[list[str]]:
        """Generate anchor queries for every chunk, optionally in parallel.

        Results are returned positionally (``results[i]`` are the queries for
        ``chunks[i]``) regardless of completion order, so downstream triplet
        assembly stays deterministic.
        """
        cfg = self.config
        results: list[list[str]] = [[] for _ in chunks]

        if cfg.max_workers == 1:
            iterator = tqdm(
                enumerate(chunks),
                total=len(chunks),
                desc="Generating queries",
                disable=not show_progress,
            )
            for position, chunk in iterator:
                results[position] = self.query_generator.generate(chunk.text, cfg.queries_per_chunk)
            return results

        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            futures = {
                executor.submit(
                    self.query_generator.generate, chunk.text, cfg.queries_per_chunk
                ): position
                for position, chunk in enumerate(chunks)
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating queries",
                disable=not show_progress,
            ):
                results[futures[future]] = future.result()
        return results

    def _generate_negatives(self, chunks: list, *, show_progress: bool) -> list[list[str]]:
        """Generate a pool of hard-negative passages per chunk via the LLM.

        One generation call is made per chunk and its passages are reused across
        that chunk's queries, so cost scales with the number of chunks (not
        queries). Results are positional, mirroring :meth:`_generate_queries`.
        """
        assert self.negative_generator is not None
        cfg = self.config
        n_pool = max(cfg.negatives_per_query, _NEG_POOL_MIN)
        results: list[list[str]] = [[] for _ in chunks]

        if cfg.max_workers == 1:
            iterator = tqdm(
                enumerate(chunks),
                total=len(chunks),
                desc="Generating negatives",
                disable=not show_progress,
            )
            for position, chunk in iterator:
                results[position] = self.negative_generator.generate(chunk.text, n_pool)
            return results

        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            futures = {
                executor.submit(self.negative_generator.generate, chunk.text, n_pool): position
                for position, chunk in enumerate(chunks)
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating negatives",
                disable=not show_progress,
            ):
                results[futures[future]] = future.result()
        return results


def _choose(pool: list[str], k: int, rng: random.Random) -> list[str]:
    """Pick ``k`` negatives from ``pool``, sampling without replacement when
    possible and cycling deterministically if the pool is smaller than ``k``."""
    if k <= 0 or not pool:
        return []
    if len(pool) >= k:
        return rng.sample(pool, k)
    return [pool[i % len(pool)] for i in range(k)]
