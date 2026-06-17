# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Retrieval evaluation comparing base vs adapted embeddings.

The evaluator embeds the corpus once, then retrieves chunks for a set of
evaluation queries using cosine similarity. It reports metrics for the raw
embeddings and for the adapter-transformed query embeddings, plus the delta.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from ..adapter.model import LinearAdapter
from ..dataset.schema import Triplet
from ..embeddings.base import EmbeddingModel, l2_normalize
from ..knowledge_base.base import KnowledgeBase
from .metrics import evaluate_rankings


class RetrievalEvaluator:
    """Evaluate retrieval quality over a knowledge base.

    Args:
        knowledge_base: The corpus to retrieve from.
        embedder: Embedding backend (must match the one used for training).
        ks: Cut-offs at which precision/recall/hit-rate/ndcg are computed.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        embedder: EmbeddingModel,
        *,
        ks: Sequence[int] = (1, 3, 5, 10),
    ) -> None:
        self.kb = knowledge_base
        self.embedder = embedder
        self.ks = tuple(ks)
        self.chunk_ids = knowledge_base.ids
        self._chunk_matrix = l2_normalize(embedder.embed(knowledge_base.texts))

    def evaluate(
        self,
        triplets: Sequence[Triplet],
        *,
        adapter: LinearAdapter | None = None,
    ) -> dict[str, float]:
        """Compute metrics for ``triplets`` (optionally through ``adapter``)."""
        queries, relevant = self._ground_truth(triplets)
        if not queries:
            return {"mrr": 0.0, "n_queries": 0.0}

        query_matrix = l2_normalize(self.embedder.embed(queries))
        if adapter is not None:
            query_matrix = l2_normalize(adapter.transform(query_matrix))

        ranked = self._rank(query_matrix)
        rankings = [(ranked[i], relevant[i]) for i in range(len(queries))]
        return evaluate_rankings(rankings, ks=self.ks)

    def compare(
        self,
        triplets: Sequence[Triplet],
        adapter: LinearAdapter,
    ) -> dict[str, dict[str, float]]:
        """Return ``{"base", "adapted", "delta"}`` metric dictionaries."""
        base = self.evaluate(triplets, adapter=None)
        adapted = self.evaluate(triplets, adapter=adapter)
        delta = {
            key: adapted[key] - base[key] for key in base if key != "n_queries" and key in adapted
        }
        return {"base": base, "adapted": adapted, "delta": delta}

    # -- internals ---------------------------------------------------------
    def _ground_truth(self, triplets: Sequence[Triplet]) -> tuple[list[str], list[set[str]]]:
        relevant_by_query: dict[str, set[str]] = defaultdict(set)
        for triplet in triplets:
            relevant_by_query[triplet.query].add(triplet.positive_id)
        queries = list(relevant_by_query)
        relevant = [relevant_by_query[q] for q in queries]
        return queries, relevant

    def _rank(self, query_matrix: np.ndarray) -> list[list[str]]:
        max_k = min(max(self.ks), len(self.chunk_ids))
        scores = query_matrix @ self._chunk_matrix.T  # (n_queries, n_chunks)
        # Partial top-k selection, then sort the small slice.
        top_unsorted = np.argpartition(-scores, kth=max_k - 1, axis=1)[:, :max_k]
        ranked: list[list[str]] = []
        for row in range(scores.shape[0]):
            cols = top_unsorted[row]
            order = cols[np.argsort(-scores[row, cols])]
            ranked.append([self.chunk_ids[c] for c in order])
        return ranked
