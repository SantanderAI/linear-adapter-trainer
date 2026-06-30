# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""TOML-driven configuration and component factories.

A single config file fully describes a run: where the knowledge base lives,
which embedding and query-generation backends to use, and the dataset/training
hyper-parameters. The CLI consumes this module; library users can also call the
factories directly.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .adapter.trainer import TrainingConfig
from .dataset.generator import DatasetConfig
from .embeddings.base import EmbeddingModel
from .knowledge_base.base import KnowledgeBase
from .knowledge_base.chunking import TextSplitter


def load_config(path: str | Path) -> dict[str, Any]:
    """Parse a TOML configuration file into a dictionary."""
    path = Path(path)
    with path.open("rb") as handle:
        return tomllib.load(handle)


def build_knowledge_base(spec: dict[str, Any]) -> KnowledgeBase:
    """Instantiate a knowledge base from a ``[knowledge_base]`` table."""
    fmt = spec.get("format", "jsonl")
    if fmt == "jsonl":
        path = spec["path"]
        kb = KnowledgeBase.from_jsonl(
            path,
            text_key=spec.get("text_key", "text"),
            id_key=spec.get("id_key", "id"),
        )
    elif fmt == "directory":
        path = spec["path"]
        kb = KnowledgeBase.from_directory(path, glob=spec.get("glob", "*.txt"))
    elif fmt == "web_fetch":
        from .knowledge_base.web import WebLoader

        client = spec.get("client")
        if client is None:
            backend = spec.get("backend")
            if not backend:
                raise ValueError(
                    "web_fetch requires either a `client` or a `backend` "
                    "naming an optional web-fetch adapter (e.g. backend = \"linkup\")."
                )
            from .knowledge_base.web_adapters import build_web_fetch_client

            client = build_web_fetch_client(backend)
        kb = WebLoader(
            client=client,
            render_js=spec.get("render_js", True),
            include_raw_html=spec.get("include_raw_html", False),
            extract_images=spec.get("extract_images", False),
        ).load_urls(spec["urls"], ids=spec.get("ids"))
    else:
        raise ValueError(f"Unsupported knowledge_base.format: {fmt!r}")

    chunking = spec.get("chunking")
    if chunking and chunking.get("enabled", False):
        splitter = TextSplitter(
            chunk_size=chunking.get("chunk_size", 512),
            chunk_overlap=chunking.get("chunk_overlap", 64),
        )
        kb = splitter.split_knowledge_base(kb)
    return kb


def build_embedder(spec: dict[str, Any]) -> EmbeddingModel:
    """Instantiate an embedding backend from an ``[embedder]`` table."""
    backend = spec.get("backend", "hashing")
    if backend == "hashing":
        from .embeddings.hashing import HashingEmbedder

        return HashingEmbedder(
            dimension=spec.get("dimension", 256),
            ngram_range=tuple(spec.get("ngram_range", (1, 2))),
            seed=spec.get("seed", 0),
        )
    if backend == "sentence-transformers":
        from .embeddings.sentence_transformer import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            model_name=spec.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            device=spec.get("device"),
            batch_size=spec.get("batch_size", 32),
            normalize=spec.get("normalize", True),
        )
    if backend == "openai":
        from .embeddings.openai import OpenAIEmbedder

        return OpenAIEmbedder(
            model=spec.get("model", "text-embedding-3-small"),
            base_url=spec.get("base_url"),
            dimensions=spec.get("dimensions"),
            batch_size=spec.get("batch_size", 128),
            normalize=spec.get("normalize", True),
        )
    raise ValueError(f"Unsupported embedder.backend: {backend!r}")


def build_query_generator(spec: dict[str, Any]):
    """Instantiate a query generator from a ``[query_generator]`` table."""
    backend = spec.get("backend", "template")
    if backend == "template":
        from .dataset.query_generation import TemplateQueryGenerator

        return TemplateQueryGenerator(
            seed=spec.get("seed", 0),
            max_keyword_words=spec.get("max_keyword_words", 3),
        )
    if backend == "llm":
        from .dataset.query_generation import LLMQueryGenerator

        return LLMQueryGenerator(
            model=spec.get("model", "gpt-5-mini"),
            base_url=spec.get("base_url"),
            temperature=spec.get("temperature"),
        )
    raise ValueError(f"Unsupported query_generator.backend: {backend!r}")


def build_negative_generator(spec: dict[str, Any]):
    """Instantiate an optional LLM negative generator from a ``[negative_generator]`` table.

    Returns ``None`` (corpus-mined negatives via :class:`NegativeSampler`) unless
    ``backend = "llm"`` is requested, in which case negatives are synthesised as
    LLM hard-negative passages.
    """
    backend = spec.get("backend", "none")
    if backend in ("none", "sampler"):
        return None
    if backend == "llm":
        from .dataset.negative_generation import LLMNegativeGenerator

        return LLMNegativeGenerator(
            model=spec.get("model", "gpt-5-mini"),
            base_url=spec.get("base_url"),
            temperature=spec.get("temperature"),
        )
    raise ValueError(f"Unsupported negative_generator.backend: {backend!r}")


def build_dataset_config(spec: dict[str, Any]) -> DatasetConfig:
    return DatasetConfig(
        queries_per_chunk=spec.get("queries_per_chunk", 3),
        negatives_per_query=spec.get("negatives_per_query", 1),
        strategy=spec.get("strategy", "semantic_opposite"),
        pool_size=spec.get("pool_size", 10),
        mix=spec.get("mix"),
        val_fraction=spec.get("val_fraction", 0.2),
        split_strategy=spec.get("split_strategy", "chunk"),
        seed=spec.get("seed", 0),
        max_workers=spec.get("max_workers", 1),
    )


def build_training_config(spec: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        epochs=spec.get("epochs", 20),
        batch_size=spec.get("batch_size", 64),
        learning_rate=spec.get("learning_rate", 1e-3),
        weight_decay=spec.get("weight_decay", 0.0),
        margin=spec.get("margin", 0.2),
        distance=spec.get("distance", "cosine"),
        residual=spec.get("residual", True),
        normalize_output=spec.get("normalize_output", True),
        eval_ks=tuple(spec.get("eval_ks", (1, 3, 5, 10))),
        monitor=spec.get("monitor", "mrr"),
        patience=spec.get("patience", 5),
        grad_clip=spec.get("grad_clip", 1.0),
        device=spec.get("device"),
        seed=spec.get("seed", 0),
    )
