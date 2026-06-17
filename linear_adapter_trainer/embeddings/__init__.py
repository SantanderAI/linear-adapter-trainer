# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Pluggable text embedding backends."""

from .base import EmbeddingModel, as_float32, l2_normalize
from .hashing import HashingEmbedder

__all__ = [
    "EmbeddingModel",
    "HashingEmbedder",
    "as_float32",
    "l2_normalize",
]


def __getattr__(name: str):  # pragma: no cover - lazy optional imports
    if name == "SentenceTransformerEmbedder":
        from .sentence_transformer import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder
    if name == "OpenAIEmbedder":
        from .openai import OpenAIEmbedder

        return OpenAIEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
