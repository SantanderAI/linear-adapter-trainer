# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Embedding model protocol and shared utilities.

Any object that turns text into dense vectors can be plugged into the rest of
the pipeline as long as it satisfies the :class:`EmbeddingModel` protocol.
This keeps the library agnostic to the provider (Sentence-Transformers,
OpenAI, a custom service, ...).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingModel(Protocol):
    """Structural type for any text embedding backend."""

    @property
    def dimension(self) -> int:
        """Dimensionality of the produced embedding vectors."""
        ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts into a ``(len(texts), dimension)`` array."""
        ...


def l2_normalize(matrix: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Return a copy of ``matrix`` with unit-norm rows."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, eps, None)


def as_float32(matrix: np.ndarray) -> np.ndarray:
    """Coerce an array to a contiguous ``float32`` matrix."""
    return np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
