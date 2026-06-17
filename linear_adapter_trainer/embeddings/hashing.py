# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free hashing embedder.

This backend produces deterministic vectors using the hashing trick over word
n-grams. It is **not** a semantic model: its purpose is to make the full
pipeline runnable in CI, tests, and offline demos without downloading weights.
Use a real backend (Sentence-Transformers, OpenAI) for production results.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from .base import l2_normalize

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


class HashingEmbedder:
    """Deterministic bag-of-n-grams embedder via the hashing trick.

    Args:
        dimension: Output vector size.
        ngram_range: Inclusive ``(min_n, max_n)`` word n-gram range.
        seed: Salt applied to the token hash for reproducible variety.
    """

    def __init__(
        self,
        dimension: int = 256,
        *,
        ngram_range: tuple[int, int] = (1, 2),
        seed: int = 0,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        min_n, max_n = ngram_range
        if not 1 <= min_n <= max_n:
            raise ValueError("ngram_range must satisfy 1 <= min_n <= max_n.")
        self._dimension = dimension
        self._ngram_range = ngram_range
        self._seed = seed

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._ngrams(text):
                index = self._bucket(token)
                matrix[row, index] += 1.0
        return l2_normalize(matrix)

    # -- internals ---------------------------------------------------------
    def _ngrams(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        min_n, max_n = self._ngram_range
        grams: list[str] = []
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                grams.append(" ".join(tokens[i : i + n]))
        return grams

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(f"{self._seed}:{token}".encode(), digest_size=8).digest()
        return int.from_bytes(digest, "little") % self._dimension
