# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""OpenAI (and OpenAI-compatible) embedding backend.

Requires the optional ``openai`` extra::

    pip install "linear-adapter-trainer[openai]"

The API key is read from the ``OPENAI_API_KEY`` environment variable by
default. Never hard-code credentials.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import as_float32, l2_normalize

_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    """Embed text with the OpenAI embeddings API.

    Args:
        model: Embedding model name.
        api_key: Optional explicit key; falls back to ``OPENAI_API_KEY``.
        base_url: Optional override for OpenAI-compatible gateways.
        dimensions: Optional output dimensionality (``text-embedding-3-*``).
        batch_size: Number of texts per request.
        normalize: Whether to L2-normalize the returned vectors.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 128,
        normalize: bool = True,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "OpenAIEmbedder requires the optional dependency. "
                'Install it with: pip install "linear-adapter-trainer[openai]"'
            ) from exc

        self.model = model
        self.batch_size = batch_size
        self.normalize = normalize
        self._dimensions = dimensions
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._dimension = dimensions or _DIMENSIONS.get(model, 1536)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            kwargs: dict[str, Any] = {"model": self.model, "input": batch}
            if self._dimensions is not None:
                kwargs["dimensions"] = self._dimensions
            response = self._client.embeddings.create(**kwargs)
            out.extend(item.embedding for item in response.data)
        vectors = as_float32(np.array(out))
        return l2_normalize(vectors) if self.normalize else vectors
