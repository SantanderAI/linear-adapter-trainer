# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Sentence-Transformers embedding backend.

Requires the optional ``sentence-transformers`` extra::

    pip install "linear-adapter-trainer[sentence-transformers]"
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import as_float32


class SentenceTransformerEmbedder:
    """Wrap a :class:`sentence_transformers.SentenceTransformer` model.

    Args:
        model_name: Any model id understood by Sentence-Transformers.
        device: Optional device override (``"cpu"``, ``"cuda"``, ...).
        batch_size: Encoding batch size.
        normalize: Whether to L2-normalize embeddings at the source.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "SentenceTransformerEmbedder requires the optional dependency. "
                'Install it with: pip install "linear-adapter-trainer[sentence-transformers]"'
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return as_float32(vectors)
