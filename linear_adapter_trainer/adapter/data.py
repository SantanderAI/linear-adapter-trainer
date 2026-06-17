# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Tensor datasets for adapter training.

Embeddings are precomputed once: query (anchor) embeddings per triplet, and a
single chunk-embedding matrix shared across triplets. Only anchors are fed
through the adapter during training, so chunk embeddings are stored once and
gathered by index to keep memory low.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ..dataset.schema import Triplet
from ..embeddings.base import EmbeddingModel, as_float32
from ..knowledge_base.base import KnowledgeBase


@dataclass(slots=True)
class EmbeddingBundle:
    """Precomputed tensors shared by train/val datasets."""

    chunk_matrix: torch.Tensor  # (n_chunks, dim)
    chunk_id_to_row: dict[str, int]
    dim: int


def build_embedding_bundle(kb: KnowledgeBase, embedder: EmbeddingModel) -> EmbeddingBundle:
    """Embed all chunks once and index them by id."""
    matrix = as_float32(embedder.embed(kb.texts))
    chunk_id_to_row = {chunk.id: i for i, chunk in enumerate(kb)}
    return EmbeddingBundle(
        chunk_matrix=torch.from_numpy(matrix),
        chunk_id_to_row=chunk_id_to_row,
        dim=matrix.shape[1],
    )


class TripletEmbeddingDataset(Dataset):
    """Yields ``(anchor, positive, negative)`` embedding tensors.

    Positives are always corpus chunks (gathered by id from ``bundle``).
    Negatives are corpus chunks too, unless a precomputed per-triplet
    ``negatives`` tensor is supplied (e.g. LLM-generated hard negatives whose
    text is not in the knowledge base).
    """

    def __init__(
        self,
        triplets: Sequence[Triplet],
        anchors: torch.Tensor,
        bundle: EmbeddingBundle,
        negatives: torch.Tensor | None = None,
    ) -> None:
        if len(triplets) != anchors.shape[0]:
            raise ValueError("triplets and anchors must align in length.")
        self._anchors = anchors
        self._chunks = bundle.chunk_matrix
        self._pos_rows = torch.tensor(
            [bundle.chunk_id_to_row[t.positive_id] for t in triplets], dtype=torch.long
        )
        if negatives is not None:
            if negatives.shape[0] != anchors.shape[0]:
                raise ValueError("negatives and anchors must align in length.")
            self._negatives = negatives
            self._neg_rows = None
        else:
            self._negatives = None
            self._neg_rows = torch.tensor(
                [bundle.chunk_id_to_row[t.negative_id] for t in triplets], dtype=torch.long
            )

    def __len__(self) -> int:
        return self._anchors.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positive = self._chunks[self._pos_rows[index]]
        if self._negatives is not None:
            negative = self._negatives[index]
        else:
            assert self._neg_rows is not None
            negative = self._chunks[self._neg_rows[index]]
        return self._anchors[index], positive, negative


def embed_queries(triplets: Sequence[Triplet], embedder: EmbeddingModel) -> torch.Tensor:
    """Embed every triplet's query (deduplicated for efficiency)."""
    unique = sorted({t.query for t in triplets})
    if not unique:
        return torch.empty((0, embedder.dimension), dtype=torch.float32)
    vectors = as_float32(embedder.embed(unique))
    lookup = {query: i for i, query in enumerate(unique)}
    rows = np.stack([vectors[lookup[t.query]] for t in triplets], axis=0)
    return torch.from_numpy(as_float32(rows))


def embed_negatives(
    triplets: Sequence[Triplet],
    embedder: EmbeddingModel,
    bundle: EmbeddingBundle,
) -> torch.Tensor | None:
    """Build a per-triplet negative-embedding tensor when needed.

    Returns ``None`` when every negative is a corpus chunk (the fast path, where
    :class:`TripletEmbeddingDataset` gathers chunk rows directly). When any
    triplet carries ``negative_text`` (e.g. an LLM-generated hard negative), the
    free text is embedded once (deduplicated) and a ``(n_triplets, dim)`` tensor
    is returned; corpus-chunk negatives in the same set are gathered from
    ``bundle`` so the result stays aligned with ``triplets``.
    """
    if not any(t.negative_text for t in triplets):
        return None

    unique = sorted({t.negative_text for t in triplets if t.negative_text})
    text_vectors = as_float32(embedder.embed(unique)) if unique else None
    lookup = {text: i for i, text in enumerate(unique)}
    chunk_matrix = bundle.chunk_matrix.numpy()

    rows = []
    for triplet in triplets:
        if triplet.negative_text:
            assert text_vectors is not None
            rows.append(text_vectors[lookup[triplet.negative_text]])
        else:
            rows.append(chunk_matrix[bundle.chunk_id_to_row[triplet.negative_id]])
    return torch.from_numpy(as_float32(np.stack(rows, axis=0)))
