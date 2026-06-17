# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Deterministic text chunking utilities.

The default splitter is a recursive, separator-aware character splitter with
configurable overlap. It is intentionally dependency-free so that ingestion
remains reproducible and easy to audit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .base import Chunk, KnowledgeBase

_DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


@dataclass(slots=True)
class TextSplitter:
    """Split long documents into overlapping chunks.

    Attributes:
        chunk_size: Target maximum number of characters per chunk.
        chunk_overlap: Number of trailing characters shared between chunks.
        separators: Ordered separators tried from coarsest to finest.
    """

    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: Sequence[str] = field(default_factory=lambda: _DEFAULT_SEPARATORS)

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size.")

    def split_text(self, text: str) -> list[str]:
        """Split a single document into a list of chunk strings."""
        segments = self._split_recursive(text, list(self.separators))
        return self._merge(segments)

    def split_knowledge_base(self, kb: KnowledgeBase) -> KnowledgeBase:
        """Re-chunk every document of a knowledge base.

        New chunk ids are derived from the parent id with a positional suffix
        (e.g. ``doc-1::2``) and the parent id is preserved in metadata.
        """
        chunks: list[Chunk] = []
        for parent in kb:
            pieces = self.split_text(parent.text)
            for position, piece in enumerate(pieces):
                chunks.append(
                    Chunk(
                        id=f"{parent.id}::{position}",
                        text=piece,
                        metadata={**parent.metadata, "parent_id": parent.id},
                    )
                )
        return KnowledgeBase(chunks)

    # -- internals ---------------------------------------------------------
    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size or not separators:
            return [text] if text else []

        separator = separators[0]
        remaining = separators[1:]
        parts = text.split(separator) if separator else list(text)

        out: list[str] = []
        for part in parts:
            piece = part if not separator else part + separator
            if len(piece) <= self.chunk_size:
                if piece:
                    out.append(piece)
            else:
                out.extend(self._split_recursive(piece, remaining))
        return out

    def _merge(self, segments: list[str]) -> list[str]:
        merged: list[str] = []
        buffer = ""
        for segment in segments:
            if len(buffer) + len(segment) <= self.chunk_size:
                buffer += segment
                continue
            if buffer.strip():
                merged.append(buffer.strip())
            buffer = (buffer[-self.chunk_overlap :] if self.chunk_overlap else "") + segment
        if buffer.strip():
            merged.append(buffer.strip())
        return merged
