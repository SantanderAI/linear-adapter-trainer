# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Core knowledge base abstractions.

A :class:`KnowledgeBase` is an ordered, immutable-ish collection of
:class:`Chunk` objects. Chunks are the atomic unit of retrieval: each one
carries an identifier, a text payload, and optional metadata.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single retrievable unit of text.

    Attributes:
        id: Stable, unique identifier for the chunk.
        text: The textual content used for embedding and retrieval.
        metadata: Arbitrary JSON-serializable metadata (source, page, etc.).
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Chunk.id must be a non-empty string.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(f"Chunk {self.id!r} has empty text.")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Chunk:
        return cls(
            id=str(payload["id"]),
            text=str(payload["text"]),
            metadata=dict(payload.get("metadata", {})),
        )


class KnowledgeBase(Sequence[Chunk]):
    """An ordered collection of unique chunks.

    The knowledge base guarantees that chunk identifiers are unique and
    provides convenient constructors for the most common ingestion formats.
    """

    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self._chunks: list[Chunk] = list(chunks)
        self._index: dict[str, int] = {}
        for position, chunk in enumerate(self._chunks):
            if chunk.id in self._index:
                raise ValueError(f"Duplicate chunk id detected: {chunk.id!r}")
            self._index[chunk.id] = position
        if not self._chunks:
            raise ValueError("KnowledgeBase cannot be empty.")

    # -- Sequence protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int) -> Chunk:  # type: ignore[override]
        return self._chunks[index]

    def __iter__(self) -> Iterator[Chunk]:
        return iter(self._chunks)

    # -- Lookups -----------------------------------------------------------
    def get(self, chunk_id: str) -> Chunk:
        """Return the chunk with ``chunk_id`` or raise ``KeyError``."""
        position = self._index.get(chunk_id)
        if position is None:
            raise KeyError(f"Unknown chunk id: {chunk_id!r}")
        return self._chunks[position]

    def position_of(self, chunk_id: str) -> int:
        """Return the integer position of ``chunk_id`` within the base."""
        position = self._index.get(chunk_id)
        if position is None:
            raise KeyError(f"Unknown chunk id: {chunk_id!r}")
        return position

    @property
    def ids(self) -> list[str]:
        return [chunk.id for chunk in self._chunks]

    @property
    def texts(self) -> list[str]:
        return [chunk.text for chunk in self._chunks]

    # -- Constructors ------------------------------------------------------
    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        *,
        ids: Sequence[str] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> KnowledgeBase:
        """Build a knowledge base from raw strings.

        Identifiers default to ``chunk-0``, ``chunk-1`` ... when not supplied.
        """
        texts = list(texts)
        if ids is not None and len(ids) != len(texts):
            raise ValueError("`ids` length must match `texts` length.")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("`metadatas` length must match `texts` length.")
        chunks = [
            Chunk(
                id=str(ids[i]) if ids is not None else f"chunk-{i}",
                text=text,
                metadata=dict(metadatas[i]) if metadatas is not None else {},
            )
            for i, text in enumerate(texts)
        ]
        return cls(chunks)

    @classmethod
    def from_jsonl(
        cls, path: str | Path, *, text_key: str = "text", id_key: str = "id"
    ) -> KnowledgeBase:
        """Load a knowledge base from a JSON Lines file.

        Each line must be a JSON object containing at least ``text_key``.
        ``id_key`` is optional; positional identifiers are used as a fallback.
        """
        path = Path(path)
        chunks: list[Chunk] = []
        with path.open("r", encoding="utf-8") as handle:
            for position, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                metadata = {k: v for k, v in payload.items() if k not in {text_key, id_key}}
                chunks.append(
                    Chunk(
                        id=str(payload.get(id_key, f"chunk-{position}")),
                        text=str(payload[text_key]),
                        metadata=metadata,
                    )
                )
        return cls(chunks)

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        glob: str = "*.txt",
        encoding: str = "utf-8",
    ) -> KnowledgeBase:
        """Load every matching text file in ``directory`` as a single chunk."""
        directory = Path(directory)
        paths = sorted(directory.glob(glob))
        if not paths:
            raise FileNotFoundError(f"No files matching {glob!r} in {directory}")
        chunks = [
            Chunk(
                id=path.stem,
                text=path.read_text(encoding=encoding),
                metadata={"source": str(path)},
            )
            for path in paths
        ]
        return cls(chunks)

    def to_jsonl(self, path: str | Path) -> None:
        """Persist the knowledge base to a JSON Lines file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
