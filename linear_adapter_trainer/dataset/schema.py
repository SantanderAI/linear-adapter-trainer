# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Data structures for the generated triplet dataset.

A training example is a *(anchor, positive, negative)* triplet expressed in
terms of the query text and the ids of the positive/negative chunks. Keeping
ids (instead of raw vectors) makes the dataset portable and reproducible: the
embeddings are recomputed at training time from the chosen backend.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Triplet:
    """A single training triplet.

    Attributes:
        query: The anchor text (a query about the positive chunk).
        positive_id: Id of the chunk the query was generated from.
        negative_id: Id of the contrasting (irrelevant) chunk.
        negative_text: Raw text of the negative when it is *not* a corpus chunk
            (e.g. an LLM-generated hard negative). When non-empty, training
            embeds this text directly instead of looking ``negative_id`` up in
            the knowledge base.
        strategy: Negative-mining strategy that produced the negative.
    """

    query: str
    positive_id: str
    negative_id: str
    negative_text: str = ""
    strategy: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Triplet:
        return cls(
            query=str(payload["query"]),
            positive_id=str(payload["positive_id"]),
            negative_id=str(payload["negative_id"]),
            negative_text=str(payload.get("negative_text", "")),
            strategy=str(payload.get("strategy", "unknown")),
        )


@dataclass(slots=True)
class TripletDataset:
    """A train/validation split of triplets plus provenance metadata."""

    train: list[Triplet] = field(default_factory=list)
    val: list[Triplet] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.train) + len(self.val)

    @property
    def splits(self) -> dict[str, list[Triplet]]:
        return {"train": self.train, "val": self.val}

    def iter_split(self, split: str) -> Iterator[Triplet]:
        if split not in {"train", "val"}:
            raise ValueError(f"Unknown split: {split!r}")
        return iter(self.train if split == "train" else self.val)

    # -- persistence -------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        """Persist the dataset to ``train.jsonl``, ``val.jsonl`` and ``meta.json``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _write_jsonl(directory / "train.jsonl", self.train)
        _write_jsonl(directory / "val.jsonl", self.val)
        (directory / "meta.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> TripletDataset:
        directory = Path(directory)
        meta_path = directory / "meta.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(
            train=_read_jsonl(directory / "train.jsonl"),
            val=_read_jsonl(directory / "val.jsonl"),
            metadata=metadata,
        )


def _write_jsonl(path: Path, triplets: Sequence[Triplet]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for triplet in triplets:
            handle.write(json.dumps(triplet.to_dict(), ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[Triplet]:
    if not path.exists():
        return []
    triplets: list[Triplet] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                triplets.append(Triplet.from_dict(json.loads(line)))
    return triplets


def triplets_to_dicts(triplets: Iterable[Triplet]) -> list[dict[str, Any]]:
    return [triplet.to_dict() for triplet in triplets]
