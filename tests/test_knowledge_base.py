# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from linear_adapter_trainer.knowledge_base import Chunk, KnowledgeBase, TextSplitter


def test_chunk_validation():
    with pytest.raises(ValueError):
        Chunk(id="", text="hello")
    with pytest.raises(ValueError):
        Chunk(id="x", text="   ")


def test_knowledge_base_lookup_and_unique_ids():
    kb = KnowledgeBase.from_texts(["alpha text", "beta text"], ids=["a", "b"])
    assert len(kb) == 2
    assert kb.get("a").text == "alpha text"
    assert kb.position_of("b") == 1
    assert kb.ids == ["a", "b"]
    with pytest.raises(KeyError):
        kb.get("missing")


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        KnowledgeBase([Chunk("a", "x"), Chunk("a", "y")])


def test_empty_kb_rejected():
    with pytest.raises(ValueError):
        KnowledgeBase([])


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "kb.jsonl"
    rows = [
        {"id": "1", "text": "first", "topic": "x"},
        {"id": "2", "text": "second", "topic": "y"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    kb = KnowledgeBase.from_jsonl(path)
    assert len(kb) == 2
    assert kb.get("1").metadata["topic"] == "x"

    out = tmp_path / "out.jsonl"
    kb.to_jsonl(out)
    reloaded = KnowledgeBase.from_jsonl(out)
    assert reloaded.texts == kb.texts


def test_text_splitter_overlap():
    splitter = TextSplitter(chunk_size=40, chunk_overlap=10)
    text = "Sentence one is here. Sentence two follows. Sentence three ends it all now."
    pieces = splitter.split_text(text)
    assert len(pieces) >= 2
    assert all(len(p) <= 60 for p in pieces)


def test_text_splitter_invalid_overlap():
    with pytest.raises(ValueError):
        TextSplitter(chunk_size=10, chunk_overlap=10)
