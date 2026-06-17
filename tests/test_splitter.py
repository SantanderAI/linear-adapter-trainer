# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

import pytest

from linear_adapter_trainer.dataset.schema import Triplet
from linear_adapter_trainer.dataset.splitter import split_triplets


def _triplets():
    out = []
    for chunk in range(10):
        for q in range(3):
            out.append(
                Triplet(
                    query=f"q{chunk}-{q}",
                    positive_id=f"c{chunk}",
                    negative_id=f"c{(chunk + 1) % 10}",
                    strategy="random",
                )
            )
    return out


def test_split_is_leakage_free():
    train, val = split_triplets(_triplets(), val_fraction=0.3, seed=0)
    train_chunks = {t.positive_id for t in train}
    val_chunks = {t.positive_id for t in val}
    assert train_chunks.isdisjoint(val_chunks)
    assert train_chunks | val_chunks == {f"c{i}" for i in range(10)}


def test_split_drops_query_string_leakage():
    # Two different chunks generate the SAME query text. After splitting they
    # may land in opposite splits; the shared query must not remain in val.
    triplets = _triplets() + [
        Triplet(
            query="shared question",
            positive_id="c0",  # forced into one split
            negative_id="c1",
            strategy="random",
        ),
        Triplet(
            query="shared question",
            positive_id="c5",  # forced into the other split
            negative_id="c6",
            strategy="random",
        ),
    ]
    train, val = split_triplets(triplets, val_fraction=0.3, seed=0)
    train_queries = {t.query for t in train}
    val_queries = {t.query for t in val}
    assert train_queries.isdisjoint(val_queries)
    if "shared question" in train_queries:
        assert "shared question" not in val_queries


def test_split_is_deterministic():
    a = split_triplets(_triplets(), val_fraction=0.2, seed=42)
    b = split_triplets(_triplets(), val_fraction=0.2, seed=42)
    assert [t.query for t in a[0]] == [t.query for t in b[0]]


def test_zero_val_fraction():
    train, val = split_triplets(_triplets(), val_fraction=0.0, seed=0)
    assert len(val) == 0
    assert len(train) == 30


def test_query_split_holds_out_queries_but_keeps_corpus():
    # With strategy="query" the queries are disjoint across splits, but every
    # positive chunk stays visible in training (the corpus is fixed/known).
    train, val = split_triplets(_triplets(), val_fraction=0.3, seed=0, strategy="query")
    train_queries = {t.query for t in train}
    val_queries = {t.query for t in val}
    assert val_queries  # non-empty holdout
    assert train_queries.isdisjoint(val_queries)
    # No positive chunk is exclusive to validation: all are seen in training.
    assert {t.positive_id for t in val} <= {t.positive_id for t in train}


def test_invalid_split_strategy():
    with pytest.raises(ValueError):
        split_triplets(_triplets(), strategy="nope")
