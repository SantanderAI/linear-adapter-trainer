# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

import math

from linear_adapter_trainer.evaluation.metrics import (
    evaluate_rankings,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["a", "b", "c", "d"]


def test_precision_at_k():
    assert precision_at_k(RANKED, {"a"}, 1) == 1.0
    assert precision_at_k(RANKED, {"c"}, 2) == 0.0
    assert precision_at_k(RANKED, {"a", "b"}, 2) == 1.0
    assert precision_at_k(RANKED, {"a", "c"}, 4) == 0.5


def test_recall_at_k():
    assert recall_at_k(RANKED, {"a", "c"}, 1) == 0.5
    assert recall_at_k(RANKED, {"a", "c"}, 3) == 1.0
    assert recall_at_k(RANKED, set(), 3) == 0.0


def test_hit_rate():
    assert hit_rate_at_k(RANKED, {"d"}, 4) == 1.0
    assert hit_rate_at_k(RANKED, {"d"}, 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, {"a"}) == 1.0
    assert reciprocal_rank(RANKED, {"c"}) == 1.0 / 3.0
    assert reciprocal_rank(RANKED, {"z"}) == 0.0


def test_ndcg():
    assert ndcg_at_k(RANKED, {"a"}, 4) == 1.0
    # relevant at rank 2 => dcg = 1/log2(3), idcg = 1
    expected = (1.0 / math.log2(3)) / 1.0
    assert math.isclose(ndcg_at_k(RANKED, {"b"}, 4), expected, rel_tol=1e-9)


def test_evaluate_rankings_aggregate():
    rankings = [
        (["a", "b", "c"], {"a"}),
        (["x", "y", "z"], {"z"}),
    ]
    metrics = evaluate_rankings(rankings, ks=(1, 3))
    assert metrics["n_queries"] == 2.0
    assert math.isclose(metrics["mrr"], (1.0 + 1.0 / 3.0) / 2.0)
    assert math.isclose(metrics["hit_rate@3"], 1.0)
    assert math.isclose(metrics["precision@1"], 0.5)


def test_evaluate_rankings_empty():
    assert evaluate_rankings([])["n_queries"] == 0.0
