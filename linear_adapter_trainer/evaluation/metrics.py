# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Information-retrieval metrics.

All functions operate on a ranked list of candidate ids (ordered by decreasing
score) and a set of relevant ids. Aggregate helpers average per-query metrics
over an evaluation set.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-k results that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive.")
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant items retrieved within the top-k."""
    if not relevant_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def hit_rate_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if any relevant item appears in the top-k, else 0.0."""
    return 1.0 if any(cid in relevant_ids for cid in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """Reciprocal of the rank of the first relevant item (0 if none)."""
    for position, cid in enumerate(ranked_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Normalized discounted cumulative gain with binary relevance."""
    dcg = 0.0
    for position, cid in enumerate(ranked_ids[:k], start=1):
        if cid in relevant_ids:
            dcg += 1.0 / math.log2(position + 1)
    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg


def evaluate_rankings(
    rankings: Iterable[tuple[Sequence[str], set[str]]],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Average IR metrics over an iterable of ``(ranked_ids, relevant_ids)``.

    Returns a flat dict with keys such as ``precision@5``, ``recall@10``,
    ``ndcg@5``, ``hit_rate@1`` and ``mrr``.
    """
    rankings = list(rankings)
    if not rankings:
        return {"mrr": 0.0, "n_queries": 0.0}

    sums: dict[str, float] = {}
    for key in (f"{m}@{k}" for k in ks for m in ("precision", "recall", "hit_rate", "ndcg")):
        sums[key] = 0.0
    sums["mrr"] = 0.0

    for ranked_ids, relevant_ids in rankings:
        for k in ks:
            sums[f"precision@{k}"] += precision_at_k(ranked_ids, relevant_ids, k)
            sums[f"recall@{k}"] += recall_at_k(ranked_ids, relevant_ids, k)
            sums[f"hit_rate@{k}"] += hit_rate_at_k(ranked_ids, relevant_ids, k)
            sums[f"ndcg@{k}"] += ndcg_at_k(ranked_ids, relevant_ids, k)
        sums["mrr"] += reciprocal_rank(ranked_ids, relevant_ids)

    n = len(rankings)
    metrics = {key: value / n for key, value in sums.items()}
    metrics["n_queries"] = float(n)
    return metrics
