# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Retrieval evaluation: metrics and evaluator."""

from .evaluator import RetrievalEvaluator
from .metrics import (
    evaluate_rankings,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "RetrievalEvaluator",
    "evaluate_rankings",
    "hit_rate_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
