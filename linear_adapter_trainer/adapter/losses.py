# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Triplet margin loss for embedding alignment.

The loss pulls the (adapted) anchor towards the positive and pushes it away
from the negative::

    L = max(0, d(a, p) - d(a, n) + margin)

Two distance functions are supported: cosine distance (``1 - cos_sim``) and
squared/plain Euclidean distance.
"""

from __future__ import annotations

import torch
from torch import nn

_DISTANCES = ("cosine", "euclidean")


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.nn.functional.cosine_similarity(a, b, dim=-1)


def _euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.norm(a - b, p=2, dim=-1)


class TripletLoss(nn.Module):
    """Margin-based triplet loss with a configurable distance.

    Args:
        margin: Desired separation between positive and negative distances.
        distance: ``"cosine"`` or ``"euclidean"``.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.
    """

    def __init__(
        self,
        margin: float = 0.2,
        *,
        distance: str = "cosine",
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if distance not in _DISTANCES:
            raise ValueError(f"distance must be one of {_DISTANCES}, got {distance!r}.")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'.")
        self.margin = margin
        self.distance = distance
        self.reduction = reduction
        self._dist_fn = _cosine_distance if distance == "cosine" else _euclidean_distance

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        pos_dist = self._dist_fn(anchor, positive)
        neg_dist = self._dist_fn(anchor, negative)
        losses = torch.relu(pos_dist - neg_dist + self.margin)
        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        return losses


class CosineSimilarityLoss(nn.Module):
    """Loss that maximizes cosine similarity between anchor a query and a chunk."""

    def __init__(
        self,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'.")
        self.reduction = reduction

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
    ) -> torch.Tensor:
        similarities = torch.nn.functional.cosine_similarity(anchor, positive, dim=-1)
        losses = 1.0 - similarities
        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        return losses
