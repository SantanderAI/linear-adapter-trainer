# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Training loop for the linear embedding adapter (Module 2).

The trainer precomputes all embeddings once, optimizes the adapter with a
triplet margin loss, and monitors a retrieval metric (default: MRR) on the
validation split for early stopping. It returns the best adapter together with
baseline (un-adapted) and adapted metrics so improvements are easy to report.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..dataset.schema import Triplet, TripletDataset
from ..embeddings.base import EmbeddingModel, l2_normalize
from ..evaluation.metrics import evaluate_rankings
from ..knowledge_base.base import KnowledgeBase
from .data import (
    TripletEmbeddingDataset,
    build_embedding_bundle,
    embed_negatives,
    embed_queries,
)
from .losses import TripletLoss
from .model import AdapterConfig, LinearAdapter


@dataclass(slots=True)
class TrainingConfig:
    """Hyper-parameters for :class:`AdapterTrainer`.

    Attributes:
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        weight_decay: L2 regularization on adapter weights.
        margin: Triplet-loss margin.
        distance: Distance for the triplet loss (``cosine``/``euclidean``).
        residual: Use a residual adapter initialized near identity.
        normalize_output: L2-normalize adapter outputs.
        eval_ks: Cut-offs for validation metrics.
        monitor: Validation metric used for model selection (e.g. ``mrr``).
        patience: Early-stopping patience in epochs (0 disables).
        grad_clip: Optional gradient-norm clip value.
        device: Torch device (``cpu``/``cuda``/``mps``); auto-detected if None.
        seed: Seed for reproducible training.
    """

    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    margin: float = 0.2
    distance: str = "cosine"
    residual: bool = True
    normalize_output: bool = True
    eval_ks: tuple[int, ...] = (1, 3, 5, 10)
    monitor: str = "mrr"
    patience: int = 5
    grad_clip: float | None = 1.0
    device: str | None = None
    seed: int = 0


@dataclass(slots=True)
class TrainResult:
    """Outcome of :meth:`AdapterTrainer.fit`."""

    adapter: LinearAdapter
    baseline_metrics: dict[str, float]
    best_metrics: dict[str, float]
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def improvement(self) -> dict[str, float]:
        return {
            key: self.best_metrics[key] - self.baseline_metrics[key]
            for key in self.best_metrics
            if key != "n_queries" and key in self.baseline_metrics
        }


class AdapterTrainer:
    """Train a :class:`LinearAdapter` from a generated triplet dataset.

    Args:
        knowledge_base: Corpus used for validation retrieval.
        embedder: Embedding backend (must match dataset generation).
        config: Training hyper-parameters.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        embedder: EmbeddingModel,
        config: TrainingConfig | None = None,
    ) -> None:
        self.kb = knowledge_base
        self.embedder = embedder
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device or _auto_device())

    def fit(self, dataset: TripletDataset, *, verbose: bool = True) -> TrainResult:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        if not dataset.train:
            raise ValueError("Training split is empty; generate a larger dataset.")

        bundle = build_embedding_bundle(self.kb, self.embedder)
        chunk_norm = l2_normalize(bundle.chunk_matrix.numpy())

        train_anchors = embed_queries(dataset.train, self.embedder)
        train_negatives = embed_negatives(dataset.train, self.embedder, bundle)
        train_ds = TripletEmbeddingDataset(
            dataset.train, train_anchors, bundle, negatives=train_negatives
        )
        loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

        val_queries, val_relevant, val_query_matrix = self._prepare_validation(dataset.val)

        adapter = LinearAdapter(
            AdapterConfig(
                input_dim=bundle.dim,
                residual=cfg.residual,
                normalize_output=cfg.normalize_output,
            )
        ).to(self.device)

        baseline_metrics = self._retrieval_metrics(None, val_query_matrix, val_relevant, chunk_norm)

        criterion = TripletLoss(margin=cfg.margin, distance=cfg.distance)
        optimizer = torch.optim.Adam(
            adapter.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        # The untrained residual adapter is a no-op, so the identity baseline is
        # itself a valid candidate: model selection starts from it and only
        # accepts epochs that strictly improve on the base embeddings. This
        # guarantees the returned adapter never degrades retrieval.
        best_score = baseline_metrics.get(cfg.monitor, -float("inf"))
        best_state = copy.deepcopy(adapter.state_dict())
        best_metrics = baseline_metrics
        epochs_without_improvement = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, cfg.epochs + 1):
            train_loss = self._train_epoch(adapter, loader, criterion, optimizer)
            val_metrics = self._retrieval_metrics(
                adapter, val_query_matrix, val_relevant, chunk_norm
            )
            score = val_metrics.get(cfg.monitor, 0.0)
            record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
            history.append(record)

            if verbose:
                print(f"epoch {epoch:>3} | loss {train_loss:.4f} | " f"{cfg.monitor} {score:.4f}")

            if score > best_score + 1e-6:
                best_score = score
                best_state = copy.deepcopy(adapter.state_dict())
                best_metrics = val_metrics
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if cfg.patience and epochs_without_improvement >= cfg.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}.")
                    break

        adapter.load_state_dict(best_state)
        adapter.eval()
        return TrainResult(
            adapter=adapter,
            baseline_metrics=baseline_metrics,
            best_metrics=best_metrics,
            history=history,
        )

    # -- training internals ------------------------------------------------
    def _train_epoch(self, adapter, loader, criterion, optimizer) -> float:
        adapter.train()
        total = 0.0
        count = 0
        for anchor, positive, negative in loader:
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            optimizer.zero_grad()
            adapted = adapter(anchor)
            loss = criterion(adapted, positive, negative)
            loss.backward()
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), self.config.grad_clip)
            optimizer.step()

            total += loss.item() * anchor.shape[0]
            count += anchor.shape[0]
        return total / max(count, 1)

    def _prepare_validation(
        self, val_triplets: list[Triplet]
    ) -> tuple[list[str], list[set[str]], np.ndarray]:
        if not val_triplets:
            return [], [], np.empty((0, self.embedder.dimension), dtype=np.float32)
        relevant_by_query: dict[str, set[str]] = defaultdict(set)
        for triplet in val_triplets:
            relevant_by_query[triplet.query].add(triplet.positive_id)
        queries = list(relevant_by_query)
        relevant = [relevant_by_query[q] for q in queries]
        matrix = l2_normalize(self.embedder.embed(queries))
        return queries, relevant, matrix

    def _retrieval_metrics(
        self,
        adapter: LinearAdapter | None,
        query_matrix: np.ndarray,
        relevant: list[set[str]],
        chunk_norm: np.ndarray,
    ) -> dict[str, float]:
        if query_matrix.shape[0] == 0:
            return {"mrr": 0.0, "n_queries": 0.0}
        queries = query_matrix
        if adapter is not None:
            queries = l2_normalize(adapter.transform(query_matrix))
        scores = queries @ chunk_norm.T
        chunk_ids = self.kb.ids
        max_k = min(max(self.config.eval_ks), len(chunk_ids))
        top_unsorted = np.argpartition(-scores, kth=max_k - 1, axis=1)[:, :max_k]
        rankings = []
        for row in range(scores.shape[0]):
            cols = top_unsorted[row]
            order = cols[np.argsort(-scores[row, cols])]
            rankings.append(([chunk_ids[c] for c in order], relevant[row]))
        return evaluate_rankings(rankings, ks=self.config.eval_ks)


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
