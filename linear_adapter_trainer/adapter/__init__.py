# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Module 2: the trainable linear embedding adapter."""

from .data import (
    EmbeddingBundle,
    TripletEmbeddingDataset,
    build_embedding_bundle,
    embed_negatives,
    embed_queries,
)
from .losses import TripletLoss
from .model import AdapterConfig, LinearAdapter
from .trainer import AdapterTrainer, TrainingConfig, TrainResult

__all__ = [
    "AdapterConfig",
    "AdapterTrainer",
    "EmbeddingBundle",
    "LinearAdapter",
    "TrainResult",
    "TrainingConfig",
    "TripletEmbeddingDataset",
    "TripletLoss",
    "build_embedding_bundle",
    "embed_negatives",
    "embed_queries",
]
