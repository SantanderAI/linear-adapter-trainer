# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""LinearAdapterTrainer.

Train linear embedding adapters with triplet loss to align retrieval
embeddings with your queries.

Two modules:

* ``dataset`` - generate ``(query, positive, negative)`` triplets from a
  knowledge base, with configurable negative mining and a leakage-free
  train/val split.
* ``adapter`` - train a linear adapter over query embeddings with triplet
  loss so relevant chunks move closer and irrelevant ones move away.

Plus an ``evaluation`` package with precision@k, recall@k, MRR, nDCG and a
base-vs-adapted comparison.
"""

from __future__ import annotations

from .adapter import (
    AdapterConfig,
    AdapterTrainer,
    LinearAdapter,
    TrainingConfig,
    TrainResult,
    TripletLoss,
)
from .dataset import (
    DatasetConfig,
    DatasetGenerator,
    LLMNegativeGenerator,
    LLMQueryGenerator,
    NegativeSampler,
    TemplateQueryGenerator,
    Triplet,
    TripletDataset,
)
from .embeddings import EmbeddingModel, HashingEmbedder
from .evaluation import RetrievalEvaluator, evaluate_rankings
from .knowledge_base import Chunk, KnowledgeBase, LinkupWebLoader, TextSplitter

__version__ = "0.1.0"

__all__ = [
    "AdapterConfig",
    "AdapterTrainer",
    "Chunk",
    "DatasetConfig",
    "DatasetGenerator",
    "EmbeddingModel",
    "HashingEmbedder",
    "KnowledgeBase",
    "LLMNegativeGenerator",
    "LLMQueryGenerator",
    "LinkupWebLoader",
    "LinearAdapter",
    "NegativeSampler",
    "RetrievalEvaluator",
    "TemplateQueryGenerator",
    "TextSplitter",
    "TrainResult",
    "TrainingConfig",
    "Triplet",
    "TripletDataset",
    "TripletLoss",
    "evaluate_rankings",
    "__version__",
]
