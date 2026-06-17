# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Module 1: triplet dataset generation from a knowledge base."""

from .generator import DatasetConfig, DatasetGenerator
from .negative_generation import (
    LLMNegativeGenerator,
    NegativeGenerator,
    resolve_negative_generator,
)
from .negatives import STRATEGIES, NegativeSampler
from .query_generation import (
    LLMQueryGenerator,
    QueryGenerator,
    TemplateQueryGenerator,
    resolve_generator,
)
from .schema import Triplet, TripletDataset
from .splitter import split_triplets

__all__ = [
    "STRATEGIES",
    "DatasetConfig",
    "DatasetGenerator",
    "LLMNegativeGenerator",
    "LLMQueryGenerator",
    "NegativeGenerator",
    "NegativeSampler",
    "QueryGenerator",
    "TemplateQueryGenerator",
    "Triplet",
    "TripletDataset",
    "resolve_generator",
    "resolve_negative_generator",
    "split_triplets",
]
