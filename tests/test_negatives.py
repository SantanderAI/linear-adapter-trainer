# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from linear_adapter_trainer.dataset.negatives import NegativeSampler


def _toy_embeddings():
    # Two clusters: 0,1 near each other; 2,3 opposite direction.
    return np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ],
        dtype=np.float32,
    )


def test_semantic_opposite_picks_far_chunk():
    sampler = NegativeSampler(_toy_embeddings(), strategy="semantic_opposite", pool_size=1, seed=1)
    ((index, strategy),) = sampler.sample(0, 1)
    assert strategy == "semantic_opposite"
    assert index in {2, 3}


def test_hard_picks_near_chunk():
    sampler = NegativeSampler(_toy_embeddings(), strategy="hard", pool_size=1, seed=1)
    ((index, strategy),) = sampler.sample(0, 1)
    assert strategy == "hard"
    assert index == 1


def test_random_excludes_positive():
    sampler = NegativeSampler(_toy_embeddings(), strategy="random", seed=3)
    for _ in range(10):
        ((index, strategy),) = sampler.sample(0, 1)
        assert strategy == "random"
        assert index != 0


def test_negatives_are_distinct_within_query():
    sampler = NegativeSampler(_toy_embeddings(), strategy="random", seed=0)
    samples = sampler.sample(0, 3)
    indices = [i for i, _ in samples]
    assert len(set(indices)) == len(indices)
    assert 0 not in indices


def test_mixed_strategy_normalizes_weights():
    sampler = NegativeSampler(
        _toy_embeddings(),
        strategy="mixed",
        mix={"semantic_opposite": 2.0, "random": 2.0},
        seed=0,
    )
    assert sampler.mix is not None
    assert abs(sum(sampler.mix.values()) - 1.0) < 1e-9


def test_invalid_strategy():
    with pytest.raises(ValueError):
        NegativeSampler(_toy_embeddings(), strategy="nope")
