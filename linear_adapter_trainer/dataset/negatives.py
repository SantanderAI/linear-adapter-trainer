# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Negative-mining strategies for triplet construction.

Given the embedding of a positive chunk, we need a contrasting *negative*
chunk. Three strategies are supported:

* ``random`` - a uniformly sampled chunk (easy negative).
* ``semantic_opposite`` - one of the least similar chunks (far in latent
  space); this is what the user calls "semantically opposite".
* ``hard`` - one of the *most* similar (but still incorrect) chunks; the
  classic hard-negative that makes triplet training effective.

Strategies can also be ``mixed`` according to user-provided weights.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from ..embeddings.base import l2_normalize

STRATEGIES: tuple[str, ...] = ("random", "semantic_opposite", "hard")


@dataclass(slots=True)
class NegativeSampler:
    """Sample negative chunks for a given positive index.

    Args:
        embeddings: ``(n_chunks, dim)`` matrix; L2-normalized on init.
        strategy: One of ``random``, ``semantic_opposite``, ``hard``, ``mixed``.
        pool_size: Candidate pool size for ``semantic_opposite`` / ``hard``.
        mix: Weights per strategy used when ``strategy == "mixed"``.
        seed: Seed for reproducible sampling.
    """

    embeddings: np.ndarray
    strategy: str = "semantic_opposite"
    pool_size: int = 10
    mix: dict[str, float] | None = None
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.embeddings = l2_normalize(self.embeddings)
        self._rng = random.Random(self.seed)
        if self.strategy not in (*STRATEGIES, "mixed"):
            raise ValueError(
                f"Unknown strategy {self.strategy!r}. " f"Choose from {(*STRATEGIES, 'mixed')}."
            )
        if self.strategy == "mixed":
            weights = self.mix or {"semantic_opposite": 0.5, "hard": 0.3, "random": 0.2}
            unknown = set(weights) - set(STRATEGIES)
            if unknown:
                raise ValueError(f"Unknown strategies in mix: {sorted(unknown)}")
            total = sum(weights.values())
            if total <= 0:
                raise ValueError("Mix weights must sum to a positive number.")
            self.mix = {k: v / total for k, v in weights.items()}

    @property
    def n_chunks(self) -> int:
        return self.embeddings.shape[0]

    def sample(self, positive_index: int, n_negatives: int = 1) -> list[tuple[int, str]]:
        """Return ``n_negatives`` ``(index, strategy)`` pairs for a positive."""
        if self.n_chunks < 2:
            raise ValueError("Need at least 2 chunks to mine negatives.")
        similarities = self.embeddings @ self.embeddings[positive_index]
        chosen: list[tuple[int, str]] = []
        used: set[int] = {positive_index}
        for _ in range(n_negatives):
            strategy = self._pick_strategy()
            index = self._sample_one(positive_index, similarities, strategy, used)
            used.add(index)
            chosen.append((index, strategy))
        return chosen

    # -- internals ---------------------------------------------------------
    def _pick_strategy(self) -> str:
        if self.strategy != "mixed":
            return self.strategy
        assert self.mix is not None
        names = list(self.mix)
        weights = [self.mix[name] for name in names]
        return self._rng.choices(names, weights=weights, k=1)[0]

    def _sample_one(
        self,
        positive_index: int,
        similarities: np.ndarray,
        strategy: str,
        used: set[int],
    ) -> int:
        if strategy == "random":
            return self._sample_random(used)

        order = np.argsort(similarities)  # ascending similarity
        if strategy == "hard":
            order = order[::-1]  # descending: most similar first

        pool: list[int] = []
        for candidate in order:
            idx = int(candidate)
            if idx in used:
                continue
            pool.append(idx)
            if len(pool) >= self.pool_size:
                break
        if not pool:
            return self._sample_random(used)
        return self._rng.choice(pool)

    def _sample_random(self, used: set[int]) -> int:
        candidates = [i for i in range(self.n_chunks) if i not in used]
        if not candidates:
            candidates = [i for i in range(self.n_chunks) if i not in {min(used)}]
        return self._rng.choice(candidates)
