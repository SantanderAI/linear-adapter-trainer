# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Train/validation splitting with two leakage-free strategies.

Which strategy is "correct" depends on what the adapter must generalize to:

* ``"chunk"`` (default) groups by *positive chunk*: every triplet whose positive
  is a given chunk goes entirely to one split. Validation therefore measures
  generalization to **unseen corpus chunks** - a strict test, but a poor match
  for a query-side adapter, whose corpus (index) is fixed and fully known at
  training time.
* ``"query"`` groups by *query text*: a fraction of the unique queries is held
  out, while the corpus stays fully visible during training. This measures
  generalization to **new queries over a fixed corpus**, which is exactly the
  deployment scenario for a query-side adapter. It is still leakage-free because
  the held-out query strings are never seen during training.

The ``"chunk"`` strategy additionally drops any val query whose text also
appears in train (generic templates / LLM paraphrases can repeat across
chunks), so a validation anchor is never one that was optimized in training.
The ``"query"`` strategy guarantees this by construction.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from .schema import Triplet

SPLIT_STRATEGIES: tuple[str, ...] = ("chunk", "query")


def split_triplets(
    triplets: Sequence[Triplet],
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
    strategy: str = "chunk",
) -> tuple[list[Triplet], list[Triplet]]:
    """Split triplets into ``(train, val)``.

    Args:
        triplets: The full list of generated triplets.
        val_fraction: Target fraction of *groups* allocated to validation.
        seed: Seed for the reproducible shuffle of groups.
        strategy: ``"chunk"`` groups by ``positive_id`` (generalize to unseen
            chunks); ``"query"`` groups by ``query`` text (generalize to unseen
            queries over a fixed corpus). See module docstring.

    Returns:
        A ``(train, val)`` tuple of triplet lists.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"Unknown split strategy {strategy!r}. Choose from {SPLIT_STRATEGIES}.")

    key = (lambda t: t.positive_id) if strategy == "chunk" else (lambda t: t.query)
    groups: dict[str, list[Triplet]] = defaultdict(list)
    for triplet in triplets:
        groups[key(triplet)].append(triplet)

    group_ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n_val = int(round(len(group_ids) * val_fraction))
    if val_fraction > 0.0 and len(group_ids) > 1:
        n_val = max(1, n_val)
    val_ids = set(group_ids[:n_val])

    train: list[Triplet] = []
    val: list[Triplet] = []
    for group_id in group_ids:
        bucket = val if group_id in val_ids else train
        bucket.extend(groups[group_id])

    if strategy == "chunk":
        # De-contaminate: a query string can be generated for two chunks; drop
        # val triplets whose query also appears in train so anchors stay unseen.
        train_queries = {triplet.query for triplet in train}
        val = [triplet for triplet in val if triplet.query not in train_queries]

    return train, val
