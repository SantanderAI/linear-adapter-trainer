# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""End-to-end quickstart for LinearAdapterTrainer.

Run it with::

    uv run python examples/quickstart.py

By default it uses Sentence-Transformers if installed, falling back to the
dependency-free HashingEmbedder so the script always runs.
"""

from __future__ import annotations

from pathlib import Path

from linear_adapter_trainer import (
    AdapterTrainer,
    DatasetConfig,
    DatasetGenerator,
    KnowledgeBase,
    TemplateQueryGenerator,
    TrainingConfig,
)

HERE = Path(__file__).resolve().parent
KB_PATH = HERE / "data" / "sample_kb.jsonl"


def build_embedder():
    """Prefer a real semantic model; gracefully fall back for offline use."""
    try:
        from linear_adapter_trainer.embeddings import SentenceTransformerEmbedder

        print("Using SentenceTransformerEmbedder (all-MiniLM-L6-v2).")
        return SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:  # noqa: BLE001 - offline / not installed
        from linear_adapter_trainer import HashingEmbedder

        print("Sentence-Transformers unavailable; using HashingEmbedder (demo only).")
        return HashingEmbedder(dimension=512)


def main() -> None:
    kb = KnowledgeBase.from_jsonl(KB_PATH)
    embedder = build_embedder()

    # Module 1: generate a triplet dataset with mixed negatives.
    generator = DatasetGenerator(
        knowledge_base=kb,
        embedder=embedder,
        query_generator=TemplateQueryGenerator(seed=0),
        config=DatasetConfig(
            queries_per_chunk=4,
            negatives_per_query=2,
            strategy="mixed",
            mix={"semantic_opposite": 0.5, "hard": 0.3, "random": 0.2},
            val_fraction=0.25,
            seed=0,
        ),
    )
    dataset = generator.generate()
    print(f"Generated {len(dataset.train)} train / {len(dataset.val)} val triplets.\n")

    # Module 2: train the linear adapter with triplet loss.
    trainer = AdapterTrainer(
        kb,
        embedder,
        TrainingConfig(epochs=30, learning_rate=5e-3, margin=0.2, monitor="mrr"),
    )
    result = trainer.fit(dataset)

    print("\nValidation metrics (base -> adapted):")
    for key in ("mrr", "precision@1", "precision@5", "recall@5", "ndcg@10"):
        base = result.baseline_metrics.get(key, 0.0)
        adapted = result.best_metrics.get(key, 0.0)
        print(f"  {key:<14} {base:.4f} -> {adapted:.4f}  ({adapted - base:+.4f})")


if __name__ == "__main__":
    main()
