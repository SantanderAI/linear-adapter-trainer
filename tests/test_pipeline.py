# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from linear_adapter_trainer import (
    AdapterTrainer,
    DatasetConfig,
    DatasetGenerator,
    HashingEmbedder,
    KnowledgeBase,
    LinearAdapter,
    RetrievalEvaluator,
    TemplateQueryGenerator,
    TrainingConfig,
    TripletDataset,
)

CORPUS = [
    "Photosynthesis lets plants convert sunlight into chemical energy and glucose.",
    "Black holes are regions of spacetime where gravity prevents light from escaping.",
    "The Roman Empire controlled vast territory across Europe and the Mediterranean.",
    "TCP/IP protocols provide reliable, ordered packet delivery across the internet.",
    "Espresso is concentrated coffee brewed by forcing hot water through fine grounds.",
    "Vaccines train the immune system to recognize and fight specific pathogens.",
    "A blockchain is a distributed ledger of cryptographically linked transaction blocks.",
    "Volcanoes release lava, ash, and gases from a magma chamber below the crust.",
]


def _kb():
    return KnowledgeBase.from_texts(CORPUS, ids=[f"c{i}" for i in range(len(CORPUS))])


def _dataset(kb):
    generator = DatasetGenerator(
        knowledge_base=kb,
        embedder=HashingEmbedder(dimension=256),
        query_generator=TemplateQueryGenerator(seed=0),
        config=DatasetConfig(
            queries_per_chunk=3,
            negatives_per_query=2,
            strategy="mixed",
            mix={"semantic_opposite": 0.5, "hard": 0.3, "random": 0.2},
            val_fraction=0.25,
            seed=0,
        ),
    )
    return generator.generate(show_progress=False)


def test_parallel_generation_matches_sequential():
    # Concurrency must not change the output: triplets are assembled in chunk
    # order regardless of how many workers run query generation.
    kb = _kb()

    def build(max_workers):
        return DatasetGenerator(
            knowledge_base=kb,
            embedder=HashingEmbedder(dimension=256),
            query_generator=TemplateQueryGenerator(seed=0),
            config=DatasetConfig(
                queries_per_chunk=3,
                negatives_per_query=2,
                strategy="mixed",
                mix={"semantic_opposite": 0.5, "hard": 0.3, "random": 0.2},
                val_fraction=0.25,
                seed=0,
                max_workers=max_workers,
            ),
        ).generate(show_progress=False)

    sequential = build(1)
    parallel = build(8)

    def as_tuples(triplets):
        return [(t.query, t.positive_id, t.negative_id, t.strategy) for t in triplets]

    assert as_tuples(sequential.train) == as_tuples(parallel.train)
    assert as_tuples(sequential.val) == as_tuples(parallel.val)


def test_dataset_generation_structure():
    kb = _kb()
    dataset = _dataset(kb)
    assert len(dataset.train) > 0
    assert len(dataset.val) > 0
    for triplet in dataset.train:
        assert triplet.positive_id != triplet.negative_id
        assert triplet.positive_id in kb.ids
        assert triplet.negative_id in kb.ids
    assert dataset.metadata["n_chunks"] == len(kb)


def test_dataset_save_load(tmp_path):
    dataset = _dataset(_kb())
    dataset.save(tmp_path)
    reloaded = TripletDataset.load(tmp_path)
    assert len(reloaded.train) == len(dataset.train)
    assert len(reloaded.val) == len(dataset.val)
    assert reloaded.metadata["n_chunks"] == dataset.metadata["n_chunks"]


def test_training_runs_and_reports_metrics():
    kb = _kb()
    dataset = _dataset(kb)
    embedder = HashingEmbedder(dimension=256)
    trainer = AdapterTrainer(
        kb,
        embedder,
        TrainingConfig(epochs=5, batch_size=16, learning_rate=5e-3, patience=0, eval_ks=(1, 3, 5)),
    )
    result = trainer.fit(dataset, verbose=False)

    assert isinstance(result.adapter, LinearAdapter)
    assert "mrr" in result.best_metrics
    assert "mrr" in result.baseline_metrics
    assert 0.0 <= result.best_metrics["mrr"] <= 1.0
    assert len(result.history) >= 1


def test_adapter_never_degrades_baseline():
    # Model selection includes the identity baseline, so the returned adapter
    # must never score below the base embeddings on the monitored metric.
    kb = _kb()
    dataset = _dataset(kb)
    embedder = HashingEmbedder(dimension=256)
    trainer = AdapterTrainer(
        kb,
        embedder,
        TrainingConfig(epochs=8, learning_rate=5e-3, monitor="mrr", patience=0),
    )
    result = trainer.fit(dataset, verbose=False)
    assert result.best_metrics["mrr"] >= result.baseline_metrics["mrr"] - 1e-9


class _StubNegativeGenerator:
    def generate(self, positive_text: str, n: int) -> list[str]:
        head = positive_text.split()[0]
        return [f"Same-topic but wrong passage {i} regarding {head}." for i in range(n)]


def test_training_runs_with_llm_text_negatives():
    # Triplets whose negatives are free text (not corpus chunks) must train
    # end-to-end: the trainer embeds negative_text directly.
    from linear_adapter_trainer import DatasetGenerator

    kb = _kb()
    embedder = HashingEmbedder(dimension=256)
    dataset = DatasetGenerator(
        knowledge_base=kb,
        embedder=embedder,
        query_generator=TemplateQueryGenerator(seed=0),
        config=DatasetConfig(queries_per_chunk=3, negatives_per_query=2, val_fraction=0.25, seed=0),
        negative_generator=_StubNegativeGenerator(),
    ).generate(show_progress=False)

    assert all(t.negative_text for t in dataset.train)
    trainer = AdapterTrainer(
        kb, embedder, TrainingConfig(epochs=3, batch_size=16, learning_rate=5e-3, patience=0)
    )
    result = trainer.fit(dataset, verbose=False)
    assert isinstance(result.adapter, LinearAdapter)
    assert "mrr" in result.best_metrics
    assert 0.0 <= result.best_metrics["mrr"] <= 1.0


def test_evaluator_compare_keys():
    kb = _kb()
    dataset = _dataset(kb)
    embedder = HashingEmbedder(dimension=256)
    adapter = LinearAdapter.load  # sanity ref, not called
    trainer = AdapterTrainer(kb, embedder, TrainingConfig(epochs=2, patience=0))
    result = trainer.fit(dataset, verbose=False)

    evaluator = RetrievalEvaluator(kb, embedder, ks=(1, 3, 5))
    comparison = evaluator.compare(dataset.val, result.adapter)
    assert set(comparison) == {"base", "adapted", "delta"}
    assert "mrr" in comparison["delta"]
    assert np.isfinite(comparison["delta"]["mrr"])
    assert callable(adapter)
