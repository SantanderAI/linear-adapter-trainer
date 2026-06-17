# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from linear_adapter_trainer import DatasetConfig, DatasetGenerator, HashingEmbedder, KnowledgeBase
from linear_adapter_trainer.adapter.data import build_embedding_bundle, embed_negatives
from linear_adapter_trainer.dataset.negative_generation import LLMNegativeGenerator
from linear_adapter_trainer.dataset.query_generation import TemplateQueryGenerator
from linear_adapter_trainer.dataset.schema import Triplet

CORPUS = [
    "Photosynthesis lets plants convert sunlight into chemical energy and glucose.",
    "Black holes are regions of spacetime where gravity prevents light from escaping.",
    "The Roman Empire controlled vast territory across Europe and the Mediterranean.",
    "TCP/IP protocols provide reliable, ordered packet delivery across the internet.",
]


def _kb():
    return KnowledgeBase.from_texts(CORPUS, ids=[f"c{i}" for i in range(len(CORPUS))])


def test_llm_negative_parser_json_array():
    parsed = LLMNegativeGenerator._parse('Sure:\n["A wrong passage.", "Another one."]', 5)
    assert parsed == ["A wrong passage.", "Another one."]


def test_llm_negative_parser_bullet_fallback():
    parsed = LLMNegativeGenerator._parse("1. First negative.\n2. Second negative.", 5)
    assert parsed == ["First negative.", "Second negative."]


class _StubNegativeGenerator:
    """Offline stand-in for an LLM negative generator (deterministic)."""

    def generate(self, positive_text: str, n: int) -> list[str]:
        head = positive_text.split()[0]
        return [f"Unrelated but same-topic passage {i} about {head}." for i in range(n)]


def test_generator_uses_llm_text_negatives():
    kb = _kb()
    dataset = DatasetGenerator(
        knowledge_base=kb,
        embedder=HashingEmbedder(dimension=128),
        query_generator=TemplateQueryGenerator(seed=0),
        config=DatasetConfig(queries_per_chunk=2, negatives_per_query=2, val_fraction=0.25, seed=0),
        negative_generator=_StubNegativeGenerator(),
    ).generate(show_progress=False)

    assert dataset.metadata["negative_backend"] == "llm"
    all_triplets = dataset.train + dataset.val
    assert all_triplets
    for triplet in all_triplets:
        assert triplet.strategy == "llm_hard"
        assert triplet.negative_text  # carries free text, not a corpus id
        assert triplet.negative_id not in kb.ids
        assert triplet.positive_id in kb.ids


def test_embed_negatives_returns_none_for_corpus_negatives():
    kb = _kb()
    embedder = HashingEmbedder(dimension=128)
    bundle = build_embedding_bundle(kb, embedder)
    triplets = [Triplet(query="q", positive_id="c0", negative_id="c1")]
    assert embed_negatives(triplets, embedder, bundle) is None


def test_embed_negatives_embeds_free_text():
    kb = _kb()
    embedder = HashingEmbedder(dimension=128)
    bundle = build_embedding_bundle(kb, embedder)
    triplets = [
        Triplet(
            query="q1", positive_id="c0", negative_id="x::0", negative_text="totally other text"
        ),
        Triplet(query="q2", positive_id="c1", negative_id="c2"),  # corpus chunk negative
    ]
    negatives = embed_negatives(triplets, embedder, bundle)
    assert negatives is not None
    assert negatives.shape == (2, embedder.dimension)
    # The corpus-chunk negative row must equal the chunk's embedding.
    expected = bundle.chunk_matrix[bundle.chunk_id_to_row["c2"]].numpy()
    np.testing.assert_allclose(negatives[1].numpy(), expected, rtol=1e-5, atol=1e-6)
