# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from linear_adapter_trainer.dataset.query_generation import (
    LLMQueryGenerator,
    TemplateQueryGenerator,
    resolve_generator,
)

PASSAGE = (
    "Photosynthesis is the process by which green plants convert light energy "
    "into chemical energy stored as glucose."
)


def test_template_generator_count_and_determinism():
    gen = TemplateQueryGenerator(seed=0)
    a = gen.generate(PASSAGE, 4)
    b = gen.generate(PASSAGE, 4)
    assert a == b
    assert 1 <= len(a) <= 4
    assert all(q.strip() for q in a)


def test_template_generator_unique_queries():
    gen = TemplateQueryGenerator(seed=1)
    queries = gen.generate(PASSAGE, 5)
    assert len(set(queries)) == len(queries)


def test_resolve_generator():
    gen = resolve_generator("template", seed=2)
    assert isinstance(gen, TemplateQueryGenerator)


def test_llm_parser_json_array():
    parsed = LLMQueryGenerator._parse('Here:\n["What is X?", "Define X."]', 5)
    assert parsed == ["What is X?", "Define X."]


def test_llm_parser_bullet_fallback():
    parsed = LLMQueryGenerator._parse("1. What is X?\n2. Define X.", 5)
    assert parsed == ["What is X?", "Define X."]
