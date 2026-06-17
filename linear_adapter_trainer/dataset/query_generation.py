# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Query generation backends.

Module 1 needs *anchor* queries that a user might ask to retrieve a given
chunk. Two backends are provided:

* :class:`TemplateQueryGenerator` - deterministic, dependency-free, offline.
  Good for CI, tests and quick demos.
* :class:`LLMQueryGenerator` - high quality questions from an OpenAI-compatible
  chat model. Recommended for real datasets.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9']+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_STOPWORDS: frozenset[str] = frozenset("""
    a an the of to in on for and or but with without within into onto from by as at
    is are was were be been being this that these those it its their our your his her
    not no nor so than then once here there when where which who whom whose what why how
    el la los las un una unos unas de del y o u pero con sin para por que se su sus es
    son era eran ser estar como cuando donde cual quien porque
    """.split())


@runtime_checkable
class QueryGenerator(Protocol):
    """Structural type for anchor-query generators."""

    def generate(self, text: str, n: int) -> list[str]:
        """Return up to ``n`` queries that the chunk ``text`` should answer."""
        ...


class TemplateQueryGenerator:
    """Deterministic, offline query generator based on keyword templates.

    It extracts salient keywords/phrases from the chunk and fills a small set
    of natural-language templates. Output is reproducible given ``seed``.
    """

    _TEMPLATES: tuple[str, ...] = (
        "What is {kw}?",
        "Tell me about {kw}.",
        "How does {kw} work?",
        "Explain {kw}.",
        "Why is {kw} important?",
        "What does the text say about {kw}?",
    )

    def __init__(self, *, seed: int = 0, max_keyword_words: int = 3) -> None:
        self._seed = seed
        self._max_keyword_words = max_keyword_words

    def generate(self, text: str, n: int) -> list[str]:
        rng = random.Random(f"{self._seed}:{text}")
        keywords = self._keywords(text)
        if not keywords:
            keywords = [self._fallback(text)]

        queries: list[str] = []
        seen: set[str] = set()
        attempts = 0
        max_attempts = n * 8 + 16
        while len(queries) < n and attempts < max_attempts:
            attempts += 1
            keyword = rng.choice(keywords)
            template = rng.choice(self._TEMPLATES)
            query = template.format(kw=keyword)
            if query.lower() not in seen:
                seen.add(query.lower())
                queries.append(query)
        return queries

    # -- internals ---------------------------------------------------------
    def _keywords(self, text: str) -> list[str]:
        sentences = _SENTENCE_RE.split(text.strip())
        candidates: list[str] = []
        for sentence in sentences:
            tokens = [t for t in _TOKEN_RE.findall(sentence) if t.lower() not in _STOPWORDS]
            for size in range(1, self._max_keyword_words + 1):
                for i in range(len(tokens) - size + 1):
                    phrase = " ".join(tokens[i : i + size])
                    if len(phrase) >= 4 and any(len(t) > 3 for t in phrase.split()):
                        candidates.append(phrase)

        ranked = sorted(set(candidates), key=lambda p: (-len(p.split()), -len(p), p))
        return ranked[:12]

    def _fallback(self, text: str) -> str:
        tokens = _TOKEN_RE.findall(text)
        return " ".join(tokens[:3]) if tokens else "the topic"


class LLMQueryGenerator:
    """Generate diverse, natural queries with an OpenAI-compatible chat model.

    Args:
        model: Chat model name.
        api_key: Optional explicit key; falls back to ``OPENAI_API_KEY``.
        base_url: Optional override for OpenAI-compatible gateways.
        temperature: Sampling temperature for query diversity. Leave as ``None``
            to use the model default. Newer models (e.g. the GPT-5 family) only
            accept the default and reject any explicit temperature, so it is not
            sent unless you set it.
        system_prompt: Optional override for the system instruction.
    """

    _DEFAULT_SYSTEM = (
        "You are a search-dataset generator. Given a passage, you write short, "
        "natural questions a user would type to retrieve it. Vary phrasing and "
        "specificity. Respond ONLY with a JSON array of strings."
    )

    def __init__(
        self,
        model: str = "gpt-5-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "LLMQueryGenerator requires the optional dependency. "
                'Install it with: pip install "linear-adapter-trainer[openai]"'
            ) from exc

        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt or self._DEFAULT_SYSTEM
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, text: str, n: int) -> list[str]:
        user = (
            f'Passage:\n"""\n{text}\n"""\n\n'
            f"Write {n} distinct questions answerable by this passage. "
            "Return a JSON array of strings only."
        )
        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or "[]"
        return self._parse(content, n)

    @staticmethod
    def _parse(content: str, n: int) -> list[str]:
        content = content.strip()
        match = re.search(r"\[.*\]", content, flags=re.DOTALL)
        if match:
            try:
                items = json.loads(match.group(0))
                queries = [str(q).strip() for q in items if str(q).strip()]
                return queries[:n]
            except json.JSONDecodeError:
                pass
        lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip() for ln in content.splitlines()]
        return [ln for ln in lines if ln][:n]


def resolve_generator(name: str, **kwargs: object) -> QueryGenerator:
    """Factory mapping a backend name to a :class:`QueryGenerator`."""
    backends: dict[str, type] = {
        "template": TemplateQueryGenerator,
        "llm": LLMQueryGenerator,
    }
    if name not in backends:
        raise ValueError(f"Unknown query generator {name!r}. Choose from {sorted(backends)}.")
    return backends[name](**kwargs)  # type: ignore[arg-type]


def _coerce_queries(raw: Sequence[str]) -> list[str]:
    return [q.strip() for q in raw if q and q.strip()]
