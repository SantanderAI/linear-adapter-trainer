# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Negative-passage generation backends.

Where :mod:`negatives` mines a contrasting chunk *from the corpus*, this module
synthesises the negative *text* with a language model. The recommended use is
**hard negatives**: short passages on the same broad topic as the positive
chunk that are plausible yet do **not** answer the query. Hard negatives are
what give triplet training a useful gradient — off-topic text sits so far away
in embedding space that the margin is satisfied trivially and the adapter
learns nothing.

The produced text is stored on :class:`~linear_adapter_trainer.dataset.schema.Triplet`
via ``negative_text`` and embedded at training time, so these negatives never
have to exist in the knowledge base.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class NegativeGenerator(Protocol):
    """Structural type for negative-passage generators."""

    def generate(self, positive_text: str, n: int) -> list[str]:
        """Return up to ``n`` negative passages contrasting ``positive_text``."""
        ...


class LLMNegativeGenerator:
    """Generate hard-negative passages with an OpenAI-compatible chat model.

    A hard negative is a short passage on the **same topic** as the source
    chunk that is plausible but does not contain its specific facts — close
    enough to confuse a retriever, yet genuinely irrelevant as an answer.

    Args:
        model: Chat model name.
        api_key: Optional explicit key; falls back to ``OPENAI_API_KEY``.
        base_url: Optional override for OpenAI-compatible gateways.
        temperature: Sampling temperature. Leave as ``None`` to use the model
            default. Newer models (e.g. the GPT-5 family) only accept the
            default and reject an explicit value, so it is not sent unless set.
        system_prompt: Optional override for the system instruction.
    """

    _DEFAULT_SYSTEM = (
        "You are a search-dataset generator that writes HARD NEGATIVE passages. "
        "Given a source passage, write short passages on the SAME broad topic "
        "that are plausible and natural but DO NOT answer the source or contain "
        "its specific facts. They must be close enough to be confusing for a "
        "retriever, yet genuinely the wrong result. Do not contradict yourself "
        "and do not copy sentences from the source. Respond ONLY with a JSON "
        "array of strings."
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
                "LLMNegativeGenerator requires the optional dependency. "
                'Install it with: pip install "linear-adapter-trainer[openai]"'
            ) from exc

        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt or self._DEFAULT_SYSTEM
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, positive_text: str, n: int) -> list[str]:
        user = (
            f'Source passage:\n"""\n{positive_text}\n"""\n\n'
            f"Write {n} distinct hard-negative passages about the same topic "
            "that do NOT answer or restate the source. Each should be 1-3 "
            "sentences. Return a JSON array of strings only."
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
                passages = [str(p).strip() for p in items if str(p).strip()]
                return passages[:n]
            except json.JSONDecodeError:
                pass
        lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip() for ln in content.splitlines()]
        return [ln for ln in lines if ln][:n]


def resolve_negative_generator(name: str, **kwargs: object) -> NegativeGenerator:
    """Factory mapping a backend name to a :class:`NegativeGenerator`."""
    backends: dict[str, type] = {"llm": LLMNegativeGenerator}
    if name not in backends:
        raise ValueError(f"Unknown negative generator {name!r}. Choose from {sorted(backends)}.")
    return backends[name](**kwargs)  # type: ignore[arg-type]
