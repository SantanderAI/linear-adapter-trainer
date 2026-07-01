# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Provider-agnostic web ingestion for retrieval knowledge bases."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .base import Chunk, KnowledgeBase
from .chunking import TextSplitter


class WebFetchClient(Protocol):
    """Minimal interface a web-fetch backend must implement.

    Any object with a ``fetch`` method that takes a URL and returns a response
    carrying the cleaned page content can be used. The concrete backend (a
    hosted fetch API, a local scraper, a stub in tests, etc.) is the caller's
    choice; the loader has no knowledge of any specific provider.
    """

    def fetch(self, **kwargs: Any) -> Any:
        """Fetch one URL and return a response with cleaned page content."""


@dataclass(slots=True)
class WebLoader:
    """Build a :class:`KnowledgeBase` from pages fetched by a pluggable client.

    Pass any ``client`` implementing :class:`WebFetchClient`, or build a neutral
    adapter from :mod:`linear_adapter_trainer.knowledge_base.web_adapters`.
    """

    client: WebFetchClient
    render_js: bool = True
    include_raw_html: bool = False
    extract_images: bool = False

    def load_urls(self, urls: Sequence[str], *, ids: Sequence[str] | None = None) -> KnowledgeBase:
        """Fetch known URLs and return one knowledge-base chunk per page."""
        if not urls:
            raise ValueError("WebLoader requires at least one URL.")
        if ids is not None and len(ids) != len(urls):
            raise ValueError("`ids` length must match `urls` length.")

        chunks: list[Chunk] = []
        for position, url in enumerate(urls):
            response = self.client.fetch(
                url=url,
                render_js=self.render_js,
                include_raw_html=self.include_raw_html,
                extract_images=self.extract_images,
            )
            metadata = {
                "source": url,
                "source_url": url,
                **_extract_metadata(response),
            }
            chunks.append(
                Chunk(
                    id=str(ids[position]) if ids is not None else f"web-{position}",
                    text=_extract_text(response),
                    metadata=metadata,
                )
            )
        return KnowledgeBase(chunks)

    def load_and_split_urls(
        self,
        urls: Sequence[str],
        *,
        ids: Sequence[str] | None = None,
        splitter: TextSplitter | None = None,
    ) -> KnowledgeBase:
        """Fetch known URLs, then split each fetched page into retrieval chunks."""
        kb = self.load_urls(urls, ids=ids)
        return (splitter or TextSplitter()).split_knowledge_base(kb)


def _extract_text(response: Any) -> str:
    for key in ("content", "markdown", "text"):
        value = _get_value(response, key)
        if isinstance(value, str) and value.strip():
            return value
    if isinstance(response, str) and response.strip():
        return response
    raise ValueError("Web fetch response did not contain cleaned page text.")


def _extract_metadata(response: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    title = _get_value(response, "title") or _get_value(response, "name")
    if isinstance(title, str) and title:
        metadata["title"] = title
    return metadata


def _get_value(response: Any, key: str) -> Any:
    if isinstance(response, dict):
        return response.get(key)
    return getattr(response, key, None)
