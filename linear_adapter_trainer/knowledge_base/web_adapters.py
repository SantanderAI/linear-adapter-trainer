# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Neutral, opt-in web-fetch backends for :class:`WebLoader`."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

from .web import WebFetchClient


@dataclass(slots=True)
class HttpWebFetchClient:
    """Fetch pages with Python's standard-library HTTP client."""

    timeout: float = 10.0
    user_agent: str = "linear-adapter-trainer/0.1"

    def fetch(self, **kwargs: Any) -> dict[str, str]:
        url = kwargs["url"]
        include_raw_html = kwargs.get("include_raw_html", False)
        request = urllib.request.Request(url, headers={"User-agent": self.user_agent})

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            raw_html = response.read().decode(charset, errors="replace")

        parser = _HTMLTextParser()
        parser.feed(raw_html)
        result = {"content": parser.content or raw_html.strip()}
        if parser.title:
            result["title"] = parser.title
        if include_raw_html:
            result["raw_html"] = raw_html
        return result


def http_fetch_client(**kwargs: Any) -> WebFetchClient:
    """Build a dependency-free HTTP fetch client."""
    return HttpWebFetchClient(**kwargs)


_BACKENDS = {"http": http_fetch_client}


def build_web_fetch_client(backend: str, **kwargs: Any) -> WebFetchClient:
    """Build a web-fetch client for a named backend."""
    try:
        factory = _BACKENDS[backend]
    except KeyError:
        known = ", ".join(sorted(_BACKENDS)) or "(none)"
        raise ValueError(
            f"Unknown web_fetch backend {backend!r}. Known backends: {known}. "
            "Alternatively, pass a `client` implementing WebFetchClient directly."
        ) from None
    return factory(**kwargs)


class _HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._title_parts: list[str] = []
        self._in_title = False
        self._in_head = False
        self._skip_depth = 0

    @property
    def content(self) -> str:
        return " ".join(" ".join(self._parts).split())

    @property
    def title(self) -> str:
        return " ".join(" ".join(self._title_parts).split())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "head":
            self._in_head = True
        elif tag == "title":
            self._in_title = True
        elif tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "head":
            self._in_head = False
        elif tag == "title":
            self._in_title = False
        elif tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self._title_parts.append(text)
        elif not self._in_head and not self._skip_depth:
            self._parts.append(text)
