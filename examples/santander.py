# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Scrape the Santander website into a :class:`KnowledgeBase` for the demo.

This is *example-only* code: it lives outside the core package so the library
stays dependency-light. It needs the optional ``examples`` extra::

    uv sync --group examples          # or: pip install beautifulsoup4 requests

The public entry point is :func:`build_knowledge_base`, which tries a live
scrape and transparently falls back to a committed JSONL snapshot
(``examples/data/santander_kb.jsonl``) so the notebook is always reproducible
even if the website changes or blocks the request.

You can regenerate the cached snapshot with::

    uv run python examples/santander.py --refresh
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path

from linear_adapter_trainer import Chunk, KnowledgeBase, TextSplitter

HERE = Path(__file__).resolve().parent
CACHE_PATH = HERE / "data" / "santander_kb.jsonl"

# Corporate pages from the Santander group site. They are reasonably stable and
# mix English content across business, strategy and investor topics. If any URL
# moves, the cached snapshot keeps the notebook working.
DEFAULT_URLS: tuple[str, ...] = (
    "https://www.santander.com/en/about-us",
    "https://www.santander.com/en/about-us/where-we-are",
    "https://www.santander.com/en/about-us/our-history",
    "https://www.santander.com/en/our-approach",
    "https://www.santander.com/en/our-approach/our-strategy",
    "https://www.santander.com/en/our-approach/inclusive-and-sustainable-growth",
)

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 LinearAdapterTrainer-demo"
)
_WHITESPACE_RE = re.compile(r"\s+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def html_to_text(html: str, *, min_block_chars: int = 40) -> str:
    """Extract clean, readable text from an HTML document.

    Drops boilerplate (scripts, styles, navigation, footers, forms) and keeps
    headings, paragraphs and list items. Pure and offline, so it is unit-tested
    without any network access.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "html_to_text requires BeautifulSoup. Install the examples extras: "
            "uv sync --group examples  (or: pip install beautifulsoup4)"
        ) from exc

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(
        ["script", "style", "noscript", "nav", "footer", "header", "form", "aside", "svg", "button"]
    ):
        tag.decompose()

    root = soup.find("main") or soup.body or soup
    blocks: list[str] = []
    seen: set[str] = set()
    for element in root.find_all(["h1", "h2", "h3", "p", "li"]):
        text = _WHITESPACE_RE.sub(" ", element.get_text(" ", strip=True)).strip()
        if len(text) >= min_block_chars and text not in seen:
            seen.add(text)
            blocks.append(text)
    return "\n\n".join(blocks)


def _slug(url: str) -> str:
    path = url.split("://", 1)[-1]
    slug = _SLUG_RE.sub("-", path.lower()).strip("-")
    return slug or "page"


def fetch_page(url: str, *, timeout: float = 20.0) -> str:
    """Download a single page and return its raw HTML."""
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "fetch_page requires requests. Install the examples extras: "
            "uv sync --group examples  (or: pip install requests)"
        ) from exc

    response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def scrape(
    urls: Sequence[str] = DEFAULT_URLS,
    *,
    min_page_chars: int = 250,
    delay: float = 1.0,
    verbose: bool = True,
) -> list[Chunk]:
    """Scrape ``urls`` into one :class:`Chunk` per page (best effort)."""
    pages: list[Chunk] = []
    for url in urls:
        try:
            text = html_to_text(fetch_page(url))
        except Exception as exc:  # noqa: BLE001 - skip individual failures
            if verbose:
                print(f"  skip {url}: {exc}")
            continue
        if len(text) < min_page_chars:
            if verbose:
                print(f"  skip {url}: only {len(text)} chars after cleaning")
            continue
        pages.append(Chunk(id=_slug(url), text=text, metadata={"source": url}))
        if verbose:
            print(f"  ok   {url} ({len(text)} chars)")
        if delay:
            time.sleep(delay)
    return pages


def _save_pages(pages: Sequence[Chunk], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for page in pages:
            record = {"id": page.id, "text": page.text, "source": page.metadata.get("source", "")}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_knowledge_base(
    urls: Sequence[str] = DEFAULT_URLS,
    *,
    cache_path: str | Path = CACHE_PATH,
    prefer_live: bool = True,
    refresh_cache: bool = False,
    min_pages: int = 3,
    chunk_size: int = 700,
    chunk_overlap: int = 80,
    verbose: bool = True,
) -> KnowledgeBase:
    """Build a chunked knowledge base from Santander, with cache fallback.

    Tries a live scrape first; if it yields fewer than ``min_pages`` usable
    pages (offline, blocked, site changed), loads the committed snapshot at
    ``cache_path``. Pages are then split with :class:`TextSplitter`.
    """
    pages: list[Chunk] = []
    if prefer_live:
        if verbose:
            print(f"Scraping {len(urls)} Santander pages...")
        pages = scrape(urls, verbose=verbose)

    if len(pages) >= min_pages:
        if refresh_cache or not Path(cache_path).exists():
            _save_pages(pages, cache_path)
            if verbose:
                print(f"Cached {len(pages)} pages -> {cache_path}")
        page_kb = KnowledgeBase(pages)
    else:
        if verbose:
            reason = "live scrape disabled" if not prefer_live else "too few live pages"
            print(f"Using cached snapshot ({reason}): {cache_path}")
        page_kb = KnowledgeBase.from_jsonl(cache_path)

    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    kb = splitter.split_knowledge_base(page_kb)
    if verbose:
        print(f"Knowledge base: {len(page_kb)} pages -> {len(kb)} chunks")
    return kb


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape Santander into a cached KB snapshot.")
    parser.add_argument("--refresh", action="store_true", help="Force-refresh the cached snapshot.")
    parser.add_argument("--no-live", action="store_true", help="Only read the cached snapshot.")
    args = parser.parse_args(argv)

    kb = build_knowledge_base(
        prefer_live=not args.no_live,
        refresh_cache=args.refresh,
    )
    print(f"\nDone. {len(kb)} chunks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
