# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

import pytest

from linear_adapter_trainer.config import build_knowledge_base
from linear_adapter_trainer.knowledge_base import LinkupWebLoader, TextSplitter


class RecordingClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def fetch(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses[kwargs["url"]]


def test_linkup_loader_fetches_known_urls_into_knowledge_base():
    client = RecordingClient(
        {
            "https://example.com/risk": {
                "content": "# Risk overview\n\nSource-backed risk text.",
                "title": "Risk overview",
            },
            "https://example.com/governance": {
                "markdown": "# Governance\n\nClean markdown from Linkup.",
            },
        }
    )

    kb = LinkupWebLoader(client=client).load_urls(
        ["https://example.com/risk", "https://example.com/governance"],
        ids=["risk", "governance"],
    )

    assert kb.ids == ["risk", "governance"]
    assert kb.get("risk").text == "# Risk overview\n\nSource-backed risk text."
    assert kb.get("risk").metadata == {
        "source": "https://example.com/risk",
        "source_url": "https://example.com/risk",
        "fetched_with": "linkup",
        "title": "Risk overview",
    }
    assert client.calls == [
        {
            "url": "https://example.com/risk",
            "render_js": True,
            "include_raw_html": False,
            "extract_images": False,
        },
        {
            "url": "https://example.com/governance",
            "render_js": True,
            "include_raw_html": False,
            "extract_images": False,
        },
    ]


def test_linkup_loader_can_split_fetched_pages():
    client = RecordingClient(
        {
            "https://example.com/report": {
                "content": "First sourced paragraph.\n\nSecond sourced paragraph.\n\nThird sourced paragraph."
            }
        }
    )

    kb = LinkupWebLoader(client=client).load_and_split_urls(
        ["https://example.com/report"],
        splitter=TextSplitter(chunk_size=35, chunk_overlap=5),
    )

    assert len(kb) > 1
    assert all(chunk.metadata["source_url"] == "https://example.com/report" for chunk in kb)
    assert all(chunk.metadata["parent_id"] == "linkup-0" for chunk in kb)


def test_linkup_loader_requires_optional_dependency_without_client():
    with pytest.raises(ImportError, match="linear-adapter-trainer\\[linkup\\]"):
        LinkupWebLoader().load_urls(["https://example.com"])


def test_config_builds_linkup_fetch_knowledge_base_with_injected_client():
    client = RecordingClient(
        {"https://example.com/docs": {"content": "Trusted source material for an AI workflow."}}
    )

    kb = build_knowledge_base(
        {
            "format": "linkup_fetch",
            "urls": ["https://example.com/docs"],
            "client": client,
            "render_js": False,
            "chunking": {"enabled": True, "chunk_size": 32, "chunk_overlap": 4},
        }
    )

    assert kb.get("linkup-0::0").metadata["source_url"] == "https://example.com/docs"
    assert client.calls[0]["render_js"] is False
