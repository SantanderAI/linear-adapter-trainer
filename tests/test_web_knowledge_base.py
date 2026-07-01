# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

import io

import pytest

from linear_adapter_trainer.config import build_knowledge_base
from linear_adapter_trainer.knowledge_base import TextSplitter, WebLoader
from linear_adapter_trainer.knowledge_base.web_adapters import build_web_fetch_client


class RecordingClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def fetch(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses[kwargs["url"]]


def test_web_loader_fetches_known_urls_into_knowledge_base():
    client = RecordingClient(
        {
            "https://example.com/risk": {
                "content": "# Risk overview\n\nSource-backed risk text.",
                "title": "Risk overview",
            },
            "https://example.com/governance": {
                "markdown": "# Governance\n\nClean markdown content.",
            },
        }
    )

    kb = WebLoader(client=client).load_urls(
        ["https://example.com/risk", "https://example.com/governance"],
        ids=["risk", "governance"],
    )

    assert kb.ids == ["risk", "governance"]
    assert kb.get("risk").text == "# Risk overview\n\nSource-backed risk text."
    assert kb.get("risk").metadata == {
        "source": "https://example.com/risk",
        "source_url": "https://example.com/risk",
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


def test_web_loader_can_split_fetched_pages():
    client = RecordingClient(
        {
            "https://example.com/report": {
                "content": "First sourced paragraph.\n\nSecond sourced paragraph.\n\nThird sourced paragraph."
            }
        }
    )

    kb = WebLoader(client=client).load_and_split_urls(
        ["https://example.com/report"],
        splitter=TextSplitter(chunk_size=35, chunk_overlap=5),
    )

    assert len(kb) > 1
    assert all(chunk.metadata["source_url"] == "https://example.com/report" for chunk in kb)
    assert all(chunk.metadata["parent_id"] == "web-0" for chunk in kb)


def test_web_loader_requires_a_client():
    with pytest.raises(TypeError):
        WebLoader()  # type: ignore[call-arg]


def test_unknown_web_fetch_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown web_fetch backend"):
        build_web_fetch_client("does-not-exist")


def test_http_web_fetch_backend_fetches_html_with_stdlib(monkeypatch):
    class FakeResponse(io.BytesIO):
        def __init__(self):
            super().__init__(
                b"<html><head><title>Example</title></head>"
                b"<body><h1>Heading</h1><p>Clean page text.</p></body></html>"
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        @property
        def headers(self):
            return self

        def get_content_charset(self):
            return "utf-8"

    def fake_urlopen(request, *, timeout):
        assert request.full_url == "https://example.com/docs"
        assert request.get_header("User-agent") == "linear-adapter-trainer/0.1"
        assert timeout == 3.0
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = build_web_fetch_client("http", timeout=3.0)
    response = client.fetch(url="https://example.com/docs", include_raw_html=True)

    assert response["title"] == "Example"
    assert response["content"] == "Heading Clean page text."
    assert response["raw_html"].startswith("<html>")


def test_web_fetch_config_requires_client_or_backend():
    with pytest.raises(ValueError, match="requires either a `client` or a `backend`"):
        build_knowledge_base({"format": "web_fetch", "urls": ["https://example.com"]})


def test_config_builds_web_fetch_knowledge_base_with_injected_client():
    client = RecordingClient(
        {"https://example.com/docs": {"content": "Trusted source material for an AI workflow."}}
    )

    kb = build_knowledge_base(
        {
            "format": "web_fetch",
            "urls": ["https://example.com/docs"],
            "client": client,
            "render_js": False,
            "chunking": {"enabled": True, "chunk_size": 32, "chunk_overlap": 4},
        }
    )

    assert kb.get("web-0::0").metadata["source_url"] == "https://example.com/docs"
    assert client.calls[0]["render_js"] is False
