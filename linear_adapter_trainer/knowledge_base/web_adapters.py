# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Optional, opt-in web-fetch backends for :class:`WebLoader`.

These adapters are not part of the public API and none is used by default.
Each builds a client satisfying
:class:`linear_adapter_trainer.knowledge_base.web.WebFetchClient`. Add your own
adapter here (or pass any compatible client directly) to plug in a different
backend.
"""

from __future__ import annotations

from typing import Any

from .web import WebFetchClient


def linkup_fetch_client(**kwargs: Any) -> WebFetchClient:
    """Build a Linkup-backed fetch client (optional).

    Requires the optional dependency ``linear-adapter-trainer[linkup]`` and a
    ``LINKUP_API_KEY``. Provided only as one example backend; the core loader
    does not depend on it.
    """
    try:
        from linkup import LinkupClient
    except ImportError as exc:  # pragma: no cover - exercised without dependency
        raise ImportError(
            "This backend requires the optional dependency: install "
            "`linear-adapter-trainer[linkup]` and set `LINKUP_API_KEY`."
        ) from exc
    return LinkupClient(**kwargs)


_BACKENDS = {"linkup": linkup_fetch_client}


def build_web_fetch_client(backend: str, **kwargs: Any) -> WebFetchClient:
    """Build a web-fetch client for a named optional backend."""
    try:
        factory = _BACKENDS[backend]
    except KeyError:
        known = ", ".join(sorted(_BACKENDS)) or "(none)"
        raise ValueError(
            f"Unknown web_fetch backend {backend!r}. Known optional backends: {known}. "
            "Alternatively, pass a `client` implementing WebFetchClient directly."
        ) from None
    return factory(**kwargs)
