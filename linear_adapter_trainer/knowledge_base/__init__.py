# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Knowledge base ingestion and chunking."""

from .base import Chunk, KnowledgeBase
from .chunking import TextSplitter
from .linkup import LinkupWebLoader

__all__ = ["Chunk", "KnowledgeBase", "LinkupWebLoader", "TextSplitter"]
