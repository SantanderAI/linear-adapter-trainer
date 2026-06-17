# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Knowledge base ingestion and chunking."""

from .base import Chunk, KnowledgeBase
from .chunking import TextSplitter

__all__ = ["Chunk", "KnowledgeBase", "TextSplitter"]
