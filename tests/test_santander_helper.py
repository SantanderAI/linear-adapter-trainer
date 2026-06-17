# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the Santander example scraper.

These exercise only the pure, network-free parts of ``examples/santander.py``
(HTML cleaning and slug derivation), keeping the suite runnable without any
network access, per CONTRIBUTING.md.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.santander import _slug, html_to_text  # noqa: E402

SAMPLE_HTML = """
<html>
  <head>
    <style>.banner { color: red; }</style>
    <script>console.log("tracking pixel");</script>
  </head>
  <body>
    <nav><a href="/">Home</a></nav>
    <main>
      <h1>Santander at a glance and our global presence</h1>
      <p>Santander operates in many countries around the world, serving
         millions of customers every day.</p>
      <p>Short.</p>
      <ul>
        <li>We focus on financial health and inclusion for our customers.</li>
      </ul>
    </main>
    <footer>Cookie settings and legal notice apply to this website.</footer>
  </body>
</html>
"""


def test_html_to_text_keeps_meaningful_blocks():
    text = html_to_text(SAMPLE_HTML)
    assert "Santander at a glance and our global presence" in text
    assert "Santander operates in many countries" in text
    assert "financial health and inclusion" in text


def test_html_to_text_drops_boilerplate_and_short_blocks():
    text = html_to_text(SAMPLE_HTML)
    # scripts/styles removed
    assert "console.log" not in text
    assert "color: red" not in text
    # nav/footer removed
    assert "Home" not in text
    assert "Cookie settings" not in text
    # blocks shorter than the threshold are dropped
    assert "Short." not in text


def test_html_to_text_separates_blocks_with_blank_lines():
    text = html_to_text(SAMPLE_HTML)
    assert "\n\n" in text


def test_slug_is_filesystem_and_id_safe():
    assert _slug("https://www.santander.com/en/about-us") == "www-santander-com-en-about-us"
    assert _slug("https://example.com/") == "example-com"
