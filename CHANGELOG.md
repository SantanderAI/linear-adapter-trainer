# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Santander demo notebook (`examples/santander_retrieval_demo.ipynb`): scrape a
  real corpus, generate a QA dataset, train the adapter, and measure the
  retrieval improvement end-to-end.
- `examples/santander.py` scraper helper with a live-scrape + cached-snapshot
  fallback (`examples/data/santander_kb.jsonl`).
- `examples` dependency group for the notebook (scraping + plotting).
- GitHub Actions CI (ruff + pytest) and issue/PR templates.
- Offline unit tests for the scraper's HTML cleaning.

### Changed
- Export `LLMQueryGenerator` from the top-level package and de-duplicate
  `__all__`.

## [0.1.0] - 2026-06-15

### Added
- Initial release.
- Module 1 — triplet **dataset generator**: query generation (template / LLM),
  negative mining (`random`, `semantic_opposite`, `hard`, `mixed`), and a
  leakage-free train/val split.
- Module 2 — **linear adapter trainer**: residual/identity-initialized adapter,
  triplet margin loss (cosine / euclidean), early stopping on a validation
  retrieval metric.
- **Evaluation**: precision@k, recall@k, hit_rate@k, MRR, nDCG, and a
  base-vs-adapted comparison.
- Pluggable embedding backends: `HashingEmbedder` (offline),
  `SentenceTransformerEmbedder`, `OpenAIEmbedder`.
- TOML-driven CLI (`linear-adapter generate|train|evaluate|run`).
- Offline-first pipeline, type hints (`py.typed`), and documentation.

[Unreleased]: https://github.com/SantanderAI/linear-adapter-trainer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SantanderAI/linear-adapter-trainer/releases/tag/v0.1.0
