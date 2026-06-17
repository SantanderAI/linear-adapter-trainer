# Contributing to LinearAdapterTrainer

Thanks for your interest in improving the project! This guide covers the
essentials.

## Development setup

```bash
uv sync --dev
```

This installs the package in editable mode along with the dev tools (`ruff`,
`pytest`).

## Before opening a pull request

Please make sure the following pass locally:

```bash
uv run ruff check .      # lint + import sorting
uv run pytest            # full test suite (runs offline)
```

New behavior should come with tests. The test suite must remain runnable
without network access or API keys — use `HashingEmbedder` and
`TemplateQueryGenerator` in tests.

## Code style

- Target Python 3.12+, with type hints on all public APIs.
- Keep modules focused (see `AGENTS.md` for the responsibility map).
- Keep defaults deterministic and seeded.
- Optional dependencies (Sentence-Transformers, OpenAI) must be imported
  lazily, with a clear error pointing to the matching extra.

## Adding an embedding backend

Implement the `EmbeddingModel` protocol (`embeddings/base.py`): a `dimension`
property and an `embed(texts) -> np.ndarray` method returning a
`(len(texts), dimension)` float32 array. Wire it into `config.build_embedder`.

## Commit messages

Use clear, imperative summaries (e.g. "Add nDCG to the evaluator"). Group
related changes together.

## License

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0.
