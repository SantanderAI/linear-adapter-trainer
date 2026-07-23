# Repo Rules

## Intent

This repo trains linear embedding adapters with triplet loss to improve
retrieval. It is two clean, composable modules: a triplet **dataset generator**
and a PyTorch **linear adapter trainer**, plus retrieval **evaluation**. The
code is public; keep it explicit, typed, and easy to read.

## Design Rules

- Prefer direct code over abstraction layers.
- Each module has one clear responsibility.
- Keep public APIs typed and small; ship type hints (`py.typed`).
- Keep defaults deterministic and seeded.
- Avoid hidden behavior and side effects.
- Add a dependency only when it removes real complexity. Core stays light
  (`numpy`, `torch`, `safetensors`, `tqdm`); heavy backends are optional extras.
- The pipeline must run fully offline via `HashingEmbedder` +
  `TemplateQueryGenerator` so CI never needs network or API keys.

## Structure

- `knowledge_base/` owns ingestion (`base.py`) and chunking (`chunking.py`).
- `embeddings/` owns embedding backends; `base.py` defines the `EmbeddingModel`
  protocol. Optional backends import their dependency lazily.
- `dataset/` is Module 1: `query_generation.py`, `negatives.py`, `splitter.py`,
  `schema.py`, orchestrated by `generator.py`.
- `adapter/` is Module 2: `model.py` (the `nn.Module`), `losses.py`,
  `data.py` (tensor datasets), `trainer.py` (training loop).
- `evaluation/` owns `metrics.py` (pure functions) and `evaluator.py`.
- `config.py` owns TOML parsing and component factories.
- `cli.py` stays thin and only wires config to modules.

## Commands

- Install: `uv sync --dev`
- Lint: `uv run ruff check .`
- Test: `uv run pytest`
- Demo (offline): `uv run python examples/quickstart.py`
- Full run: `uv run linear-adapter run examples/config.toml`
