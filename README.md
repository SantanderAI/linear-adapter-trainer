<div align="center">

# LinearAdapterTrainer

**Train linear embedding adapters with triplet loss to align retrieval embeddings with your queries.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/SantanderAI/linear-adapter-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/SantanderAI/linear-adapter-trainer/actions/workflows/ci.yml)
[![CodeQL](https://github.com/SantanderAI/linear-adapter-trainer/actions/workflows/codeql.yml/badge.svg)](https://github.com/SantanderAI/linear-adapter-trainer/actions/workflows/codeql.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Typed](https://img.shields.io/badge/typing-PEP%20561-blue.svg)](https://peps.python.org/pep-0561/)

</div>

---

> **Open source by [Santander AI Lab](https://github.com/SantanderAI).** An
> **AI / machine learning** Python library for retrieval-augmented generation
> (RAG): it trains linear embedding adapters with triplet loss to align your
> retrieval embeddings with real user queries. Part of
> [Santander AI Open Source](https://github.com/SantanderAI) — see also
> [santander.com](https://www.santander.com).

LinearAdapterTrainer fine-tunes retrieval **without retraining your embedding
model**. It learns a small linear transform that is applied to *query*
embeddings at search time, nudging them closer to relevant chunks and away from
irrelevant ones. Your vector index stays exactly as it is — you only adapt the
query side.

It is built around two composable modules:

1. **Dataset generator** — point it at a knowledge base and it produces
   `(query, positive, negative)` triplets, with configurable negative mining
   (semantically *opposite*, *hard*, or *random*) and a leakage-free train/val
   split.
2. **Linear adapter trainer** — trains a PyTorch linear adapter on those
   triplets with **triplet loss**, then reports retrieval gains with
   **precision@k**, **recall@k**, **MRR** and **nDCG**.

## Why a linear adapter?

| Approach | Cost | Re-index corpus? | Reversible |
|---|---|---|---|
| Fine-tune the embedding model | High (GPU, data) | Yes | No |
| **Linear query adapter (this repo)** | **Low (CPU-friendly)** | **No** | **Yes** |
| Re-ranking model | Medium (latency) | No | Yes |

The adapter is a single matrix (initialized at identity), so training is fast,
stable, and easy to audit. At inference you simply do
`adapted_query = adapter(query_embedding)` before your usual nearest-neighbor
search.

## How it works

```
                  Module 1: dataset                Module 2: adapter
 ┌───────────────┐   ┌──────────────────┐      ┌────────────────────────┐
 │ Knowledge base│──▶│ query generation │──┐   │  anchor = adapter(q)    │
 │   (chunks)    │   │ negative mining  │  ├──▶│  triplet loss:          │
 └───────────────┘   │ train/val split  │  │   │   pull anchor→positive  │
         │           └──────────────────┘  │   │   push anchor↮negative  │
         ▼                                  │   └────────────┬───────────┘
   embed chunks ────────────────────────────              evaluate
                                                  precision@k · recall@k · MRR · nDCG
```

The triplet objective is:

```
L = max(0, d(adapter(query), positive) − d(adapter(query), negative) + margin)
```

where `d` is cosine (default) or Euclidean distance.

## Installation

```bash
# with uv (recommended)
uv sync --dev

# for the example notebook (scraping + plotting)
uv sync --group examples

# or with pip, choosing the backends you need
pip install "linear-adapter-trainer[sentence-transformers]"   # local models
pip install "linear-adapter-trainer[openai]"                  # OpenAI API
pip install "linear-adapter-trainer[linkup]"                  # optional web-fetch backend
pip install "linear-adapter-trainer[all]"
```

The core install is dependency-light (`numpy`, `torch`, `tqdm`). A
dependency-free `HashingEmbedder` and `TemplateQueryGenerator` let you run the
whole pipeline offline (great for CI and demos).

When the corpus should come from current public web pages instead of local
files, use the provider-agnostic `WebLoader`. It fetches each URL through a
pluggable backend that you supply, so no specific provider is baked into the
package or its default install. Optional adapters for concrete backends are
available behind extras (for example `[linkup]`), and you can also pass your own
fetch client implementing the small `WebFetchClient` interface.

## Quickstart (Python)

```python
from linear_adapter_trainer import (
    AdapterTrainer, DatasetConfig, DatasetGenerator, KnowledgeBase,
    TemplateQueryGenerator, TrainingConfig,
)
from linear_adapter_trainer.embeddings import SentenceTransformerEmbedder

kb = KnowledgeBase.from_jsonl("examples/data/sample_kb.jsonl")
embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")

# Module 1 — generate triplets
dataset = DatasetGenerator(
    knowledge_base=kb,
    embedder=embedder,
    query_generator=TemplateQueryGenerator(seed=0),  # or LLMQueryGenerator(...)
    config=DatasetConfig(queries_per_chunk=4, strategy="mixed", val_fraction=0.2),
).generate()

# Module 2 — train the adapter
result = AdapterTrainer(kb, embedder, TrainingConfig(epochs=30)).fit(dataset)

print(result.improvement)        # delta per metric vs the base embeddings
result.adapter.save("adapter.pt")
```

At query time:

```python
import numpy as np
from linear_adapter_trainer import LinearAdapter

adapter = LinearAdapter.load("adapter.pt")
query_vec = embedder.embed(["how do plants make energy?"])
adapted = adapter.transform(query_vec)   # use this for nearest-neighbor search
```

## Quickstart (CLI)

Everything is driven by one TOML file (see `examples/config.toml`):

```bash
uv run linear-adapter generate examples/config.toml   # build the dataset
uv run linear-adapter train    examples/config.toml   # train + report metrics
uv run linear-adapter evaluate examples/config.toml   # base vs adapted
uv run linear-adapter run      examples/config.toml   # generate -> train
```

To build the knowledge base from known web pages, switch to
`examples/web_fetch_config.toml`, which selects a pluggable fetch backend. The
example uses the optional `linkup` adapter, so install that extra and set its
key first:

```bash
pip install "linear-adapter-trainer[linkup]"
export LINKUP_API_KEY=...
uv run linear-adapter generate examples/web_fetch_config.toml
```

Example output (with a Sentence-Transformers backend on a paraphrased query set):

```
metric              base     adapted     delta
------------------------------------------------
precision@1       0.5200      0.7100   +0.1900
mrr               0.6310      0.8050   +0.1740
ndcg@10           0.6890      0.8420   +0.1530
```

> The bundled `examples/config.toml` is **offline** (hashing embedder + template
> queries) so it runs anywhere. In that setup the baseline is already optimal —
> the queries reuse chunk tokens — so the adapter correctly reports a ~0 delta.
> Model selection always includes the identity baseline, so **the trained
> adapter can never score worse than your base embeddings**. Switch to a
> semantic backend (and `llm` query generation) to see real gains.

## Example: improving retrieval on real data

The notebook
[`examples/santander_retrieval_demo.ipynb`](examples/santander_retrieval_demo.ipynb)
walks through the full workflow on a corpus scraped from the
[Santander](https://www.santander.com) website:

1. **Scrape & chunk** the site into a knowledge base (live, with a cached
   snapshot fallback so it always runs).
2. **Generate a QA dataset** of triplets with an LLM (or offline templates).
3. **Train** the linear adapter.
4. **Measure** the base-vs-adapted retrieval gain with tables and plots.

```bash
uv sync --group examples
export OPENAI_API_KEY=sk-...     # for natural, LLM-generated queries
uv run jupyter lab examples/santander_retrieval_demo.ipynb
```

The notebook prefers a Sentence-Transformers model + LLM queries (real gains)
and automatically falls back to the offline `HashingEmbedder` +
`TemplateQueryGenerator` when no model or API key is available.

## Negative mining strategies

Set `strategy` in `[dataset]` to control how negatives are sampled:

- **`semantic_opposite`** — least similar chunks (far in latent space).
- **`hard`** — most similar but incorrect chunks (the strongest training signal).
- **`random`** — uniformly sampled easy negatives.
- **`mixed`** — a weighted blend (configure `[dataset.mix]`).

## Project layout

```
linear_adapter_trainer/
├── knowledge_base/   # ingestion + chunking
├── embeddings/       # pluggable backends (hashing, sentence-transformers, openai)
├── dataset/          # Module 1: query generation, negatives, split
├── adapter/          # Module 2: model, triplet loss, trainer
├── evaluation/       # precision@k, recall@k, MRR, nDCG, comparison
├── config.py         # TOML config + factories
└── cli.py            # command-line interface
```

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for the full API reference and design
notes.

## Development

```bash
uv sync --dev
uv run ruff check .
uv run pytest
uv run python examples/quickstart.py
```

## Contributing

Contributions are welcome — please read [`CONTRIBUTING.md`](CONTRIBUTING.md).
Notable changes are tracked in [`CHANGELOG.md`](CHANGELOG.md).

## Security

Please report security issues responsibly — see [`.github/SECURITY.md`](.github/SECURITY.md)
(contact `security-opensource@gruposantander.com` or open a GitHub Security
Advisory). Do not file public issues for vulnerabilities.

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
Copyright (c) 2026 Santander Group. Originally authored by Pedro Martin Minguez
(Santander AI Lab).

## Citation

If you use this software, please cite it — see [`CITATION.cff`](CITATION.cff).

```bibtex
@software{linear_adapter_trainer_2026,
  title  = {LinearAdapterTrainer: linear embedding adapters with triplet loss for retrieval},
  author = {{Santander AI Lab}},
  year   = {2026},
  url    = {https://github.com/SantanderAI/linear-adapter-trainer},
  license = {Apache-2.0}
}
```
