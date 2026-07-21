# Documentation

A reference for the LinearAdapterTrainer API and its design. For a quick tour,
see the [`README`](README.md).

## Table of contents

1. [Concepts](#concepts)
2. [Knowledge base](#knowledge-base)
3. [Embedding backends](#embedding-backends)
4. [Module 1 — dataset generation](#module-1--dataset-generation)
5. [Module 2 — the linear adapter](#module-2--the-linear-adapter)
6. [Evaluation](#evaluation)
7. [Configuration reference](#configuration-reference)
8. [Design notes](#design-notes)

---

## Concepts

A **chunk** is a unit of retrievable text. A **query** is text a user might
search with. A **triplet** is `(query, positive_chunk, negative_chunk)` where
the positive is relevant to the query and the negative is not.

The **linear adapter** is a learned matrix applied to *query* embeddings only.
Corpus embeddings are untouched, so you never re-index. Training minimizes a
**triplet margin loss** so adapted queries land near positives and far from
negatives.

---

## Knowledge base

```python
from linear_adapter_trainer import KnowledgeBase, Chunk, TextSplitter

kb = KnowledgeBase.from_jsonl("kb.jsonl")           # {"id","text", ...} per line
kb = KnowledgeBase.from_texts(["a", "b"], ids=...)  # from raw strings
kb = KnowledgeBase.from_directory("docs/", glob="*.txt")

kb.get("chunk-1")        # Chunk lookup by id
kb.ids, kb.texts         # parallel lists
kb.to_jsonl("out.jsonl")
```

`Chunk` is a frozen dataclass with `id`, `text`, and `metadata`. Ids must be
unique and text non-empty.

### Chunking

```python
splitter = TextSplitter(chunk_size=512, chunk_overlap=64)
chunked_kb = splitter.split_knowledge_base(kb)   # child ids like "doc::0"
```

The splitter is recursive and separator-aware (`\n\n`, `\n`, `. `, ` `), with
deterministic output.

---

## Embedding backends

Any object satisfying the `EmbeddingModel` protocol works:

```python
class EmbeddingModel(Protocol):
    @property
    def dimension(self) -> int: ...
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...   # (N, dim) float32
```

Provided backends:

| Backend | Import | Extra | Notes |
|---|---|---|---|
| `HashingEmbedder` | `linear_adapter_trainer` | none | Deterministic, offline. Demo/CI only. |
| `SentenceTransformerEmbedder` | `linear_adapter_trainer.embeddings` | `sentence-transformers` | Local models. |
| `OpenAIEmbedder` | `linear_adapter_trainer.embeddings` | `openai` | Reads `OPENAI_API_KEY`. |

Optional backends are imported lazily; a missing dependency raises a clear
`ImportError` naming the extra to install.

---

## Module 1 — dataset generation

```python
from linear_adapter_trainer import (
    DatasetGenerator, DatasetConfig, TemplateQueryGenerator, LLMQueryGenerator,
)

generator = DatasetGenerator(
    knowledge_base=kb,
    embedder=embedder,
    query_generator=TemplateQueryGenerator(seed=0),
    config=DatasetConfig(
        queries_per_chunk=4,
        negatives_per_query=2,
        strategy="mixed",                 # random | semantic_opposite | hard | mixed
        mix={"semantic_opposite": 0.5, "hard": 0.3, "random": 0.2},
        pool_size=10,
        val_fraction=0.2,
        seed=0,
    ),
)
dataset = generator.generate()
dataset.save("artifacts/dataset")        # train.jsonl, val.jsonl, meta.json
```

### Query generators

- `TemplateQueryGenerator` — offline, deterministic keyword templates.
- `LLMQueryGenerator` — OpenAI-compatible chat model producing natural queries.
  Defaults to `gpt-5-mini`. `temperature` is optional and left unset by
  default, because newer models (e.g. the GPT-5 family) only accept the
  default temperature and reject any explicit value. Set it only when
  targeting an older model that supports sampling.

Both satisfy the `QueryGenerator` protocol: `generate(text, n) -> list[str]`.

### Negative mining

`NegativeSampler` mines negatives from chunk embeddings:

- `semantic_opposite` — sampled from the least similar chunks.
- `hard` — sampled from the most similar (but incorrect) chunks.
- `random` — uniform.
- `mixed` — weighted blend.

### Splitting

`split_triplets` partitions by **positive chunk**, so no chunk appears in both
train and val (no leakage). The split is seeded and deterministic.

`TripletDataset` holds `train`, `val`, and `metadata`, with `save`/`load`.

---

## Module 2 — the linear adapter

### Model

```python
from linear_adapter_trainer import LinearAdapter, AdapterConfig

adapter = LinearAdapter(AdapterConfig(
    input_dim=384,
    residual=True,          # learn y = x + W x, initialized near identity
    normalize_output=True,  # L2-normalize outputs
))
adapted = adapter.transform(query_matrix)   # numpy in/out for inference
adapter.save("adapter.safetensors"); LinearAdapter.load("adapter.safetensors")
```

Safetensors checkpoints contain the adapter tensors plus a format version and
strictly validated JSON configuration in the file metadata. New writes must
use the `.safetensors` extension. Legacy `.pt` and `.pth` checkpoints are read
only through `torch.load(..., weights_only=True)` and emit a deprecation
warning. Convert them with:

```python
LinearAdapter.migrate_checkpoint("adapter.pt", "adapter.safetensors")
```

`load(..., map_location=...)` accepts a device string or `torch.device` for
safetensors checkpoints. Parser selection is extension-based; corrupt
safetensors files never fall back to the legacy loader.

A residual adapter starts as a no-op (`W = 0`), so early training never hurts
the base embeddings. A non-residual square adapter is initialized at identity.

### Loss

```python
from linear_adapter_trainer import TripletLoss
loss = TripletLoss(margin=0.2, distance="cosine")  # or "euclidean"
```

### Trainer

```python
from linear_adapter_trainer import AdapterTrainer, TrainingConfig

result = AdapterTrainer(kb, embedder, TrainingConfig(
    epochs=30, batch_size=64, learning_rate=1e-3,
    margin=0.2, distance="cosine", monitor="mrr", patience=5,
)).fit(dataset)

result.adapter            # best adapter (by monitored metric)
result.baseline_metrics   # metrics with raw embeddings
result.best_metrics       # metrics with the adapter
result.improvement        # per-metric delta
result.history            # per-epoch log
```

The trainer precomputes embeddings once, optimizes with Adam, monitors a
validation retrieval metric for early stopping, and restores the best weights.
Device is auto-selected (`cuda` → `mps` → `cpu`) unless overridden.

---

## Evaluation

Pure metric functions operate on `(ranked_ids, relevant_ids)`:

```python
from linear_adapter_trainer.evaluation import (
    precision_at_k, recall_at_k, hit_rate_at_k, reciprocal_rank, ndcg_at_k,
    evaluate_rankings,
)
```

`RetrievalEvaluator` runs end-to-end retrieval and compares base vs adapted:

```python
from linear_adapter_trainer import RetrievalEvaluator

evaluator = RetrievalEvaluator(kb, embedder, ks=(1, 3, 5, 10))
report = evaluator.compare(dataset.val, adapter)
report["base"], report["adapted"], report["delta"]
```

Aggregated keys include `precision@k`, `recall@k`, `hit_rate@k`, `ndcg@k`, and
`mrr`.

---

## Configuration reference

`linear-adapter <command> config.toml` reads these tables:

```toml
[knowledge_base]   # path, format ("jsonl"|"directory"), text_key, id_key, glob
[embedder]         # backend, model, dimension/dimensions, device, batch_size
[query_generator]  # backend ("template"|"llm"), model, temperature (optional), seed
[dataset]          # queries_per_chunk, negatives_per_query, strategy, pool_size, val_fraction, seed, max_workers
[dataset.mix]      # weights per strategy when strategy = "mixed"
[training]         # epochs, batch_size, learning_rate, margin, distance, residual, monitor, patience, eval_ks
[output]           # dataset_dir, adapter_path, metrics_path
```

See [`examples/config.toml`](examples/config.toml) for a complete, runnable
example.

---

## Design notes

- **Query-side only.** Adapting queries (not the corpus) keeps the vector index
  fixed and the change reversible. This is the cheapest way to get retrieval
  gains from an existing index.
- **Identity initialization.** Residual/identity init means the adapter cannot
  degrade retrieval at step zero; it can only learn improvements.
- **Reproducibility.** Generation, splitting, and training are seeded.
- **Offline-first.** The hashing embedder and template generator make the whole
  pipeline runnable without downloads or keys, which keeps CI fast and the
  examples trustworthy.
- **Portability.** Datasets store ids and text, not vectors, so they remain
  valid across embedding backends; vectors are recomputed at train time.
