# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for LinearAdapterTrainer.

Subcommands::

    linear-adapter generate CONFIG    # build the triplet dataset (Module 1)
    linear-adapter train    CONFIG    # train the adapter (Module 2)
    linear-adapter evaluate CONFIG    # report base vs adapted metrics
    linear-adapter run      CONFIG    # generate -> train -> evaluate

All behavior is driven by a single TOML config file (see examples/).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adapter.model import LinearAdapter
from .adapter.trainer import AdapterTrainer
from .config import (
    build_dataset_config,
    build_embedder,
    build_knowledge_base,
    build_negative_generator,
    build_query_generator,
    build_training_config,
    load_config,
)
from .dataset.generator import DatasetGenerator
from .dataset.schema import TripletDataset
from .evaluation.evaluator import RetrievalEvaluator


def _output(cfg: dict[str, Any]) -> dict[str, Any]:
    out = cfg.get("output", {})
    return {
        "dataset_dir": out.get("dataset_dir", "artifacts/dataset"),
        "adapter_path": out.get("adapter_path", "artifacts/adapter.safetensors"),
        "metrics_path": out.get("metrics_path", "artifacts/metrics.json"),
    }


def _validate_adapter_output_path(path: str | Path) -> None:
    if Path(path).suffix.lower() != ".safetensors":
        raise ValueError(
            "Change output.adapter_path to use the '.safetensors' extension. "
            "Existing '.pt' or '.pth' checkpoints can be converted with "
            "LinearAdapter.migrate_checkpoint()."
        )


def cmd_generate(cfg: dict[str, Any]) -> TripletDataset:
    kb = build_knowledge_base(cfg.get("knowledge_base", {}))
    embedder = build_embedder(cfg.get("embedder", {}))
    query_generator = build_query_generator(cfg.get("query_generator", {}))
    negative_generator = build_negative_generator(cfg.get("negative_generator", {}))
    dataset_config = build_dataset_config(cfg.get("dataset", {}))

    generator = DatasetGenerator(
        kb,
        embedder,
        query_generator,
        dataset_config,
        negative_generator=negative_generator,
    )
    dataset = generator.generate()

    out = _output(cfg)
    dataset.save(out["dataset_dir"])
    print(
        f"Dataset: {len(dataset.train)} train / {len(dataset.val)} val triplets "
        f"-> {out['dataset_dir']}"
    )
    return dataset


def cmd_train(cfg: dict[str, Any]) -> None:
    out = _output(cfg)
    _validate_adapter_output_path(out["adapter_path"])
    kb = build_knowledge_base(cfg.get("knowledge_base", {}))
    embedder = build_embedder(cfg.get("embedder", {}))
    dataset = TripletDataset.load(out["dataset_dir"])

    trainer = AdapterTrainer(kb, embedder, build_training_config(cfg.get("training", {})))
    result = trainer.fit(dataset)

    result.adapter.save(out["adapter_path"])
    _write_metrics(
        out["metrics_path"],
        {
            "baseline": result.baseline_metrics,
            "adapted": result.best_metrics,
            "improvement": result.improvement,
            "history": result.history,
        },
    )
    print(f"\nAdapter saved -> {out['adapter_path']}")
    _print_comparison(result.baseline_metrics, result.best_metrics)


def cmd_evaluate(cfg: dict[str, Any]) -> None:
    out = _output(cfg)
    kb = build_knowledge_base(cfg.get("knowledge_base", {}))
    embedder = build_embedder(cfg.get("embedder", {}))
    dataset = TripletDataset.load(out["dataset_dir"])
    adapter = LinearAdapter.load(out["adapter_path"])

    eval_ks = tuple(cfg.get("training", {}).get("eval_ks", (1, 3, 5, 10)))
    evaluator = RetrievalEvaluator(kb, embedder, ks=eval_ks)
    comparison = evaluator.compare(dataset.val or dataset.train, adapter)
    _print_comparison(comparison["base"], comparison["adapted"])


def cmd_run(cfg: dict[str, Any]) -> None:
    _validate_adapter_output_path(_output(cfg)["adapter_path"])
    cmd_generate(cfg)
    cmd_train(cfg)


def _write_metrics(path: str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_comparison(base: dict[str, float], adapted: dict[str, float]) -> None:
    keys = [k for k in adapted if k != "n_queries"]
    print(f"\n{'metric':<16}{'base':>10}{'adapted':>12}{'delta':>10}")
    print("-" * 48)
    for key in keys:
        b = base.get(key, 0.0)
        a = adapted.get(key, 0.0)
        print(f"{key:<16}{b:>10.4f}{a:>12.4f}{a - b:>+10.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linear-adapter",
        description="Train linear embedding adapters with triplet loss.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("generate", "Generate the triplet dataset (Module 1)."),
        ("train", "Train the linear adapter (Module 2)."),
        ("evaluate", "Evaluate base vs adapted retrieval metrics."),
        ("run", "Run generate then train end-to-end."),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("config", type=str, help="Path to a TOML config file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    dispatch = {
        "generate": cmd_generate,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "run": cmd_run,
    }
    dispatch[args.command](cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
