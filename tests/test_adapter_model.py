# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from linear_adapter_trainer.adapter.losses import TripletLoss
from linear_adapter_trainer.adapter.model import AdapterConfig, LinearAdapter


def _metadata(config: AdapterConfig, *, version: str = "1") -> dict[str, str]:
    return {
        "linear_adapter_format_version": version,
        "linear_adapter_config": json.dumps(asdict(config), sort_keys=True, separators=(",", ":")),
    }


def test_residual_adapter_starts_as_identity():
    adapter = LinearAdapter(AdapterConfig(input_dim=8, residual=True, normalize_output=False))
    x = torch.randn(4, 8)
    out = adapter(x)
    assert torch.allclose(out, x, atol=1e-6)


def test_normalize_output_unit_norm():
    adapter = LinearAdapter(AdapterConfig(input_dim=8, residual=True, normalize_output=True))
    x = torch.randn(4, 8)
    out = adapter(x)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_transform_numpy_shape():
    adapter = LinearAdapter(AdapterConfig(input_dim=6))
    embeddings = np.random.randn(5, 6).astype(np.float32)
    out = adapter.transform(embeddings)
    assert out.shape == (5, 6)


def test_residual_adapter_rejects_mismatched_output_dimension():
    with pytest.raises(ValueError, match="output_dim == input_dim"):
        LinearAdapter(AdapterConfig(input_dim=4, output_dim=3, residual=True))


@pytest.mark.parametrize(
    "config",
    [
        AdapterConfig(input_dim=4),
        AdapterConfig(input_dim=4, bias=False, normalize_output=False),
        AdapterConfig(input_dim=4, residual=False, normalize_output=False),
        AdapterConfig(input_dim=4, output_dim=3, residual=False),
    ],
)
def test_save_and_load(tmp_path, config):
    adapter = LinearAdapter(config)
    with torch.no_grad():
        adapter.linear.weight.add_(0.05)
    path = tmp_path / "adapter.safetensors"
    adapter.save(path)

    loaded = LinearAdapter.load(path)
    x = torch.randn(3, 4)
    assert torch.allclose(adapter(x), loaded(x), atol=1e-6)
    assert loaded.config == config
    assert not loaded.training

    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        assert checkpoint.metadata() == _metadata(config)
        assert set(checkpoint.keys()) == set(adapter.state_dict())


def test_save_rejects_legacy_extension(tmp_path):
    adapter = LinearAdapter(AdapterConfig(input_dim=4))
    with pytest.raises(ValueError, match="must use the '.safetensors' extension"):
        adapter.save(tmp_path / "adapter.pt")


def test_save_cleans_partial_temporary_file_on_failure(tmp_path, monkeypatch):
    adapter = LinearAdapter(AdapterConfig(input_dim=4))
    path = tmp_path / "adapter.safetensors"
    path.write_bytes(b"existing")

    def fail_save(tensors, filename, *, metadata):
        del tensors, metadata
        Path(filename).write_bytes(b"partial")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr("linear_adapter_trainer.adapter.model.save_file", fail_save)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        adapter.save(path)
    assert path.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [path]


def test_save_atomically_replaces_existing_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "adapter.safetensors"
    original = LinearAdapter(AdapterConfig(input_dim=4, normalize_output=False))
    original.save(path)
    original_bytes = path.read_bytes()
    replace_observations = []
    real_replace = os.replace

    def observe_replace(source, destination):
        replace_observations.append(
            {
                "source_is_temporary": source != destination and source.exists(),
                "destination_is_unchanged": destination.read_bytes() == original_bytes,
            }
        )
        real_replace(source, destination)

    monkeypatch.setattr("linear_adapter_trainer.adapter.model.os.replace", observe_replace)

    replacement = LinearAdapter(AdapterConfig(input_dim=4, normalize_output=False))
    with torch.no_grad():
        replacement.linear.weight.add_(0.25)
    replacement.save(path)

    loaded = LinearAdapter.load(path)
    assert replace_observations == [{"source_is_temporary": True, "destination_is_unchanged": True}]
    for key, tensor in replacement.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[key])


def test_load_rejects_unsupported_format_version(tmp_path):
    config = AdapterConfig(input_dim=4)
    adapter = LinearAdapter(config)
    path = tmp_path / "adapter.safetensors"
    save_file(adapter.state_dict(), path, metadata=_metadata(config, version="2"))

    with pytest.raises(ValueError, match="Unsupported.*format version '2'"):
        LinearAdapter.load(path)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "Invalid checkpoint metadata keys"),
        ({"linear_adapter_format_version": "1"}, "Invalid checkpoint metadata keys"),
        (
            {
                **_metadata(AdapterConfig(input_dim=4)),
                "unexpected": "metadata",
            },
            "Invalid checkpoint metadata keys",
        ),
        (
            {
                "linear_adapter_format_version": "1",
                "linear_adapter_config": "not-json",
            },
            "not valid JSON",
        ),
    ],
)
def test_load_rejects_invalid_metadata(tmp_path, metadata, message):
    adapter = LinearAdapter(AdapterConfig(input_dim=4))
    path = tmp_path / "adapter.safetensors"
    save_file(adapter.state_dict(), path, metadata=metadata)

    with pytest.raises(ValueError, match=message):
        LinearAdapter.load(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: config.pop("bias"), "Invalid adapter configuration keys"),
        (lambda config: config.update(extra=True), "Invalid adapter configuration keys"),
        (lambda config: config.update(input_dim=True), "input_dim must be a positive integer"),
        (lambda config: config.update(input_dim=-1), "input_dim must be a positive integer"),
        (lambda config: config.update(input_dim=4.0), "input_dim must be a positive integer"),
        (lambda config: config.update(input_dim="4"), "input_dim must be a positive integer"),
        (
            lambda config: config.update(output_dim=0),
            "output_dim must be null or a positive integer",
        ),
        (
            lambda config: config.update(output_dim=-1),
            "output_dim must be null or a positive integer",
        ),
        (
            lambda config: config.update(output_dim=4.0),
            "output_dim must be null or a positive integer",
        ),
        (
            lambda config: config.update(output_dim="4"),
            "output_dim must be null or a positive integer",
        ),
        (lambda config: config.update(residual=1), "residual must be a boolean"),
        (
            lambda config: config.update(output_dim=3, residual=True),
            "residual adapters require output_dim == input_dim",
        ),
    ],
)
def test_load_rejects_invalid_config(tmp_path, mutate, message):
    config = asdict(AdapterConfig(input_dim=4))
    mutate(config)
    metadata = _metadata(AdapterConfig(input_dim=4))
    metadata["linear_adapter_config"] = json.dumps(config)
    adapter = LinearAdapter(AdapterConfig(input_dim=4))
    path = tmp_path / "adapter.safetensors"
    save_file(adapter.state_dict(), path, metadata=metadata)

    with pytest.raises(ValueError, match=message):
        LinearAdapter.load(path)


def test_load_rejects_missing_unexpected_and_mismatched_tensors(tmp_path):
    config = AdapterConfig(input_dim=4)
    metadata = _metadata(config)

    missing_path = tmp_path / "missing.safetensors"
    save_file({"linear.weight": torch.zeros(4, 4)}, missing_path, metadata=metadata)
    with pytest.raises(ValueError, match=r"missing=\['linear.bias'\]"):
        LinearAdapter.load(missing_path)

    unexpected_path = tmp_path / "unexpected.safetensors"
    state_dict = LinearAdapter(config).state_dict()
    state_dict["extra"] = torch.zeros(1)
    save_file(state_dict, unexpected_path, metadata=metadata)
    with pytest.raises(ValueError, match=r"unexpected=\['extra'\]"):
        LinearAdapter.load(unexpected_path)

    mismatched_path = tmp_path / "mismatched.safetensors"
    save_file(
        {"linear.weight": torch.zeros(3, 4), "linear.bias": torch.zeros(4)},
        mismatched_path,
        metadata=metadata,
    )
    with pytest.raises(ValueError, match="tensors do not match"):
        LinearAdapter.load(mismatched_path)


def test_load_rejects_corrupt_safetensors_without_legacy_fallback(tmp_path):
    marker = tmp_path / "pwned-via-fallback.txt"

    class _Evil:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    path = tmp_path / "evil.safetensors"
    torch.save(_Evil(), path)

    with pytest.raises(SafetensorError):
        LinearAdapter.load(path)
    assert not marker.exists()


def test_load_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match="Unsupported checkpoint extension"):
        LinearAdapter.load(tmp_path / "adapter.bin")


def test_safetensors_rejects_unsupported_map_location(tmp_path):
    adapter = LinearAdapter(AdapterConfig(input_dim=4))
    path = tmp_path / "adapter.safetensors"
    adapter.save(path)

    with pytest.raises(TypeError, match="string or torch.device"):
        LinearAdapter.load(path, map_location={"cpu": "cpu"})


def test_safetensors_accepts_torch_device_map_location(tmp_path):
    adapter = LinearAdapter(AdapterConfig(input_dim=4))
    path = tmp_path / "adapter.safetensors"
    adapter.save(path)

    loaded = LinearAdapter.load(path, map_location=torch.device("cpu"))
    assert loaded.config == adapter.config


def test_safetensors_accepts_string_map_location_and_preserves_legacy_dtype_semantics(tmp_path):
    adapter = LinearAdapter(AdapterConfig(input_dim=4)).double()
    path = tmp_path / "adapter.safetensors"
    adapter.save(str(path))

    loaded = LinearAdapter.load(str(path), map_location="cpu")
    assert next(loaded.parameters()).device.type == "cpu"
    assert next(loaded.parameters()).dtype == torch.get_default_dtype()


def test_safetensors_extension_is_case_insensitive(tmp_path):
    adapter = LinearAdapter(AdapterConfig(input_dim=4))
    path = tmp_path / "adapter.SAFETENSORS"
    adapter.save(path)

    loaded = LinearAdapter.load(path)
    assert loaded.config == adapter.config


def test_migration_rejects_non_legacy_source(tmp_path):
    with pytest.raises(ValueError, match="Migration source"):
        LinearAdapter.migrate_checkpoint(
            tmp_path / "adapter.safetensors", tmp_path / "migrated.safetensors"
        )


def test_migration_rejects_legacy_destination_extension(tmp_path):
    adapter = LinearAdapter(AdapterConfig(input_dim=4))
    source = tmp_path / "adapter.pt"
    torch.save({"config": asdict(adapter.config), "state_dict": adapter.state_dict()}, source)

    with pytest.raises(ValueError, match="Migration destination"):
        LinearAdapter.migrate_checkpoint(source, tmp_path / "migrated.pt")


def test_loads_and_migrates_legacy_checkpoint(tmp_path):
    adapter = LinearAdapter(AdapterConfig(input_dim=4, normalize_output=False))
    with torch.no_grad():
        adapter.linear.weight.add_(0.05)
    legacy_path = tmp_path / "adapter.pt"
    torch.save({"config": asdict(adapter.config), "state_dict": adapter.state_dict()}, legacy_path)

    with pytest.warns(
        DeprecationWarning, match="Legacy PyTorch checkpoints are deprecated"
    ) as load_warnings:
        loaded = LinearAdapter.load(legacy_path)
    assert load_warnings[0].filename == __file__
    assert loaded.config == adapter.config
    for key, tensor in adapter.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[key])

    destination = tmp_path / "adapter.safetensors"
    with pytest.warns(DeprecationWarning) as migration_warnings:
        migrated = LinearAdapter.migrate_checkpoint(legacy_path, destination)
    assert migration_warnings[0].filename == __file__
    reloaded = LinearAdapter.load(destination)
    assert migrated.config == reloaded.config == adapter.config
    for key, tensor in adapter.state_dict().items():
        assert torch.equal(tensor, reloaded.state_dict()[key])


def test_load_rejects_invalid_legacy_payload(tmp_path):
    adapter = LinearAdapter(AdapterConfig(input_dim=4))
    path = tmp_path / "adapter.pt"
    torch.save(
        {
            "config": asdict(adapter.config),
            "state_dict": adapter.state_dict(),
            "unexpected": True,
        },
        path,
    )

    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="exactly"):
        LinearAdapter.load(path)


def test_load_rejects_invalid_legacy_state_dict(tmp_path):
    config = AdapterConfig(input_dim=4)
    invalid_value_path = tmp_path / "invalid-value.pt"
    torch.save(
        {"config": asdict(config), "state_dict": {"linear.weight": "no"}}, invalid_value_path
    )
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="strings to tensors"):
        LinearAdapter.load(invalid_value_path)

    missing_key_path = tmp_path / "missing-key.pt"
    torch.save(
        {"config": asdict(config), "state_dict": {"linear.weight": torch.zeros(4, 4)}},
        missing_key_path,
    )
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="missing=.*linear.bias"):
        LinearAdapter.load(missing_key_path)

    mismatched_path = tmp_path / "mismatched.pt"
    torch.save(
        {
            "config": asdict(config),
            "state_dict": {
                "linear.weight": torch.zeros(3, 4),
                "linear.bias": torch.zeros(4),
            },
        },
        mismatched_path,
    )
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ValueError, match="Legacy checkpoint tensors do not match"),
    ):
        LinearAdapter.load(mismatched_path)


def test_load_rejects_malicious_pickle(tmp_path):
    """A crafted checkpoint must not execute code on load (CWE-502 regression).

    ``LinearAdapter.load`` uses ``weights_only=True``, so a pickle payload that
    defines ``__reduce__`` must raise instead of running its side effect.
    """
    marker = tmp_path / "pwned.txt"

    class _Evil:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    evil_path = tmp_path / "evil.pt"
    torch.save({"config": {"input_dim": 4}, "state_dict": {}, "x": _Evil()}, evil_path)

    with pytest.warns(DeprecationWarning), pytest.raises(pickle.UnpicklingError):
        LinearAdapter.load(evil_path)
    assert not marker.exists()


def test_triplet_loss_zero_when_separated():
    loss = TripletLoss(margin=0.1, distance="cosine")
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[-1.0, 0.0]])
    assert loss(anchor, positive, negative).item() == 0.0


def test_triplet_loss_positive_when_violated():
    loss = TripletLoss(margin=0.5, distance="cosine")
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[-1.0, 0.0]])
    negative = torch.tensor([[1.0, 0.0]])
    assert loss(anchor, positive, negative).item() > 0.0
