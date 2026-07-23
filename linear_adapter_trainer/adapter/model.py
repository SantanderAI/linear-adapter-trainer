# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""The linear embedding adapter (Module 2).

A :class:`LinearAdapter` is a small ``torch`` module applied to *query*
embeddings. It learns to shift them in latent space so that they land closer
to relevant chunks and farther from irrelevant ones, while leaving the chunk
("corpus") embeddings untouched. This means the adapter is cheap to train and
can be dropped in front of any existing vector index at query time.
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

_FORMAT_VERSION = "1"
_CONFIG_METADATA_KEY = "linear_adapter_config"
_VERSION_METADATA_KEY = "linear_adapter_format_version"
_METADATA_KEYS = {_CONFIG_METADATA_KEY, _VERSION_METADATA_KEY}
_CONFIG_KEYS = {"input_dim", "output_dim", "bias", "residual", "normalize_output"}
_LEGACY_SUFFIXES = {".pt", ".pth"}


@dataclass(slots=True)
class AdapterConfig:
    """Architecture of a :class:`LinearAdapter`.

    Attributes:
        input_dim: Dimensionality of the input embeddings.
        output_dim: Output dimensionality (defaults to ``input_dim``).
        bias: Whether to include a bias term.
        residual: If ``True``, learn a residual ``y = x + W x`` (requires
            ``output_dim == input_dim``); initialized near identity.
        normalize_output: L2-normalize the output vectors.
    """

    input_dim: int
    output_dim: int | None = None
    bias: bool = True
    residual: bool = True
    normalize_output: bool = True

    def resolved_output_dim(self) -> int:
        return self.output_dim if self.output_dim is not None else self.input_dim


class LinearAdapter(nn.Module):
    """A learned linear transform over query embeddings."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        out_dim = config.resolved_output_dim()
        if config.residual and out_dim != config.input_dim:
            raise ValueError("residual adapters require output_dim == input_dim.")
        self.config = config
        self.linear = nn.Linear(config.input_dim, out_dim, bias=config.bias)
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.config.residual:
                # Start as a near no-op: y = x + 0.
                self.linear.weight.zero_()
            elif self.linear.weight.shape[0] == self.linear.weight.shape[1]:
                # Square non-residual: start from identity.
                self.linear.weight.copy_(torch.eye(self.linear.weight.shape[0]))
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.config.residual:
            out = x + out
        if self.config.normalize_output:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out

    # -- inference helpers -------------------------------------------------
    @torch.inference_mode()
    def transform(self, embeddings: np.ndarray, *, batch_size: int = 1024) -> np.ndarray:
        """Apply the adapter to a matrix of embeddings (numpy in/out)."""
        self.eval()
        device = next(self.parameters()).device
        tensor = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
        outputs: list[np.ndarray] = []
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[start : start + batch_size].to(device)
            outputs.append(self(batch).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save weights and architecture to a pickle-free safetensors file."""
        path = Path(path)
        if path.suffix.lower() != ".safetensors":
            raise ValueError(
                "New checkpoints must use the '.safetensors' extension. "
                "Use LinearAdapter.migrate_checkpoint() for legacy '.pt' checkpoints."
            )
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            _VERSION_METADATA_KEY: _FORMAT_VERSION,
            _CONFIG_METADATA_KEY: json.dumps(
                asdict(self.config), sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
        }
        tensors = {
            name: tensor.detach().cpu().contiguous() for name, tensor in self.state_dict().items()
        }

        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=".linear-adapter-", suffix=".tmp"
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            save_file(tensors, str(temporary_path), metadata=metadata)
            with temporary_path.open("r+b") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            if os.name != "nt":
                directory_descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        finally:
            temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path, *, map_location: Any = "cpu") -> LinearAdapter:
        """Load a safetensors checkpoint or an explicitly named legacy checkpoint.

        ``.safetensors`` files use strict, versioned JSON metadata and never
        invoke pickle. Legacy ``.pt`` and ``.pth`` files remain readable through
        PyTorch's restricted ``weights_only=True`` loader and emit a deprecation
        warning. Parser selection is extension-based: a malformed safetensors
        file is never retried through the legacy loader. ``map_location`` controls
        tensor deserialization; as in the legacy implementation, the returned
        adapter uses the module's default device and dtype.

        Only load checkpoints from sources you trust and verify artifact
        checksums before loading, especially for legacy PyTorch checkpoints.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".safetensors":
            return cls._load_safetensors(path, map_location=map_location)
        if suffix in _LEGACY_SUFFIXES:
            return cls._load_legacy(path, map_location=map_location, warning_stacklevel=3)
        raise ValueError(
            "Unsupported checkpoint extension. Expected '.safetensors', '.pt', or '.pth'."
        )

    @classmethod
    def migrate_checkpoint(
        cls,
        source: str | Path,
        destination: str | Path,
        *,
        map_location: Any = "cpu",
    ) -> LinearAdapter:
        """Migrate a legacy ``.pt`` or ``.pth`` checkpoint to safetensors."""
        source = Path(source)
        destination = Path(destination)
        if source.suffix.lower() not in _LEGACY_SUFFIXES:
            raise ValueError("Migration source must use the '.pt' or '.pth' extension.")
        if destination.suffix.lower() != ".safetensors":
            raise ValueError("Migration destination must use the '.safetensors' extension.")
        adapter = cls._load_legacy(source, map_location=map_location, warning_stacklevel=3)
        adapter.save(destination)
        return adapter

    @classmethod
    def _load_safetensors(cls, path: Path, *, map_location: Any) -> LinearAdapter:
        if not isinstance(map_location, (str, torch.device)):
            raise TypeError("Safetensors map_location must be a string or torch.device.")
        device = str(map_location)
        with safe_open(str(path), framework="pt", device=device) as checkpoint:
            metadata = checkpoint.metadata()
            config = _config_from_metadata(metadata)
            adapter = cls(config)
            expected_keys = set(adapter.state_dict())
            actual_keys = set(checkpoint.keys())
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)
                unexpected = sorted(actual_keys - expected_keys)
                raise ValueError(
                    f"Invalid checkpoint tensor keys: missing={missing}, unexpected={unexpected}."
                )
            state_dict = {name: checkpoint.get_tensor(name) for name in sorted(actual_keys)}

        try:
            adapter.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise ValueError("Checkpoint tensors do not match the adapter architecture.") from error
        adapter.eval()
        return adapter

    @classmethod
    def _load_legacy(
        cls, path: Path, *, map_location: Any, warning_stacklevel: int
    ) -> LinearAdapter:
        warnings.warn(
            "Legacy PyTorch checkpoints are deprecated. Migrate with "
            "LinearAdapter.migrate_checkpoint(source, destination).",
            DeprecationWarning,
            stacklevel=warning_stacklevel,
        )
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(payload, Mapping) or set(payload) != {"config", "state_dict"}:
            raise ValueError("Legacy checkpoint must contain exactly 'config' and 'state_dict'.")
        config = _validate_config(payload["config"])
        state_dict = payload["state_dict"]
        if not isinstance(state_dict, Mapping) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state_dict.items()
        ):
            raise ValueError("Legacy checkpoint state_dict must map strings to tensors.")

        adapter = cls(config)
        expected_keys = set(adapter.state_dict())
        actual_keys = set(state_dict)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"Invalid legacy checkpoint tensor keys: missing={missing}, "
                f"unexpected={unexpected}."
            )
        try:
            adapter.load_state_dict(dict(state_dict), strict=True)
        except RuntimeError as error:
            raise ValueError(
                "Legacy checkpoint tensors do not match the adapter architecture."
            ) from error
        adapter.eval()
        return adapter


def _config_from_metadata(metadata: dict[str, str] | None) -> AdapterConfig:
    if metadata is None or set(metadata) != _METADATA_KEYS:
        actual_keys = sorted(metadata or {})
        raise ValueError(
            f"Invalid checkpoint metadata keys: expected={sorted(_METADATA_KEYS)}, "
            f"actual={actual_keys}."
        )
    version = metadata[_VERSION_METADATA_KEY]
    if version != _FORMAT_VERSION:
        raise ValueError(
            f"Unsupported linear adapter checkpoint format version {version!r}; "
            f"supported version is {_FORMAT_VERSION}."
        )
    try:
        config = json.loads(metadata[_CONFIG_METADATA_KEY])
    except json.JSONDecodeError as error:
        raise ValueError("Checkpoint adapter configuration is not valid JSON.") from error
    return _validate_config(config)


def _validate_config(value: Any) -> AdapterConfig:
    if not isinstance(value, Mapping) or set(value) != _CONFIG_KEYS:
        actual_keys = sorted(value) if isinstance(value, Mapping) else []
        raise ValueError(
            f"Invalid adapter configuration keys: expected={sorted(_CONFIG_KEYS)}, "
            f"actual={actual_keys}."
        )

    input_dim = value["input_dim"]
    output_dim = value["output_dim"]
    if type(input_dim) is not int or input_dim <= 0:
        raise ValueError("Adapter configuration input_dim must be a positive integer.")
    if output_dim is not None and (type(output_dim) is not int or output_dim <= 0):
        raise ValueError("Adapter configuration output_dim must be null or a positive integer.")
    for key in ("bias", "residual", "normalize_output"):
        if type(value[key]) is not bool:
            raise ValueError(f"Adapter configuration {key} must be a boolean.")

    config = AdapterConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        bias=value["bias"],
        residual=value["residual"],
        normalize_output=value["normalize_output"],
    )
    if config.residual and config.resolved_output_dim() != config.input_dim:
        raise ValueError("residual adapters require output_dim == input_dim.")
    return config
