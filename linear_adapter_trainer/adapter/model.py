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

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


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
        """Save weights and architecture to a single ``.pt`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: Any = "cpu") -> LinearAdapter:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        adapter = cls(AdapterConfig(**payload["config"]))
        adapter.load_state_dict(payload["state_dict"])
        adapter.eval()
        return adapter
