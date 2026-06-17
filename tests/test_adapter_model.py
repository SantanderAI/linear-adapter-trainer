# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from linear_adapter_trainer.adapter.losses import TripletLoss
from linear_adapter_trainer.adapter.model import AdapterConfig, LinearAdapter


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


def test_save_and_load(tmp_path):
    adapter = LinearAdapter(AdapterConfig(input_dim=4, residual=True))
    with torch.no_grad():
        adapter.linear.weight.add_(0.05)
    path = tmp_path / "adapter.pt"
    adapter.save(path)

    loaded = LinearAdapter.load(path)
    x = torch.randn(3, 4)
    assert torch.allclose(adapter(x), loaded(x), atol=1e-6)


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
