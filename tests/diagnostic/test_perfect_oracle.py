"""Tests for the PerfectSAE oracle.

We don't load a real SynthSAEBench model here — we build a tiny synthetic
``(hidden_pool, features_pool, gt_feature_vectors)`` triple by hand and check
the lookup table behaves correctly.
"""

from __future__ import annotations

import pytest
import torch

from saebench_audit.diagnostic.perfect_oracle import PerfectSAE


def _toy_pool(
    n_pool: int = 64, d_in: int = 8, num_gt_features: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    gt = torch.randn(num_gt_features, d_in)
    gt = gt / gt.norm(dim=-1, keepdim=True)
    features = (torch.rand(n_pool, num_gt_features) > 0.7).float()
    hidden = features @ gt + 0.01 * torch.randn(n_pool, d_in)
    return hidden, features, gt


def test_perfect_sae_encode_returns_first_d_sae_features() -> None:
    hidden, features, gt = _toy_pool(num_gt_features=16)
    d_sae = 8
    sae = PerfectSAE(hidden, features, gt, d_sae=d_sae, device="cpu")
    out = sae.encode(hidden)
    expected = features[:, :d_sae]
    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-5)


def test_perfect_sae_decode_uses_gt_directions() -> None:
    hidden, features, gt = _toy_pool(num_gt_features=16)
    d_sae = 8
    sae = PerfectSAE(hidden, features, gt, d_sae=d_sae, device="cpu")
    f = features[:, :d_sae]
    decoded = sae.decode(f)
    assert torch.allclose(decoded, f @ gt[:d_sae], atol=1e-5)


def test_perfect_sae_lookup_handles_3d_input() -> None:
    """Encode should preserve leading batch dims."""
    hidden, features, gt = _toy_pool(n_pool=32, num_gt_features=8)
    sae = PerfectSAE(hidden, features, gt, d_sae=4, device="cpu")
    # Build a (B, T, d_in) input where every row is a row from the pool.
    sub = hidden[:6].reshape(2, 3, -1)
    out = sae.encode(sub)
    assert out.shape == (2, 3, 4)
    expected = features[:6, :4].reshape(2, 3, 4)
    assert torch.allclose(out, expected, atol=1e-5)


def test_perfect_sae_rejects_size_mismatched_pool() -> None:
    hidden, features, gt = _toy_pool()
    bad_features = features[:-1]
    with pytest.raises(ValueError, match="same N"):
        PerfectSAE(hidden, bad_features, gt, d_sae=4, device="cpu")


def test_perfect_sae_rejects_d_in_mismatch() -> None:
    hidden, features, gt = _toy_pool(d_in=8)
    bad_gt = gt[:, :-1]
    with pytest.raises(ValueError, match="d_in"):
        PerfectSAE(hidden, features, bad_gt, d_sae=4, device="cpu")
