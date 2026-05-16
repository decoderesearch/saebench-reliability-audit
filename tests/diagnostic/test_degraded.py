"""Tests for degraded-SAE constructors.

CPU-only; uses tiny JumpReLU SAEs.
"""

from __future__ import annotations

from typing import cast

import torch
from sae_lens import JumpReLUSAE, JumpReLUSAEConfig

from saebench_audit.diagnostic.degraded import (
    best_k_variant,
    permute_decoder,
    random_init_variant,
    random_l0_matched,
)


def _make_jumprelu(d_in: int = 8, d_sae: int = 32) -> JumpReLUSAE:
    cfg = JumpReLUSAEConfig(d_in=d_in, d_sae=d_sae, device="cpu", dtype="float32")
    return JumpReLUSAE(cfg)


def test_permute_decoder_keeps_encoder_unchanged() -> None:
    base = _make_jumprelu()
    permuted = permute_decoder(base, seed=0)
    # Encoder weights are identical → encoder output is identical.
    x = torch.randn(4, 8)
    assert torch.allclose(base.encode(x), permuted.encode(x))
    # Decoder rows are a permutation of base.W_dec.
    base_rows = {tuple(r.tolist()) for r in base.W_dec}
    permuted_rows = {tuple(r.tolist()) for r in permuted.W_dec}
    assert base_rows == permuted_rows
    # And the order has actually changed.
    assert not torch.allclose(base.W_dec, permuted.W_dec)


def test_permute_decoder_does_not_mutate_base() -> None:
    base = _make_jumprelu()
    base_W_dec_before = base.W_dec.detach().clone()
    permute_decoder(base, seed=0)
    assert torch.allclose(base.W_dec, base_W_dec_before)


def test_random_init_variant_zeroes_biases() -> None:
    base = _make_jumprelu()
    new = random_init_variant(base, seed=0)
    b_enc = cast(torch.Tensor, new.b_enc)
    b_dec = cast(torch.Tensor, new.b_dec)
    threshold = cast(torch.Tensor, new.threshold)
    assert torch.equal(b_enc, torch.zeros_like(b_enc))
    assert torch.equal(b_dec, torch.zeros_like(b_dec))
    assert torch.equal(threshold, torch.ones_like(threshold))
    assert not torch.allclose(
        cast(torch.Tensor, base.W_enc), cast(torch.Tensor, new.W_enc)
    )


def test_random_init_variant_is_deterministic_per_seed() -> None:
    base = _make_jumprelu()
    a = random_init_variant(base, seed=42)
    b = random_init_variant(base, seed=42)
    assert torch.allclose(a.W_enc, b.W_enc)
    assert torch.allclose(a.W_dec, b.W_dec)


def test_random_l0_matched_threshold_yields_target_l0() -> None:
    """The tuned threshold should put L0 close to the target on calibration data."""
    base = _make_jumprelu(d_sae=64)
    torch.manual_seed(0)
    calibration = torch.randn(2000, 8)
    target_l0 = 5.0
    new = random_l0_matched(base, calibration, target_l0=target_l0, seed=0)
    z = new.encode(calibration)
    measured_l0 = (z > 0).float().sum(-1).mean().item()
    # Allow ±50% slack — the threshold is a quantile, not a constraint.
    assert 0.5 * target_l0 < measured_l0 < 1.5 * target_l0


def test_best_k_variant_zeros_excess_latents() -> None:
    base = _make_jumprelu(d_sae=32)
    torch.manual_seed(0)
    calibration = torch.randn(200, 8)
    new = best_k_variant(base, calibration, k=4)
    # Exactly 4 latents should remain non-zero in W_dec; the rest must be zero.
    nonzero_rows = (new.W_dec.abs().sum(dim=-1) > 0).sum().item()
    assert nonzero_rows == 4
    # Threshold for the dropped latents pushed to ∞ effectively.
    threshold = cast(torch.Tensor, new.threshold)
    W_dec = cast(torch.Tensor, new.W_dec)
    dropped = ~(W_dec.abs().sum(dim=-1) > 0)
    assert (threshold[dropped] >= 1e8).all().item()


def test_best_k_variant_returns_unchanged_if_k_ge_d_sae() -> None:
    base = _make_jumprelu(d_sae=16)
    calibration = torch.randn(50, 8)
    new = best_k_variant(base, calibration, k=16)
    assert torch.allclose(base.W_dec, new.W_dec)
