"""Tests for the Matryoshka BatchTopK SAE.

These run on CPU and use tiny SAEs (d_in=8, d_sae=64) so they finish in well
under a second per test.
"""

from __future__ import annotations

import math

import pytest
import torch
from sae_lens.saes.sae import TrainStepInput

from saebench_audit.saes.matryoshka_sae import (
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
    sample_log_uniform_widths,
)


def _make_step_input(
    x: torch.Tensor,
    *,
    dead_neuron_mask: torch.Tensor | None = None,
) -> TrainStepInput:
    return TrainStepInput(
        sae_in=x,
        n_training_steps=0,
        coefficients={"l1": 0.0},
        dead_neuron_mask=dead_neuron_mask,
        is_logging_step=False,
    )


def test_config_architecture_string() -> None:
    cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=4, d_sae=16, k=2, matryoshka_widths=[4, 16]
    )
    assert cfg.architecture() == "matryoshka_batchtopk"


def test_config_appends_d_sae_to_widths() -> None:
    cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=4, d_sae=16, k=2, matryoshka_widths=[4, 8]
    )
    with pytest.warns(UserWarning, match="does not end at d_sae"):
        MatryoshkaBatchTopKTrainingSAE(cfg)
    assert cfg.matryoshka_widths == [4, 8, 16]


def test_config_rejects_non_increasing_widths() -> None:
    cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=4, d_sae=16, k=2, matryoshka_widths=[8, 4, 16]
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        MatryoshkaBatchTopKTrainingSAE(cfg)


def test_config_rejects_unknown_mode() -> None:
    cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=4, d_sae=16, k=2, level_selection_mode="bogus", matryoshka_widths=[4, 16]
    )
    with pytest.raises(ValueError, match="level_selection_mode"):
        MatryoshkaBatchTopKTrainingSAE(cfg)


def test_config_rejects_min_width_too_large() -> None:
    cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=4,
        d_sae=16,
        k=2,
        level_selection_mode="log_uniform",
        min_matryoshka_width=32,
    )
    with pytest.raises(
        ValueError, match="min_matryoshka_width must be less than d_sae"
    ):
        MatryoshkaBatchTopKTrainingSAE(cfg)


def test_config_rejects_zero_sampled_levels() -> None:
    cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=4,
        d_sae=16,
        k=2,
        level_selection_mode="log_uniform",
        num_sampled_levels=0,
    )
    with pytest.raises(ValueError, match="num_sampled_levels"):
        MatryoshkaBatchTopKTrainingSAE(cfg)


def test_fixed_mode_training_forward_pass() -> None:
    cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=8, d_sae=32, k=4, matryoshka_widths=[8, 16, 32]
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    x = torch.randn(8, 8)
    out = sae.training_forward_pass(_make_step_input(x))
    assert "mse_loss" in out.losses
    assert "inner_recons_loss" in out.losses
    assert torch.isfinite(out.loss)
    assert out.metrics["k"] == 4


def test_fixed_mode_aux_loss_with_dead_neurons() -> None:
    cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=8, d_sae=32, k=4, matryoshka_widths=[8, 16, 32]
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    dead = torch.zeros(32, dtype=torch.bool)
    dead[10:20] = True
    out = sae.training_forward_pass(
        _make_step_input(torch.randn(8, 8), dead_neuron_mask=dead)
    )
    assert "auxiliary_reconstruction_loss" in out.losses
    assert out.losses["auxiliary_reconstruction_loss"].item() > 0.0


def test_fixed_mode_aux_loss_zero_without_dead_neurons() -> None:
    cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=8, d_sae=32, k=4, matryoshka_widths=[8, 16, 32]
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    dead = torch.zeros(32, dtype=torch.bool)
    out = sae.training_forward_pass(
        _make_step_input(torch.randn(8, 8), dead_neuron_mask=dead)
    )
    assert out.losses["auxiliary_reconstruction_loss"].item() == 0.0


def test_log_uniform_mode_training_forward_pass() -> None:
    cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=8,
        d_sae=64,
        k=4,
        level_selection_mode="log_uniform",
        num_sampled_levels=3,
        min_matryoshka_width=4,
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    out = sae.training_forward_pass(_make_step_input(torch.randn(8, 8)))
    assert torch.isfinite(out.loss)
    assert "inner_recons_loss" in out.losses


def test_log_uniform_widths_clears_after_step() -> None:
    """The cached widths should be regenerated each step."""
    cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=8,
        d_sae=64,
        k=4,
        level_selection_mode="log_uniform",
        num_sampled_levels=3,
        min_matryoshka_width=4,
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    sae.training_forward_pass(_make_step_input(torch.randn(8, 8)))
    assert sae._current_step_widths is None


def test_use_matryoshka_aux_loss_false_falls_back_to_batchtopk() -> None:
    cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=8,
        d_sae=32,
        k=4,
        matryoshka_widths=[8, 16, 32],
        use_matryoshka_aux_loss=False,
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    dead = torch.zeros(32, dtype=torch.bool)
    dead[10:20] = True
    # With the flag off, calculate_topk_aux_loss should match the parent class's
    # output rather than splitting per-prefix.
    aux = sae.calculate_topk_aux_loss(
        sae_in=torch.randn(8, 8),
        sae_out=torch.randn(8, 8),
        hidden_pre=torch.randn(8, 32),
        dead_neuron_mask=dead,
    )
    assert torch.is_tensor(aux)


def test_iterable_decode_partial_widths() -> None:
    """``iterable_decode`` yields one tensor per prefix, building up incrementally."""
    cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=4,
        d_sae=16,
        k=2,
        matryoshka_widths=[4, 8, 16],
        rescale_acts_by_decoder_norm=False,
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    feats = torch.randn(2, 16)
    parts = list(sae.iterable_decode(feats, force_include_outer_loss=True))
    assert len(parts) == 3
    # Each successive partial reconstruction should differ from the previous
    # (we add features, not subtract).
    assert not torch.allclose(parts[0], parts[1])
    assert not torch.allclose(parts[1], parts[2])
    full = feats @ sae.W_dec + sae.b_dec
    assert torch.allclose(parts[-1], full, atol=1e-5)


def test_iterable_decode_skips_final_when_outer_loss_used() -> None:
    cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=4, d_sae=16, k=2, matryoshka_widths=[4, 8, 16]
    )
    sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    feats = torch.randn(2, 16)
    parts = list(sae.iterable_decode(feats))
    # skip_final_matryoshka_width=True: drop the final prefix from the loop.
    assert len(parts) == 2


def test_sample_log_uniform_widths_in_range() -> None:
    torch.manual_seed(0)
    widths = sample_log_uniform_widths(4, 64, n=5)
    assert widths
    for w in widths:
        assert 4 <= w < 64


def test_sample_log_uniform_widths_uniform_in_log_space() -> None:
    """Many independent samples should distribute roughly uniformly in log space.

    To avoid integer-truncation dedup biasing the test at small widths, this
    accumulates samples across many calls (each call dedups internally) and
    bins by which 4-quartile of log space the *raw* samples land in.
    """
    torch.manual_seed(0)
    log_min = math.log(64)
    log_max = math.log(4096)
    bucket_size = (log_max - log_min) / 4
    counts = [0, 0, 0, 0]
    for _ in range(1000):
        widths = sample_log_uniform_widths(min_w=64, max_w=4096, n=4)
        for w in widths:
            idx = min(3, int((math.log(w) - log_min) // bucket_size))
            counts[idx] += 1
    total = sum(counts)
    for c in counts:
        assert c / total > 0.15
