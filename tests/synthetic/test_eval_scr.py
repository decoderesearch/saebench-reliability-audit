"""End-to-end CPU smoke test for SCR synthetic eval."""

from __future__ import annotations

import sae_bench.sae_bench_utils.dataset_info as dataset_info
import torch
from sae_lens import StandardSAE, StandardSAEConfig

from saebench_audit.synthetic.eval_scr import (
    build_scr_class_acts,
    compute_scr_metrics,
    run_scr_for_task,
    scr_paired_keys,
)


def test_scr_paired_keys_context_restores_original() -> None:
    original = dataset_info.PAIRED_CLASS_KEYS
    with scr_paired_keys():
        assert dataset_info.PAIRED_CLASS_KEYS == {
            "S": "S_neg",
            "T": "T_neg",
            "bias": "bias_neg",
        }
    assert dataset_info.PAIRED_CLASS_KEYS == original


def test_build_scr_class_acts_returns_six_balanced_classes() -> None:
    torch.manual_seed(0)
    n = 600
    hidden = torch.randn(n, 8)
    t_label = (torch.rand(n) > 0.5).long()
    s_label = (torch.rand(n) > 0.5).long()
    out = build_scr_class_acts(
        hidden, t_label, s_label, samples_per_class=30, device="cpu"
    )
    assert set(out) == {"S", "S_neg", "T", "T_neg", "bias", "bias_neg"}
    sizes = {k: v.shape[0] for k, v in out.items()}
    # All six balanced to the same size, capped at samples_per_class.
    assert len(set(sizes.values())) == 1
    assert next(iter(sizes.values())) <= 30


def test_build_scr_class_acts_returns_empty_when_too_few() -> None:
    hidden = torch.randn(20, 8)
    # All labels = 0 → S_neg, T_neg, bias_neg are full but S, T, bias empty.
    t_label = torch.zeros(20, dtype=torch.long)
    s_label = torch.zeros(20, dtype=torch.long)
    out = build_scr_class_acts(
        hidden, t_label, s_label, samples_per_class=10, device="cpu"
    )
    assert out == {}


def test_compute_scr_metrics_uses_canonical_formula() -> None:
    """Hand-written check that the SCR formula matches the one in the docstring."""
    class_accuracies = {
        "S": {10: {"bias probe on T data": 0.7, "bias probe on S data": 0.6}},
        "T": {10: {"bias probe on T data": 0.6, "bias probe on S data": 0.5}},
        "bias": {10: {"bias probe on T data": 0.5, "bias probe on S data": 0.5}},
    }
    clean_accs = {
        "T": 0.9,
        "S": 0.8,
        "bias probe on T data": 0.4,
        "bias probe on S data": 0.4,
    }
    out = compute_scr_metrics(class_accuracies, clean_accs)
    # dir1 ablates S, evaluates bias probe on T data → (changed - orig) / (clean - orig)
    expected_dir1 = (0.7 - 0.4) / (0.9 - 0.4)
    expected_dir2 = (0.5 - 0.4) / (0.8 - 0.4)
    assert abs(out["scr_dir1_threshold_10"] - expected_dir1) < 1e-9
    assert abs(out["scr_dir2_threshold_10"] - expected_dir2) < 1e-9
    # bias_on_T_orig and bias_on_S_orig are equal here, so canonical metric
    # falls back to the average of dir1 and dir2.
    assert (
        abs(out["scr_metric_threshold_10"] - 0.5 * (expected_dir1 + expected_dir2))
        < 1e-9
    )


def test_run_scr_returns_metrics_dict() -> None:
    """End-to-end SCR eval on a contrived but balanced (T, S) task."""
    cfg = StandardSAEConfig(d_in=8, d_sae=16, device="cpu", dtype="float32")
    sae = StandardSAE(cfg)
    torch.manual_seed(0)
    n = 600
    hidden = torch.randn(n, 8)
    # Balanced T and S labels with some co-firing.
    t_label = (torch.rand(n) > 0.5).long()
    s_label = (torch.rand(n) > 0.5).long()
    result = run_scr_for_task(
        sae,
        hidden,
        t_label,
        s_label,
        n_values=[2],
        device="cpu",
        samples_per_class=40,
        sae_batch_size=32,
        probe_test_batch_size=32,
        probe_train_batch_size=8,
        probe_epochs=2,
    )
    assert "metrics" in result
    assert "scr_metric_threshold_2" in result["metrics"]
