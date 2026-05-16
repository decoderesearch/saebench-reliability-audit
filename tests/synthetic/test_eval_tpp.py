"""End-to-end CPU smoke test for TPP synthetic eval."""

from __future__ import annotations

import torch
from sae_lens import StandardSAE, StandardSAEConfig

from saebench_audit.synthetic.eval_tpp import run_tpp_for_sibling_group


def test_tpp_returns_expected_keys() -> None:
    cfg = StandardSAEConfig(d_in=8, d_sae=16, device="cpu", dtype="float32")
    sae = StandardSAE(cfg)
    torch.manual_seed(0)
    hidden = torch.randn(200, 8)
    per_class_indices = {
        f"class_{i}": torch.arange(50 * i, 50 * (i + 1)) for i in range(4)
    }
    result = run_tpp_for_sibling_group(
        sae,
        hidden,
        per_class_indices,
        n_values=[2, 5],
        device="cpu",
        samples_per_class=40,
        sae_batch_size=32,
        probe_test_batch_size=32,
        probe_train_batch_size=8,
        probe_epochs=2,
    )
    assert {"per_task", "overall", "clean_accs"} <= set(result)
    overall = result["overall"]
    assert "tpp_threshold_2_total_metric" in overall
    assert "tpp_threshold_5_total_metric" in overall


def test_tpp_returns_empty_with_too_few_classes() -> None:
    cfg = StandardSAEConfig(d_in=8, d_sae=16, device="cpu", dtype="float32")
    sae = StandardSAE(cfg)
    hidden = torch.randn(200, 8)
    # Only one class has enough samples.
    per_class_indices = {
        "class_0": torch.arange(100),
        "class_1": torch.arange(100, 110),  # too few
        "class_2": torch.arange(110, 115),  # too few
        "class_3": torch.arange(115, 120),  # too few
    }
    result = run_tpp_for_sibling_group(
        sae,
        hidden,
        per_class_indices,
        n_values=[2],
        device="cpu",
        samples_per_class=40,
        sae_batch_size=32,
        probe_test_batch_size=32,
        probe_train_batch_size=8,
        probe_epochs=2,
    )
    # min_pos defaults to 25, so only class_0 is eligible — a single class
    # is below the 2-class minimum, so the eval returns an empty result.
    assert result == {"per_task": {}, "overall": {}}
