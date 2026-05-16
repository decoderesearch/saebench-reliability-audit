"""Tests for sparse-probing helpers.

These exercise ``balanced_split``, ``encode_in_chunks``, and
``run_sae_probes_for_task`` on a tiny CPU SAE without touching SAEBench's
heavy code paths.
"""

from __future__ import annotations

import torch
from sae_lens import StandardSAE, StandardSAEConfig

from saebench_audit.synthetic.eval_sae_probes import (
    balanced_split,
    encode_in_chunks,
    run_sae_probes_for_task,
)


def test_balanced_split_balances_classes() -> None:
    torch.manual_seed(0)
    labels = torch.cat([torch.ones(60), torch.zeros(60)])
    train_idx, test_idx = balanced_split(
        labels,
        max_train=80,
        target_train_pos=20,
        target_test_pos=10,
        seed=0,
    )
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    # Both train and test should have equal positive/negative counts.
    assert (train_labels == 1).sum().item() == (train_labels == 0).sum().item()
    assert (test_labels == 1).sum().item() == (test_labels == 0).sum().item()


def test_balanced_split_returns_empty_when_not_enough_positives() -> None:
    labels = torch.zeros(100)
    labels[:3] = 1.0  # only 3 positives
    train_idx, test_idx = balanced_split(
        labels,
        max_train=80,
        target_train_pos=20,
        target_test_pos=10,
        seed=0,
    )
    assert len(train_idx) == 0 and len(test_idx) == 0


def test_encode_in_chunks_matches_one_shot() -> None:
    cfg = StandardSAEConfig(d_in=8, d_sae=16, device="cpu", dtype="float32")
    sae = StandardSAE(cfg)
    torch.manual_seed(0)
    hidden = torch.randn(32, 8)
    z_chunked = encode_in_chunks(sae, hidden, batch_size=7, device="cpu")
    z_full = sae.encode(hidden).detach()
    assert torch.allclose(z_chunked, z_full, atol=1e-5)


def test_run_sae_probes_returns_metrics_dict() -> None:
    """Run sae_bench-mode sparse probing on a contrived but separable task."""
    torch.manual_seed(0)
    n = 200
    d_sae = 8
    # latent 0 fires only when label is 1 → top-k=1 probe should solve it.
    sae_acts = torch.zeros(n, d_sae)
    labels = torch.zeros(n, dtype=torch.long)
    labels[: n // 2] = 1
    sae_acts[: n // 2, 0] = 1.0
    perm = torch.randperm(n)
    sae_acts = sae_acts[perm]
    labels = labels[perm]
    train_idx = torch.arange(160)
    test_idx = torch.arange(160, n)
    out = run_sae_probes_for_task(
        sae_acts, labels, train_idx, test_idx, ks=[1], mode="sae_bench"
    )
    assert 1 in out
    # Top-1 probe should achieve essentially perfect accuracy on this task.
    assert out[1]["test_acc"] > 0.95


def test_run_sae_probes_handles_empty_split() -> None:
    sae_acts = torch.zeros(10, 4)
    labels = torch.zeros(10, dtype=torch.long)
    out = run_sae_probes_for_task(
        sae_acts,
        labels,
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        ks=[1, 4],
    )
    assert out[1]["test_acc"] == 0.5
    assert out[4]["test_acc"] == 0.5
