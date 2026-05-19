"""Tests for the sampled-Matryoshka training script.

The smoke test monkey-patches ``run_training`` so it never starts an LM, then
inspects the config, override SAE, and snapshot schedule the training entry
point would have handed to SAELens. This catches regressions without paying the
cost of loading Gemma-2-2b.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from saebench_audit.saes.matryoshka_sae import (
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
)
from saebench_audit.training import common
from saebench_audit.training import train_sampled_matryoshka as sampled_mat
from saebench_audit.training.train_sampled_matryoshka import _snapshot_token_amounts


@pytest.fixture
def captured_call(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Replace ``run_training`` so we capture the would-be call instead of running."""
    captured: dict[str, object] = {}

    def fake_run(cfg, *, snapshots=None, override_sae=None):  # type: ignore[no-untyped-def]
        captured["cfg"] = cfg
        captured["snapshots"] = snapshots
        captured["override_sae"] = override_sae
        return common.TrainingResult(final_dir="x", checkpoints_dir="x/c")

    monkeypatch.setattr(sampled_mat, "run_training", fake_run)
    return captured


# --------------------------------------------------------------------------- #
# train_sampled_matryoshka
# --------------------------------------------------------------------------- #


def test_train_sampled_matryoshka_uses_log_uniform(
    captured_call: dict[str, object], tmp_path: Path
) -> None:
    sampled_mat.train_sampled_matryoshka(
        n_levels=3,
        output_path=str(tmp_path),
        training_tokens=1_000_000,
    )
    cfg = cast(Any, captured_call["cfg"])
    matryoshka_cfg = cfg.sae
    assert isinstance(matryoshka_cfg, MatryoshkaBatchTopKTrainingSAEConfig)
    assert matryoshka_cfg.level_selection_mode == "log_uniform"
    assert matryoshka_cfg.num_sampled_levels == 3
    assert matryoshka_cfg.min_matryoshka_width == 64
    assert matryoshka_cfg.use_matryoshka_aux_loss is True
    assert matryoshka_cfg.k == sampled_mat.K_DEFAULT
    assert isinstance(captured_call["override_sae"], MatryoshkaBatchTopKTrainingSAE)
    # Snapshots must be wired through to run_training, keyed by training step.
    snapshots = cast(dict[int, str], captured_call["snapshots"])
    assert snapshots, "sampled-matryoshka training should request snapshots"
    assert 1 in snapshots


# --------------------------------------------------------------------------- #
# Snapshot schedule
# --------------------------------------------------------------------------- #


def test_sampled_matryoshka_schedule_matches_paper() -> None:
    """The 300M-token sampled-Matryoshka panel: every 30M tokens + step 1 = 11."""
    amounts = _snapshot_token_amounts(300_000_000)
    assert amounts == list(range(30_000_000, 300_000_001, 30_000_000))
    assert len(amounts) == 10

    snapshots = common.build_snapshots("/out", amounts)
    assert len(snapshots) == 11
    assert 1 in snapshots
