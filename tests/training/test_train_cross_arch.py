"""Tests for the cross-architecture training script.

The smoke tests monkey-patch ``run_training`` so they never start an LM, then
inspect the config, override SAE, and snapshot schedule the training entry
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
from saebench_audit.training import train_cross_arch as cross_arch
from saebench_audit.training.train_cross_arch import _snapshot_token_amounts


@pytest.fixture
def captured_call(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Replace ``run_training`` so we capture the would-be call instead of running."""
    captured: dict[str, object] = {}

    def fake_run(cfg, *, snapshots=None, override_sae=None):  # type: ignore[no-untyped-def]
        captured["cfg"] = cfg
        captured["snapshots"] = snapshots
        captured["override_sae"] = override_sae
        return common.TrainingResult(final_dir="x", checkpoints_dir="x/c")

    monkeypatch.setattr(cross_arch, "run_training", fake_run)
    return captured


# --------------------------------------------------------------------------- #
# train_cross_arch
# --------------------------------------------------------------------------- #


def test_train_cross_arch_btk_uses_batchtopk_no_override(
    captured_call: dict[str, object], tmp_path: Path
) -> None:
    cross_arch.train_cross_arch(
        variant="btk",
        k=50,
        output_path=str(tmp_path),
        training_tokens=1_000_000,
    )
    assert captured_call["override_sae"] is None
    cfg = cast(Any, captured_call["cfg"])
    assert cfg.sae.architecture() == "batchtopk"
    assert cfg.sae.k == 50
    assert cfg.training_tokens == 1_000_000
    # Snapshots must be wired through to run_training, keyed by training step.
    snapshots = cast(dict[int, str], captured_call["snapshots"])
    assert snapshots, "cross-arch training should request snapshots"
    assert all(isinstance(step, int) for step in snapshots)
    assert 1 in snapshots


def test_train_cross_arch_matryoshka_uses_three_fixed_widths(
    captured_call: dict[str, object], tmp_path: Path
) -> None:
    cross_arch.train_cross_arch(
        variant="matryoshka",
        k=100,
        output_path=str(tmp_path),
        training_tokens=1_000_000,
    )
    cfg = cast(Any, captured_call["cfg"])
    matryoshka_cfg = cfg.sae
    assert isinstance(matryoshka_cfg, MatryoshkaBatchTopKTrainingSAEConfig)
    assert matryoshka_cfg.k == 100
    # Paper uses (d_sae/16, d_sae/4, d_sae) for the cross-arch panel.
    assert matryoshka_cfg.matryoshka_widths == [
        common.D_SAE // 16,
        common.D_SAE // 4,
        common.D_SAE,
    ]
    assert matryoshka_cfg.level_selection_mode == "fixed"
    assert isinstance(captured_call["override_sae"], MatryoshkaBatchTopKTrainingSAE)


def test_train_cross_arch_rejects_unknown_variant(
    captured_call: dict[str, object],  # noqa: ARG001 — fixture protects against accidental real training
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="variant"):
        cross_arch.train_cross_arch(
            variant="unknown",  # type: ignore[arg-type]
            k=50,
            output_path=str(tmp_path),
            training_tokens=1_000,
        )


# --------------------------------------------------------------------------- #
# Snapshot schedule
# --------------------------------------------------------------------------- #


def test_cross_arch_schedule_matches_paper() -> None:
    """The 1.5B-token cross-arch panel: 27 token amounts + step 1 = 28 snapshots."""
    amounts = _snapshot_token_amounts(1_500_000_000)
    # Dense early (10M, 25M, every 50M to 1B), sparse late (every 100M to 1.5B).
    expected = [10_000_000, 25_000_000]
    expected += list(range(50_000_000, 1_000_000_001, 50_000_000))
    expected += list(range(1_100_000_000, 1_500_000_001, 100_000_000))
    assert amounts == expected
    assert len(amounts) == 27

    snapshots = common.build_snapshots("/out", amounts)
    assert len(snapshots) == 28
    assert 1 in snapshots


def test_cross_arch_schedule_drops_amounts_past_training_budget() -> None:
    # A short run should only schedule snapshots it will actually reach.
    amounts = _snapshot_token_amounts(60_000_000)
    assert amounts == [10_000_000, 25_000_000, 50_000_000]
