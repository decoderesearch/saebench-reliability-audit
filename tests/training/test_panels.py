"""Smoke tests for the cross-architecture and sampled-Matryoshka training scripts.

We monkey-patch ``run_training`` so the test never actually starts an LM, then
inspect the config and override SAE that the training entry point would have
handed to SAELens. This catches config regressions without paying the cost of
loading Gemma-2-2b.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from saebench_audit.saes.matryoshka_sae import (
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
)
from saebench_audit.training import common
from saebench_audit.training import train_cross_arch as cross_arch
from saebench_audit.training import train_sampled_matryoshka as sampled_mat


@pytest.fixture
def captured_call(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Replace ``run_training`` so we capture the would-be call instead of running."""
    captured: dict[str, object] = {}

    def fake_run(cfg, *, override_sae=None):  # type: ignore[no-untyped-def]
        captured["cfg"] = cfg
        captured["override_sae"] = override_sae
        return common.TrainingResult(final_dir="x", checkpoints_dir="x/c")

    monkeypatch.setattr(cross_arch, "run_training", fake_run)
    monkeypatch.setattr(sampled_mat, "run_training", fake_run)
    return captured


def test_train_cross_arch_btk_uses_batchtopk_no_override(
    captured_call: dict[str, object],
) -> None:
    cross_arch.train_cross_arch(
        variant="btk",
        k=50,
        output_path="/tmp/out",
        training_tokens=1_000_000,
        n_checkpoints=4,
    )
    assert captured_call["override_sae"] is None
    cfg = cast(Any, captured_call["cfg"])
    assert cfg.sae.architecture() == "batchtopk"
    assert cfg.sae.k == 50
    assert cfg.training_tokens == 1_000_000


def test_train_cross_arch_matryoshka_uses_three_fixed_widths(
    captured_call: dict[str, object],
) -> None:
    cross_arch.train_cross_arch(
        variant="matryoshka",
        k=100,
        output_path="/tmp/out",
        training_tokens=1_000_000,
        n_checkpoints=4,
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
) -> None:
    with pytest.raises(ValueError, match="variant"):
        cross_arch.train_cross_arch(
            variant="unknown",  # type: ignore[arg-type]
            k=50,
            output_path="/tmp/out",
            training_tokens=1_000,
            n_checkpoints=1,
        )


def test_train_sampled_matryoshka_uses_log_uniform(
    captured_call: dict[str, object],
) -> None:
    sampled_mat.train_sampled_matryoshka(
        n_levels=3,
        output_path="/tmp/out",
        training_tokens=1_000_000,
        n_checkpoints=2,
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
