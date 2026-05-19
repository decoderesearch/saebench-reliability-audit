"""Tests for the shared training helpers in ``common.py``.

Covers three things:

* :func:`make_runner_config` — building the SAELens runner config. We build the
  config but never run training; that needs Gemma-2-2b.
* :func:`build_snapshots` — mapping snapshot token counts to training steps.
* :class:`SnapshotSAETrainer` — actually writing a loadable inference-mode SAE
  at each scheduled step. This runs real training, but on a tiny SAE fed
  synthetic activations, so it needs neither Gemma-2-2b nor a dataset.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from sae_lens import (
    SAE,
    BatchTopKTrainingSAEConfig,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
)
from sae_lens.config import SAETrainerConfig

from saebench_audit.training.common import (
    BATCH_SIZE,
    D_IN,
    D_SAE,
    GEMMA_2_2B,
    HOOK_NAME,
    LR,
    SnapshotSAETrainer,
    build_snapshots,
    make_runner_config,
)

# --------------------------------------------------------------------------- #
# make_runner_config
# --------------------------------------------------------------------------- #


def test_make_runner_config_sets_paper_hyperparameters(tmp_path: Path) -> None:
    sae_cfg = BatchTopKTrainingSAEConfig(d_in=D_IN, d_sae=D_SAE, k=50)
    cfg = make_runner_config(
        sae_cfg=sae_cfg,
        training_tokens=1_500_000_000,
        n_checkpoints=28,
        seed=0,
        output_path=str(tmp_path),
        run_name="test",
    )
    assert cfg.model_name == GEMMA_2_2B
    assert cfg.hook_name == HOOK_NAME
    assert cfg.lr == LR
    assert cfg.train_batch_size_tokens == BATCH_SIZE
    assert cfg.training_tokens == 1_500_000_000
    assert cfg.n_checkpoints == 28
    # LR decays over the final fifth of training.
    assert cfg.lr_decay_steps == (1_500_000_000 // BATCH_SIZE) // 5
    assert cfg.lr_warm_up_steps == 0
    assert cfg.autocast and cfg.autocast_lm
    assert cfg.adam_beta1 == 0.9
    assert cfg.exclude_special_tokens is True
    assert cfg.save_final_checkpoint is True
    assert cfg.seed == 0
    # Logger should be in offline mode by default.
    assert cfg.logger.log_to_wandb is False


def test_make_runner_config_defaults_to_no_checkpoints(tmp_path: Path) -> None:
    # Intermediate SAEs are saved as snapshots, not SAELens checkpoints.
    sae_cfg = BatchTopKTrainingSAEConfig(d_in=D_IN, d_sae=D_SAE, k=50)
    cfg = make_runner_config(
        sae_cfg=sae_cfg,
        training_tokens=1_000_000,
        seed=0,
        output_path=str(tmp_path),
        run_name="test",
    )
    assert cfg.n_checkpoints == 0


def test_make_runner_config_enables_wandb_when_project_passed(tmp_path: Path) -> None:
    sae_cfg = BatchTopKTrainingSAEConfig(d_in=D_IN, d_sae=D_SAE, k=50)
    cfg = make_runner_config(
        sae_cfg=sae_cfg,
        training_tokens=1_000_000,
        n_checkpoints=2,
        seed=0,
        output_path=str(tmp_path),
        run_name="test-run",
        wandb_project="some-project",
        wandb_entity="some-entity",
    )
    assert cfg.logger.log_to_wandb is True
    assert cfg.logger.wandb_project == "some-project"
    assert cfg.logger.wandb_entity == "some-entity"
    assert cfg.logger.run_name == "test-run"


# --------------------------------------------------------------------------- #
# build_snapshots
# --------------------------------------------------------------------------- #


def test_build_snapshots_always_includes_step_1() -> None:
    snapshots = build_snapshots("/out", [])
    assert snapshots == {1: "/out/snapshots/step-1-tokens-0"}


def test_build_snapshots_maps_tokens_to_steps() -> None:
    # Each token count maps to a step via tokens // batch_size.
    snapshots = build_snapshots("/out", [10_000_000, 25_000_000])
    assert snapshots[10_000_000 // BATCH_SIZE] == (
        f"/out/snapshots/step-{10_000_000 // BATCH_SIZE}-tokens-10000000"
    )
    assert snapshots[25_000_000 // BATCH_SIZE] == (
        f"/out/snapshots/step-{25_000_000 // BATCH_SIZE}-tokens-25000000"
    )
    # step 1 is implicit on top of the two requested amounts.
    assert len(snapshots) == 3


def test_build_snapshots_respects_custom_batch_size() -> None:
    snapshots = build_snapshots("/out", [1000], batch_size=100)
    assert 10 in snapshots
    assert snapshots[10] == "/out/snapshots/step-10-tokens-1000"


# --------------------------------------------------------------------------- #
# SnapshotSAETrainer
# --------------------------------------------------------------------------- #


def _synthetic_activations(batch: int, d_in: int) -> Iterator[torch.Tensor]:
    """An endless stream of random activations standing in for an LLM's."""
    while True:
        yield torch.randn(batch, d_in)


def _make_trainer(
    snapshots: Mapping[Any, str | Path],
    *,
    total_samples: int,
    batch: int = 4,
    d_in: int = 8,
    d_sae: int = 16,
) -> SnapshotSAETrainer:
    sae = StandardTrainingSAE(StandardTrainingSAEConfig(d_in=d_in, d_sae=d_sae))
    cfg = SAETrainerConfig(
        total_training_samples=total_samples,
        train_batch_size_samples=batch,
        lr_end=3e-5,
        device="cpu",
    )
    return SnapshotSAETrainer(
        sae=sae,
        data_provider=_synthetic_activations(batch, d_in),
        evaluator=None,
        save_checkpoint_fn=None,
        cfg=cfg,
        snapshots=snapshots,  # type: ignore[arg-type]
    )


def _assert_is_loadable_snapshot(path: Path) -> None:
    """A snapshot directory must hold an inference-mode SAE plus its sparsity."""
    assert (path / "cfg.json").exists()
    assert (path / "sae_weights.safetensors").exists()
    assert (path / "sparsity.safetensors").exists()
    # The whole point of a snapshot: it loads exactly like a finished SAE.
    SAE.load_from_disk(str(path), device="cpu")


def test_snapshot_saved_at_each_scheduled_step(tmp_path: Path) -> None:
    # batch=4, total=20 -> 5 training steps; snapshot at steps 1, 3, 5.
    snapshots = {
        1: str(tmp_path / "s1"),
        3: str(tmp_path / "s3"),
        5: str(tmp_path / "s5"),
    }
    trainer = _make_trainer(snapshots, total_samples=20)
    trainer.fit()

    assert trainer.n_training_steps == 5
    for path in snapshots.values():
        _assert_is_loadable_snapshot(Path(path))


def test_no_snapshot_for_unreached_step(tmp_path: Path) -> None:
    # Only 5 steps run, so the step-9 snapshot must never be written.
    snapshots = {3: str(tmp_path / "reached"), 9: str(tmp_path / "unreached")}
    trainer = _make_trainer(snapshots, total_samples=20)
    trainer.fit()

    assert (tmp_path / "reached").exists()
    assert not (tmp_path / "unreached").exists()


def test_snapshot_steps_accept_string_keys(tmp_path: Path) -> None:
    # JSON/CLI plumbing can hand us str keys; they must still match n_training_steps.
    trainer = _make_trainer({"3": str(tmp_path / "s3")}, total_samples=20)
    assert trainer.snapshots == {3: str(tmp_path / "s3")}
    trainer.fit()
    _assert_is_loadable_snapshot(tmp_path / "s3")


def test_snapshots_capture_distinct_training_states(tmp_path: Path) -> None:
    """Each snapshot is the SAE *as of that step*, not a copy of the final SAE."""
    snapshots = {1: str(tmp_path / "early"), 5: str(tmp_path / "late")}
    trainer = _make_trainer(snapshots, total_samples=20)
    trainer.fit()

    early = SAE.load_from_disk(str(tmp_path / "early"), device="cpu")
    late = SAE.load_from_disk(str(tmp_path / "late"), device="cpu")
    # Training moves the weights, so an early snapshot differs from a later one.
    assert not torch.allclose(early.W_dec, late.W_dec)
