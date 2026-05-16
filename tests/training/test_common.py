"""Tests for the SAELens runner-config builder.

We build the config but never actually run training — that needs Gemma-2-2b.
"""

from __future__ import annotations

from sae_lens import BatchTopKTrainingSAEConfig

from saebench_audit.training.common import (
    BATCH_SIZE,
    D_IN,
    D_SAE,
    GEMMA_2_2B,
    HOOK_NAME,
    LR,
    make_runner_config,
)


def test_make_runner_config_sets_paper_hyperparameters() -> None:
    sae_cfg = BatchTopKTrainingSAEConfig(d_in=D_IN, d_sae=D_SAE, k=50)
    cfg = make_runner_config(
        sae_cfg=sae_cfg,
        training_tokens=1_500_000_000,
        n_checkpoints=28,
        seed=0,
        output_path="/tmp/out",
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


def test_make_runner_config_enables_wandb_when_project_passed() -> None:
    sae_cfg = BatchTopKTrainingSAEConfig(d_in=D_IN, d_sae=D_SAE, k=50)
    cfg = make_runner_config(
        sae_cfg=sae_cfg,
        training_tokens=1_000_000,
        n_checkpoints=2,
        seed=0,
        output_path="/tmp/out",
        run_name="test-run",
        wandb_project="some-project",
        wandb_entity="some-entity",
    )
    assert cfg.logger.log_to_wandb is True
    assert cfg.logger.wandb_project == "some-project"
    assert cfg.logger.wandb_entity == "some-entity"
    assert cfg.logger.run_name == "test-run"
