"""Shared training defaults for the discriminability panels.

The two panels in the paper share most of their hyperparameters; only the
sparsifying activation, training-token count, and matryoshka configuration
differ. This module factors the common values and a small helper that wraps
SAELens's ``LanguageModelSAETrainingRunner`` with checkpoint scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass

from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    LoggingConfig,
    TrainingSAE,
    TrainingSAEConfig,
)

# Paper's canonical training settings (Appendix C "SAE training setup").
GEMMA_2_2B = "google/gemma-2-2b"
LAYER = 12
HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"
D_IN = 2304
D_SAE = 16 * D_IN  # 32_768
LR = 3e-4
BATCH_SIZE = 4096
DATASET_PATH = "monology/pile-uncopyrighted"
CONTEXT_SIZE = 1024


@dataclass
class TrainingResult:
    """Where a training run ended up on disk."""

    final_dir: str
    checkpoints_dir: str | None


def _make_logger(
    *,
    run_name: str,
    wandb_project: str | None,
    wandb_entity: str | None,
) -> LoggingConfig:
    if wandb_project is None:
        return LoggingConfig(log_to_wandb=False)
    return LoggingConfig(
        log_to_wandb=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity or "",
        wandb_log_frequency=100,
        run_name=run_name,
    )


def make_runner_config(
    sae_cfg: TrainingSAEConfig,
    *,
    training_tokens: int,
    n_checkpoints: int,
    seed: int,
    output_path: str,
    run_name: str,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
) -> LanguageModelSAERunnerConfig[TrainingSAEConfig]:
    """Build the SAELens runner config used by both training panels.

    Mirrors the hyperparameters listed in Appendix C of the paper:

    * Adam optimizer with default $\\beta_1, \\beta_2$ from SAELens, peak
      learning rate $3 \\times 10^{-4}$, no warmup, linear LR decay over the
      final fifth of training.
    * Train batch size 4{,}096 tokens, ``n_batches_in_buffer=256``.
    * The Pile dataset, context length 1024.
    * Mixed-precision (bf16 autocast) for both the LM and the SAE.

    ``n_checkpoints`` is forwarded to SAELens's checkpoint scheduler; the
    paper uses 28 snapshots for the 1.5B-token panel and 11 snapshots for the
    300M-token sampled-matryoshka panel.
    """
    total_steps = training_tokens // BATCH_SIZE
    lr_decay_steps = total_steps // 5
    return LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=GEMMA_2_2B,
        hook_name=HOOK_NAME,
        dataset_path=DATASET_PATH,
        is_dataset_tokenized=False,
        streaming=True,
        context_size=CONTEXT_SIZE,
        training_tokens=training_tokens,
        device="cuda",
        lr=LR,
        lr_warm_up_steps=0,
        lr_decay_steps=lr_decay_steps,
        n_batches_in_buffer=256,
        train_batch_size_tokens=BATCH_SIZE,
        store_batch_size_prompts=12,
        eval_batch_size_prompts=6,
        autocast=True,
        autocast_lm=True,
        adam_beta1=0.9,
        n_checkpoints=n_checkpoints,
        save_final_checkpoint=True,
        checkpoint_path=f"{output_path}/checkpoints",
        output_path=output_path,
        seed=seed,
        exclude_special_tokens=True,
        logger=_make_logger(
            run_name=run_name,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        ),
    )


def run_training(
    cfg: LanguageModelSAERunnerConfig[TrainingSAEConfig],
    *,
    override_sae: TrainingSAE | None = None,
) -> TrainingResult:
    """Run a single SAE training and return the on-disk paths.

    ``override_sae`` is forwarded to ``LanguageModelSAETrainingRunner`` and is only needed
    for SAE classes that the SAELens config registry does not know how to
    instantiate from a config alone (e.g. our matryoshka SAE before
    registration).
    """
    runner = LanguageModelSAETrainingRunner(cfg, override_sae=override_sae)
    runner.run()
    return TrainingResult(
        final_dir=str(cfg.output_path),
        checkpoints_dir=cfg.checkpoint_path,
    )
