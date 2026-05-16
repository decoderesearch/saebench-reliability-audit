"""Tests for the SAEBench eval wrappers in ``runners/eval.py``.

These tests focus on plumbing: that the ``random_seed`` parameter is
forwarded to the underlying SAEBench / sae-probes code, and that the
``has_eval_run`` thresholds use the right reference counts. We don't run
the real evals here — we monkey-patch the SAEBench/sae-probes entry points
and assert on what they were called with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from saebench_audit.runners.eval import (
    SAEProbesEval,
    SAEProbesOptions,
    ScrAndTppOptions,
    SCREval,
    SparseProbingEval,
    SparseProbingOptions,
    TPPEval,
)


class _StubMetadata:
    model_name = "fake-model"
    hook_name = "blocks.0.hook_resid_post"


class _StubCfg:
    metadata = _StubMetadata()
    apply_b_dec_to_input = False
    d_in = 4
    d_sae = 8


class _StubSAE:
    cfg = _StubCfg()

    def fold_W_dec_norm(self) -> None:  # pragma: no cover — used by wrappers
        return None


# ---- SAEProbesEval ------------------------------------------------------


def test_sae_probes_run_forwards_random_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Critical-issue-1 fix: SAEProbesEval.run must pass random_seed
    through to ``run_sae_probes_evals`` as the ``seed=`` argument.
    Without this, every reseed produces identical sae-probes output."""
    captured: dict[str, Any] = {}

    def fake_run_sae_probes_evals(**kwargs: Any) -> None:
        captured.update(kwargs)

    import saebench_audit.runners.eval as eval_mod

    monkeypatch.setattr(eval_mod, "run_sae_probes_evals", fake_run_sae_probes_evals)

    sae = _StubSAE()
    SAEProbesEval(SAEProbesOptions(device="cpu")).run(
        sae=sae,  # type: ignore[arg-type]
        results_dir=tmp_path / "r",
        shared_dir=tmp_path / "s",
        random_seed=2024,
    )
    assert captured.get("seed") == 2024, (
        f"random_seed=2024 was not forwarded to run_sae_probes_evals; "
        f"captured kwargs: {sorted(captured)}"
    )


def test_sae_probes_run_default_seed_is_42(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_sae_probes_evals(**kwargs: Any) -> None:
        captured.update(kwargs)

    import saebench_audit.runners.eval as eval_mod

    monkeypatch.setattr(eval_mod, "run_sae_probes_evals", fake_run_sae_probes_evals)
    SAEProbesEval(SAEProbesOptions(device="cpu")).run(
        sae=_StubSAE(),  # type: ignore[arg-type]
        results_dir=tmp_path / "r",
        shared_dir=tmp_path / "s",
    )
    assert captured.get("seed") == 42


def test_sae_probes_has_eval_run_uses_dataset_count(tmp_path: Path) -> None:
    """has_eval_run must compare against the actual sae-probes dataset count
    (113), not a stale hard-coded 112."""
    from saebench_audit.runners.eval import _SAE_PROBES_DATASETS

    eval_obj = SAEProbesEval(SAEProbesOptions(settings=["normal"]))
    setting_dir = tmp_path / "sae_probes" / "fake_sae" / "normal_setting"
    setting_dir.mkdir(parents=True)

    # Below-threshold: not done.
    for i in range(len(_SAE_PROBES_DATASETS) - 1):
        (setting_dir / f"dataset_{i}.json").write_text("{}")
    assert not eval_obj.has_eval_run(tmp_path), (
        f"has_eval_run returned True at {len(_SAE_PROBES_DATASETS) - 1} of "
        f"{len(_SAE_PROBES_DATASETS)} JSONs; threshold is wrong."
    )

    # At-threshold: done.
    (setting_dir / f"dataset_{len(_SAE_PROBES_DATASETS) - 1}.json").write_text("{}")
    assert eval_obj.has_eval_run(tmp_path)


# ---- SparseProbingEval / SCREval / TPPEval ------------------------------


def test_sparse_probing_forwards_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(*, config: Any, **kwargs: Any) -> None:
        captured["random_seed"] = config.random_seed
        captured.update(kwargs)

    import saebench_audit.runners.eval as eval_mod

    monkeypatch.setattr(eval_mod, "run_sparse_probing_eval", fake_run)
    SparseProbingEval(SparseProbingOptions(device="cpu")).run(
        sae=_StubSAE(),  # type: ignore[arg-type]
        results_dir=tmp_path / "r",
        shared_dir=tmp_path / "s",
        random_seed=789,
    )
    assert captured["random_seed"] == 789


def test_scr_forwards_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(*, config: Any, **kwargs: Any) -> None:
        captured["random_seed"] = config.random_seed
        captured["perform_scr"] = config.perform_scr
        captured.update(kwargs)

    import saebench_audit.runners.eval as eval_mod

    monkeypatch.setattr(eval_mod, "run_scr_tpp_eval", fake_run)
    SCREval(ScrAndTppOptions(device="cpu")).run(
        sae=_StubSAE(),  # type: ignore[arg-type]
        results_dir=tmp_path / "r",
        shared_dir=tmp_path / "s",
        random_seed=456,
    )
    assert captured["random_seed"] == 456
    assert captured["perform_scr"] is True


def test_tpp_forwards_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(*, config: Any, **kwargs: Any) -> None:
        captured["random_seed"] = config.random_seed
        captured["perform_scr"] = config.perform_scr
        captured.update(kwargs)

    import saebench_audit.runners.eval as eval_mod

    monkeypatch.setattr(eval_mod, "run_scr_tpp_eval", fake_run)
    TPPEval(ScrAndTppOptions(device="cpu")).run(
        sae=_StubSAE(),  # type: ignore[arg-type]
        results_dir=tmp_path / "r",
        shared_dir=tmp_path / "s",
        random_seed=123,
    )
    assert captured["random_seed"] == 123
    assert captured["perform_scr"] is False
