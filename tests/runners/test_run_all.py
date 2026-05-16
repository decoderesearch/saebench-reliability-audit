"""Tests for the ``run_all_evals`` driver and ``Eval`` interface.

Uses a tiny stub ``Eval`` to verify control flow without invoking real
SAEBench code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from sae_lens import StandardSAE, StandardSAEConfig

from saebench_audit.runners.eval import Eval
from saebench_audit.runners.run_all import run_all_evals


class _StubEval(Eval):
    """Records calls and writes a sentinel file the next has_eval_run picks up."""

    def __init__(self, name: str, *, raise_on_run: bool = False) -> None:
        self.name = name
        self.raise_on_run = raise_on_run
        self.calls: list[int] = []

    def has_eval_run(self, results_dir: Path) -> bool:
        return (results_dir / f"{self.name}.done").exists()

    def run(
        self,
        sae: Any,  # noqa: ARG002
        results_dir: Path,
        shared_dir: Path,  # noqa: ARG002
        random_seed: int = 42,
    ) -> None:
        self.calls.append(random_seed)
        if self.raise_on_run:
            raise RuntimeError("boom")
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / f"{self.name}.done").touch()


def _tiny_sae() -> StandardSAE:
    cfg = StandardSAEConfig(d_in=4, d_sae=8, device="cpu", dtype="float32")
    return StandardSAE(cfg)


def test_run_all_evals_skips_when_already_done(tmp_path: Path) -> None:
    sae = _tiny_sae()
    eval_a = _StubEval("a")
    # First call writes the sentinel; second call must skip.
    run_all_evals(
        sae, results_dir=tmp_path / "r", shared_dir=tmp_path / "s", evals=[eval_a]
    )
    run_all_evals(
        sae, results_dir=tmp_path / "r", shared_dir=tmp_path / "s", evals=[eval_a]
    )
    assert len(eval_a.calls) == 1


def test_run_all_evals_force_reruns(tmp_path: Path) -> None:
    sae = _tiny_sae()
    eval_a = _StubEval("a")
    run_all_evals(
        sae, results_dir=tmp_path / "r", shared_dir=tmp_path / "s", evals=[eval_a]
    )
    run_all_evals(
        sae,
        results_dir=tmp_path / "r",
        shared_dir=tmp_path / "s",
        evals=[eval_a],
        force=True,
    )
    assert len(eval_a.calls) == 2


def test_run_all_evals_threads_seed(tmp_path: Path) -> None:
    sae = _tiny_sae()
    eval_a = _StubEval("a")
    run_all_evals(
        sae,
        results_dir=tmp_path / "r",
        shared_dir=tmp_path / "s",
        evals=[eval_a],
        random_seed=99,
    )
    assert eval_a.calls == [99]


def test_run_all_evals_does_not_abort_on_one_failure(tmp_path: Path) -> None:
    sae = _tiny_sae()
    failing = _StubEval("fail", raise_on_run=True)
    succeeding = _StubEval("ok")
    run_all_evals(
        sae,
        results_dir=tmp_path / "r",
        shared_dir=tmp_path / "s",
        evals=[failing, succeeding],
    )
    # ``succeeding`` should have run despite ``failing`` raising.
    assert (tmp_path / "r" / "ok.done").exists()


def test_run_all_evals_crash_on_error_propagates(tmp_path: Path) -> None:
    sae = _tiny_sae()
    failing = _StubEval("fail", raise_on_run=True)
    with pytest.raises(RuntimeError, match="boom"):
        run_all_evals(
            sae,
            results_dir=tmp_path / "r",
            shared_dir=tmp_path / "s",
            evals=[failing],
            crash_on_error=True,
        )
    # Just to silence "torch unused" warnings on macOS:
    assert torch.is_tensor(torch.zeros(1))
