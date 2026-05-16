"""Tests for synthetic-task helpers.

Loading a real SynthSAEBench model would download a HuggingFace artifact, so
the end-to-end ``generate`` test is not run here. Instead we test the
helpers that don't need a model.
"""

from __future__ import annotations

import numpy as np
import torch

from saebench_audit.synthetic.data_gen import (
    PAPER_FIXTURES_DIR,
    SCRTask,
    SPTask,
    TPPTask,
    _construct_label_np,
    _eval_op,
    _load_tasks_from_fixture,
    model_tag_for_repo,
)


def test_model_tag_for_repo_is_filesystem_safe() -> None:
    tag = model_tag_for_repo("decoderesearch/synth-sae-bench-16k-v1", revision=None)
    assert "/" not in tag
    assert "main" in tag

    tag2 = model_tag_for_repo(
        "anon/synth-sae-bench-variations",
        revision="firing-magnitude-stdev/std-2.5",
    )
    assert "/" not in tag2
    assert "std-2.5" in tag2


def test_eval_op_single() -> None:
    fb = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=torch.float32)
    assert torch.equal(_eval_op(fb, "single", [0]), fb[:, 0])


def test_eval_op_and_or() -> None:
    fb = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=torch.float32)
    expected_and = torch.tensor([0.0, 0.0, 1.0])
    assert torch.equal(_eval_op(fb, "and", [0, 1]), expected_and)
    expected_or = torch.tensor([1.0, 1.0, 1.0])
    assert torch.equal(_eval_op(fb, "or", [0, 1]), expected_or)


def test_construct_label_np_matches_torch() -> None:
    fb_np = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=bool)
    fb = torch.from_numpy(fb_np.astype(np.float32))
    for op, feats in [("single", [0]), ("and", [0, 1]), ("or", [1, 2])]:
        np_out = _construct_label_np(fb_np, op, feats)
        torch_out = _eval_op(fb, op, feats).bool().numpy()
        assert (np_out == torch_out).all()


def test_sp_task_name_round_trip() -> None:
    task = SPTask(id=1, op="and", feats=[1, 5], type="bool_in", pos_rate=0.1)
    assert task.name() == "bool_in__and__1_5"


def test_tpp_and_scr_task_dataclasses_constructible() -> None:
    tpp = TPPTask(name="t", parent_idx=0, sibling_feats=[1, 2, 3, 4], category="all_in")
    scr = SCRTask(
        name="s",
        t_op="single",
        t_feats=[1],
        t_cat="in",
        s_op="single",
        s_feats=[2],
        s_cat="in",
        root_t=0,
        root_s=1,
        cell_counts={"00": 1, "01": 2, "10": 3, "11": 4},
    )
    assert tpp.category == "all_in"
    assert scr.cell_counts["11"] == 4


def test_paper_fixture_v1_seed_1234_loads_with_paper_picks() -> None:
    """The shipped paper fixture matches the feature picks used to build the
    paper's Figure 1 / Table 2."""
    fixture_dir = PAPER_FIXTURES_DIR / "v1_seed_1234"
    assert fixture_dir.is_dir()

    sp_tasks, tpp_tasks, scr_tasks = _load_tasks_from_fixture(fixture_dir)

    # Paper-config task counts: 24 single_in + 24 single_out + 16 bool_in
    # + 12 bool_out + 16 bool_mixed = 92 SP tasks; 30 + 30 = 60 TPP tasks; 9 SCR.
    assert len(sp_tasks) == 92
    assert len(tpp_tasks) == 60
    assert len(scr_tasks) == 9

    # The first single_in feats must match what the paper figures used —
    # different from what the current data_gen RNG produces (the multi-seed
    # rewrite offset the seeds: TPP +100, SCR +1000, SP boolean +5000,
    # SP single +9000; the paper used the unoffset RandomState(SEED)).
    single_in = [t for t in sp_tasks if t.type == "single_in"]
    assert [t.feats[0] for t in single_in[:5]] == [390, 656, 43, 925, 768]


def test_paper_fixture_round_trips_through_loader() -> None:
    """Loader returns dataclass instances with the JSON keys preserved."""
    fixture_dir = PAPER_FIXTURES_DIR / "v1_seed_1234"
    sp_tasks, tpp_tasks, scr_tasks = _load_tasks_from_fixture(fixture_dir)
    # Spot-check types and a couple of fields.
    assert isinstance(sp_tasks[0], SPTask)
    assert isinstance(tpp_tasks[0], TPPTask)
    assert isinstance(scr_tasks[0], SCRTask)
    assert sp_tasks[0].op in {"single", "and", "or"}
    assert tpp_tasks[0].category in {"all_in", "all_out"}
    assert set(scr_tasks[0].cell_counts) == {"00", "01", "10", "11"}
