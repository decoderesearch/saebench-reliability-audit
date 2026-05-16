"""Tests for the snapshot driver's snapshot-discovery helper."""

from __future__ import annotations

from pathlib import Path

from saebench_audit.runners.snapshots import list_snapshot_paths


def test_list_snapshot_paths_finds_cfg_json_dirs(tmp_path: Path) -> None:
    a = tmp_path / "btk" / "k50" / "step-100"
    b = tmp_path / "matryoshka" / "k100" / "step-200"
    nested = tmp_path / "btk" / "extra" / "skipped"
    for d in (a, b, nested):
        d.mkdir(parents=True)
    (a / "cfg.json").write_text("{}")
    (b / "cfg.json").write_text("{}")
    # nested has no cfg.json so it should be ignored.
    found = list_snapshot_paths(tmp_path)
    assert set(found) == {a, b}


def test_list_snapshot_paths_returns_empty_for_empty_tree(tmp_path: Path) -> None:
    assert list_snapshot_paths(tmp_path) == []
