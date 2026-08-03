from __future__ import annotations

from types import SimpleNamespace

import pytest

from mech_int_vla.runtime_cli import _require_clean_worktree, build_parser


def test_discovery_rollout_parser_fixes_discovery_ranges() -> None:
    args = build_parser().parse_args(
        [
            "discovery-rollout",
            "--task-rank",
            "2",
            "--init-id",
            "9",
            "--condition-index",
            "3",
            "--cache-dir",
            "/tmp/cache",
        ]
    )
    assert args.task_rank == 2
    assert args.init_id == 9
    assert args.condition_index == 3
    assert str(args.cache_dir) == "/tmp/cache"


def test_episode_execution_rejects_dirty_worktree(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "mech_int_vla.runtime_cli.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" M runtime.py\n"),
    )
    with pytest.raises(RuntimeError, match="clean committed worktree"):
        _require_clean_worktree(tmp_path)
