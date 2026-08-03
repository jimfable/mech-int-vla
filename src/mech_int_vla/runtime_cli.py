"""Command line entry points for pinned model and pre-rollout LIBERO checks."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .config import ConditionSpec, SplitName, load_protocol_config
from .instrumentation import SmolVLAInstrumentation
from .libero_runtime import RawLiberoEpisode
from .manifest import generate_episode_manifest
from .rollout import run_single_episode
from .snapshots import (
    load_locked_smolvla,
    load_model_input_lock,
    resolve_snapshot_paths,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def _head_commit(root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def _require_clean_worktree(root: Path) -> None:
    process = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if process.stdout.strip():
        raise RuntimeError("episode execution requires a clean committed worktree")


def _snapshots(args: argparse.Namespace) -> int:
    paths = resolve_snapshot_paths(
        args.environment_lock,
        cache_dir=args.cache_dir,
        local_files_only=not args.download,
    )
    print(
        json.dumps(
            {
                "policy": str(paths.policy),
                "policy_revision": paths.lock.policy_revision,
                "base_vlm": str(paths.base_vlm),
                "base_vlm_revision": paths.lock.base_vlm_revision,
                "local_files_only": not args.download,
            },
            sort_keys=True,
        )
    )
    return 0


def _load_policy(args: argparse.Namespace) -> int:
    paths = resolve_snapshot_paths(
        args.environment_lock,
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    runtime = load_locked_smolvla(paths, device=args.device)
    print(
        json.dumps(
            {
                "policy_revision": paths.lock.policy_revision,
                "base_vlm_revision": paths.lock.base_vlm_revision,
                "device": str(runtime.policy.config.device),
                "num_steps": runtime.policy.config.num_steps,
                "chunk_size": runtime.policy.config.chunk_size,
                "n_action_steps": runtime.policy.config.n_action_steps,
                "empty_cameras": runtime.policy.config.empty_cameras,
                "tokenizer_path": str(paths.base_vlm),
                "original_checkpoint_state_shape": runtime.original_state_shape,
                "runtime_state_shape": runtime.runtime_state_shape,
                "normalization_state_shapes": runtime.normalization_state_shapes,
            },
            sort_keys=True,
        )
    )
    return 0


def _discovery_reset(args: argparse.Namespace) -> int:
    root = args.repo_root.resolve()
    config = load_protocol_config(root / "configs")
    task = config.task_order.tasks[args.task_rank - 1]
    manifest = generate_episode_manifest(
        SplitName.DISCOVERY,
        task,
        config,
        policy_revision=load_model_input_lock(
            root / "environment.lock"
        ).policy_revision,
        code_commit=_head_commit(root),
    )
    matches = [
        episode
        for episode in manifest.episodes
        if episode.base_init_state_id == args.init_id
        and episode.condition_index == args.condition_index
    ]
    if len(matches) != 1:
        available = sorted(
            episode.condition_index
            for episode in manifest.episodes
            if episode.base_init_state_id == args.init_id
        )
        raise ValueError(
            f"no unique Discovery episode for init={args.init_id}, "
            f"condition={args.condition_index}; available={available}"
        )
    episode_spec = matches[0]
    # The manifest has already realized any assignment-dependent values.
    condition = ConditionSpec(
        episode_spec.condition_name,
        episode_spec.condition_family,
        episode_spec.condition_index,
        episode_spec.condition_parameters,
    )
    runtime = RawLiberoEpisode.create(
        task,
        base_init_state_id=args.init_id,
        execution=config.split.policy_execution,
        validity=config.perturbations.validity,
    )
    try:
        result = runtime.reset(seed=episode_spec.reset_seed, condition=condition)
        payload = {
            "episode_id": episode_spec.episode_id,
            "reset_seed": episode_spec.reset_seed,
            "condition": episode_spec.condition_name,
            "validity": {
                "valid": result.validity.valid,
                "reasons": result.validity.reasons,
                "penetration_m": result.validity.deepest_primary_penetration_m,
                "linear_speed_m_s": result.validity.linear_speed_m_s,
                "angular_speed_rad_s": result.validity.angular_speed_rad_s,
                "in_workspace": result.validity.in_workspace,
                "initial_success": result.validity.initial_success,
            },
            "policy_state_length": int(result.frame.policy_state.size),
            "phase": result.frame.phase,
            "settle_steps": len(result.settle_actions),
        }
        print(json.dumps(payload, default=_json_default, sort_keys=True))
    finally:
        runtime.close()
    return 0


def _discovery_rollout(args: argparse.Namespace) -> int:
    root = args.repo_root.resolve()
    _require_clean_worktree(root)
    config = load_protocol_config(root / "configs")
    task = config.task_order.tasks[args.task_rank - 1]
    lock = load_model_input_lock(args.environment_lock)
    manifest = generate_episode_manifest(
        SplitName.DISCOVERY,
        task,
        config,
        policy_revision=lock.policy_revision,
        code_commit=_head_commit(root),
    )
    matches = [
        episode
        for episode in manifest.episodes
        if episode.base_init_state_id == args.init_id
        and episode.condition_index == args.condition_index
    ]
    if len(matches) != 1:
        raise ValueError("no unique manifested Discovery rollout cell")
    episode_spec = matches[0]
    condition = ConditionSpec(
        episode_spec.condition_name,
        episode_spec.condition_family,
        episode_spec.condition_index,
        episode_spec.condition_parameters,
    )
    snapshots = resolve_snapshot_paths(
        args.environment_lock,
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    policy_runtime = load_locked_smolvla(snapshots, device=args.device)
    episode = RawLiberoEpisode.create(
        task,
        base_init_state_id=args.init_id,
        execution=config.split.policy_execution,
        validity=config.perturbations.validity,
    )
    instrumentation = SmolVLAInstrumentation(policy_runtime.policy)
    try:
        result = run_single_episode(
            policy_runtime,
            episode,
            instrumentation,
            episode_spec,
            condition,
            validity_retry_factory=lambda: RawLiberoEpisode.create(
                task,
                base_init_state_id=args.init_id,
                execution=config.split.policy_execution,
                validity=config.perturbations.validity,
            ),
            artifact_root=args.artifact_root,
        )
        print(
            json.dumps(
                {
                    "episode_id": episode_spec.episode_id,
                    "artifact_path": result.artifact_path,
                    "status": result.status,
                    "valid_reset": result.valid_reset,
                    "success": result.success,
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                    "control_steps": result.control_steps,
                },
                default=_json_default,
                sort_keys=True,
            )
        )
    finally:
        instrumentation.remove()
        episode.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshots = subparsers.add_parser(
        "snapshots", help="resolve exact policy and VLM snapshots"
    )
    snapshots.add_argument(
        "--environment-lock", type=Path, default=Path("environment.lock")
    )
    snapshots.add_argument("--cache-dir", type=Path)
    snapshots.add_argument(
        "--download",
        action="store_true",
        help="allow pinned Hub downloads; without this flag resolution is offline",
    )
    snapshots.set_defaults(handler=_snapshots)

    policy = subparsers.add_parser(
        "load-policy", help="offline-load and verify the pinned SmolVLA runtime"
    )
    policy.add_argument(
        "--environment-lock", type=Path, default=Path("environment.lock")
    )
    policy.add_argument("--cache-dir", type=Path)
    policy.add_argument("--device", default="cuda")
    policy.set_defaults(handler=_load_policy)

    reset = subparsers.add_parser(
        "discovery-reset", help="construct and validate one Discovery reset"
    )
    reset.add_argument("--repo-root", type=Path, default=Path.cwd())
    reset.add_argument("--task-rank", type=int, choices=(1, 2, 3), default=1)
    reset.add_argument("--init-id", type=int, choices=range(10), required=True)
    reset.add_argument("--condition-index", type=int, choices=range(4), default=0)
    reset.set_defaults(handler=_discovery_reset)

    rollout = subparsers.add_parser(
        "discovery-rollout",
        help="execute and atomically record one manifested Discovery episode",
    )
    rollout.add_argument("--repo-root", type=Path, default=Path.cwd())
    rollout.add_argument(
        "--environment-lock", type=Path, default=Path("environment.lock")
    )
    rollout.add_argument("--cache-dir", type=Path)
    rollout.add_argument("--device", default="cuda")
    rollout.add_argument("--artifact-root", type=Path, default=Path("artifacts/raw"))
    rollout.add_argument("--task-rank", type=int, choices=(1, 2, 3), default=1)
    rollout.add_argument("--init-id", type=int, choices=range(10), required=True)
    rollout.add_argument("--condition-index", type=int, choices=range(4), default=0)
    rollout.set_defaults(handler=_discovery_rollout)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
