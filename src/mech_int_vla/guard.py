"""Fail-closed repository and payload guards for protected split access."""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    CalibrationGuardConfig,
    CalibrationSelectionConfig,
    LockedTestGuardConfig,
    PredictorCandidateConfig,
    TaskSpec,
)

_PLACEHOLDERS = frozenset(
    {
        "changeme",
        "n/a",
        "na",
        "none",
        "null",
        "pending",
        "placeholder",
        "replace_me",
        "tbd",
        "todo",
        "unknown",
    }
)
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_TASK_FIELDS = (
    "rank",
    "suite",
    "task_id",
    "language",
    "primary_object",
    "planar_symmetry_order",
)
_LOCKED_FIELDS = frozenset(
    {
        "selected_task",
        "selected_variable",
        "policy_revision",
        "representation_probe",
        "predictor",
        "artifact_hashes",
        "alarm_thresholds",
        "patch_strength",
        "calibration_metrics",
    }
)


class ProtocolGuardError(RuntimeError):
    """Base class for protected-split access failures."""


class CalibrationGuardError(ProtocolGuardError):
    """Raised when the preregistration lock does not authorize Calibration."""


class LockedTestGuardError(ProtocolGuardError):
    """Raised when the calibration lock does not authorize Locked Test."""


@dataclass(frozen=True)
class CalibrationReceipt:
    repository: Path
    head_commit: str
    tag: str
    freeze_file: Path
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class LockedTestReceipt:
    repository: Path
    head_commit: str
    tag: str
    freeze_file: Path
    payload: Mapping[str, Any]


def _git(repo: Path, error_type: type[ProtocolGuardError], *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = (
            exc.stderr.strip()
            if isinstance(exc, subprocess.CalledProcessError)
            else str(exc)
        )
        raise error_type(f"git {' '.join(args)} failed: {detail}") from exc
    return result.stdout.strip()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _read_lock(
    repo_root: str | Path,
    config: CalibrationGuardConfig | LockedTestGuardConfig,
    *,
    stage: str,
    error_type: type[ProtocolGuardError],
) -> tuple[Path, str, Path, dict[str, Any]]:
    repo = Path(repo_root).resolve()
    actual_root = Path(_git(repo, error_type, "rev-parse", "--show-toplevel")).resolve()
    if actual_root != repo:
        raise error_type(
            f"repo_root must be the git worktree root ({actual_root}), got {repo}"
        )

    freeze_file = (repo / config.required_file).resolve()
    try:
        relative_freeze = freeze_file.relative_to(repo).as_posix()
    except ValueError as exc:
        raise error_type("freeze file must be inside the repository") from exc
    if not freeze_file.is_file():
        raise error_type(f"required {stage} freeze file is missing: {freeze_file}")
    try:
        frozen = json.loads(
            freeze_file.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise error_type(
            f"{stage} freeze file is not valid finite JSON: {exc}"
        ) from exc
    if not isinstance(frozen, dict) or not frozen:
        raise error_type(f"{stage} freeze file must contain a nonempty JSON object")

    try:
        _git(
            repo,
            error_type,
            "ls-files",
            "--error-unmatch",
            "--",
            relative_freeze,
        )
    except ProtocolGuardError as exc:
        raise error_type(
            f"{stage} freeze file must be tracked in the lock commit"
        ) from exc

    head = _git(repo, error_type, "rev-parse", "HEAD")
    tag_commit = _git(
        repo,
        error_type,
        "rev-parse",
        "--verify",
        f"refs/tags/{config.required_tag}^{{commit}}",
    )
    if tag_commit != head:
        raise error_type(
            f"required tag {config.required_tag!r} is not exactly at HEAD ({head})"
        )

    if config.require_clean_worktree:
        status = _git(
            repo,
            error_type,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        if status:
            raise error_type(f"{stage} requires a clean worktree")

    return repo, head, freeze_file, frozen


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return (
        not normalized
        or normalized in _PLACEHOLDERS
        or normalized.startswith(("todo:", "tbd:", "pending:", "placeholder:"))
        or (normalized.startswith("<") and normalized.endswith(">"))
    )


def _reject_placeholders(
    value: Any, *, path: str, error_type: type[ProtocolGuardError]
) -> None:
    if isinstance(value, str):
        if _is_placeholder(value):
            raise error_type(f"placeholder value at {path}")
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and _is_placeholder(key):
                raise error_type(f"placeholder key at {path}")
            _reject_placeholders(nested, path=f"{path}.{key}", error_type=error_type)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_placeholders(nested, path=f"{path}[{index}]", error_type=error_type)


def _required_mapping(
    payload: Mapping[str, Any], key: str, error_type: type[ProtocolGuardError]
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict) or not value:
        raise error_type(f"{key} must be a nonempty object")
    return value


def _validate_task(
    payload: Mapping[str, Any], task: TaskSpec, error_type: type[ProtocolGuardError]
) -> None:
    selected = _required_mapping(payload, "selected_task", error_type)
    expected = {
        "rank": task.rank,
        "suite": task.suite,
        "task_id": task.task_id,
        "language": task.language,
        "primary_object": task.primary_object,
        "planar_symmetry_order": task.planar_symmetry_order,
    }
    missing = set(_TASK_FIELDS) - selected.keys()
    if missing:
        raise error_type(
            "selected_task is missing semantic fields: " + ", ".join(sorted(missing))
        )
    mismatched = [field for field in _TASK_FIELDS if selected[field] != expected[field]]
    if mismatched:
        raise error_type(
            "selected_task does not match manifest task fields: "
            + ", ".join(mismatched)
        )


def _validate_policy_revision(
    payload: Mapping[str, Any],
    policy_revision: str,
    error_type: type[ProtocolGuardError],
) -> None:
    frozen_revision = payload.get("policy_revision")
    if not isinstance(frozen_revision, str) or _is_placeholder(frozen_revision):
        raise error_type("policy_revision must be a non-placeholder string")
    if frozen_revision != policy_revision:
        raise error_type(
            "frozen policy_revision does not match manifest policy_revision"
        )


def _validate_selected_variable(
    payload: Mapping[str, Any],
    task: TaskSpec,
    error_type: type[ProtocolGuardError],
) -> None:
    selected = payload.get("selected_variable")
    if isinstance(selected, str) and _is_placeholder(selected):
        raise error_type("selected_variable must not be a placeholder")
    if not isinstance(selected, dict) or not selected:
        raise error_type("selected_variable must be a nonempty object with name")
    name = selected.get("name")
    allowed = {"theta_rel", "relative_planar_position", "object_gripper_contact"}
    if not isinstance(name, str) or name not in allowed:
        raise error_type("selected_variable.name is not preregistered")
    if name == "theta_rel":
        symmetry = selected.get("symmetry_order")
        if (
            isinstance(symmetry, bool)
            or not isinstance(symmetry, int)
            or symmetry != task.planar_symmetry_order
        ):
            raise error_type(
                "selected_variable.symmetry_order does not match the manifest task"
            )


def _validate_sha256(value: Any, path: str) -> None:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or len(set(value.lower())) == 1
    ):
        raise LockedTestGuardError(
            f"{path} must be a non-placeholder SHA-256 hex digest"
        )


def _validate_representation(
    payload: Mapping[str, Any], selection: CalibrationSelectionConfig
) -> None:
    representation = _required_mapping(
        payload, "representation_probe", LockedTestGuardError
    )
    required = {"candidate", "ridge_alpha", "coefficient_hash"}
    if set(representation) != required:
        raise LockedTestGuardError(
            "representation_probe must have exactly: " + ", ".join(sorted(required))
        )
    candidate = representation.get("candidate")
    if candidate not in selection.representation_candidates:
        raise LockedTestGuardError(
            "representation_probe.candidate is not one of the five configured candidates"
        )
    ridge_alpha = representation.get("ridge_alpha")
    if isinstance(ridge_alpha, bool) or not isinstance(ridge_alpha, (int, float)):
        raise LockedTestGuardError("representation_probe.ridge_alpha must be numeric")
    if float(ridge_alpha) not in selection.ridge_alpha_candidates:
        raise LockedTestGuardError(
            "representation_probe.ridge_alpha is outside the configured candidates"
        )
    _validate_sha256(
        representation.get("coefficient_hash"),
        "representation_probe.coefficient_hash",
    )


def _value_is_allowed(value: Any, allowed: tuple[Any, ...]) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return float(value) in {float(candidate) for candidate in allowed}


def _validate_predictor_hyperparameters(
    hyperparameters: Mapping[str, Any], candidate: PredictorCandidateConfig
) -> None:
    expected = set(candidate.hyperparameter_grid)
    if candidate.max_iter_min is not None or candidate.max_iter_max is not None:
        expected.add("max_iter")
    if set(hyperparameters) != expected:
        raise LockedTestGuardError(
            f"predictor.hyperparameters for {candidate.family} must have exactly: "
            + ", ".join(sorted(expected))
        )
    for name, allowed in candidate.hyperparameter_grid.items():
        if not _value_is_allowed(hyperparameters[name], allowed):
            raise LockedTestGuardError(
                f"predictor.hyperparameters.{name} is outside the configured candidates"
            )
    if "max_iter" in expected:
        max_iter = hyperparameters["max_iter"]
        if isinstance(max_iter, bool) or not isinstance(max_iter, int):
            raise LockedTestGuardError(
                "predictor.hyperparameters.max_iter must be an integer"
            )
        if candidate.max_iter_min is not None and max_iter < candidate.max_iter_min:
            raise LockedTestGuardError(
                "predictor.hyperparameters.max_iter is too small"
            )
        if candidate.max_iter_max is not None and max_iter > candidate.max_iter_max:
            raise LockedTestGuardError(
                "predictor.hyperparameters.max_iter is too large"
            )


def _validate_predictor(
    payload: Mapping[str, Any], selection: CalibrationSelectionConfig
) -> None:
    predictor = _required_mapping(payload, "predictor", LockedTestGuardError)
    required = {"family", "hyperparameters", "coefficient_hash"}
    if set(predictor) != required:
        raise LockedTestGuardError(
            "predictor must have exactly: " + ", ".join(sorted(required))
        )
    family = predictor.get("family")
    if not isinstance(family, str) or family not in selection.predictor_candidates:
        raise LockedTestGuardError("predictor.family is not a configured candidate")
    hyperparameters = predictor.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        raise LockedTestGuardError("predictor.hyperparameters must be an object")
    _validate_predictor_hyperparameters(
        hyperparameters, selection.predictor_candidates[family]
    )
    _validate_sha256(predictor.get("coefficient_hash"), "predictor.coefficient_hash")


def _validate_artifact_hashes(payload: Mapping[str, Any]) -> None:
    hashes = _required_mapping(payload, "artifact_hashes", LockedTestGuardError)
    required = {
        "m0_predictor",
        "m1_predictor",
        "m2_predictor",
        "probe",
        "calibration_manifest",
    }
    missing = required - hashes.keys()
    if missing:
        raise LockedTestGuardError(
            "artifact_hashes is missing: " + ", ".join(sorted(missing))
        )
    for name, digest in hashes.items():
        if not isinstance(name, str) or _is_placeholder(name):
            raise LockedTestGuardError(
                "artifact_hashes keys must be meaningful strings"
            )
        _validate_sha256(digest, f"artifact_hashes.{name}")
    predictor_hashes = {
        hashes["m0_predictor"],
        hashes["m1_predictor"],
        hashes["m2_predictor"],
    }
    if len(predictor_hashes) != 3:
        raise LockedTestGuardError(
            "M0, M1, and M2 predictor artifact hashes must be distinct"
        )


def _validate_alarm_thresholds(payload: Mapping[str, Any]) -> None:
    thresholds = _required_mapping(payload, "alarm_thresholds", LockedTestGuardError)
    missing = {"m0", "m1", "m2"} - thresholds.keys()
    if missing:
        raise LockedTestGuardError(
            "alarm_thresholds is missing: " + ", ".join(sorted(missing))
        )
    for model, threshold in thresholds.items():
        if (
            not isinstance(model, str)
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise LockedTestGuardError(
                f"alarm_thresholds.{model} must be a finite probability"
            )


def _validate_numeric_tree(value: Any, path: str) -> None:
    if isinstance(value, bool):
        raise LockedTestGuardError(f"{path} must contain numeric metrics, not booleans")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise LockedTestGuardError(f"{path} contains a non-finite metric")
        return
    if isinstance(value, dict) and value:
        for key, nested in value.items():
            if not isinstance(key, str) or _is_placeholder(key):
                raise LockedTestGuardError(f"{path} has an invalid metric name")
            _validate_numeric_tree(nested, f"{path}.{key}")
        return
    if isinstance(value, list) and value:
        for index, nested in enumerate(value):
            _validate_numeric_tree(nested, f"{path}[{index}]")
        return
    raise LockedTestGuardError(f"{path} must be a nonempty numeric metric structure")


def _validate_calibration_metrics(payload: Mapping[str, Any]) -> None:
    metrics = _required_mapping(payload, "calibration_metrics", LockedTestGuardError)
    missing = {"m0", "m1", "m2"} - metrics.keys()
    if missing:
        raise LockedTestGuardError(
            "calibration_metrics is missing: " + ", ".join(sorted(missing))
        )
    _validate_numeric_tree(metrics, "calibration_metrics")
    for model in ("m0", "m1", "m2"):
        model_metrics = metrics[model]
        required = {"log_loss", "brier", "auroc"}
        if not isinstance(model_metrics, dict) or not required <= model_metrics.keys():
            raise LockedTestGuardError(
                f"calibration_metrics.{model} must contain log_loss, brier, and auroc"
            )
        for nonnegative_name in ("log_loss", "brier"):
            nonnegative = model_metrics[nonnegative_name]
            if (
                isinstance(nonnegative, bool)
                or not isinstance(nonnegative, (int, float))
                or float(nonnegative) < 0.0
            ):
                raise LockedTestGuardError(
                    f"calibration_metrics.{model}.{nonnegative_name} "
                    "must be finite and nonnegative"
                )
        for bounded_name in ("auroc",):
            bounded = model_metrics[bounded_name]
            if (
                isinstance(bounded, bool)
                or not isinstance(bounded, (int, float))
                or not 0.0 <= float(bounded) <= 1.0
            ):
                raise LockedTestGuardError(
                    f"calibration_metrics.{model}.{bounded_name} must be in [0, 1]"
                )
        if float(model_metrics["brier"]) > 1.0:
            raise LockedTestGuardError(
                f"calibration_metrics.{model}.brier must be in [0, 1]"
            )


def assert_calibration_ready(
    repo_root: str | Path,
    config: CalibrationGuardConfig,
    *,
    task: TaskSpec,
    policy_revision: str,
) -> CalibrationReceipt:
    """Authorize Calibration only from the committed Reality Gate selection."""

    repo, head, freeze_file, payload = _read_lock(
        repo_root,
        config,
        stage="Calibration",
        error_type=CalibrationGuardError,
    )
    _reject_placeholders(payload, path="freeze", error_type=CalibrationGuardError)
    _validate_task(payload, task, CalibrationGuardError)
    _validate_selected_variable(payload, task, CalibrationGuardError)
    _validate_policy_revision(payload, policy_revision, CalibrationGuardError)
    return CalibrationReceipt(repo, head, config.required_tag, freeze_file, payload)


def assert_locked_test_ready(
    repo_root: str | Path,
    config: LockedTestGuardConfig,
    *,
    task: TaskSpec,
    policy_revision: str,
    selection: CalibrationSelectionConfig,
) -> LockedTestReceipt:
    """Authorize Locked Test only from a complete committed Calibration freeze."""

    repo, head, freeze_file, payload = _read_lock(
        repo_root,
        config,
        stage="Locked Test",
        error_type=LockedTestGuardError,
    )
    _reject_placeholders(payload, path="freeze", error_type=LockedTestGuardError)
    missing = _LOCKED_FIELDS - payload.keys()
    if missing:
        raise LockedTestGuardError(
            "calibration freeze file is incomplete; missing: "
            + ", ".join(sorted(missing))
        )
    _validate_task(payload, task, LockedTestGuardError)
    _validate_selected_variable(payload, task, LockedTestGuardError)
    _validate_policy_revision(payload, policy_revision, LockedTestGuardError)
    _validate_representation(payload, selection)
    _validate_predictor(payload, selection)
    _validate_artifact_hashes(payload)
    _validate_alarm_thresholds(payload)
    patch_strength = payload.get("patch_strength")
    if not _value_is_allowed(patch_strength, selection.patch_strength_candidates):
        raise LockedTestGuardError(
            "patch_strength must be one of the configured values 0.25, 0.5, or 1.0"
        )
    _validate_calibration_metrics(payload)
    return LockedTestReceipt(repo, head, config.required_tag, freeze_file, payload)
