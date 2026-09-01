"""Fail-closed repository and payload guards for protected split access."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .config import (
    CalibrationGuardConfig,
    CalibrationSelectionConfig,
    LockedTestGuardConfig,
    PredictorCandidateConfig,
    ProtocolConfig,
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
_REALITY_GATE_LOCK_PROTOCOL = "preregistered-reality-gate-lock-v1"
_REALITY_GATE_LOCK_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "policy_revision",
        "code_commit",
        "selected_task",
        "selected_variable",
        "reality_gate_lock_receipt_sha256",
        "reality_gate_lock_receipt",
        "reality_gate_receipt_sha256",
        "reality_gate_receipt",
        "orientation_eligibility_sha256",
        "orientation_eligibility",
        "selected_task_artifacts",
        "failure_event_freeze_sha256",
        "failure_event_freeze",
    }
)
_MAX_LOCK_JSON_BYTES = 16 * 1024 * 1024


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


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


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

    freeze_file = repo / config.required_file
    try:
        relative_freeze = freeze_file.relative_to(repo).as_posix()
    except ValueError as exc:
        raise error_type("freeze file must be inside the repository") from exc
    current = repo
    for component in PurePosixPath(relative_freeze).parts:
        current = current / component
        if current.is_symlink():
            raise error_type(f"{stage} freeze file path must not contain symlinks")
    try:
        file_stat = freeze_file.stat()
    except OSError as exc:
        raise error_type(f"required {stage} freeze file is missing: {freeze_file}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise error_type(f"required {stage} freeze file is missing: {freeze_file}")
    if file_stat.st_size > _MAX_LOCK_JSON_BYTES:
        raise error_type(f"{stage} freeze file exceeds the maximum safe JSON size")
    try:
        encoded = freeze_file.read_bytes()
        frozen = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
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


def _canonical_sha256(
    value: Any, path: str, error_type: type[ProtocolGuardError]
) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise error_type(f"{path} is not finite canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _validate_lock_digest(
    value: Any, path: str, error_type: type[ProtocolGuardError]
) -> None:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
        or len(set(value)) == 1
    ):
        raise error_type(f"{path} must be a non-placeholder lowercase SHA-256")


def _validate_code_commit(
    value: Any, path: str, error_type: type[ProtocolGuardError]
) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise error_type(f"{path} must be a lowercase 40-character Git SHA")


def _require_exact_artifact_list(
    value: Any,
    expected: list[Mapping[str, Any]],
    path: str,
    error_type: type[ProtocolGuardError],
) -> None:
    if not isinstance(value, list) or value != expected:
        raise error_type(f"{path} does not exactly match the selected raw artifacts")
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {
            "episode_id",
            "metadata_sha256",
            "trajectory_sha256",
        }:
            raise error_type(f"{path}[{index}] is not a complete artifact identity")
        if not isinstance(item["episode_id"], str) or not item["episode_id"]:
            raise error_type(f"{path}[{index}].episode_id is invalid")
        _validate_lock_digest(item["metadata_sha256"], f"{path}[{index}].metadata_sha256", error_type)
        _validate_lock_digest(item["trajectory_sha256"], f"{path}[{index}].trajectory_sha256", error_type)


def _validate_reality_gate_lock_payload(
    payload: Mapping[str, Any],
    task: TaskSpec,
    policy_revision: str,
    protocol: ProtocolConfig,
    repo: Path,
    head: str,
    error_type: type[ProtocolGuardError],
) -> None:
    """Rehydrate and cross-bind the complete Reality-Gate lock bundle.

    Every nested receipt is parsed by its owning domain module.  Serialized
    ``passes``/``eligible`` flags are therefore compared with recomputed typed
    decisions, not merely checked for internal hash consistency.
    """

    try:
        from .failure_artifacts import (
            FailureArtifactError,
            failure_event_freeze_from_metadata,
        )
        from .reality_gate import (
            RealityGateError,
            RealityGateLockReceipt,
            _artifact_identity_from_metadata,
            orientation_eligibility_from_metadata,
            reality_gate_receipt_from_metadata,
        )

        if set(payload) != _REALITY_GATE_LOCK_FIELDS:
            missing = sorted(_REALITY_GATE_LOCK_FIELDS - set(payload))
            extra = sorted(set(payload) - _REALITY_GATE_LOCK_FIELDS)
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if extra:
                detail.append("unexpected " + ", ".join(extra))
            raise error_type(
                "Reality-Gate lock payload keys differ: " + "; ".join(detail)
            )
        if payload["schema_version"] != 1 or payload["protocol"] != _REALITY_GATE_LOCK_PROTOCOL:
            raise error_type("Reality-Gate lock schema or protocol is unsupported")
        _validate_code_commit(payload["code_commit"], "code_commit", error_type)
        _validate_task(payload, task, error_type)
        _validate_policy_revision(payload, policy_revision, error_type)
        if payload["selected_variable"] != {
            "name": "theta_rel",
            "symmetry_order": task.planar_symmetry_order,
        }:
            raise error_type("Reality-Gate selected_variable is not the frozen theta_rel choice")

        selected_artifacts = payload["selected_task_artifacts"]
        if not isinstance(selected_artifacts, list) or len(selected_artifacts) != 40:
            raise error_type("selected_task_artifacts must contain exactly 40 identities")
        parsed_selected_artifacts = tuple(
            _artifact_identity_from_metadata(item, f"selected_task_artifacts[{index}]")
            for index, item in enumerate(selected_artifacts)
        )
        if tuple(item.episode_id for item in parsed_selected_artifacts) != tuple(
            sorted(item.episode_id for item in parsed_selected_artifacts)
        ):
            raise error_type("selected_task_artifacts must be sorted by episode ID")
        if len({item.episode_id for item in parsed_selected_artifacts}) != 40:
            raise error_type("selected_task_artifacts contains duplicate episode IDs")

        gate_payload = payload["reality_gate_receipt"]
        _validate_lock_digest(
            payload["reality_gate_receipt_sha256"],
            "reality_gate_receipt_sha256",
            error_type,
        )
        if _canonical_sha256(gate_payload, "reality_gate_receipt", error_type) != payload[
            "reality_gate_receipt_sha256"
        ]:
            raise error_type("reality_gate_receipt hash does not match its content")
        gate = reality_gate_receipt_from_metadata(gate_payload, protocol)
        if gate.sha256 != payload["reality_gate_receipt_sha256"]:
            raise error_type("rehydrated Reality-Gate receipt hash differs")
        try:
            _git(
                repo,
                error_type,
                "rev-parse",
                "--verify",
                f"{gate.code_commit}^{{commit}}",
            )
        except ProtocolGuardError as exc:
            raise error_type(
                "task-gate Discovery code commit is not present in the repository"
            ) from exc
        if gate.policy_revision != policy_revision:
            raise error_type("task-gate policy revision differs from the lock")
        if gate.selected_task != task or gate.selected_task is None:
            raise error_type("task-gate selected task differs from the lock")
        selected_decision = gate.attempts[-1]
        expected_artifacts = tuple(sorted(selected_decision.raw_artifacts, key=lambda item: item.episode_id))
        if expected_artifacts != parsed_selected_artifacts:
            raise error_type("selected task artifacts differ from recomputed gate cells")

        orientation_payload = payload["orientation_eligibility"]
        _validate_lock_digest(
            payload["orientation_eligibility_sha256"],
            "orientation_eligibility_sha256",
            error_type,
        )
        if _canonical_sha256(orientation_payload, "orientation_eligibility", error_type) != payload[
            "orientation_eligibility_sha256"
        ]:
            raise error_type("orientation eligibility hash does not match its content")
        orientation = orientation_eligibility_from_metadata(orientation_payload)
        if orientation.sha256 != payload["orientation_eligibility_sha256"]:
            raise error_type("rehydrated orientation eligibility hash differs")
        if orientation.source_artifacts != parsed_selected_artifacts:
            raise error_type("orientation sources differ from selected task artifacts")
        if (
            not orientation.eligible
            or orientation.symmetry_order != task.planar_symmetry_order
            or orientation._validation_marker is None
        ):
            raise error_type("orientation eligibility does not authorize theta_rel")

        lock_payload = payload["reality_gate_lock_receipt"]
        _validate_lock_digest(
            payload["reality_gate_lock_receipt_sha256"],
            "reality_gate_lock_receipt_sha256",
            error_type,
        )
        if _canonical_sha256(lock_payload, "reality_gate_lock_receipt", error_type) != payload[
            "reality_gate_lock_receipt_sha256"
        ]:
            raise error_type("Reality-Gate lock receipt hash does not match its content")
        lock = RealityGateLockReceipt(
            task_gate_receipt=gate,
            selected_task_artifacts=parsed_selected_artifacts,
            orientation_eligibility=orientation,
        )
        if lock.to_dict() != lock_payload or lock.sha256 != payload[
            "reality_gate_lock_receipt_sha256"
        ]:
            raise error_type("nested Reality-Gate lock receipt is not reconstructable")
        if lock_payload.get("task_gate_receipt") != gate_payload:
            raise error_type("nested task-gate receipt differs from the top-level receipt")
        if lock_payload.get("orientation_eligibility") != orientation_payload:
            raise error_type("nested orientation eligibility differs")

        freeze_payload = payload["failure_event_freeze"]
        _validate_lock_digest(
            payload["failure_event_freeze_sha256"],
            "failure_event_freeze_sha256",
            error_type,
        )
        if _canonical_sha256(freeze_payload, "failure_event_freeze", error_type) != payload[
            "failure_event_freeze_sha256"
        ]:
            raise error_type("failure-event freeze hash does not match its content")
        freeze = failure_event_freeze_from_metadata(freeze_payload)
        if freeze.sha256 != payload["failure_event_freeze_sha256"]:
            raise error_type("rehydrated failure-event freeze hash differs")
        expected_freeze_task = {
            "suite": task.suite,
            "task_id": task.task_id,
            "task_rank": task.rank,
            "language": task.language,
            "primary_object": task.primary_object,
            "planar_symmetry_order": task.planar_symmetry_order,
        }
        if freeze.task.to_dict() != expected_freeze_task:
            raise error_type("failure-event freeze task differs from the selected task")
        expected_ids = tuple(item.episode_id for item in parsed_selected_artifacts)
        if tuple(item.episode_id for item in freeze.bounds.expected_artifacts) != expected_ids:
            raise error_type("failure-event bounds do not cover the selected artifacts")
        if tuple(freeze.video_audit_episode_ids) != expected_ids:
            raise error_type("failure-event video audit does not cover selected artifacts")
        if tuple(item.artifact.episode_id for item in freeze.bounds.provenance) != expected_ids:
            raise error_type("failure-event provenance order differs")
        if tuple(item.episode_id for item in freeze.annotations) != expected_ids:
            raise error_type("failure-event annotation order differs")
        provenance_by_id = {item.artifact.episode_id: item for item in freeze.bounds.provenance}
        cell_by_id = {cell.episode_id: cell for cell in selected_decision.cells}
        for annotation in freeze.annotations:
            cell = cell_by_id.get(annotation.episode_id)
            provenance = provenance_by_id.get(annotation.episode_id)
            if cell is None or provenance is None or not cell.executed:
                raise error_type("failure-event annotation is not bound to a final gate cell")
            if (
                annotation.valid_reset != cell.valid
                or annotation.success != cell.success
                or provenance.valid_reset != cell.valid
                or provenance.success != cell.success
            ):
                raise error_type("failure-event outcomes disagree with final gate cells")

        implementation_commit = freeze.implementation_commit
        _validate_code_commit(
            implementation_commit,
            "failure_event_freeze.implementation_commit",
            error_type,
        )
        try:
            _git(
                repo,
                error_type,
                "merge-base",
                "--is-ancestor",
                implementation_commit,
                head,
            )
        except ProtocolGuardError as exc:
            raise error_type(
                "failure-event implementation commit is not an ancestor of the lock HEAD"
            ) from exc
    except error_type:
        raise
    except (RealityGateError, FailureArtifactError, KeyError, TypeError, ValueError) as exc:
        raise error_type(f"Reality-Gate lock evidence is invalid: {exc}") from exc


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
        if candidate.family == "histogram_gradient_boosting" and max_iter != 200:
            raise LockedTestGuardError(
                "predictor.hyperparameters.max_iter must be exactly 200 for "
                "histogram_gradient_boosting"
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


def _safe_artifact_path(repo: Path, raw_path: Any, *, name: str) -> tuple[Path, str]:
    if not isinstance(raw_path, str) or _is_placeholder(raw_path):
        raise LockedTestGuardError(
            f"artifact_hashes.{name}.path must be a meaningful repository-relative path"
        )
    if "\\" in raw_path:
        raise LockedTestGuardError(
            f"artifact_hashes.{name}.path must use repository-relative POSIX syntax"
        )
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != raw_path
    ):
        raise LockedTestGuardError(
            f"artifact_hashes.{name}.path must be a safe repository-relative path"
        )

    candidate = repo.joinpath(*relative.parts)
    current = repo
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise LockedTestGuardError(
                f"artifact_hashes.{name}.path must not contain symlinks"
            )
    try:
        mode = candidate.stat().st_mode
    except OSError as exc:
        raise LockedTestGuardError(
            f"artifact_hashes.{name}.path is missing or unreadable: {candidate}"
        ) from exc
    if not stat.S_ISREG(mode):
        raise LockedTestGuardError(
            f"artifact_hashes.{name}.path must identify a regular file"
        )
    return candidate, relative.as_posix()


def _stream_sha256(path: Path, *, name: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise LockedTestGuardError(
            f"could not hash artifact_hashes.{name}.path: {exc}"
        ) from exc
    return digest.hexdigest()


def _validate_artifact_hashes(repo: Path, payload: Mapping[str, Any]) -> None:
    hashes = _required_mapping(payload, "artifact_hashes", LockedTestGuardError)
    required = {
        "bound_probe",
        "feature_reference_arrays",
        "feature_reference_metadata",
        "predictor_bundle",
        "predictor_metadata",
        "probe",
        "reality_gate_manifest",
        "calibration_manifest",
    }
    missing = required - hashes.keys()
    if missing:
        raise LockedTestGuardError(
            "artifact_hashes is missing: " + ", ".join(sorted(missing))
        )
    declarations: dict[str, tuple[Any, str]] = {}
    for name, declaration in hashes.items():
        if not isinstance(name, str) or _is_placeholder(name):
            raise LockedTestGuardError(
                "artifact_hashes keys must be meaningful strings"
            )
        if not isinstance(declaration, dict) or set(declaration) != {"path", "sha256"}:
            raise LockedTestGuardError(
                f"artifact_hashes.{name} must have exactly: path, sha256"
            )
        digest = declaration.get("sha256")
        _validate_sha256(digest, f"artifact_hashes.{name}.sha256")
        declarations[name] = (declaration.get("path"), str(digest))
    for name, (raw_path, expected_digest) in declarations.items():
        artifact, relative = _safe_artifact_path(repo, raw_path, name=name)
        try:
            _git(
                repo,
                LockedTestGuardError,
                "ls-files",
                "--error-unmatch",
                "--",
                relative,
            )
        except ProtocolGuardError as exc:
            raise LockedTestGuardError(
                f"artifact_hashes.{name}.path must be tracked in the lock commit"
            ) from exc
        actual_digest = _stream_sha256(artifact, name=name)
        if actual_digest != expected_digest:
            raise LockedTestGuardError(
                f"artifact_hashes.{name}.sha256 does not match the artifact bytes"
            )


def _validate_alarm_thresholds(payload: Mapping[str, Any]) -> None:
    thresholds = _required_mapping(payload, "alarm_thresholds", LockedTestGuardError)
    missing = {"m0", "m1", "m2"} - thresholds.keys()
    if missing:
        raise LockedTestGuardError(
            "alarm_thresholds is missing: " + ", ".join(sorted(missing))
        )
    never_alarm = math.nextafter(1.0, math.inf)
    for model, threshold in thresholds.items():
        numeric = (
            float(threshold)
            if not isinstance(threshold, bool) and isinstance(threshold, (int, float))
            else math.nan
        )
        if (
            not isinstance(model, str)
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(numeric)
            or not (0.0 <= numeric <= 1.0 or numeric == never_alarm)
        ):
            raise LockedTestGuardError(
                f"alarm_thresholds.{model} must be in [0, 1] or the exact "
                "nextafter(1, +inf) never-alarm sentinel"
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
    protocol: ProtocolConfig,
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
    _validate_reality_gate_lock_payload(
        payload,
        task,
        policy_revision,
        protocol,
        repo,
        head,
        CalibrationGuardError,
    )
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
    _validate_artifact_hashes(repo, payload)
    _validate_alarm_thresholds(payload)
    patch_strength = payload.get("patch_strength")
    if not _value_is_allowed(patch_strength, selection.patch_strength_candidates):
        raise LockedTestGuardError(
            "patch_strength must be one of the configured values 0.25, 0.5, or 1.0"
        )
    _validate_calibration_metrics(payload)
    return LockedTestReceipt(repo, head, config.required_tag, freeze_file, payload)
