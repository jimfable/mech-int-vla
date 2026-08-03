"""Safe persistence for the post-Discovery failure-event freeze.

The numerical event rules live in :mod:`mech_int_vla.failure_events`.  This
module supplies the deliberately smaller storage boundary: canonical JSON in a
content-addressed directory, written by rename and loaded without pickle.  The
loader reconstructs the validated domain objects instead of trusting decoded
JSON, then requires byte-for-byte canonical equivalence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .failure_events import (
    AnnotationStatus,
    ArtifactIdentity,
    DiscoveryBoundsDerivation,
    DiscoveryEpisodeProvenance,
    FailureEventFreezeManifest,
    FailureEventResult,
    FailureEventType,
    ReachableBounds,
    TaskIdentity,
    failure_event_protocol_metadata,
)

FAILURE_ARTIFACT_SCHEMA_VERSION = 1
FAILURE_FREEZE_FILENAME = "failure-freeze.json"
FAILURE_EVENT_FILENAME = "failure-event.json"
_REALITY_GATE_TAG = "prereg-locked-v1"
_SHA256_LENGTH = 64
_PROTECTED_PATH_PARTS = frozenset({"configs", "locks"})
_MAX_JSON_BYTES = 16 * 1024 * 1024


class FailureArtifactError(ValueError):
    """Raised when a persisted failure artifact is unsafe or invalid."""


@dataclass(frozen=True)
class FailureEventArtifact:
    """One annotated Discovery episode bound to its raw input and freeze."""

    source_artifact: ArtifactIdentity
    annotation: FailureEventResult
    freeze_sha256: str
    reality_gate_tag: str = _REALITY_GATE_TAG
    schema_version: int = FAILURE_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.source_artifact, ArtifactIdentity):
            raise FailureArtifactError(
                "source_artifact must be a validated ArtifactIdentity"
            )
        if not isinstance(self.annotation, FailureEventResult):
            raise FailureArtifactError(
                "annotation must be a validated FailureEventResult"
            )
        if self.source_artifact.episode_id != self.annotation.episode_id:
            raise FailureArtifactError(
                "source artifact and annotation episode IDs differ"
            )
        if not _is_lower_sha256(self.freeze_sha256):
            raise FailureArtifactError(
                "freeze_sha256 must be 64 lowercase hexadecimal digits"
            )
        if (
            type(self.schema_version) is not int
            or self.schema_version != FAILURE_ARTIFACT_SCHEMA_VERSION
        ):
            raise FailureArtifactError("unsupported failure-event artifact schema")
        if self.reality_gate_tag != _REALITY_GATE_TAG:
            raise FailureArtifactError("unsupported failure-event Reality Gate tag")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_kind": "failure-event-record",
            "reality_gate_tag": self.reality_gate_tag,
            "freeze_sha256": self.freeze_sha256,
            "source_artifact": self.source_artifact.to_dict(),
            "annotation": self.annotation.to_dict(),
        }

    def canonical_json(self) -> bytes:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()


def create_failure_event_artifact(
    freeze: FailureEventFreezeManifest, episode_id: str
) -> FailureEventArtifact:
    """Create the unique per-episode record authorized by ``freeze``."""

    if not isinstance(freeze, FailureEventFreezeManifest):
        raise FailureArtifactError("freeze must be a validated freeze manifest")
    if not isinstance(episode_id, str) or not episode_id:
        raise FailureArtifactError("episode_id must be a non-empty string")
    source_by_id = {item.episode_id: item for item in freeze.bounds.expected_artifacts}
    annotation_by_id = {item.episode_id: item for item in freeze.annotations}
    if (
        episode_id not in source_by_id
        or episode_id not in annotation_by_id
        or episode_id not in freeze.video_audit_episode_ids
    ):
        raise FailureArtifactError(
            "episode is not in the freeze's exact artifact/annotation/video audit"
        )
    return FailureEventArtifact(
        source_artifact=source_by_id[episode_id],
        annotation=annotation_by_id[episode_id],
        freeze_sha256=freeze.sha256,
    )


def create_failure_event_artifacts(
    freeze: FailureEventFreezeManifest,
) -> tuple[FailureEventArtifact, ...]:
    """Materialize all per-episode records in the freeze's canonical order."""

    if not isinstance(freeze, FailureEventFreezeManifest):
        raise FailureArtifactError("freeze must be a validated freeze manifest")
    return tuple(
        create_failure_event_artifact(freeze, item.episode_id)
        for item in freeze.bounds.expected_artifacts
    )


def write_failure_freeze(
    freeze: FailureEventFreezeManifest, output_root: str | Path
) -> Path:
    """Atomically publish one content-addressed failure freeze."""

    if not isinstance(freeze, FailureEventFreezeManifest):
        raise FailureArtifactError("freeze must be a validated freeze manifest")
    return _write_artifact(
        canonical=freeze.canonical_json(),
        output_root=output_root,
        filename=FAILURE_FREEZE_FILENAME,
        description="failure freeze",
    )


def load_failure_freeze(
    path: str | Path, expected_sha256: str | None = None
) -> FailureEventFreezeManifest:
    """Load and fully validate a canonical failure freeze without pickle."""

    canonical, digest = _read_artifact(
        path=path,
        expected_sha256=expected_sha256,
        filename=FAILURE_FREEZE_FILENAME,
        description="failure freeze",
    )
    metadata = _decode_json(canonical, FAILURE_FREEZE_FILENAME)
    try:
        freeze = _freeze_from_metadata(metadata)
    except FailureArtifactError:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise FailureArtifactError(f"invalid failure freeze: {exc}") from exc
    reconstructed = freeze.canonical_json()
    if reconstructed != canonical:
        raise FailureArtifactError(
            "failure-freeze.json bytes are not the exact canonical encoding"
        )
    if hashlib.sha256(reconstructed).hexdigest() != digest:
        raise FailureArtifactError(
            "failure freeze directory does not match its SHA-256"
        )
    return freeze


def failure_event_freeze_from_metadata(value: Any) -> FailureEventFreezeManifest:
    """Rehydrate one strict canonical freeze object from decoded JSON metadata.

    The guard uses this public wrapper so it shares the exact parser and domain
    validation used by the on-disk loader, rather than maintaining a second,
    weaker schema implementation.
    """

    try:
        return _freeze_from_metadata(value)
    except FailureArtifactError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise FailureArtifactError("failure freeze metadata is malformed") from exc


def write_failure_event_artifact(
    artifact: FailureEventArtifact,
    output_root: str | Path,
    *,
    freeze: FailureEventFreezeManifest,
) -> Path:
    """Atomically publish one content-addressed per-episode event record."""

    if not isinstance(artifact, FailureEventArtifact):
        raise FailureArtifactError("artifact must be a validated FailureEventArtifact")
    expected = create_failure_event_artifact(freeze, artifact.annotation.episode_id)
    if artifact != expected:
        raise FailureArtifactError("failure-event artifact differs from the freeze")
    return _write_artifact(
        canonical=artifact.canonical_json(),
        output_root=output_root,
        filename=FAILURE_EVENT_FILENAME,
        description="failure-event artifact",
    )


def load_failure_event_artifact(
    path: str | Path,
    *,
    freeze: FailureEventFreezeManifest,
    expected_sha256: str | None = None,
) -> FailureEventArtifact:
    """Load a per-episode record and optionally prove its freeze membership."""

    canonical, digest = _read_artifact(
        path=path,
        expected_sha256=expected_sha256,
        filename=FAILURE_EVENT_FILENAME,
        description="failure-event artifact",
    )
    metadata = _decode_json(canonical, FAILURE_EVENT_FILENAME)
    try:
        artifact = _event_artifact_from_metadata(metadata)
    except FailureArtifactError:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise FailureArtifactError(f"invalid failure-event artifact: {exc}") from exc
    reconstructed = artifact.canonical_json()
    if reconstructed != canonical:
        raise FailureArtifactError(
            "failure-event.json bytes are not the exact canonical encoding"
        )
    if hashlib.sha256(reconstructed).hexdigest() != digest:
        raise FailureArtifactError(
            "failure-event artifact directory does not match its SHA-256"
        )
    expected = create_failure_event_artifact(freeze, artifact.annotation.episode_id)
    if artifact != expected:
        raise FailureArtifactError(
            "failure-event artifact differs from the supplied freeze"
        )
    return artifact


def _write_artifact(
    *,
    canonical: bytes,
    output_root: str | Path,
    filename: str,
    description: str,
) -> Path:
    root = _normalized_absolute_path(output_root)
    if any(part in _PROTECTED_PATH_PARTS for part in root.parts):
        raise FailureArtifactError(
            "failure artifacts may not be written into lock/config paths"
        )
    if _has_symlink_component(root):
        raise FailureArtifactError("output path may not contain a symlink component")
    if root.exists() and not root.is_dir():
        raise FailureArtifactError("output_root must be a directory")
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FailureArtifactError(f"could not create output_root: {exc}") from exc
    if _has_symlink_component(root) or root.is_symlink() or not root.is_dir():
        raise FailureArtifactError("output path may not contain a symlink component")

    digest = hashlib.sha256(canonical).hexdigest()
    destination = root / digest
    if _lexists(destination):
        raise FileExistsError(f"refusing to overwrite {description} {destination}")

    lock_path = root / f".{digest}.publish.lock"
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"another writer is publishing {description} {destination}"
        ) from exc
    except OSError as exc:
        raise FailureArtifactError(
            f"could not acquire publication lock: {exc}"
        ) from exc
    os.close(lock_descriptor)

    staging: Path | None = None
    try:
        _fsync_directory(root)
        staging = Path(tempfile.mkdtemp(prefix=f".{digest}.tmp-", dir=root))
        artifact_path = staging / filename
        with artifact_path.open("xb") as stream:
            stream.write(canonical)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(staging)
        if _lexists(destination):
            raise FileExistsError(f"refusing to overwrite {description} {destination}")
        os.rename(staging, destination)
        staging = None
        _fsync_directory(root)
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        finally:
            _fsync_directory(root)
    return destination


def _read_artifact(
    *,
    path: str | Path,
    expected_sha256: str | None,
    filename: str,
    description: str,
) -> tuple[bytes, str]:
    if expected_sha256 is not None and not _is_lower_sha256(expected_sha256):
        raise FailureArtifactError(
            "expected_sha256 must be 64 lowercase hexadecimal digits"
        )
    directory = _normalized_absolute_path(path)
    if _has_symlink_component(directory):
        raise FailureArtifactError(
            f"{description} path may not contain a symlink component"
        )
    if not directory.is_dir():
        raise FailureArtifactError(f"{description} path must be a directory")
    try:
        directory_before = directory.lstat()
    except OSError as exc:
        raise FailureArtifactError(f"could not inspect {description}: {exc}") from exc
    try:
        entries = list(os.scandir(directory))
    except OSError as exc:
        raise FailureArtifactError(f"could not inspect {description}: {exc}") from exc
    if len(entries) != 1 or entries[0].name != filename:
        raise FailureArtifactError(f"{description} must contain exactly one {filename}")
    entry = entries[0]
    try:
        entry_stat = entry.stat(follow_symlinks=False)
    except OSError as exc:
        raise FailureArtifactError(f"could not inspect {filename}: {exc}") from exc
    if entry.is_symlink() or not stat.S_ISREG(entry_stat.st_mode):
        raise FailureArtifactError(f"{filename} must be a regular file, not a symlink")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(entry.path, flags)
        try:
            opened_before = os.fstat(descriptor)
            if not stat.S_ISREG(opened_before.st_mode):
                raise FailureArtifactError(f"{filename} must be a regular file")
            if _stat_identity(entry_stat) != _stat_identity(opened_before):
                raise FailureArtifactError(f"{description} changed before it was read")
            if opened_before.st_size > _MAX_JSON_BYTES:
                raise FailureArtifactError(f"{description} exceeds the size limit")
            canonical = _read_all(descriptor)
            opened_after = os.fstat(descriptor)
            if _stat_identity(opened_before) != _stat_identity(opened_after):
                raise FailureArtifactError(f"{description} changed while being read")
        finally:
            os.close(descriptor)
    except FailureArtifactError:
        raise
    except OSError as exc:
        raise FailureArtifactError(f"could not read {filename} safely: {exc}") from exc

    try:
        path_after = Path(entry.path).lstat()
        directory_after = directory.lstat()
        entries_after = list(os.scandir(directory))
    except OSError as exc:
        raise FailureArtifactError(
            f"could not re-inspect {description}: {exc}"
        ) from exc
    if (
        stat.S_ISLNK(path_after.st_mode)
        or _stat_identity(path_after) != _stat_identity(entry_stat)
        or _stat_identity(directory_after) != _stat_identity(directory_before)
        or len(entries_after) != 1
        or entries_after[0].name != filename
    ):
        raise FailureArtifactError(f"{description} changed while being read")

    digest = directory.name
    if not _is_lower_sha256(digest):
        raise FailureArtifactError(
            f"{description} directory name must be its lowercase SHA-256"
        )
    if expected_sha256 is not None and digest != expected_sha256:
        raise FailureArtifactError(
            f"{description} SHA-256 does not match expected_sha256"
        )
    return canonical, digest


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > _MAX_JSON_BYTES:
            raise FailureArtifactError("failure artifact exceeds the size limit")
        chunks.append(chunk)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _decode_json(canonical: bytes, filename: str) -> Any:
    def duplicate_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FailureArtifactError(f"{filename} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise FailureArtifactError(
            f"{filename} contains non-finite JSON constant {value!r}"
        )

    try:
        return json.loads(
            canonical.decode("utf-8"),
            object_pairs_hook=duplicate_guard,
            parse_constant=reject_constant,
        )
    except FailureArtifactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FailureArtifactError(
            f"{filename} is not strict UTF-8 JSON: {exc}"
        ) from exc


def _freeze_from_metadata(value: Any) -> FailureEventFreezeManifest:
    root = _mapping(value, FAILURE_FREEZE_FILENAME)
    _exact_keys(
        root,
        {
            "schema_version",
            "reality_gate_tag",
            "task",
            "protocol",
            "bounds",
            "primary_placement_predicate_keys",
            "annotations",
            "video_audit_episode_ids",
            "implementation_commit",
        },
        FAILURE_FREEZE_FILENAME,
    )
    if _integer(root["schema_version"], "schema_version") != 1:
        raise FailureArtifactError("failure freeze schema_version is unsupported")
    if _string(root["reality_gate_tag"], "reality_gate_tag") != _REALITY_GATE_TAG:
        raise FailureArtifactError("failure freeze Reality Gate tag is unsupported")
    if root["protocol"] != failure_event_protocol_metadata():
        raise FailureArtifactError("failure freeze protocol metadata has drifted")

    task_value = _mapping(root["task"], "task")
    _exact_keys(
        task_value,
        {
            "suite",
            "task_id",
            "task_rank",
            "language",
            "primary_object",
            "planar_symmetry_order",
        },
        "task",
    )
    task = TaskIdentity(
        suite=_string(task_value["suite"], "task.suite"),
        task_id=_integer(task_value["task_id"], "task.task_id"),
        task_rank=_integer(task_value["task_rank"], "task.task_rank"),
        language=_string(task_value["language"], "task.language"),
        primary_object=_string(task_value["primary_object"], "task.primary_object"),
        planar_symmetry_order=_integer(
            task_value["planar_symmetry_order"], "task.planar_symmetry_order"
        ),
    )

    bounds_value = _mapping(root["bounds"], "bounds")
    _exact_keys(
        bounds_value,
        {
            "rule",
            "margin_m",
            "raw_bounds",
            "expanded_bounds",
            "expected_artifacts",
            "provenance",
        },
        "bounds",
    )
    if bounds_value["rule"] != (
        "all_valid_frame0_plus_valid_successful_frames1_to_terminal"
    ):
        raise FailureArtifactError("failure freeze bounds rule has drifted")
    expected_values = _list(bounds_value["expected_artifacts"], "expected_artifacts")
    provenance_values = _list(bounds_value["provenance"], "provenance")
    bounds = DiscoveryBoundsDerivation(
        raw_bounds=_bounds_from_metadata(bounds_value["raw_bounds"], "raw_bounds"),
        expanded_bounds=_bounds_from_metadata(
            bounds_value["expanded_bounds"], "expanded_bounds"
        ),
        margin_m=_number(bounds_value["margin_m"], "bounds.margin_m"),
        expected_artifacts=tuple(
            _identity_from_metadata(item, f"expected_artifacts[{index}]")
            for index, item in enumerate(expected_values)
        ),
        provenance=tuple(
            _provenance_from_metadata(item, f"provenance[{index}]")
            for index, item in enumerate(provenance_values)
        ),
    )

    annotations = tuple(
        _result_from_metadata(item, f"annotations[{index}]")
        for index, item in enumerate(_list(root["annotations"], "annotations"))
    )
    return FailureEventFreezeManifest(
        task=task,
        bounds=bounds,
        primary_placement_predicate_keys=_string_tuple(
            root["primary_placement_predicate_keys"],
            "primary_placement_predicate_keys",
        ),
        annotations=annotations,
        video_audit_episode_ids=_string_tuple(
            root["video_audit_episode_ids"], "video_audit_episode_ids"
        ),
        implementation_commit=_string(
            root["implementation_commit"], "implementation_commit"
        ),
        reality_gate_tag=_string(root["reality_gate_tag"], "reality_gate_tag"),
        schema_version=_integer(root["schema_version"], "schema_version"),
    )


def _event_artifact_from_metadata(value: Any) -> FailureEventArtifact:
    root = _mapping(value, FAILURE_EVENT_FILENAME)
    _exact_keys(
        root,
        {
            "schema_version",
            "artifact_kind",
            "reality_gate_tag",
            "freeze_sha256",
            "source_artifact",
            "annotation",
        },
        FAILURE_EVENT_FILENAME,
    )
    if _integer(root["schema_version"], "schema_version") != 1:
        raise FailureArtifactError(
            "failure-event artifact schema_version is unsupported"
        )
    if _string(root["artifact_kind"], "artifact_kind") != "failure-event-record":
        raise FailureArtifactError("failure-event artifact kind is unsupported")
    return FailureEventArtifact(
        source_artifact=_identity_from_metadata(
            root["source_artifact"], "source_artifact"
        ),
        annotation=_result_from_metadata(root["annotation"], "annotation"),
        freeze_sha256=_string(root["freeze_sha256"], "freeze_sha256"),
        reality_gate_tag=_string(root["reality_gate_tag"], "reality_gate_tag"),
        schema_version=_integer(root["schema_version"], "schema_version"),
    )


def _identity_from_metadata(value: Any, where: str) -> ArtifactIdentity:
    metadata = _mapping(value, where)
    _exact_keys(
        metadata,
        {"episode_id", "metadata_sha256", "trajectory_sha256"},
        where,
    )
    return ArtifactIdentity(
        episode_id=_string(metadata["episode_id"], f"{where}.episode_id"),
        metadata_sha256=_string(
            metadata["metadata_sha256"], f"{where}.metadata_sha256"
        ),
        trajectory_sha256=_string(
            metadata["trajectory_sha256"], f"{where}.trajectory_sha256"
        ),
    )


def _bounds_from_metadata(value: Any, where: str) -> ReachableBounds:
    metadata = _mapping(value, where)
    _exact_keys(metadata, {"lower_xyz", "upper_xyz"}, where)
    return ReachableBounds(
        lower_xyz=_fixed_float_tuple(
            metadata["lower_xyz"], f"{where}.lower_xyz", length=3
        ),
        upper_xyz=_fixed_float_tuple(
            metadata["upper_xyz"], f"{where}.upper_xyz", length=3
        ),
    )


def _provenance_from_metadata(value: Any, where: str) -> DiscoveryEpisodeProvenance:
    metadata = _mapping(value, where)
    _exact_keys(
        metadata,
        {
            "episode_id",
            "metadata_sha256",
            "trajectory_sha256",
            "valid_reset",
            "success",
            "validity_reasons",
            "frame_zero_included",
            "successful_path_included",
        },
        where,
    )
    identity = _identity_from_metadata(
        {
            key: metadata[key]
            for key in ("episode_id", "metadata_sha256", "trajectory_sha256")
        },
        f"{where}.artifact",
    )
    return DiscoveryEpisodeProvenance(
        artifact=identity,
        valid_reset=_boolean(metadata["valid_reset"], f"{where}.valid_reset"),
        success=_boolean(metadata["success"], f"{where}.success"),
        validity_reasons=_string_tuple(
            metadata["validity_reasons"], f"{where}.validity_reasons"
        ),
        frame_zero_included=_boolean(
            metadata["frame_zero_included"], f"{where}.frame_zero_included"
        ),
        successful_path_included=_boolean(
            metadata["successful_path_included"],
            f"{where}.successful_path_included",
        ),
    )


def _result_from_metadata(value: Any, where: str) -> FailureEventResult:
    metadata = _mapping(value, where)
    _exact_keys(
        metadata,
        {
            "episode_id",
            "valid_reset",
            "success",
            "status",
            "event_type",
            "onset_step",
            "confirmation_step",
        },
        where,
    )
    status_raw = _string(metadata["status"], f"{where}.status")
    try:
        status = AnnotationStatus(status_raw)
    except ValueError as exc:
        raise FailureArtifactError(f"{where}.status is unknown") from exc
    event_raw = metadata["event_type"]
    if event_raw is None:
        event_type = None
    else:
        try:
            event_type = FailureEventType(_string(event_raw, f"{where}.event_type"))
        except ValueError as exc:
            raise FailureArtifactError(f"{where}.event_type is unknown") from exc
    return FailureEventResult(
        episode_id=_string(metadata["episode_id"], f"{where}.episode_id"),
        valid_reset=_boolean(metadata["valid_reset"], f"{where}.valid_reset"),
        success=_boolean(metadata["success"], f"{where}.success"),
        status=status,
        event_type=event_type,
        onset_step=_optional_integer(metadata["onset_step"], f"{where}.onset_step"),
        confirmation_step=_optional_integer(
            metadata["confirmation_step"], f"{where}.confirmation_step"
        ),
    )


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FailureArtifactError(f"{where} must be a JSON object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        raise FailureArtifactError(
            f"{where} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise FailureArtifactError(f"{where} must be a JSON array")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise FailureArtifactError(f"{where} must be a string")
    return value


def _string_tuple(value: Any, where: str) -> tuple[str, ...]:
    return tuple(
        _string(item, f"{where}[{index}]")
        for index, item in enumerate(_list(value, where))
    )


def _integer(value: Any, where: str) -> int:
    if type(value) is not int:
        raise FailureArtifactError(f"{where} must be an integer")
    return value


def _optional_integer(value: Any, where: str) -> int | None:
    if value is None:
        return None
    return _integer(value, where)


def _boolean(value: Any, where: str) -> bool:
    if type(value) is not bool:
        raise FailureArtifactError(f"{where} must be boolean")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FailureArtifactError(f"{where} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise FailureArtifactError(f"{where} must be a finite number") from exc
    if not math.isfinite(result):
        raise FailureArtifactError(f"{where} must be a finite number")
    return result


def _fixed_float_tuple(value: Any, where: str, *, length: int) -> tuple[float, ...]:
    items = _list(value, where)
    if len(items) != length:
        raise FailureArtifactError(f"{where} must contain exactly {length} numbers")
    return tuple(_number(item, f"{where}[{index}]") for index, item in enumerate(items))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _normalized_absolute_path(path: str | Path) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    except (TypeError, ValueError, OSError) as exc:
        raise FailureArtifactError("artifact path is invalid") from exc


def _has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise FailureArtifactError(
                f"could not inspect path component {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode):
            return True
    return False


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "FAILURE_ARTIFACT_SCHEMA_VERSION",
    "FAILURE_EVENT_FILENAME",
    "FAILURE_FREEZE_FILENAME",
    "FailureArtifactError",
    "FailureEventArtifact",
    "create_failure_event_artifact",
    "create_failure_event_artifacts",
    "failure_event_freeze_from_metadata",
    "load_failure_event_artifact",
    "load_failure_freeze",
    "write_failure_event_artifact",
    "write_failure_freeze",
]
