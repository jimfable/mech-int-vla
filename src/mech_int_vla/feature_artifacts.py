"""Safe, canonical persistence for frozen feature-pipeline outputs.

Feature artifacts are two-file, content-addressed directories.  ``metadata.json``
contains the complete logical provenance and row metadata; ``arrays.npz`` contains
only explicitly typed NumPy arrays and is never loaded with pickle enabled.  The
directory name binds the canonical metadata bytes and deterministic NPZ bytes.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from .feature_pipeline import (
    CohortIdentity,
    EpisodeSourceHashes,
    FeatureCohort,
    FeatureReferenceBundle,
    FeatureStateRecord,
    TaskIdentity,
)
from .features import (
    ACTION_DIMENSION,
    COVERAGE_VECTOR_NAMES,
    ActionScale,
    CoverageState,
    FeatureHierarchy,
    NamedFeatureRow,
    ProbeNormState,
)

FEATURE_ARTIFACT_SCHEMA_VERSION: Final = 1
FEATURE_ARTIFACT_FORMAT: Final = "mech_int_vla_feature_artifact"
REFERENCE_KIND: Final = "feature_reference_bundle"
COHORT_KIND: Final = "feature_cohort"
_FILE_NAMES: Final = frozenset({"metadata.json", "arrays.npz"})
_REFERENCE_ARRAY_NAMES: Final = frozenset(
    {
        "action_scale_values",
        "action_scale_replaced_by_one",
        "coverage_vectors",
        "probe_norm_mean",
    }
)
_COHORT_ARRAY_NAMES: Final = frozenset({"m0_matrix", "m1_matrix", "m2_matrix"})
_MAX_FEATURE_STATES: Final = 160 * 104
_MAX_METADATA_BYTES: Final = 64 * 1024 * 1024
_MAX_ARCHIVE_BYTES: Final = 256 * 1024 * 1024
_MAX_NPY_HEADER_BYTES: Final = 4096


class FeatureArtifactError(ValueError):
    """Raised when a feature artifact is unsafe, malformed, or inconsistent."""


def _exact_keys(value: Any, expected: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FeatureArtifactError(f"{where} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise FeatureArtifactError(
            f"{where} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return value


def _json_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FeatureArtifactError(f"metadata.json has duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise FeatureArtifactError(f"metadata.json contains non-finite constant {value}")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FeatureArtifactError(
            "feature metadata is not canonical JSON data"
        ) from exc


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(nested) for nested in value]
    if isinstance(value, list):
        return [_plain_json(nested) for nested in value]
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, where: str) -> str:
    if not _is_sha256(value):
        raise FeatureArtifactError(f"{where} must be a lowercase SHA-256")
    return str(value)


def _logical_array_sha256(value: NDArray[Any]) -> str:
    array = np.array(value, copy=True, order="C")
    if array.dtype.kind == "f" and np.isnan(array).any():
        array[np.isnan(array)] = np.nan
    canonical = np.ascontiguousarray(
        array.astype(array.dtype.newbyteorder("<"), copy=False)
    )
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _normalize_float_nan_payloads(value: NDArray[Any]) -> NDArray[Any]:
    result = np.ascontiguousarray(value)
    if result.dtype.kind == "f" and np.isnan(result).any():
        result = np.array(result, copy=True, order="C")
        result[np.isnan(result)] = np.nan
    return result


def _canonical_array(value: NDArray[Any], dtype: str) -> NDArray[Any]:
    result = np.array(value, dtype=np.dtype(dtype), copy=True, order="C")
    return _normalize_float_nan_payloads(result)


def _array_spec(value: NDArray[Any]) -> dict[str, Any]:
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "logical_sha256": _logical_array_sha256(value),
    }


def _deterministic_npz_bytes(arrays: Mapping[str, NDArray[Any]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(arrays):
            npy = io.BytesIO()
            canonical = _normalize_float_nan_payloads(arrays[name])
            np.lib.format.write_array(npy, canonical, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(info, npy.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return buffer.getvalue()


def _directory_digest(metadata_bytes: bytes, array_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"mech-int-vla-feature-artifact-v1\0metadata\0")
    digest.update(metadata_bytes)
    digest.update(b"\0arrays\0")
    digest.update(array_bytes)
    return digest.hexdigest()


def _reference_record_metadata(
    coverage: CoverageState, probe_norm: ProbeNormState
) -> dict[str, Any]:
    return {
        "coverage": {
            "episode_id": coverage.episode_id,
            "base_init_state_id": coverage.base_init_state_id,
            "control_step": coverage.control_step,
            "split": coverage.split,
            "phase": coverage.phase,
            "success": coverage.success,
        },
        "probe_norm": {
            "episode_id": probe_norm.episode_id,
            "base_init_state_id": probe_norm.base_init_state_id,
            "control_step": probe_norm.control_step,
            "split": probe_norm.split,
            "phase": probe_norm.phase,
            "success": probe_norm.success,
        },
    }


def _cohort_record_metadata(record: FeatureStateRecord) -> dict[str, Any]:
    return {
        "episode_id": record.episode_id,
        "base_init_state_id": record.base_init_state_id,
        "split": record.split,
        "control_step": record.control_step,
        "terminal_failure_label": record.terminal_failure_label,
        "phase": record.phase,
        "source_hashes": record.source_hashes.to_metadata(),
        "hierarchy_metadata": {
            "M0": _plain_json(record.m0.metadata),
            "M1": _plain_json(record.m1.metadata),
            "M2": _plain_json(record.m2.metadata),
        },
    }


def _manifest(
    *,
    kind: str,
    provenance: Mapping[str, Any],
    records: list[dict[str, Any]],
    arrays: Mapping[str, NDArray[Any]],
) -> tuple[dict[str, Any], bytes, bytes, str]:
    array_bytes = _deterministic_npz_bytes(arrays)
    metadata = {
        "schema_version": FEATURE_ARTIFACT_SCHEMA_VERSION,
        "format": FEATURE_ARTIFACT_FORMAT,
        "kind": kind,
        "provenance": _plain_json(provenance),
        "records": records,
        "arrays": {
            "file": "arrays.npz",
            "sha256": hashlib.sha256(array_bytes).hexdigest(),
            "members": {name: _array_spec(arrays[name]) for name in sorted(arrays)},
        },
    }
    metadata_bytes = _canonical_json_bytes(metadata)
    digest = _directory_digest(metadata_bytes, array_bytes)
    return metadata, metadata_bytes, array_bytes, digest


def _reference_payload(
    bundle: FeatureReferenceBundle,
) -> tuple[dict[str, Any], bytes, bytes, str]:
    arrays = {
        "action_scale_values": _canonical_array(bundle.action_scale.values, "<f8"),
        "action_scale_replaced_by_one": _canonical_array(
            bundle.action_scale.replaced_by_one, "|b1"
        ),
        "coverage_vectors": _canonical_array(
            np.stack([state.vector for state in bundle.coverage_states]), "<f8"
        ),
        "probe_norm_mean": _canonical_array(
            np.asarray([state.mean_norm for state in bundle.probe_norm_states]),
            "<f8",
        ),
    }
    records = [
        _reference_record_metadata(coverage, probe_norm)
        for coverage, probe_norm in zip(
            bundle.coverage_states, bundle.probe_norm_states, strict=True
        )
    ]
    return _manifest(
        kind=REFERENCE_KIND,
        provenance=bundle.to_metadata(),
        records=records,
        arrays=arrays,
    )


def _cohort_payload(
    cohort: FeatureCohort,
) -> tuple[dict[str, Any], bytes, bytes, str]:
    arrays = {
        "m0_matrix": _canonical_array(cohort.m0_matrix, "<f8"),
        "m1_matrix": _canonical_array(cohort.m1_matrix, "<f8"),
        "m2_matrix": _canonical_array(cohort.m2_matrix, "<f8"),
    }
    return _manifest(
        kind=COHORT_KIND,
        provenance=cohort.to_metadata(),
        records=[_cohort_record_metadata(record) for record in cohort.records],
        arrays=arrays,
    )


def _normalized_absolute_path(value: str | Path) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    except (TypeError, ValueError, OSError) as exc:
        raise FeatureArtifactError("artifact path is invalid") from exc


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if not current.exists() and not current.is_symlink():
            break
        if current.is_symlink():
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


def _publish(
    output_root: str | Path,
    *,
    metadata_bytes: bytes,
    array_bytes: bytes,
    digest: str,
) -> Path:
    root = _normalized_absolute_path(output_root)
    if any(part in {"configs", "locks"} for part in root.parts):
        raise FeatureArtifactError(
            "feature artifacts may not be written into config or lock paths"
        )
    if _has_symlink_component(root):
        raise FeatureArtifactError("output path may not contain a symlink component")
    if root.exists() and not root.is_dir():
        raise FeatureArtifactError("output_root must be a directory")
    root.mkdir(parents=True, exist_ok=True)
    if _has_symlink_component(root):
        raise FeatureArtifactError("output path may not contain a symlink component")
    destination = root / digest
    if _lexists(destination):
        raise FileExistsError(f"refusing to overwrite feature artifact {destination}")

    lock_path = root / f".{digest}.publish.lock"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        lock_descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"another writer is publishing feature artifact {destination}"
        ) from exc
    os.close(lock_descriptor)

    staging: Path | None = None
    try:
        _fsync_directory(root)
        staging = Path(tempfile.mkdtemp(prefix=f".{digest}.tmp-", dir=root))
        for name, content in (
            ("arrays.npz", array_bytes),
            ("metadata.json", metadata_bytes),
        ):
            with (staging / name).open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        _fsync_directory(staging)
        if _lexists(destination):
            raise FileExistsError(
                f"refusing to overwrite feature artifact {destination}"
            )
        os.rename(staging, destination)
        staging = None
        _fsync_directory(root)
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        lock_path.unlink(missing_ok=True)
        _fsync_directory(root)
    return destination


def write_feature_reference_bundle(
    bundle: FeatureReferenceBundle, output_root: str | Path
) -> Path:
    """Atomically publish a frozen Calibration feature reference bundle."""

    if not isinstance(bundle, FeatureReferenceBundle):
        raise FeatureArtifactError("bundle must be a FeatureReferenceBundle")
    _, metadata_bytes, array_bytes, digest = _reference_payload(bundle)
    return _publish(
        output_root,
        metadata_bytes=metadata_bytes,
        array_bytes=array_bytes,
        digest=digest,
    )


def write_feature_cohort(cohort: FeatureCohort, output_root: str | Path) -> Path:
    """Atomically publish a Calibration or Locked Test feature cohort."""

    if not isinstance(cohort, FeatureCohort):
        raise FeatureArtifactError("cohort must be a FeatureCohort")
    _, metadata_bytes, array_bytes, digest = _cohort_payload(cohort)
    return _publish(
        output_root,
        metadata_bytes=metadata_bytes,
        array_bytes=array_bytes,
        digest=digest,
    )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    try:
        before_path = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise FeatureArtifactError(f"cannot inspect {path.name}: {exc}") from exc
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise FeatureArtifactError(f"{path.name} must be a regular file, not a symlink")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            before_open = os.fstat(descriptor)
            if not stat.S_ISREG(before_open.st_mode) or (
                before_open.st_dev,
                before_open.st_ino,
            ) != (before_path.st_dev, before_path.st_ino):
                raise FeatureArtifactError(f"{path.name} changed while opening")
            if before_open.st_size > max_bytes:
                raise FeatureArtifactError(f"{path.name} exceeds the size limit")
            chunks = []
            total = 0
            while chunk := os.read(descriptor, 1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise FeatureArtifactError(f"{path.name} exceeds the size limit")
                chunks.append(chunk)
            after_open = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise FeatureArtifactError(f"could not read {path.name} safely: {exc}") from exc
    if _stat_identity(before_open) != _stat_identity(after_open) or _stat_identity(
        before_path
    ) != _stat_identity(after_path):
        raise FeatureArtifactError(f"{path.name} changed while it was being read")
    return b"".join(chunks)


def _read_directory(
    path: str | Path, expected_sha256: str | None
) -> tuple[Path, Mapping[str, Any], bytes]:
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, "expected_sha256")
    directory = _normalized_absolute_path(path)
    if _has_symlink_component(directory):
        raise FeatureArtifactError(
            "feature artifact path may not contain a symlink component"
        )
    if not directory.is_dir():
        raise FeatureArtifactError("feature artifact path must be a directory")
    try:
        entries = list(os.scandir(directory))
    except OSError as exc:
        raise FeatureArtifactError(f"cannot inspect feature artifact: {exc}") from exc
    names = {entry.name for entry in entries}
    if len(entries) != 2 or names != _FILE_NAMES:
        raise FeatureArtifactError(
            "feature artifact must contain exactly metadata.json and arrays.npz"
        )
    if any(entry.is_symlink() for entry in entries):
        raise FeatureArtifactError("feature artifact files may not be symlinks")
    metadata_bytes = _read_regular_file(
        directory / "metadata.json", max_bytes=_MAX_METADATA_BYTES
    )
    array_bytes = _read_regular_file(
        directory / "arrays.npz", max_bytes=_MAX_ARCHIVE_BYTES
    )
    try:
        metadata = json.loads(
            metadata_bytes.decode("utf-8"),
            object_pairs_hook=_json_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeatureArtifactError(
            f"metadata.json is not strict UTF-8 JSON: {exc}"
        ) from exc
    if not isinstance(metadata, Mapping):
        raise FeatureArtifactError("metadata.json top level must be an object")
    if _canonical_json_bytes(metadata) != metadata_bytes:
        raise FeatureArtifactError("metadata.json is not the exact canonical encoding")
    digest = _directory_digest(metadata_bytes, array_bytes)
    if directory.name != digest:
        raise FeatureArtifactError(
            "feature artifact directory does not match its SHA-256 digest"
        )
    if expected_sha256 is not None and digest != expected_sha256:
        raise FeatureArtifactError(
            "feature artifact SHA-256 does not match expected_sha256"
        )
    return directory, metadata, array_bytes


def _load_arrays(
    metadata: Mapping[str, Any],
    array_bytes: bytes,
    expected_names: frozenset[str],
    *,
    expected_shapes: Mapping[str, tuple[int, ...]],
    expected_dtypes: Mapping[str, str],
) -> dict[str, NDArray[Any]]:
    top = _exact_keys(
        metadata,
        {"schema_version", "format", "kind", "provenance", "records", "arrays"},
        "metadata",
    )
    if (
        isinstance(top["schema_version"], bool)
        or not isinstance(top["schema_version"], int)
        or top["schema_version"] != FEATURE_ARTIFACT_SCHEMA_VERSION
    ):
        raise FeatureArtifactError("unsupported feature artifact schema_version")
    if top["format"] != FEATURE_ARTIFACT_FORMAT:
        raise FeatureArtifactError("feature artifact format is invalid")
    arrays_metadata = _exact_keys(
        top["arrays"], {"file", "sha256", "members"}, "arrays"
    )
    if arrays_metadata["file"] != "arrays.npz":
        raise FeatureArtifactError("arrays.file must be arrays.npz")
    expected_physical = _require_sha256(arrays_metadata["sha256"], "arrays.sha256")
    if hashlib.sha256(array_bytes).hexdigest() != expected_physical:
        raise FeatureArtifactError("arrays.npz SHA-256 does not match metadata")
    members = arrays_metadata["members"]
    if not isinstance(members, Mapping) or set(members) != expected_names:
        raise FeatureArtifactError("logical array member set differs from schema")
    if set(expected_shapes) != expected_names or set(expected_dtypes) != expected_names:
        raise FeatureArtifactError("internal expected array schema is incomplete")
    for name in sorted(expected_names):
        spec = _exact_keys(
            members[name], {"dtype", "shape", "logical_sha256"}, f"arrays.{name}"
        )
        if spec["dtype"] != expected_dtypes[name]:
            raise FeatureArtifactError(f"array {name} dtype differs from schema")
        shape = spec["shape"]
        if not isinstance(shape, list) or tuple(shape) != expected_shapes[name]:
            raise FeatureArtifactError(f"array {name} shape differs from schema")
        _require_sha256(spec["logical_sha256"], f"arrays.{name}.logical_sha256")
    try:
        with zipfile.ZipFile(io.BytesIO(array_bytes), mode="r") as archive:
            archive_names = archive.namelist()
            archive_info = {item.filename: item for item in archive.infolist()}
    except (OSError, zipfile.BadZipFile) as exc:
        raise FeatureArtifactError("arrays.npz is not a valid ZIP archive") from exc
    expected_archive_names = [f"{name}.npy" for name in sorted(expected_names)]
    if archive_names != expected_archive_names or len(set(archive_names)) != len(
        archive_names
    ):
        raise FeatureArtifactError(
            "arrays.npz member names or order differ from schema"
        )
    for name in sorted(expected_names):
        dtype = np.dtype(expected_dtypes[name])
        element_count = math.prod(expected_shapes[name])
        maximum_size = element_count * dtype.itemsize + _MAX_NPY_HEADER_BYTES
        if archive_info[f"{name}.npy"].file_size > maximum_size:
            raise FeatureArtifactError(
                f"array {name} compressed member exceeds its schema size"
            )
    try:
        with np.load(io.BytesIO(array_bytes), allow_pickle=False) as loaded:
            if set(loaded.files) != expected_names:
                raise FeatureArtifactError("arrays.npz array set differs from schema")
            arrays = {
                name: np.array(loaded[name], copy=True, order="C")
                for name in loaded.files
            }
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise FeatureArtifactError(
            "arrays.npz could not be loaded without pickle"
        ) from exc
    if _deterministic_npz_bytes(arrays) != array_bytes:
        raise FeatureArtifactError("arrays.npz is not the exact deterministic encoding")
    for name in sorted(expected_names):
        spec = members[name]
        array = arrays[name]
        if spec["dtype"] != array.dtype.str:
            raise FeatureArtifactError(f"array {name} dtype differs from metadata")
        if tuple(spec["shape"]) != array.shape:
            raise FeatureArtifactError(f"array {name} shape differs from metadata")
        logical = _require_sha256(
            spec["logical_sha256"], f"arrays.{name}.logical_sha256"
        )
        if _logical_array_sha256(array) != logical:
            raise FeatureArtifactError(f"array {name} logical SHA-256 differs")
        array.setflags(write=False)
    return arrays


def _task_identity(value: Any) -> TaskIdentity:
    item = _exact_keys(
        value,
        {"rank", "suite", "task_id", "language", "primary_object", "symmetry_order"},
        "task_identity",
    )
    try:
        return TaskIdentity(**item)
    except (TypeError, ValueError) as exc:
        raise FeatureArtifactError(f"invalid task_identity: {exc}") from exc


def _cohort_identity(value: Any) -> CohortIdentity:
    item = _exact_keys(
        value,
        {
            "policy_revision",
            "base_vlm_revision",
            "code_commit",
            "config_sha256",
            "code_sha256",
        },
        "cohort_identity",
    )
    try:
        return CohortIdentity(**item)
    except (TypeError, ValueError) as exc:
        raise FeatureArtifactError(f"invalid cohort_identity: {exc}") from exc


def _source_hashes(value: Any, where: str) -> EpisodeSourceHashes:
    item = _exact_keys(
        value,
        {
            "episode_id",
            "raw_metadata_sha256",
            "raw_trajectory_sha256",
            "score_metadata_sha256",
            "score_primitives_sha256",
        },
        where,
    )
    try:
        return EpisodeSourceHashes(**item)
    except (TypeError, ValueError) as exc:
        raise FeatureArtifactError(f"invalid {where}: {exc}") from exc


def _provenance_mapping(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = metadata["provenance"]
    if not isinstance(provenance, Mapping):
        raise FeatureArtifactError("provenance must be a JSON object")
    return provenance


def _records_sequence(metadata: Mapping[str, Any]) -> Sequence[Any]:
    records = metadata["records"]
    if not isinstance(records, list):
        raise FeatureArtifactError("records must be a JSON array")
    return records


def load_feature_reference_bundle(
    path: str | Path, expected_sha256: str | None = None
) -> FeatureReferenceBundle:
    """Load and fully validate a frozen reference bundle without pickle."""

    _, metadata, array_bytes = _read_directory(path, expected_sha256)
    if metadata.get("kind") != REFERENCE_KIND:
        raise FeatureArtifactError("feature artifact kind is not a reference bundle")
    provenance = _exact_keys(
        _provenance_mapping(metadata),
        {
            "schema_version",
            "kind",
            "probe_sha256",
            "selected_candidate",
            "task_identity",
            "cohort_identity",
            "action_scale",
            "reference_state_count",
            "reference_state_sha256",
            "sources",
        },
        "reference provenance",
    )
    records = _records_sequence(metadata)
    count = provenance["reference_state_count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 1 <= count <= _MAX_FEATURE_STATES
    ):
        raise FeatureArtifactError("reference_state_count must be a positive integer")
    if len(records) != count:
        raise FeatureArtifactError("reference record count differs from provenance")
    expected_shapes = {
        "action_scale_values": (ACTION_DIMENSION,),
        "action_scale_replaced_by_one": (ACTION_DIMENSION,),
        "coverage_vectors": (count, len(COVERAGE_VECTOR_NAMES)),
        "probe_norm_mean": (count,),
    }
    expected_dtypes = {
        "action_scale_values": "<f8",
        "action_scale_replaced_by_one": "|b1",
        "coverage_vectors": "<f8",
        "probe_norm_mean": "<f8",
    }
    arrays = _load_arrays(
        metadata,
        array_bytes,
        _REFERENCE_ARRAY_NAMES,
        expected_shapes=expected_shapes,
        expected_dtypes=expected_dtypes,
    )
    for name in _REFERENCE_ARRAY_NAMES:
        if arrays[name].shape != expected_shapes[name]:
            raise FeatureArtifactError(f"array {name} has an invalid logical shape")
        if arrays[name].dtype.str != expected_dtypes[name]:
            raise FeatureArtifactError(f"array {name} has an invalid logical dtype")

    coverage_states = []
    probe_norm_states = []
    state_keys = {
        "episode_id",
        "base_init_state_id",
        "control_step",
        "split",
        "phase",
        "success",
    }
    for index, record in enumerate(records):
        pair = _exact_keys(record, {"coverage", "probe_norm"}, f"records[{index}]")
        coverage = _exact_keys(
            pair["coverage"], state_keys, f"records[{index}].coverage"
        )
        probe_norm = _exact_keys(
            pair["probe_norm"], state_keys, f"records[{index}].probe_norm"
        )
        try:
            coverage_states.append(
                CoverageState(**coverage, vector=arrays["coverage_vectors"][index])
            )
            probe_norm_states.append(
                ProbeNormState(
                    **probe_norm,
                    mean_norm=float(arrays["probe_norm_mean"][index]),
                )
            )
        except (TypeError, ValueError) as exc:
            raise FeatureArtifactError(
                f"invalid reference record {index}: {exc}"
            ) from exc
    action_metadata = _exact_keys(
        provenance["action_scale"],
        {
            "schema_version",
            "definition",
            "source",
            "floor",
            "values",
            "replaced_by_one",
            "episode_ids",
        },
        "action_scale",
    )
    sources_value = provenance["sources"]
    if not isinstance(sources_value, list):
        raise FeatureArtifactError("sources must be a JSON array")
    try:
        action_scale = ActionScale(
            arrays["action_scale_values"],
            arrays["action_scale_replaced_by_one"],
            tuple(action_metadata["episode_ids"]),
        )
        bundle = FeatureReferenceBundle(
            action_scale=action_scale,
            coverage_states=tuple(coverage_states),
            probe_norm_states=tuple(probe_norm_states),
            probe_sha256=provenance["probe_sha256"],
            selected_candidate=provenance["selected_candidate"],
            task_identity=_task_identity(provenance["task_identity"]),
            cohort_identity=_cohort_identity(provenance["cohort_identity"]),
            source_hashes=tuple(
                _source_hashes(value, f"sources[{index}]")
                for index, value in enumerate(sources_value)
            ),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, FeatureArtifactError):
            raise
        raise FeatureArtifactError(f"invalid reference bundle: {exc}") from exc
    if _canonical_json_bytes(bundle.to_metadata()) != _canonical_json_bytes(provenance):
        raise FeatureArtifactError(
            "reconstructed reference provenance differs from metadata"
        )
    _, reconstructed_metadata, reconstructed_arrays, _ = _reference_payload(bundle)
    if reconstructed_metadata != _canonical_json_bytes(metadata) or (
        reconstructed_arrays != array_bytes
    ):
        raise FeatureArtifactError(
            "reference artifact differs from its canonical reconstruction"
        )
    return bundle


def load_feature_cohort(
    path: str | Path, expected_sha256: str | None = None
) -> FeatureCohort:
    """Load and fully validate a feature cohort and hierarchy metadata."""

    _, metadata, array_bytes = _read_directory(path, expected_sha256)
    if metadata.get("kind") != COHORT_KIND:
        raise FeatureArtifactError("feature artifact kind is not a feature cohort")
    provenance = _exact_keys(
        _provenance_mapping(metadata),
        {
            "schema_version",
            "split",
            "probe_sha256",
            "reference_bundle_sha256",
            "task_identity",
            "cohort_identity",
            "rows",
            "columns",
            "matrix_sha256",
        },
        "cohort provenance",
    )
    provenance_rows = provenance["rows"]
    records_metadata = _records_sequence(metadata)
    if not isinstance(provenance_rows, list) or len(provenance_rows) != len(
        records_metadata
    ):
        raise FeatureArtifactError("cohort row records are missing or misaligned")
    if not records_metadata:
        raise FeatureArtifactError("cohort records cannot be empty")
    columns = _exact_keys(provenance["columns"], {"M0", "M1", "M2"}, "columns")
    for name in ("M0", "M1", "M2"):
        if not isinstance(columns[name], list) or any(
            not isinstance(column, str) for column in columns[name]
        ):
            raise FeatureArtifactError(f"columns.{name} must be a string array")
    row_count = len(records_metadata)
    if row_count > _MAX_FEATURE_STATES:
        raise FeatureArtifactError("cohort row count exceeds the frozen maximum")
    expected_shapes = {
        "m0_matrix": (row_count, len(columns["M0"])),
        "m1_matrix": (row_count, len(columns["M1"])),
        "m2_matrix": (row_count, len(columns["M2"])),
    }
    expected_dtypes = {name: "<f8" for name in _COHORT_ARRAY_NAMES}
    arrays = _load_arrays(
        metadata,
        array_bytes,
        _COHORT_ARRAY_NAMES,
        expected_shapes=expected_shapes,
        expected_dtypes=expected_dtypes,
    )
    for array_name, column_name in (
        ("m0_matrix", "M0"),
        ("m1_matrix", "M1"),
        ("m2_matrix", "M2"),
    ):
        if arrays[array_name].dtype.str != "<f8":
            raise FeatureArtifactError(
                f"array {array_name} has an invalid logical dtype"
            )
        if arrays[array_name].shape != (row_count, len(columns[column_name])):
            raise FeatureArtifactError(
                f"array {array_name} has an invalid logical shape"
            )
    matrix_hashes = _exact_keys(
        provenance["matrix_sha256"], {"M0", "M1", "M2"}, "matrix_sha256"
    )
    for array_name, level in (
        ("m0_matrix", "M0"),
        ("m1_matrix", "M1"),
        ("m2_matrix", "M2"),
    ):
        if _logical_array_sha256(arrays[array_name]) != _require_sha256(
            matrix_hashes[level], f"matrix_sha256.{level}"
        ):
            raise FeatureArtifactError(
                f"array {array_name} differs from cohort provenance"
            )

    row_keys = {
        "episode_id",
        "base_init_state_id",
        "split",
        "control_step",
        "terminal_failure_label",
        "phase",
        "source_hashes",
        "hierarchy_metadata",
    }
    provenance_row_keys = {
        "episode_id",
        "base_init_state_id",
        "control_step",
        "terminal_failure_label",
        "phase",
        "source_hashes",
    }
    records = []
    for index, (record_value, provenance_value) in enumerate(
        zip(records_metadata, provenance_rows, strict=True)
    ):
        record = _exact_keys(record_value, row_keys, f"records[{index}]")
        provenance_row = _exact_keys(
            provenance_value, provenance_row_keys, f"provenance.rows[{index}]"
        )
        if {
            key: record[key] for key in provenance_row_keys
        } != provenance_row or record["split"] != provenance["split"]:
            raise FeatureArtifactError(
                f"records[{index}] differs from cohort provenance"
            )
        hierarchy_metadata = _exact_keys(
            record["hierarchy_metadata"],
            {"M0", "M1", "M2"},
            f"records[{index}].hierarchy_metadata",
        )
        if any(
            not isinstance(hierarchy_metadata[level], Mapping)
            for level in hierarchy_metadata
        ):
            raise FeatureArtifactError("hierarchy metadata levels must be JSON objects")
        try:
            hierarchy = FeatureHierarchy(
                NamedFeatureRow(
                    tuple(columns["M0"]),
                    arrays["m0_matrix"][index],
                    hierarchy_metadata["M0"],
                ),
                NamedFeatureRow(
                    tuple(columns["M1"]),
                    arrays["m1_matrix"][index],
                    hierarchy_metadata["M1"],
                ),
                NamedFeatureRow(
                    tuple(columns["M2"]),
                    arrays["m2_matrix"][index],
                    hierarchy_metadata["M2"],
                ),
            )
            records.append(
                FeatureStateRecord(
                    episode_id=record["episode_id"],
                    base_init_state_id=record["base_init_state_id"],
                    split=record["split"],
                    control_step=record["control_step"],
                    terminal_failure_label=record["terminal_failure_label"],
                    phase=record["phase"],
                    hierarchy=hierarchy,
                    source_hashes=_source_hashes(
                        record["source_hashes"], f"records[{index}].source_hashes"
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, FeatureArtifactError):
                raise
            raise FeatureArtifactError(
                f"invalid feature record {index}: {exc}"
            ) from exc
    try:
        cohort = FeatureCohort(
            split=provenance["split"],
            records=tuple(records),
            m0_names=tuple(columns["M0"]),
            m1_names=tuple(columns["M1"]),
            m2_names=tuple(columns["M2"]),
            m0_matrix=arrays["m0_matrix"],
            m1_matrix=arrays["m1_matrix"],
            m2_matrix=arrays["m2_matrix"],
            probe_sha256=provenance["probe_sha256"],
            reference_bundle_sha256=provenance["reference_bundle_sha256"],
            task_identity=_task_identity(provenance["task_identity"]),
            cohort_identity=_cohort_identity(provenance["cohort_identity"]),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, FeatureArtifactError):
            raise
        raise FeatureArtifactError(f"invalid feature cohort: {exc}") from exc
    if _canonical_json_bytes(cohort.to_metadata()) != _canonical_json_bytes(provenance):
        raise FeatureArtifactError(
            "reconstructed cohort provenance differs from metadata"
        )
    _, reconstructed_metadata, reconstructed_arrays, _ = _cohort_payload(cohort)
    if reconstructed_metadata != _canonical_json_bytes(metadata) or (
        reconstructed_arrays != array_bytes
    ):
        raise FeatureArtifactError(
            "feature cohort differs from its canonical reconstruction"
        )
    return cohort


__all__ = [
    "COHORT_KIND",
    "FEATURE_ARTIFACT_FORMAT",
    "FEATURE_ARTIFACT_SCHEMA_VERSION",
    "REFERENCE_KIND",
    "FeatureArtifactError",
    "load_feature_cohort",
    "load_feature_reference_bundle",
    "write_feature_cohort",
    "write_feature_reference_bundle",
]
