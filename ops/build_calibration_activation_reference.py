#!/usr/bin/env python3
"""Build the frozen natural-activation reference from Calibration only.

The selected representation is fixed by the already-frozen BoundProbe. Each
natural vector is the float64 arithmetic mean of the eight factual/original
activation draws stored in an existing validated Calibration rescore sidecar.
The sidecars' raw-file hashes are checked against the frozen Calibration
feature reference, so this builder needs no labels, refit, or local raw bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
import time
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np

# Make the standalone ops CLI work from any current working directory without
# requiring callers to arrange PYTHONPATH.
REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
SOURCE_ROOT: Final = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SOURCE_ROOT))

from mech_int_vla.allocation import ScoreAllocationReceipt, ScoringArtifactIdentity
from mech_int_vla.artifacts import ACTIVATION_SCORE_STRIDE_STEPS
from mech_int_vla.causal import five_nearest_neighbor_distance
from mech_int_vla.config import SplitName, load_protocol_config
from mech_int_vla.feature_artifacts import load_feature_reference_bundle
from mech_int_vla.probe_artifacts import load_bound_probe_artifact
from mech_int_vla.provenance import frozen_config_sha256, scoring_source_sha256
from mech_int_vla.scoring import ORIGINAL_DRAWS, load_scoring_sidecar

SCHEMA_VERSION: Final = 1
ARTIFACT_FORMAT: Final = "mech_int_vla_calibration_activation_reference"
ARTIFACT_KIND: Final = "calibration_activation_reference"
ROW_ARRAY_NAMES: Final = frozenset(
    {"episode_index", "base_init_state_id", "control_step"}
)
GEOMETRY_ARRAY_NAMES: Final = frozenset(
    {"natural_query_index", "natural_five_nn_distance"}
)
ARRAY_NAMES: Final = frozenset(
    {"activation_vectors", *ROW_ARRAY_NAMES, *GEOMETRY_ARRAY_NAMES}
)
METADATA_FILENAME: Final = "metadata.json"
ARRAYS_FILENAME: Final = "arrays.npz"
EXPECTED_EPISODES: Final = 160
MAX_METADATA_BYTES: Final = 64 * 1024 * 1024
MAX_ARRAY_BYTES: Final = 512 * 1024 * 1024
NATURAL_QUERY_SEED: Final = 260803
NATURAL_QUERY_COUNT: Final = 3000
NATURAL_NEIGHBORS: Final = 5
NATURAL_PERCENTILE: Final = 95.0
FROZEN_NATURAL_95TH_PERCENTILE: Final = 3.890758912438606


class ActivationReferenceError(ValueError):
    """Raised when a source or published activation reference is inconsistent."""


@dataclass(frozen=True)
class LoadedActivationReference:
    path: Path
    metadata: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    sha256: str


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(nested) for nested in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ActivationReferenceError("metadata is not canonical JSON data") from exc


def _strict_json_bytes(payload: bytes, where: Path | str) -> dict[str, Any]:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ActivationReferenceError(f"duplicate JSON key in {where}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ActivationReferenceError(
                    f"non-finite JSON constant in {where}: {value}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActivationReferenceError(f"invalid JSON in {where}") from exc
    if not isinstance(value, dict):
        raise ActivationReferenceError(f"{where} must contain a JSON object")
    return value


def _strict_json(path: Path) -> dict[str, Any]:
    return _strict_json_bytes(path.read_bytes(), path)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    payload = _read_regular(path, MAX_ARRAY_BYTES)
    digest.update(payload)
    return digest.hexdigest()


def _require_sha256(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ActivationReferenceError(f"{where} must be a lowercase SHA-256")
    return value


def _exact_keys(value: Any, expected: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActivationReferenceError(f"{where} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise ActivationReferenceError(
            f"{where} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return value


def _logical_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    dtype = array.dtype.str.encode("ascii")
    digest.update(len(dtype).to_bytes(8, "big"))
    digest.update(dtype)
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_spec(value: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "logical_sha256": _logical_array_sha256(value),
    }


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(arrays):
            npy = io.BytesIO()
            np.lib.format.write_array(npy, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(info, npy.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return buffer.getvalue()


def _canonical_arrays(values: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if set(values) != ARRAY_NAMES:
        raise ActivationReferenceError("activation reference array names differ")
    arrays = {
        "activation_vectors": np.ascontiguousarray(
            values["activation_vectors"], dtype=np.dtype("<f8")
        ),
        "episode_index": np.ascontiguousarray(
            values["episode_index"], dtype=np.dtype("<i2")
        ),
        "base_init_state_id": np.ascontiguousarray(
            values["base_init_state_id"], dtype=np.dtype("<i4")
        ),
        "control_step": np.ascontiguousarray(
            values["control_step"], dtype=np.dtype("<i4")
        ),
        "natural_query_index": np.ascontiguousarray(
            values["natural_query_index"], dtype=np.dtype("<i4")
        ),
        "natural_five_nn_distance": np.ascontiguousarray(
            values["natural_five_nn_distance"], dtype=np.dtype("<f8")
        ),
    }
    vectors = arrays["activation_vectors"]
    if vectors.ndim != 2 or vectors.shape[0] < 1 or vectors.shape[1] < 1:
        raise ActivationReferenceError("activation vectors must be a nonempty matrix")
    rows = vectors.shape[0]
    if any(
        arrays[name].shape != (rows,) for name in ROW_ARRAY_NAMES
    ):
        raise ActivationReferenceError(
            "row-identity arrays must align with activation rows"
        )
    if not np.isfinite(vectors).all():
        raise ActivationReferenceError("activation vectors must be finite")
    if np.any(arrays["episode_index"] < 0) or np.any(arrays["base_init_state_id"] < 0):
        raise ActivationReferenceError("activation row identities must be nonnegative")
    if np.any(arrays["control_step"] < 0) or np.any(
        arrays["control_step"] % ACTIVATION_SCORE_STRIDE_STEPS != 0
    ):
        raise ActivationReferenceError(
            "activation control steps are off frozen cadence"
        )
    query_index = arrays["natural_query_index"]
    distances = arrays["natural_five_nn_distance"]
    if (
        query_index.shape != (NATURAL_QUERY_COUNT,)
        or distances.shape != (NATURAL_QUERY_COUNT,)
        or np.any(query_index < 0)
        or np.any(query_index >= rows)
        or len(np.unique(query_index)) != NATURAL_QUERY_COUNT
        or not np.isfinite(distances).all()
        or np.any(distances < 0.0)
    ):
        raise ActivationReferenceError("natural geometry arrays are invalid")
    expected_query = np.random.Generator(np.random.PCG64(NATURAL_QUERY_SEED)).choice(
        rows, size=NATURAL_QUERY_COUNT, replace=False
    )
    if not np.array_equal(query_index, expected_query):
        raise ActivationReferenceError("natural geometry query indices differ")
    for array in arrays.values():
        array.setflags(write=False)
    return arrays


def _validate_metadata(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> None:
    root = _exact_keys(
        metadata,
        {
            "schema_version",
            "format",
            "kind",
            "split",
            "selection",
            "geometry",
            "counts",
            "source",
            "episodes",
            "arrays",
            "files",
        },
        "metadata",
    )
    if (
        root["schema_version"] != SCHEMA_VERSION
        or root["format"] != ARTIFACT_FORMAT
        or root["kind"] != ARTIFACT_KIND
        or root["split"] != SplitName.CALIBRATION.value
    ):
        raise ActivationReferenceError("activation reference identity differs")
    selection = _exact_keys(
        root["selection"],
        {
            "source",
            "selected_candidate",
            "cadence_control_steps",
            "outcome_conditioned",
            "labels_used",
            "refit_performed",
            "transformation",
            "natural_draws",
        },
        "selection",
    )
    if (
        selection["source"] != "rescore_factual_original_selected_activation"
        or not isinstance(selection["selected_candidate"], str)
        or not selection["selected_candidate"]
        or selection["cadence_control_steps"] != ACTIVATION_SCORE_STRIDE_STEPS
        or selection["outcome_conditioned"] is not False
        or selection["labels_used"] is not False
        or selection["refit_performed"] is not False
        or selection["transformation"]
        != "float64_arithmetic_mean_over_all_frozen_original_draws"
        or selection["natural_draws"] != ORIGINAL_DRAWS
    ):
        raise ActivationReferenceError("activation selection protocol differs")
    geometry = _exact_keys(
        root["geometry"],
        {
            "method",
            "query_seed",
            "query_count",
            "percentile",
            "natural_95th_percentile",
        },
        "geometry",
    )
    reconstructed_percentile = float(
        np.percentile(arrays["natural_five_nn_distance"], NATURAL_PERCENTILE)
    )
    if geometry != {
        "method": "mean_euclidean_distance_to_5_nearest_leave_self_out",
        "query_seed": NATURAL_QUERY_SEED,
        "query_count": NATURAL_QUERY_COUNT,
        "percentile": NATURAL_PERCENTILE,
        "natural_95th_percentile": FROZEN_NATURAL_95TH_PERCENTILE,
    } or reconstructed_percentile != geometry["natural_95th_percentile"]:
        raise ActivationReferenceError("natural geometry protocol differs")
    counts = _exact_keys(root["counts"], {"episodes", "rows", "width"}, "counts")
    rows, width = arrays["activation_vectors"].shape
    if counts != {"episodes": len(root["episodes"]), "rows": rows, "width": width}:
        raise ActivationReferenceError("activation counts differ from arrays")
    source = _exact_keys(
        root["source"],
        {
            "predecessor_calibration_freeze_sha256",
            "manifest_sha256",
            "rollout_allocation_sha256",
            "score_allocation_sha256",
            "bound_probe_sha256",
            "feature_reference_sha256",
            "feature_reference_metadata_sha256",
            "feature_reference_arrays_sha256",
            "config_sha256",
            "code_sha256",
            "raw_bytes_reverified_locally",
            "raw_hash_links_verified_against_frozen_cohort",
        },
        "source",
    )
    for name, value in source.items():
        if name in {
            "raw_bytes_reverified_locally",
            "raw_hash_links_verified_against_frozen_cohort",
        }:
            continue
        _require_sha256(value, f"source.{name}")
    if (
        source["raw_bytes_reverified_locally"] is not False
        or source["raw_hash_links_verified_against_frozen_cohort"] is not True
    ):
        raise ActivationReferenceError("raw-source verification mode differs")
    array_metadata = _exact_keys(root["arrays"], set(ARRAY_NAMES), "arrays")
    for name in ARRAY_NAMES:
        spec = _exact_keys(
            array_metadata[name], {"dtype", "shape", "logical_sha256"}, f"arrays.{name}"
        )
        if spec != _array_spec(arrays[name]):
            raise ActivationReferenceError(f"array metadata differs for {name}")
    files = _exact_keys(root["files"], {"arrays", "arrays_sha256"}, "files")
    if files["arrays"] != ARRAYS_FILENAME:
        raise ActivationReferenceError("arrays filename differs")
    _require_sha256(files["arrays_sha256"], "files.arrays_sha256")
    episodes = root["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise ActivationReferenceError("episodes must be a nonempty JSON array")
    expected_start = 0
    observed_ids: list[str] = []
    for index, value in enumerate(episodes):
        episode = _exact_keys(
            value,
            {
                "episode_index",
                "episode_id",
                "base_init_state_id",
                "row_start",
                "row_stop",
                "raw_metadata_sha256",
                "raw_trajectory_sha256",
                "score_metadata_sha256",
                "score_primitives_sha256",
            },
            f"episodes[{index}]",
        )
        if episode["episode_index"] != index:
            raise ActivationReferenceError("episode indices are not canonical")
        episode_id = episode["episode_id"]
        if not isinstance(episode_id, str) or not episode_id:
            raise ActivationReferenceError("episode ID is invalid")
        observed_ids.append(episode_id)
        start, stop = episode["row_start"], episode["row_stop"]
        if (
            type(start) is not int
            or type(stop) is not int
            or start != expected_start
            or stop <= start
        ):
            raise ActivationReferenceError("episode row spans are not contiguous")
        if (
            type(episode["base_init_state_id"]) is not int
            or episode["base_init_state_id"] < 0
        ):
            raise ActivationReferenceError("episode base-init ID is invalid")
        for name in (
            "raw_metadata_sha256",
            "raw_trajectory_sha256",
            "score_metadata_sha256",
            "score_primitives_sha256",
        ):
            _require_sha256(episode[name], f"episodes[{index}].{name}")
        row_slice = slice(start, stop)
        if not np.all(arrays["episode_index"][row_slice] == index) or not np.all(
            arrays["base_init_state_id"][row_slice] == episode["base_init_state_id"]
        ):
            raise ActivationReferenceError(
                "episode row identities differ from metadata"
            )
        steps = arrays["control_step"][row_slice]
        if not np.array_equal(
            steps,
            np.arange(
                0, int(steps[-1]) + 1, ACTIVATION_SCORE_STRIDE_STEPS, dtype="<i4"
            ),
        ):
            raise ActivationReferenceError("episode activation cadence is not exact")
        expected_start = stop
    if observed_ids != sorted(observed_ids) or len(observed_ids) != len(
        set(observed_ids)
    ):
        raise ActivationReferenceError("episode IDs must be unique and sorted")
    if expected_start != rows:
        raise ActivationReferenceError("episode row spans do not cover all rows")


def _artifact_payload(
    metadata_base: Mapping[str, Any], arrays_value: Mapping[str, Any]
) -> tuple[dict[str, Any], bytes, bytes, str]:
    arrays = _canonical_arrays(arrays_value)
    array_bytes = _deterministic_npz_bytes(arrays)
    metadata = dict(_jsonable(metadata_base))
    metadata["arrays"] = {name: _array_spec(arrays[name]) for name in sorted(arrays)}
    metadata["files"] = {
        "arrays": ARRAYS_FILENAME,
        "arrays_sha256": _sha256(array_bytes),
    }
    _validate_metadata(metadata, arrays)
    metadata_bytes = _canonical(metadata)
    digest = _sha256(metadata_bytes + array_bytes)
    return metadata, metadata_bytes, array_bytes, digest


def _safe_output_root(value: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(value)))
    _reject_symlink_components(root, allow_missing=True)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ActivationReferenceError("output root must be a real directory")
    root.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(root)
    return root


def publish_activation_reference(
    metadata_base: Mapping[str, Any], arrays: Mapping[str, Any], output_root: Path
) -> Path:
    """Publish or exactly resume one content-addressed activation reference."""

    _, metadata_bytes, array_bytes, digest = _artifact_payload(metadata_base, arrays)
    root = _safe_output_root(output_root)
    destination = root / digest
    for entry in root.iterdir():
        if entry.name == f".{digest}.publish.lock":
            continue
        try:
            _require_sha256(entry.name, "prior content-addressed directory")
            if entry.name == digest:
                load_activation_reference(entry, expected_sha256=digest)
            else:
                _validate_prior_content_directory(entry)
        except ActivationReferenceError as exc:
            raise ActivationReferenceError(
                f"ambiguous output-root artifact: {entry}"
            ) from exc
    if destination.exists():
        load_activation_reference(destination, expected_sha256=digest)
        return destination
    lock = root / f".{digest}.publish.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if destination.exists():
                load_activation_reference(destination, expected_sha256=digest)
                return destination
            time.sleep(0.1)
        raise ActivationReferenceError(
            "timed out waiting for activation-reference publisher"
        )
    os.close(descriptor)
    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{digest}.tmp-", dir=root))
        for filename, payload in (
            (METADATA_FILENAME, metadata_bytes),
            (ARRAYS_FILENAME, array_bytes),
        ):
            with (staging / filename).open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        if destination.exists() or destination.is_symlink():
            raise ActivationReferenceError("refusing to overwrite activation reference")
        os.rename(staging, destination)
        staging = None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        lock.unlink(missing_ok=True)
    return destination


def _validate_prior_content_directory(directory: Path) -> None:
    """Validate immutable predecessor bytes without imposing the latest schema."""

    if directory.is_symlink() or not directory.is_dir():
        raise ActivationReferenceError("prior content entry is not a real directory")
    entries = {entry.name: entry for entry in directory.iterdir()}
    if set(entries) != {METADATA_FILENAME, ARRAYS_FILENAME}:
        raise ActivationReferenceError("prior content topology differs")
    metadata_bytes = _read_regular(entries[METADATA_FILENAME], MAX_METADATA_BYTES)
    array_bytes = _read_regular(entries[ARRAYS_FILENAME], MAX_ARRAY_BYTES)
    metadata = _strict_json_bytes(metadata_bytes, entries[METADATA_FILENAME])
    if (
        metadata_bytes != _canonical(metadata)
        or metadata.get("files", {}).get("arrays_sha256") != _sha256(array_bytes)
        or directory.name != _sha256(metadata_bytes + array_bytes)
    ):
        raise ActivationReferenceError("prior content identity differs")


def _reject_symlink_components(path: Path, *, allow_missing: bool = False) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if allow_missing:
                return
            raise ActivationReferenceError(
                f"artifact path component is missing: {current}"
            ) from None
        if stat.S_ISLNK(mode):
            raise ActivationReferenceError(
                f"artifact path contains a symlink: {current}"
            )


def _read_regular(path: Path, maximum: int) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_components(absolute)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ActivationReferenceError(f"artifact file is unsafe: {absolute}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ActivationReferenceError(
                f"artifact file is unsafe or exceeds size limit: {absolute}"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if remaining or len(payload) != before.st_size or identity_before != identity_after:
            raise ActivationReferenceError(
                f"artifact file changed while reading: {absolute}"
            )
        return payload
    finally:
        os.close(descriptor)


def load_activation_reference(
    path: Path, *, expected_sha256: str | None = None
) -> LoadedActivationReference:
    directory = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_components(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ActivationReferenceError("activation reference must be a real directory")
    entries = {entry.name: entry for entry in directory.iterdir()}
    if set(entries) != {METADATA_FILENAME, ARRAYS_FILENAME}:
        raise ActivationReferenceError("activation reference topology differs")
    metadata_bytes = _read_regular(entries[METADATA_FILENAME], MAX_METADATA_BYTES)
    array_bytes = _read_regular(entries[ARRAYS_FILENAME], MAX_ARRAY_BYTES)
    digest = _sha256(metadata_bytes + array_bytes)
    _require_sha256(directory.name, "activation reference directory name")
    if directory.name != digest or (
        expected_sha256 is not None and expected_sha256 != digest
    ):
        raise ActivationReferenceError("activation reference content hash differs")
    metadata = _strict_json_bytes(metadata_bytes, entries[METADATA_FILENAME])
    if metadata_bytes != _canonical(metadata):
        raise ActivationReferenceError("activation reference metadata is not canonical")
    if metadata.get("files", {}).get("arrays_sha256") != _sha256(array_bytes):
        raise ActivationReferenceError("activation array file hash differs")
    try:
        with zipfile.ZipFile(io.BytesIO(array_bytes), mode="r") as archive:
            infos = archive.infolist()
            expected_members = {f"{name}.npy" for name in ARRAY_NAMES}
            observed_members = [info.filename for info in infos]
            if (
                set(observed_members) != expected_members
                or len(observed_members) != len(expected_members)
                or any(
                    info.is_dir()
                    or info.flag_bits & 0x1
                    or info.file_size > MAX_ARRAY_BYTES
                    for info in infos
                )
                or sum(info.file_size for info in infos) > MAX_ARRAY_BYTES
            ):
                raise ActivationReferenceError("activation NPZ topology is unsafe")
        with np.load(io.BytesIO(array_bytes), allow_pickle=False) as archive:
            if set(archive.files) != ARRAY_NAMES:
                raise ActivationReferenceError("activation NPZ members differ")
            loaded = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ActivationReferenceError):
            raise
        raise ActivationReferenceError(
            "activation arrays cannot be loaded safely"
        ) from exc
    arrays = _canonical_arrays(loaded)
    if _deterministic_npz_bytes(arrays) != array_bytes:
        raise ActivationReferenceError("activation NPZ is not canonical")
    _validate_metadata(metadata, arrays)
    return LoadedActivationReference(
        directory,
        MappingProxyType(metadata),
        MappingProxyType(arrays),
        digest,
    )


def _frozen_artifact(
    freeze: Mapping[str, Any], name: str, *, root: Path
) -> tuple[Path, str]:
    artifacts = freeze.get("artifact_hashes")
    if not isinstance(artifacts, Mapping) or not isinstance(
        artifacts.get(name), Mapping
    ):
        raise ActivationReferenceError(f"Calibration freeze is missing {name}")
    item = _exact_keys(artifacts[name], {"path", "sha256"}, f"artifact_hashes.{name}")
    declared_path = item["path"]
    if not isinstance(declared_path, str) or not declared_path:
        raise ActivationReferenceError(f"Calibration freeze path is invalid for {name}")
    sha = _require_sha256(item["sha256"], f"artifact_hashes.{name}.sha256")
    path = (root / declared_path).resolve()
    if _sha256_file(path) != sha:
        raise ActivationReferenceError(f"Calibration freeze artifact differs: {name}")
    return path, sha


def _require_same_path(actual: Path, expected: Path, where: str) -> None:
    if actual.resolve() != expected.resolve():
        raise ActivationReferenceError(f"{where} path differs from Calibration freeze")


def _exact_split_directories(
    root: Path, split: str, expected: set[str], where: str
) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ActivationReferenceError(f"{where} root must be a real directory")
    split_root = root / split
    if split_root.is_symlink() or not split_root.is_dir():
        raise ActivationReferenceError(f"{where} split root must be a real directory")
    entries = {entry.name: entry for entry in split_root.iterdir()}
    missing = sorted(expected - set(entries))
    extra = sorted(set(entries) - expected)
    if missing or extra:
        raise ActivationReferenceError(
            f"{where} topology differs: missing={missing}, extra={extra}"
        )
    if any(entry.is_symlink() or not entry.is_dir() for entry in entries.values()):
        raise ActivationReferenceError(
            f"{where} episode paths must be real directories"
        )
    return split_root


def _require_exact_mapping(
    actual: Any, expected: Mapping[str, Any], where: str
) -> None:
    if not isinstance(actual, Mapping) or dict(actual) != dict(expected):
        raise ActivationReferenceError(f"{where} differs")


def _natural_activation_vectors(
    original_draws: Any,
    *,
    state_count: int,
    expected_width: int,
) -> np.ndarray:
    """Return the label-free natural vector defined by the frozen score schema."""

    draws = np.asarray(original_draws, dtype="<f8")
    if (
        draws.ndim != 3
        or draws.shape != (state_count, ORIGINAL_DRAWS, expected_width)
        or not np.isfinite(draws).all()
    ):
        raise ActivationReferenceError("selected natural activations differ")
    return np.ascontiguousarray(draws.mean(axis=1, dtype=np.float64), dtype="<f8")


def _frozen_natural_geometry(
    natural_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the preregistered natural leave-self-out 5-NN distribution."""

    natural = np.asarray(natural_vectors, dtype="<f8")
    if (
        natural.ndim != 2
        or natural.shape[0] < NATURAL_QUERY_COUNT
        or not np.isfinite(natural).all()
    ):
        raise ActivationReferenceError("natural geometry matrix is invalid")
    query_index = np.asarray(
        np.random.Generator(np.random.PCG64(NATURAL_QUERY_SEED)).choice(
            natural.shape[0], size=NATURAL_QUERY_COUNT, replace=False
        ),
        dtype="<i4",
    )
    distances = np.empty(NATURAL_QUERY_COUNT, dtype="<f8")
    for output_index, natural_index in enumerate(query_index):
        reference = np.delete(natural, int(natural_index), axis=0)
        distances[output_index] = five_nearest_neighbor_distance(
            natural[int(natural_index)], reference
        )
    percentile = float(np.percentile(distances, NATURAL_PERCENTILE))
    if percentile != FROZEN_NATURAL_95TH_PERCENTILE:
        raise ActivationReferenceError(
            "natural geometry percentile differs from the frozen Calibration value: "
            f"expected {FROZEN_NATURAL_95TH_PERCENTILE}, observed {percentile}"
        )
    return query_index, distances


def _require_row_alignment(
    generated: tuple[tuple[str, int, int], ...],
    bound: tuple[tuple[str, int, int], ...],
    reference: tuple[tuple[str, int, int], ...],
    probe_norm: tuple[tuple[str, int, int], ...],
) -> None:
    if not (
        generated == bound == reference == probe_norm
        and len(generated) == len(set(generated))
    ):
        raise ActivationReferenceError(
            "sidecar rows differ from frozen BoundProbe/feature-reference membership"
        )


def _build_from_frozen_sources(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = args.repo_root.resolve()
    protocol = load_protocol_config(root / "configs")
    freeze_path = args.calibration_freeze.resolve()
    _require_same_path(
        freeze_path, root / "locks" / "calibration_frozen.json", "freeze"
    )
    freeze = _strict_json(freeze_path)
    manifest_file, manifest_sha = _frozen_artifact(
        freeze, "calibration_manifest", root=root
    )
    bound_file, bound_sha = _frozen_artifact(freeze, "bound_probe", root=root)
    reference_metadata_file, reference_metadata_sha = _frozen_artifact(
        freeze, "feature_reference_metadata", root=root
    )
    reference_arrays_file, reference_arrays_sha = _frozen_artifact(
        freeze, "feature_reference_arrays", root=root
    )
    _require_same_path(args.manifest.resolve(), manifest_file, "Calibration manifest")
    _require_same_path(
        args.bound_probe.resolve() / "bound_probe.json", bound_file, "BoundProbe"
    )
    _require_same_path(
        args.feature_reference.resolve() / "metadata.json",
        reference_metadata_file,
        "feature-reference metadata",
    )
    _require_same_path(
        args.feature_reference.resolve() / "arrays.npz",
        reference_arrays_file,
        "feature-reference arrays",
    )
    manifest_payload = _strict_json(manifest_file)
    if _canonical(manifest_payload) != manifest_file.read_bytes():
        raise ActivationReferenceError("Calibration manifest is not canonical")
    bound = load_bound_probe_artifact(
        args.bound_probe.resolve(),
        protocol=protocol,
        repo_root=root,
        expected_sha256=bound_sha,
    )
    manifest = bound.rollout.manifest
    if (
        bound.rollout.source.split is not SplitName.CALIBRATION
        or bound.rollout.invalid_episode_ids
        or len(bound.rollout.valid_episode_ids) != EXPECTED_EPISODES
        or bound.rollout.source.manifest_sha256 != manifest_sha
        or _canonical(manifest.to_dict()) != _canonical(manifest_payload)
    ):
        raise ActivationReferenceError(
            "BoundProbe Calibration allocation differs from manifest"
        )
    reference_digest = args.feature_reference.resolve().name
    reference = load_feature_reference_bundle(
        args.feature_reference.resolve(), expected_sha256=reference_digest
    )
    if (
        reference.probe_sha256 != bound.sha256
        or reference.selected_candidate != bound.probe.candidate
    ):
        raise ActivationReferenceError(
            "feature reference differs from selected BoundProbe"
        )
    expected_ids = set(bound.rollout.valid_episode_ids)
    score_split = _exact_split_directories(
        args.score_root.resolve(), SplitName.CALIBRATION.value, expected_ids, "score"
    )
    allocation_raw = {item.episode_id: item for item in bound.rollout.valid_artifacts}
    reference_sources = {item.episode_id: item for item in reference.source_hashes}
    if set(reference_sources) != expected_ids:
        raise ActivationReferenceError("feature-reference source membership differs")
    specifications = {item.episode_id: item for item in manifest.episodes}
    config_sha = frozen_config_sha256(root)
    code_sha = scoring_source_sha256(root)
    vector_parts: list[np.ndarray] = []
    episode_index_parts: list[np.ndarray] = []
    base_parts: list[np.ndarray] = []
    step_parts: list[np.ndarray] = []
    episode_metadata: list[dict[str, Any]] = []
    score_identities: list[ScoringArtifactIdentity] = []
    row_start = 0
    width: int | None = None
    candidate = bound.probe.candidate
    selected_identities = tuple(
        identity
        for identity in bound.candidate_features
        if identity.candidate == candidate
    )
    if (
        len(selected_identities) != 1
        or selected_identities[0].rows != len(bound.rows)
    ):
        raise ActivationReferenceError("selected BoundProbe candidate identity differs")
    selected_width = selected_identities[0].width
    for episode_index, episode_id in enumerate(sorted(expected_ids)):
        specification = specifications[episode_id]
        raw_identity = allocation_raw[episode_id]
        sidecar = load_scoring_sidecar(
            score_split / episode_id, expected_episode_id=episode_id
        )
        links = sidecar.metadata.get("links")
        expected_links = {
            "raw_metadata_sha256": raw_identity.metadata_sha256,
            "raw_trajectory_sha256": raw_identity.trajectory_sha256,
            "probe_sha256": bound.sha256,
            "config_sha256": config_sha,
            "code_sha256": code_sha,
        }
        if sidecar.metadata.get("split") != SplitName.CALIBRATION.value:
            raise ActivationReferenceError(f"{episode_id}: score source links differ")
        _require_exact_mapping(
            links, expected_links, f"{episode_id}: score source links"
        )
        reference_source = reference_sources[episode_id]
        if (
            reference_source.raw_metadata_sha256 != raw_identity.metadata_sha256
            or reference_source.raw_trajectory_sha256
            != raw_identity.trajectory_sha256
            or reference_source.score_metadata_sha256 != sidecar.metadata_sha256
            or reference_source.score_primitives_sha256 != sidecar.primitives_sha256
        ):
            raise ActivationReferenceError(
                f"{episode_id}: feature-reference source differs"
            )
        score_steps = np.asarray(sidecar.arrays["control_step"], dtype="<i4")
        vectors = _natural_activation_vectors(
            sidecar.arrays["original_activation"],
            state_count=score_steps.size,
            expected_width=selected_width,
        )
        observed_width = vectors.shape[1]
        if width is None:
            width = observed_width
        if (
            observed_width != width
            or sidecar.metadata["capture"]["selected_activation_width"] != width
        ):
            raise ActivationReferenceError(
                f"{episode_id}: selected activation width differs"
            )
        rows = vectors.shape[0]
        row_stop = row_start + rows
        vector_parts.append(np.array(vectors, copy=True, order="C"))
        episode_index_parts.append(np.full(rows, episode_index, dtype="<i2"))
        base_parts.append(np.full(rows, specification.base_init_state_id, dtype="<i4"))
        step_parts.append(np.array(score_steps, copy=True, order="C"))
        score_identity = ScoringArtifactIdentity(
            episode_id=episode_id,
            metadata_sha256=sidecar.metadata_sha256,
            primitives_sha256=sidecar.primitives_sha256,
            **expected_links,
        )
        score_identities.append(score_identity)
        episode_metadata.append(
            {
                "episode_index": episode_index,
                "episode_id": episode_id,
                "base_init_state_id": specification.base_init_state_id,
                "row_start": row_start,
                "row_stop": row_stop,
                "raw_metadata_sha256": raw_identity.metadata_sha256,
                "raw_trajectory_sha256": raw_identity.trajectory_sha256,
                "score_metadata_sha256": sidecar.metadata_sha256,
                "score_primitives_sha256": sidecar.primitives_sha256,
            }
        )
        row_start = row_stop
    score_allocation = ScoreAllocationReceipt(
        rollout=bound.rollout,
        probe_sha256=bound.sha256,
        config_sha256=config_sha,
        code_sha256=code_sha,
        score_artifacts=tuple(score_identities),
    )
    arrays = {
        "activation_vectors": np.concatenate(vector_parts, axis=0),
        "episode_index": np.concatenate(episode_index_parts),
        "base_init_state_id": np.concatenate(base_parts),
        "control_step": np.concatenate(step_parts),
    }
    natural_query_index, natural_five_nn_distance = _frozen_natural_geometry(
        arrays["activation_vectors"]
    )
    arrays["natural_query_index"] = natural_query_index
    arrays["natural_five_nn_distance"] = natural_five_nn_distance
    generated_rows = tuple(
        (
            episode_metadata[int(episode_index)]["episode_id"],
            int(base_init_state_id),
            int(control_step),
        )
        for episode_index, base_init_state_id, control_step in zip(
            arrays["episode_index"],
            arrays["base_init_state_id"],
            arrays["control_step"],
            strict=True,
        )
    )
    bound_rows = tuple(
        (row.episode_id, row.base_init_state_id, row.control_step)
        for row in bound.rows
    )
    reference_rows = tuple(
        (state.episode_id, state.base_init_state_id, state.control_step)
        for state in reference.coverage_states
    )
    norm_rows = tuple(
        (state.episode_id, state.base_init_state_id, state.control_step)
        for state in reference.probe_norm_states
    )
    _require_row_alignment(generated_rows, bound_rows, reference_rows, norm_rows)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "format": ARTIFACT_FORMAT,
        "kind": ARTIFACT_KIND,
        "split": SplitName.CALIBRATION.value,
        "selection": {
            "source": "rescore_factual_original_selected_activation",
            "selected_candidate": candidate,
            "cadence_control_steps": ACTIVATION_SCORE_STRIDE_STEPS,
            "outcome_conditioned": False,
            "labels_used": False,
            "refit_performed": False,
            "transformation": "float64_arithmetic_mean_over_all_frozen_original_draws",
            "natural_draws": ORIGINAL_DRAWS,
        },
        "geometry": {
            "method": "mean_euclidean_distance_to_5_nearest_leave_self_out",
            "query_seed": NATURAL_QUERY_SEED,
            "query_count": NATURAL_QUERY_COUNT,
            "percentile": NATURAL_PERCENTILE,
            "natural_95th_percentile": FROZEN_NATURAL_95TH_PERCENTILE,
        },
        "counts": {
            "episodes": len(episode_metadata),
            "rows": int(arrays["activation_vectors"].shape[0]),
            "width": int(arrays["activation_vectors"].shape[1]),
        },
        "source": {
            "predecessor_calibration_freeze_sha256": _sha256_file(freeze_path),
            "manifest_sha256": manifest_sha,
            "rollout_allocation_sha256": bound.rollout.sha256,
            "score_allocation_sha256": score_allocation.sha256,
            "bound_probe_sha256": bound.sha256,
            "feature_reference_sha256": reference_digest,
            "feature_reference_metadata_sha256": reference_metadata_sha,
            "feature_reference_arrays_sha256": reference_arrays_sha,
            "config_sha256": config_sha,
            "code_sha256": code_sha,
            "raw_bytes_reverified_locally": False,
            "raw_hash_links_verified_against_frozen_cohort": True,
        },
        "episodes": episode_metadata,
    }
    return metadata, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--calibration-freeze", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bound-probe", type=Path, required=True)
    parser.add_argument("--feature-reference", type=Path, required=True)
    parser.add_argument(
        "--score-root",
        type=Path,
        required=True,
        help="score artifact root before the calibration split directory",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    metadata, arrays = _build_from_frozen_sources(args)
    path = publish_activation_reference(metadata, arrays, args.output_root)
    loaded = load_activation_reference(path, expected_sha256=path.name)
    print(
        json.dumps(
            {
                "kind": "calibration_activation_reference_published",
                "path": str(path),
                "sha256": loaded.sha256,
                "metadata_sha256": _sha256((path / METADATA_FILENAME).read_bytes()),
                "arrays_sha256": _sha256((path / ARRAYS_FILENAME).read_bytes()),
                "episodes": loaded.metadata["counts"]["episodes"],
                "rows": loaded.metadata["counts"]["rows"],
                "width": loaded.metadata["counts"]["width"],
                "selected_candidate": loaded.metadata["selection"][
                    "selected_candidate"
                ],
                "labels_used": False,
                "refit_performed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
