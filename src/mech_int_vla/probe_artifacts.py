"""Content-addressed bindings between a probe and its Calibration inputs.

The numerical :class:`~mech_int_vla.probes.ProbeArtifact` intentionally contains
only fitted parameters and selection statistics.  This module adds the missing
provenance edge: it binds that artifact to an audited Calibration rollout
allocation and to the exact rows and activation matrices in a
:class:`~mech_int_vla.artifacts.ProbeCohort`.

Callers cannot provide identity strings to :func:`bind_probe_artifact`.  Every
hash is derived from the supplied validated objects and the current frozen
repository bytes.  Both construction and loading revalidate the exact
Calibration topology, configuration hash, scoring-source hash, cohort content,
and a deterministic numerical refit before the binding can cross a trust
boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from .allocation import (
    AllocationError,
    RolloutAllocationReceipt,
    revalidate_rollout_receipt,
    rollout_receipt_from_metadata,
)
from .artifacts import (
    ACTIVATION_CANDIDATES,
    CohortManifest,
    ProbeCohort,
    probe_cohort_array_sha256,
)
from .config import ProtocolConfig, SplitName
from .probes import (
    ProbeArtifact,
    ProbeError,
    probe_artifact_from_metadata,
    select_and_fit_circular_probe,
)
from .provenance import (
    ProvenanceError,
    frozen_config_sha256,
    scoring_source_sha256,
)

BOUND_PROBE_SCHEMA_VERSION: Final = 1
BOUND_PROBE_FORMAT: Final = "mech_int_vla_bound_probe"
_FILENAME: Final = "bound_probe.json"
_MAX_ARTIFACT_BYTES: Final = 16 * 1024 * 1024
_MAX_TRAINING_ROWS: Final = 160 * 104
_ROW_DOMAIN: Final = b"mech-int-vla/probe-row-alignment/v1\0"
_UNTRUSTED = object()
_LIVE_REFIT_TRUST = object()
_EXPECTED_LOAD_TRUST = object()


class BoundProbeError(ValueError):
    """Raised when a probe binding is unsafe, malformed, or inconsistent."""


@dataclass(frozen=True)
class ProbeTrainingRow:
    """One exact row in the pre-action probe-training alignment."""

    episode_id: str
    base_init_state_id: int
    control_step: int

    def __post_init__(self) -> None:
        _safe_name(self.episode_id, "training_row.episode_id")
        _integer(self.base_init_state_id, "training_row.base_init_state_id")
        step = _integer(self.control_step, "training_row.control_step")
        if step < 0:
            raise BoundProbeError("training_row.control_step must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "base_init_state_id": self.base_init_state_id,
            "control_step": self.control_step,
        }


@dataclass(frozen=True)
class CandidateFeatureIdentity:
    """Logical content identity of one aligned activation matrix."""

    candidate: str
    rows: int
    width: int
    dtype: str
    logical_sha256: str

    def __post_init__(self) -> None:
        _safe_name(self.candidate, "candidate_feature.candidate")
        rows = _integer(self.rows, "candidate_feature.rows")
        width = _integer(self.width, "candidate_feature.width")
        if not 1 <= rows <= _MAX_TRAINING_ROWS:
            raise BoundProbeError("candidate_feature.rows is outside the frozen limit")
        if width < 1:
            raise BoundProbeError("candidate_feature.width must be positive")
        if self.dtype != "<f4":
            raise BoundProbeError("candidate_feature.dtype must be canonical float32")
        _sha256(self.logical_sha256, "candidate_feature.logical_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "rows": self.rows,
            "width": self.width,
            "dtype": self.dtype,
            "logical_sha256": self.logical_sha256,
        }


@dataclass(frozen=True, init=False)
class BoundProbeArtifact:
    """A numerical probe plus an exact, non-asserted Calibration binding.

    Direct construction is disabled.  Use :func:`bind_probe_artifact` for live
    inputs or :func:`load_bound_probe_artifact` for persisted bytes.
    """

    probe: ProbeArtifact
    rollout: RolloutAllocationReceipt
    cohort_manifest: CohortManifest
    rows: tuple[ProbeTrainingRow, ...]
    candidate_features: tuple[CandidateFeatureIdentity, ...]
    theta_rel_sha256: str
    failure_label_sha256: str
    config_sha256: str
    code_sha256: str
    _trust_marker: object = field(compare=False, repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "BoundProbeArtifact cannot be constructed directly; use "
            "bind_probe_artifact or load_bound_probe_artifact"
        )

    @classmethod
    def _validated(
        cls,
        *,
        probe: ProbeArtifact,
        rollout: RolloutAllocationReceipt,
        cohort_manifest: CohortManifest,
        rows: tuple[ProbeTrainingRow, ...],
        candidate_features: tuple[CandidateFeatureIdentity, ...],
        theta_rel_sha256: str,
        failure_label_sha256: str,
        config_sha256: str,
        code_sha256: str,
    ) -> BoundProbeArtifact:
        result = object.__new__(cls)
        object.__setattr__(result, "probe", probe)
        object.__setattr__(result, "rollout", rollout)
        object.__setattr__(result, "cohort_manifest", cohort_manifest)
        object.__setattr__(result, "rows", rows)
        object.__setattr__(result, "candidate_features", candidate_features)
        object.__setattr__(result, "theta_rel_sha256", theta_rel_sha256)
        object.__setattr__(result, "failure_label_sha256", failure_label_sha256)
        object.__setattr__(result, "config_sha256", config_sha256)
        object.__setattr__(result, "code_sha256", code_sha256)
        object.__setattr__(result, "_trust_marker", _UNTRUSTED)
        _validate_bound(result)
        return result

    @property
    def numerical_probe_sha256(self) -> str:
        return self.probe.sha256()

    @property
    def rollout_receipt_sha256(self) -> str:
        return self.rollout.sha256

    @property
    def cohort_manifest_sha256(self) -> str:
        return self.cohort_manifest.sha256

    @property
    def row_alignment_sha256(self) -> str:
        return _row_alignment_sha256(self.rows)

    def to_metadata(self) -> dict[str, Any]:
        valid_ids = self.rollout.valid_episode_ids
        invalid_ids = self.rollout.invalid_episode_ids
        return {
            "schema_version": BOUND_PROBE_SCHEMA_VERSION,
            "format": BOUND_PROBE_FORMAT,
            "kind": "bound_probe",
            "numerical_probe": {
                "sha256": self.numerical_probe_sha256,
                "metadata": self.probe.to_metadata(),
            },
            "calibration_inputs": {
                "split": SplitName.CALIBRATION.value,
                "episode_manifest_sha256": self.rollout.source.manifest_sha256,
                "rollout_receipt_sha256": self.rollout_receipt_sha256,
                "rollout_receipt": self.rollout.to_metadata(),
                "cohort_manifest_sha256": self.cohort_manifest_sha256,
                "cohort_manifest": _plain_json(self.cohort_manifest.payload),
                "source": self.rollout.source.to_dict(),
                "configuration_identity": {
                    "episode_manifest_sha256": self.rollout.source.manifest_sha256,
                    "full_config_sha256": self.config_sha256,
                    "scoring_source_sha256": self.code_sha256,
                },
                "valid_episode_ids": list(valid_ids),
                "invalid_reset_episode_ids": list(invalid_ids),
                "raw_artifacts": [
                    identity.to_dict() for identity in self.rollout.raw_artifacts
                ],
            },
            "training": {
                "selected_candidate": self.probe.candidate,
                "rows": len(self.rows),
                "episodes": len(valid_ids),
                "base_init_state_ids": list(self.probe.training_base_init_state_ids),
                "symmetry_order": self.probe.model.symmetry_order,
                "row_alignment_sha256": self.row_alignment_sha256,
                "theta_rel_sha256": self.theta_rel_sha256,
                "failure_label_sha256": self.failure_label_sha256,
                "row_alignment": [row.to_dict() for row in self.rows],
                "candidate_features": [
                    identity.to_dict() for identity in self.candidate_features
                ],
                "derivation": "exact_refit_matches_numerical_probe",
            },
        }

    def canonical_json(self) -> bytes:
        return _canonical_json(self.to_metadata())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()


def bind_probe_artifact(
    probe: ProbeArtifact,
    rollout: RolloutAllocationReceipt,
    cohort: ProbeCohort,
    *,
    protocol: ProtocolConfig,
    repo_root: str | Path,
) -> BoundProbeArtifact:
    """Bind one fitted probe to actual audited Calibration inputs.

    No hashes are accepted from the caller.  Logical hashes are calculated from
    copied arrays after all cross-object identities and dimensions are checked.
    """

    if not isinstance(probe, ProbeArtifact):
        raise BoundProbeError("probe must be a validated ProbeArtifact")
    if not isinstance(rollout, RolloutAllocationReceipt):
        raise BoundProbeError("rollout must be a RolloutAllocationReceipt")
    if not isinstance(cohort, ProbeCohort):
        raise BoundProbeError("cohort must be a ProbeCohort")
    if rollout.source.split is not SplitName.CALIBRATION:
        raise BoundProbeError("bound probe inputs must use the calibration split")
    try:
        rollout = revalidate_rollout_receipt(
            rollout, protocol=protocol, repo_root=repo_root
        )
        config_sha256 = frozen_config_sha256(repo_root)
        code_sha256 = scoring_source_sha256(repo_root)
    except (AllocationError, ProvenanceError) as exc:
        raise BoundProbeError(
            f"could not revalidate Calibration inputs: {exc}"
        ) from exc

    episode_ids = _copy_vector(cohort.episode_id, np.str_, "cohort.episode_id")
    base_ids = _copy_vector(
        cohort.base_init_state_id, np.int64, "cohort.base_init_state_id"
    )
    control_steps = _copy_vector(cohort.control_step, np.int64, "cohort.control_step")
    theta_rel = _copy_vector(cohort.theta_rel, np.float64, "cohort.theta_rel")
    failure_label = _copy_vector(cohort.failure_label, np.bool_, "cohort.failure_label")
    row_count = episode_ids.size
    if row_count == 0 or row_count > _MAX_TRAINING_ROWS:
        raise BoundProbeError("cohort row count is outside the frozen limit")
    if any(
        value.size != row_count
        for value in (base_ids, control_steps, theta_rel, failure_label)
    ):
        raise BoundProbeError("cohort row arrays are not exactly aligned")
    if not np.isfinite(theta_rel).all():
        raise BoundProbeError("cohort.theta_rel contains non-finite values")
    if np.any(control_steps < 0):
        raise BoundProbeError("cohort.control_step contains negative values")

    feature_identities: list[CandidateFeatureIdentity] = []
    if set(cohort.activation_features) != set(ACTIVATION_CANDIDATES):
        raise BoundProbeError("cohort activation candidates differ from the schema")
    for candidate in ACTIVATION_CANDIDATES:
        matrix = np.array(cohort.activation_features[candidate], copy=True, order="C")
        if matrix.dtype != np.dtype(np.float32):
            raise BoundProbeError(f"activation {candidate} must have dtype float32")
        if matrix.ndim != 2 or matrix.shape[0] != row_count or matrix.shape[1] < 1:
            raise BoundProbeError(f"activation {candidate} has an invalid shape")
        if not np.isfinite(matrix).all():
            raise BoundProbeError(f"activation {candidate} contains non-finite values")
        feature_identities.append(
            CandidateFeatureIdentity(
                candidate=candidate,
                rows=row_count,
                width=int(matrix.shape[1]),
                dtype="<f4",
                logical_sha256=_logical_array_sha256(matrix),
            )
        )

    rows = tuple(
        ProbeTrainingRow(str(episode_id), int(base_id), int(control_step))
        for episode_id, base_id, control_step in zip(
            episode_ids.tolist(),
            base_ids.tolist(),
            control_steps.tolist(),
            strict=True,
        )
    )
    result = BoundProbeArtifact._validated(
        probe=probe,
        rollout=rollout,
        cohort_manifest=cohort.manifest,
        rows=rows,
        candidate_features=tuple(feature_identities),
        theta_rel_sha256=_logical_array_sha256(theta_rel),
        failure_label_sha256=_logical_array_sha256(failure_label),
        config_sha256=config_sha256,
        code_sha256=code_sha256,
    )
    _validate_live_cohort(result, cohort)
    try:
        recomputed = select_and_fit_circular_probe(
            cohort.activation_features,
            cohort.samples,
            selection_config=protocol.split.calibration_selection,
        ).artifact
    except (ProbeError, TypeError, ValueError) as exc:
        raise BoundProbeError(
            f"could not deterministically refit probe: {exc}"
        ) from exc
    if recomputed.canonical_json() != probe.canonical_json():
        raise BoundProbeError(
            "numerical probe does not exactly match a deterministic refit on the cohort"
        )
    return _mark_trusted(result, _LIVE_REFIT_TRUST)


def validate_bound_probe_artifact(
    artifact: BoundProbeArtifact,
    *,
    protocol: ProtocolConfig,
    repo_root: str | Path,
) -> BoundProbeArtifact:
    """Revalidate a bound probe against current protocol and repository bytes."""

    if not isinstance(artifact, BoundProbeArtifact):
        raise BoundProbeError("artifact must be a BoundProbeArtifact")
    _require_trusted(artifact)
    _validate_bound_against_context(artifact, protocol=protocol, repo_root=repo_root)
    return artifact


def write_bound_probe_artifact(
    artifact: BoundProbeArtifact,
    output_root: str | Path,
    *,
    protocol: ProtocolConfig,
    repo_root: str | Path,
) -> Path:
    """Publish a canonical bound probe without reusing or replacing a digest.

    The final directory is reserved with exclusive ``mkdir`` and the payload is
    installed with an exclusive hard link.  A crash may therefore leave an
    incomplete (and unloadable) digest directory, but neither a cooperating nor
    a racing writer can replace an existing path.
    """

    if not isinstance(artifact, BoundProbeArtifact):
        raise BoundProbeError("artifact must be a BoundProbeArtifact")
    _require_trusted(artifact)
    _validate_bound_against_context(artifact, protocol=protocol, repo_root=repo_root)
    canonical = artifact.canonical_json()
    if len(canonical) > _MAX_ARTIFACT_BYTES:
        raise BoundProbeError("bound probe exceeds the artifact size limit")
    root = _safe_root(output_root)
    digest = hashlib.sha256(canonical).hexdigest()
    destination = root / digest
    if _lexists(destination):
        raise FileExistsError(f"refusing to overwrite bound probe {destination}")

    lock = root / f".{digest}.publish.lock"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock, flags, 0o600)
    except FileExistsError as exc:
        raise FileExistsError("another writer is publishing this bound probe") from exc
    os.close(descriptor)

    staging: Path | None = None
    destination_inode: tuple[int, int] | None = None
    payload_inode: tuple[int, int] | None = None
    try:
        _fsync(root)
        staging = Path(tempfile.mkdtemp(prefix=f".{digest}.tmp-", dir=root))
        target = staging / _FILENAME
        with target.open("xb") as stream:
            stream.write(canonical)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync(staging)
        payload_inode = _inode_identity(target.stat(follow_symlinks=False))
        if _lexists(destination):
            raise FileExistsError(f"refusing to overwrite bound probe {destination}")
        os.mkdir(destination, 0o700)
        destination_inode = _inode_identity(destination.stat(follow_symlinks=False))
        _fsync(root)
        os.link(target, destination / _FILENAME, follow_symlinks=False)
        _fsync(destination)
        target.unlink()
        staging.rmdir()
        staging = None
        _fsync(root)
    except BaseException:
        if destination_inode is not None and payload_inode is not None:
            _cleanup_failed_destination(
                root,
                digest,
                destination_inode=destination_inode,
                payload_inode=payload_inode,
            )
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        lock.unlink(missing_ok=True)
        _fsync(root)
    return destination


def load_bound_probe_artifact(
    path: str | Path,
    *,
    protocol: ProtocolConfig,
    repo_root: str | Path,
    expected_sha256: str,
) -> BoundProbeArtifact:
    """Load bytes anchored by an expected digest and reconstruct without pickle."""

    _sha256(expected_sha256, "expected_sha256")
    directory = _absolute(path)
    if _has_symlink_component(directory) or not directory.is_dir():
        raise BoundProbeError("bound probe path must be a real directory")
    try:
        before_directory = directory.stat(follow_symlinks=False)
        entries = list(os.scandir(directory))
    except OSError as exc:
        raise BoundProbeError(f"cannot inspect bound probe directory: {exc}") from exc
    if len(entries) != 1 or entries[0].name != _FILENAME:
        raise BoundProbeError(f"bound probe must contain exactly one {_FILENAME}")
    entry = entries[0]
    try:
        before_path = entry.stat(follow_symlinks=False)
    except OSError as exc:
        raise BoundProbeError(f"cannot inspect {_FILENAME}: {exc}") from exc
    if entry.is_symlink() or not stat.S_ISREG(before_path.st_mode):
        raise BoundProbeError(f"{_FILENAME} must be a regular file, not a symlink")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(entry.path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise BoundProbeError(f"{_FILENAME} must be a regular file")
            if before.st_size > _MAX_ARTIFACT_BYTES:
                raise BoundProbeError("bound probe exceeds the artifact size limit")
            chunks: list[bytes] = []
            size = 0
            while chunk := os.read(descriptor, 1024 * 1024):
                size += len(chunk)
                if size > _MAX_ARTIFACT_BYTES:
                    raise BoundProbeError("bound probe exceeds the artifact size limit")
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = (directory / _FILENAME).stat(follow_symlinks=False)
        after_directory = directory.stat(follow_symlinks=False)
    except OSError as exc:
        raise BoundProbeError(f"could not read {_FILENAME} safely: {exc}") from exc
    if _stat_identity(before) != _stat_identity(after) or _stat_identity(
        before_path
    ) != _stat_identity(after_path):
        raise BoundProbeError("bound probe changed while it was being read")
    if _stat_identity(before_directory) != _stat_identity(after_directory):
        raise BoundProbeError("bound probe directory changed while it was being read")
    canonical = b"".join(chunks)
    digest = hashlib.sha256(canonical).hexdigest()
    if directory.name != digest:
        raise BoundProbeError("bound probe directory name differs from content hash")
    if digest != expected_sha256:
        raise BoundProbeError("bound probe differs from expected_sha256")
    try:
        metadata = json.loads(
            canonical.decode("utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise BoundProbeError(f"bound probe is not strict UTF-8 JSON: {exc}") from exc
    if _canonical_json(metadata) != canonical:
        raise BoundProbeError("bound probe is not exact canonical JSON")
    artifact = _bound_from_metadata(metadata)
    _validate_bound_against_context(artifact, protocol=protocol, repo_root=repo_root)
    if artifact.canonical_json() != canonical:
        raise BoundProbeError("bound probe is not exactly reconstructable")
    if _has_symlink_component(directory) or not directory.is_dir():
        raise BoundProbeError("bound probe path changed while it was being read")
    return _mark_trusted(artifact, _EXPECTED_LOAD_TRUST)


def _mark_trusted(artifact: BoundProbeArtifact, marker: object) -> BoundProbeArtifact:
    if marker is not _LIVE_REFIT_TRUST and marker is not _EXPECTED_LOAD_TRUST:
        raise BoundProbeError("bound probe trust marker is invalid")
    _validate_bound(artifact)
    object.__setattr__(artifact, "_trust_marker", marker)
    return artifact


def _require_trusted(artifact: BoundProbeArtifact) -> None:
    marker = getattr(artifact, "_trust_marker", _UNTRUSTED)
    if marker is not _LIVE_REFIT_TRUST and marker is not _EXPECTED_LOAD_TRUST:
        raise BoundProbeError(
            "bound probe is untrusted; use exact live refit or expected-digest loading"
        )


def _validate_live_cohort(artifact: BoundProbeArtifact, cohort: ProbeCohort) -> None:
    if cohort.manifest is not artifact.cohort_manifest and (
        cohort.manifest.canonical_json() != artifact.cohort_manifest.canonical_json()
    ):
        raise BoundProbeError("cohort manifest changed during binding")
    valid_ids = tuple(cohort.valid_episode_ids)
    invalid_ids = tuple(cohort.invalid_reset_episode_ids)
    if valid_ids != artifact.rollout.valid_episode_ids:
        raise BoundProbeError("cohort valid episode IDs differ from rollout receipt")
    if invalid_ids != artifact.rollout.invalid_episode_ids:
        raise BoundProbeError("cohort invalid episode IDs differ from rollout receipt")
    if (
        cohort.samples.symmetry_order
        != artifact.rollout.source.task.planar_symmetry_order
    ):
        raise BoundProbeError("cohort symmetry order differs from rollout task")
    content = _cohort_training_content(artifact.cohort_manifest)
    actual = _training_content_from_cohort(cohort)
    if content != actual:
        raise BoundProbeError(
            "cohort arrays differ from the content hashes frozen by cohort assembly"
        )


def _validate_bound_against_context(
    artifact: BoundProbeArtifact,
    *,
    protocol: ProtocolConfig,
    repo_root: str | Path,
) -> None:
    _validate_bound(artifact)
    try:
        revalidate_rollout_receipt(
            artifact.rollout, protocol=protocol, repo_root=repo_root
        )
        config_sha256 = frozen_config_sha256(repo_root)
        code_sha256 = scoring_source_sha256(repo_root)
    except (AllocationError, ProvenanceError) as exc:
        raise BoundProbeError(f"bound probe context is invalid: {exc}") from exc
    if artifact.config_sha256 != config_sha256 or artifact.code_sha256 != code_sha256:
        raise BoundProbeError("bound probe config/source hashes differ from repository")


def _validate_bound(artifact: BoundProbeArtifact) -> None:
    if not isinstance(artifact.probe, ProbeArtifact):
        raise BoundProbeError("bound numerical probe is invalid")
    if not isinstance(artifact.rollout, RolloutAllocationReceipt):
        raise BoundProbeError("bound rollout receipt is invalid")
    if artifact.rollout.source.split is not SplitName.CALIBRATION:
        raise BoundProbeError("bound rollout split must be calibration")
    if not isinstance(artifact.cohort_manifest, CohortManifest):
        raise BoundProbeError("bound cohort manifest is invalid")
    rows = tuple(artifact.rows)
    if (
        not rows
        or len(rows) > _MAX_TRAINING_ROWS
        or not all(isinstance(row, ProbeTrainingRow) for row in rows)
    ):
        raise BoundProbeError("bound training rows are invalid")
    identities = tuple(artifact.candidate_features)
    if not all(isinstance(item, CandidateFeatureIdentity) for item in identities):
        raise BoundProbeError("bound candidate identities are invalid")
    if tuple(item.candidate for item in identities) != ACTIVATION_CANDIDATES:
        raise BoundProbeError("bound candidates differ from the frozen order")
    if any(item.rows != len(rows) for item in identities):
        raise BoundProbeError("candidate feature row counts differ")
    by_candidate = {item.candidate: item for item in identities}
    if by_candidate[artifact.probe.candidate].width != int(
        artifact.probe.model.feature_center.size
    ):
        raise BoundProbeError("selected candidate width differs from fitted probe")
    _sha256(artifact.theta_rel_sha256, "theta_rel_sha256")
    _sha256(artifact.failure_label_sha256, "failure_label_sha256")
    _sha256(artifact.config_sha256, "config_sha256")
    _sha256(artifact.code_sha256, "code_sha256")

    valid_ids = artifact.rollout.valid_episode_ids
    invalid_ids = artifact.rollout.invalid_episode_ids
    row_episode_ids = tuple(row.episode_id for row in rows)
    if tuple(sorted(set(row_episode_ids))) != valid_ids:
        raise BoundProbeError("training rows do not exactly cover valid episodes")
    if set(row_episode_ids) & set(invalid_ids):
        raise BoundProbeError("invalid-reset episode appears in training rows")
    expected_base = {
        episode.episode_id: episode.base_init_state_id
        for episode in artifact.rollout.manifest.episodes
    }
    for row in rows:
        if row.base_init_state_id != expected_base.get(row.episode_id):
            raise BoundProbeError("training row base-init ID differs from manifest")
    for episode_id in valid_ids:
        episode_rows = [row for row in rows if row.episode_id == episode_id]
        steps = tuple(row.control_step for row in episode_rows)
        if steps != tuple(sorted(set(steps))):
            raise BoundProbeError(
                "training control steps must be strictly increasing per episode"
            )
    if row_episode_ids != tuple(sorted(row_episode_ids)):
        raise BoundProbeError("training rows must be grouped in episode-ID order")

    groups = tuple(sorted({row.base_init_state_id for row in rows}))
    probe = artifact.probe
    if probe.training_rows != len(rows):
        raise BoundProbeError("probe training row count differs from cohort")
    if probe.training_episodes != len(valid_ids):
        raise BoundProbeError("probe training episode count differs from cohort")
    if probe.training_base_init_state_ids != groups:
        raise BoundProbeError("probe base-init groups differ from cohort")
    if probe.model.symmetry_order != artifact.rollout.source.task.planar_symmetry_order:
        raise BoundProbeError("probe symmetry order differs from task")
    _validate_cohort_manifest(artifact)


def _validate_cohort_manifest(artifact: BoundProbeArtifact) -> None:
    payload = _mapping(_plain_json(artifact.cohort_manifest.payload), "cohort_manifest")
    _exact_keys(
        payload,
        {
            "schema_version",
            "kind",
            "split",
            "task",
            "model",
            "selection",
            "activation_candidates",
            "training_content",
            "episodes",
            "invalid_reset_episode_ids",
        },
        "cohort_manifest",
    )
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise BoundProbeError("cohort manifest schema version differs")
    if payload["kind"] != "probe_cohort":
        raise BoundProbeError("cohort manifest schema/kind differs")
    if payload["split"] != SplitName.CALIBRATION.value:
        raise BoundProbeError("cohort manifest split must be calibration")
    source = artifact.rollout.source
    task = _mapping(payload["task"], "cohort_manifest.task")
    for name in ("task_id", "task_rank", "planar_symmetry_order"):
        if type(task.get(name)) is not int:
            raise BoundProbeError(f"cohort task {name} must be an integer")
    expected_task = {
        "suite": source.task.suite,
        "task_id": source.task.task_id,
        "task_rank": source.task.rank,
        "language": source.task.language,
        "primary_object": source.task.primary_object,
        "planar_symmetry_order": source.task.planar_symmetry_order,
    }
    if task != expected_task:
        raise BoundProbeError("cohort task differs from rollout task")
    model = _mapping(payload["model"], "cohort_manifest.model")
    if model != {
        "policy_revision": source.policy_revision,
        "base_vlm_revision": source.base_vlm_revision,
        "code_commit": source.code_commit,
    }:
        raise BoundProbeError("cohort model/source differs from rollout source")
    selection = _mapping(payload["selection"], "cohort_manifest.selection")
    _exact_keys(
        selection, {"kind", "stride", "outcome_conditioned"}, "cohort.selection"
    )
    stride = selection["stride"]
    if (
        selection["kind"] != "valid_pre_action_control_step_stride"
        or selection["outcome_conditioned"] is not False
        or type(stride) is not int
        or stride != 5
    ):
        raise BoundProbeError("cohort row-selection rule differs from protocol")
    if payload["activation_candidates"] != list(ACTIVATION_CANDIDATES):
        raise BoundProbeError("cohort activation candidate order differs")
    content = _cohort_training_content(artifact.cohort_manifest)
    if content["row_count"] != len(artifact.rows):
        raise BoundProbeError("cohort training row count differs from bound rows")
    expected_vector_hashes = {
        "episode_id_sha256": _logical_array_sha256(
            np.asarray([row.episode_id for row in artifact.rows], dtype=np.str_)
        ),
        "base_init_state_id_sha256": _logical_array_sha256(
            np.asarray(
                [row.base_init_state_id for row in artifact.rows], dtype=np.int64
            )
        ),
        "control_step_sha256": _logical_array_sha256(
            np.asarray([row.control_step for row in artifact.rows], dtype=np.int64)
        ),
        "theta_rel_sha256": artifact.theta_rel_sha256,
        "failure_label_sha256": artifact.failure_label_sha256,
    }
    if any(content[name] != value for name, value in expected_vector_hashes.items()):
        raise BoundProbeError("cohort vector content hashes differ from binding")
    expected_features = {
        identity.candidate: {
            "shape": [identity.rows, identity.width],
            "dtype": identity.dtype,
            "logical_sha256": identity.logical_sha256,
        }
        for identity in artifact.candidate_features
    }
    if content["activation_features"] != expected_features:
        raise BoundProbeError("cohort activation content hashes differ from binding")
    expected_artifacts = [
        identity.to_dict() for identity in artifact.rollout.raw_artifacts
    ]
    if payload["episodes"] != expected_artifacts:
        raise BoundProbeError("cohort raw identities differ from rollout receipt")
    if payload["invalid_reset_episode_ids"] != list(
        artifact.rollout.invalid_episode_ids
    ):
        raise BoundProbeError("cohort invalid-reset set differs from rollout receipt")
    if any(row.control_step % stride for row in artifact.rows):
        raise BoundProbeError("training control step violates cohort stride")


def _bound_from_metadata(value: Any) -> BoundProbeArtifact:
    metadata = _mapping(value, "bound_probe")
    _exact_keys(
        metadata,
        {
            "schema_version",
            "format",
            "kind",
            "numerical_probe",
            "calibration_inputs",
            "training",
        },
        "bound_probe",
    )
    if (
        metadata["schema_version"] != BOUND_PROBE_SCHEMA_VERSION
        or metadata["format"] != BOUND_PROBE_FORMAT
        or metadata["kind"] != "bound_probe"
    ):
        raise BoundProbeError("unsupported bound-probe schema/format/kind")
    numerical = _mapping(metadata["numerical_probe"], "numerical_probe")
    _exact_keys(numerical, {"sha256", "metadata"}, "numerical_probe")
    try:
        probe = probe_artifact_from_metadata(numerical["metadata"])
    except (ProbeError, TypeError, ValueError) as exc:
        raise BoundProbeError(f"invalid numerical probe: {exc}") from exc
    if _sha256(numerical["sha256"], "numerical_probe.sha256") != probe.sha256():
        raise BoundProbeError("numerical probe SHA-256 differs from its metadata")

    inputs = _mapping(metadata["calibration_inputs"], "calibration_inputs")
    _exact_keys(
        inputs,
        {
            "split",
            "episode_manifest_sha256",
            "rollout_receipt_sha256",
            "rollout_receipt",
            "cohort_manifest_sha256",
            "cohort_manifest",
            "source",
            "configuration_identity",
            "valid_episode_ids",
            "invalid_reset_episode_ids",
            "raw_artifacts",
        },
        "calibration_inputs",
    )
    try:
        rollout = rollout_receipt_from_metadata(inputs["rollout_receipt"])
    except (AllocationError, TypeError, ValueError) as exc:
        raise BoundProbeError(f"invalid rollout receipt: {exc}") from exc
    cohort_manifest = CohortManifest(
        _mapping(inputs["cohort_manifest"], "calibration_inputs.cohort_manifest")
    )
    configuration_identity = _mapping(
        inputs["configuration_identity"], "configuration_identity"
    )
    _exact_keys(
        configuration_identity,
        {
            "episode_manifest_sha256",
            "full_config_sha256",
            "scoring_source_sha256",
        },
        "configuration_identity",
    )

    training = _mapping(metadata["training"], "training")
    _exact_keys(
        training,
        {
            "selected_candidate",
            "rows",
            "episodes",
            "base_init_state_ids",
            "symmetry_order",
            "row_alignment_sha256",
            "theta_rel_sha256",
            "failure_label_sha256",
            "row_alignment",
            "candidate_features",
            "derivation",
        },
        "training",
    )
    raw_rows = _list(training["row_alignment"], "training.row_alignment")
    if len(raw_rows) > _MAX_TRAINING_ROWS:
        raise BoundProbeError("training row count exceeds the frozen limit")
    rows = tuple(_row_from_metadata(item, index) for index, item in enumerate(raw_rows))
    raw_features = _list(training["candidate_features"], "candidate_features")
    identities = tuple(
        _candidate_from_metadata(item, index) for index, item in enumerate(raw_features)
    )
    result = BoundProbeArtifact._validated(
        probe=probe,
        rollout=rollout,
        cohort_manifest=cohort_manifest,
        rows=rows,
        candidate_features=identities,
        theta_rel_sha256=_sha256(
            training["theta_rel_sha256"], "training.theta_rel_sha256"
        ),
        failure_label_sha256=_sha256(
            training["failure_label_sha256"], "training.failure_label_sha256"
        ),
        config_sha256=_sha256(
            configuration_identity["full_config_sha256"],
            "configuration_identity.full_config_sha256",
        ),
        code_sha256=_sha256(
            configuration_identity["scoring_source_sha256"],
            "configuration_identity.scoring_source_sha256",
        ),
    )
    if training["derivation"] != "exact_refit_matches_numerical_probe":
        raise BoundProbeError("bound probe derivation marker differs")
    if result.to_metadata() != metadata:
        raise BoundProbeError("bound probe contains inconsistent derived metadata")
    return result


def _row_from_metadata(value: Any, index: int) -> ProbeTrainingRow:
    item = _mapping(value, f"row_alignment[{index}]")
    _exact_keys(
        item,
        {"episode_id", "base_init_state_id", "control_step"},
        f"row_alignment[{index}]",
    )
    return ProbeTrainingRow(
        episode_id=_safe_name(item["episode_id"], f"row_alignment[{index}].episode_id"),
        base_init_state_id=_integer(
            item["base_init_state_id"], f"row_alignment[{index}].base_init_state_id"
        ),
        control_step=_integer(
            item["control_step"], f"row_alignment[{index}].control_step"
        ),
    )


def _candidate_from_metadata(value: Any, index: int) -> CandidateFeatureIdentity:
    item = _mapping(value, f"candidate_features[{index}]")
    expected = {"candidate", "rows", "width", "dtype", "logical_sha256"}
    _exact_keys(item, expected, f"candidate_features[{index}]")
    return CandidateFeatureIdentity(**{name: item[name] for name in expected})


def _logical_array_sha256(value: NDArray[Any]) -> str:
    return probe_cohort_array_sha256(value)


def _cohort_training_content(manifest: CohortManifest) -> dict[str, Any]:
    payload = _mapping(_plain_json(manifest.payload), "cohort_manifest")
    return _mapping(payload.get("training_content"), "cohort_manifest.training_content")


def _training_content_from_cohort(cohort: ProbeCohort) -> dict[str, Any]:
    return {
        "row_count": cohort.samples.n_rows,
        "episode_id_sha256": _logical_array_sha256(cohort.episode_id),
        "base_init_state_id_sha256": _logical_array_sha256(cohort.base_init_state_id),
        "control_step_sha256": _logical_array_sha256(cohort.control_step),
        "theta_rel_sha256": _logical_array_sha256(cohort.theta_rel),
        "failure_label_sha256": _logical_array_sha256(cohort.failure_label),
        "activation_features": {
            candidate: {
                "shape": list(cohort.activation_features[candidate].shape),
                "dtype": cohort.activation_features[candidate].dtype.str,
                "logical_sha256": _logical_array_sha256(
                    cohort.activation_features[candidate]
                ),
            }
            for candidate in ACTIVATION_CANDIDATES
        },
    }


def _row_alignment_sha256(rows: tuple[ProbeTrainingRow, ...]) -> str:
    canonical = _canonical_json([row.to_dict() for row in rows])
    digest = hashlib.sha256()
    digest.update(_ROW_DOMAIN)
    digest.update(len(canonical).to_bytes(8, "big"))
    digest.update(canonical)
    return digest.hexdigest()


def _copy_vector(value: Any, dtype: Any, where: str) -> NDArray[Any]:
    raw = np.asarray(value)
    expected = np.dtype(dtype)
    dtype_matches = (
        raw.dtype.kind == expected.kind
        if expected.kind in "US"
        else raw.dtype == expected
    )
    if raw.ndim != 1 or not dtype_matches:
        raise BoundProbeError(f"{where} must be a one-dimensional {expected} array")
    return np.array(raw, dtype=raw.dtype, copy=True, order="C")


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise BoundProbeError("bound probe is not finite canonical JSON") from exc


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(nested) for nested in value]
    return value


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BoundProbeError(f"{where} must be a JSON object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise BoundProbeError(f"{where} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        raise BoundProbeError(
            f"{where} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _integer(value: Any, where: str) -> int:
    if type(value) is not int:
        raise BoundProbeError(f"{where} must be an integer")
    return value


def _safe_name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise BoundProbeError(f"{where} must be a safe nonempty name")
    return value


def _sha256(value: Any, where: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise BoundProbeError(f"{where} must be a lowercase SHA-256")
    return value


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BoundProbeError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise BoundProbeError(f"non-finite JSON constant {value!r}")


def _absolute(value: str | Path) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    except (TypeError, ValueError, OSError) as exc:
        raise BoundProbeError("artifact path is invalid") from exc


def _has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise BoundProbeError(
                f"cannot inspect path component {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode):
            return True
    return False


def _safe_root(value: str | Path) -> Path:
    root = _absolute(value)
    if any(part in {"configs", "locks"} for part in root.parts):
        raise BoundProbeError("bound probes may not be written under config/lock paths")
    if _has_symlink_component(root):
        raise BoundProbeError("output path may not contain a symlink component")
    if root.exists() and not root.is_dir():
        raise BoundProbeError("output root must be a directory")
    root.mkdir(parents=True, exist_ok=True)
    if _has_symlink_component(root):
        raise BoundProbeError("output path may not contain a symlink component")
    return root


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_failed_destination(
    root: Path,
    digest: str,
    *,
    destination_inode: tuple[int, int],
    payload_inode: tuple[int, int],
) -> None:
    """Remove only the failed directory and payload created by this writer."""

    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    root_descriptor: int | None = None
    destination_descriptor: int | None = None
    try:
        root_descriptor = os.open(root, directory_flags)
        destination_descriptor = os.open(
            digest, directory_flags, dir_fd=root_descriptor
        )
        if _inode_identity(os.fstat(destination_descriptor)) != destination_inode:
            return
        entries = os.listdir(destination_descriptor)
        if entries == [_FILENAME]:
            payload = os.stat(
                _FILENAME,
                dir_fd=destination_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(payload.st_mode)
                or _inode_identity(payload) != payload_inode
            ):
                return
            os.unlink(_FILENAME, dir_fd=destination_descriptor)
            entries = os.listdir(destination_descriptor)
        if entries:
            return
        os.fsync(destination_descriptor)
        os.close(destination_descriptor)
        destination_descriptor = None
        os.rmdir(digest, dir_fd=root_descriptor)
        os.fsync(root_descriptor)
    except OSError:
        # Cleanup is best effort and must never hide the publication failure.
        return
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        if root_descriptor is not None:
            os.close(root_descriptor)


def _inode_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


__all__ = [
    "BOUND_PROBE_FORMAT",
    "BOUND_PROBE_SCHEMA_VERSION",
    "BoundProbeArtifact",
    "BoundProbeError",
    "CandidateFeatureIdentity",
    "ProbeTrainingRow",
    "bind_probe_artifact",
    "load_bound_probe_artifact",
    "validate_bound_probe_artifact",
    "write_bound_probe_artifact",
]
