"""Fail-closed provenance hashes for deterministic score sidecars.

The two file sets below are deliberately explicit.  Adding a file to the
repository cannot silently change either digest, while removing, replacing, or
symlinking any allowlisted file makes hashing fail.  Each digest is domain
separated and frames the file count, UTF-8 path length, path, byte length, and
contents, so neither concatenation ambiguity nor a config/source collision is
possible.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Final, Protocol

from .config import ProtocolConfig
from .failure_events import ArtifactIdentity
from .scoring import ContentLinks

FROZEN_CONFIG_FILES: Final[tuple[str, ...]] = (
    "start.md",
    "PREREG.md",
    "AMENDMENTS.md",
    "environment.lock",
    "configs/perturbations.yaml",
    "configs/split_protocol.yaml",
    "configs/task_order.yaml",
)

SCORING_SOURCE_FILES: Final[tuple[str, ...]] = (
    "src/mech_int_vla/allocation.py",
    "src/mech_int_vla/artifacts.py",
    "src/mech_int_vla/config.py",
    "src/mech_int_vla/determinism.py",
    "src/mech_int_vla/feature_artifacts.py",
    "src/mech_int_vla/feature_pipeline.py",
    "src/mech_int_vla/features.py",
    "src/mech_int_vla/instrumentation.py",
    "src/mech_int_vla/libero_runtime.py",
    "src/mech_int_vla/manifest.py",
    "src/mech_int_vla/probes.py",
    "src/mech_int_vla/probe_artifacts.py",
    "src/mech_int_vla/provenance.py",
    "src/mech_int_vla/rollout.py",
    "src/mech_int_vla/scoring.py",
    "src/mech_int_vla/scoring_runtime.py",
    "src/mech_int_vla/snapshots.py",
)

_CONFIG_DOMAIN: Final[bytes] = b"mech-int-vla:frozen-config:v1"
_SOURCE_DOMAIN: Final[bytes] = b"mech-int-vla:scoring-source:v1"
_FRAME_BYTES: Final[int] = 8


class ProvenanceError(ValueError):
    """Raised when provenance inputs cannot be hashed without ambiguity."""


class _Digest(Protocol):
    def update(self, value: bytes) -> None: ...


def frozen_config_sha256(repo_root: str | Path) -> str:
    """Hash the exact immutable protocol/configuration file allowlist."""

    return _allowlisted_sha256(repo_root, FROZEN_CONFIG_FILES, _CONFIG_DOMAIN)


def scoring_source_sha256(repo_root: str | Path) -> str:
    """Hash source that determines score sidecars and downstream features."""

    return _allowlisted_sha256(repo_root, SCORING_SOURCE_FILES, _SOURCE_DOMAIN)


def content_links_for(
    raw: ArtifactIdentity,
    bound_probe: object,
    repo_root: str | Path,
    *,
    protocol: ProtocolConfig,
) -> ContentLinks:
    """Compute all links for one raw artifact, frozen probe, and repository.

    Config and source hashes are intentionally computed here rather than
    accepted from a caller.  This prevents stale or fabricated repository
    digests from being attached to a score sidecar.
    """

    if not isinstance(raw, ArtifactIdentity):
        raise ProvenanceError("raw must be a validated ArtifactIdentity")
    # Local import avoids allocation -> provenance -> bound-probe import cycles.
    from .probe_artifacts import (
        BoundProbeArtifact,
        BoundProbeError,
        validate_bound_probe_artifact,
    )

    if not isinstance(bound_probe, BoundProbeArtifact):
        raise ProvenanceError("probe must be a validated BoundProbeArtifact")
    try:
        validate_bound_probe_artifact(
            bound_probe, protocol=protocol, repo_root=repo_root
        )
    except (BoundProbeError, TypeError, ValueError) as exc:
        raise ProvenanceError(f"bound probe validation failed: {exc}") from exc
    probe_sha256 = bound_probe.sha256
    if not _is_lower_sha256(probe_sha256):
        raise ProvenanceError("probe produced an invalid SHA-256 digest")
    return ContentLinks(
        raw_metadata_sha256=raw.metadata_sha256,
        raw_trajectory_sha256=raw.trajectory_sha256,
        probe_sha256=probe_sha256,
        config_sha256=frozen_config_sha256(repo_root),
        code_sha256=scoring_source_sha256(repo_root),
    )


def _allowlisted_sha256(
    repo_root: str | Path,
    relative_paths: tuple[str, ...],
    domain: bytes,
) -> str:
    root = _repo_root(repo_root)
    paths = _validate_allowlist(relative_paths)
    digest = hashlib.sha256()
    _frame(digest, domain)
    digest.update(len(paths).to_bytes(_FRAME_BYTES, "big"))
    before = tuple(
        _file_identity(_inspect_regular_file(root, relative)[1]) for relative in paths
    )
    contents = tuple(_read_regular_file(root, relative) for relative in paths)
    after = tuple(
        _file_identity(_inspect_regular_file(root, relative)[1]) for relative in paths
    )
    if before != after:
        raise ProvenanceError(
            "allowlisted files did not form one stable repository snapshot"
        )
    for relative, content in zip(paths, contents, strict=True):
        path_bytes = relative.encode("utf-8")
        _frame(digest, path_bytes)
        _frame(digest, content)
    result = digest.hexdigest()
    if not _is_lower_sha256(result):  # pragma: no cover - hashlib invariant
        raise ProvenanceError("hash implementation produced an invalid SHA-256")
    return result


def _repo_root(value: str | Path) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise ProvenanceError("repo_root must be a filesystem path") from exc
    if not isinstance(raw, (str, bytes)) or not raw:
        raise ProvenanceError("repo_root must be a non-empty filesystem path")
    try:
        root = Path(os.path.abspath(os.path.expanduser(raw)))
        root_stat = root.lstat()
    except (OSError, TypeError, ValueError) as exc:
        raise ProvenanceError(f"could not inspect repo_root: {exc}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ProvenanceError("repo_root must be a real directory, not a symlink")
    return root


def _validate_allowlist(relative_paths: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(relative_paths, tuple) or not relative_paths:
        raise ProvenanceError("file allowlist must be a non-empty tuple")
    if len(relative_paths) != len(set(relative_paths)):
        raise ProvenanceError("file allowlist must not contain duplicates")
    for relative in relative_paths:
        if not isinstance(relative, str) or not relative:
            raise ProvenanceError("allowlisted paths must be non-empty strings")
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ProvenanceError(
                f"allowlisted path escapes or is not canonical: {relative!r}"
            )
    return relative_paths


def _inspect_regular_file(root: Path, relative: str) -> tuple[Path, os.stat_result]:
    parts = PurePosixPath(relative).parts
    parent = root
    for component in parts[:-1]:
        parent = parent / component
        try:
            component_stat = parent.lstat()
        except OSError as exc:
            raise ProvenanceError(
                f"could not inspect allowlisted path {relative!r}: {exc}"
            ) from exc
        if stat.S_ISLNK(component_stat.st_mode) or not stat.S_ISDIR(
            component_stat.st_mode
        ):
            raise ProvenanceError(
                f"allowlisted path {relative!r} has a non-directory or symlink component"
            )

    path = root.joinpath(*parts)
    try:
        before = path.lstat()
    except OSError as exc:
        raise ProvenanceError(
            f"could not inspect allowlisted file {relative!r}: {exc}"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ProvenanceError(
            f"allowlisted file {relative!r} must be regular and not a symlink"
        )
    return path, before


def _read_regular_file(root: Path, relative: str) -> bytes:
    path, before = _inspect_regular_file(root, relative)

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            opened_before = os.fstat(descriptor)
            if not stat.S_ISREG(opened_before.st_mode):
                raise ProvenanceError(
                    f"allowlisted file {relative!r} must be a regular file"
                )
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            opened_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        _, after = _inspect_regular_file(root, relative)
    except ProvenanceError:
        raise
    except OSError as exc:
        raise ProvenanceError(
            f"could not read allowlisted file {relative!r} safely: {exc}"
        ) from exc

    identities = {
        _file_identity(before),
        _file_identity(opened_before),
        _file_identity(opened_after),
        _file_identity(after),
    }
    if len(identities) != 1 or stat.S_ISLNK(after.st_mode):
        raise ProvenanceError(f"allowlisted file {relative!r} changed while hashing")
    return b"".join(chunks)


def _frame(digest: _Digest, value: bytes) -> None:
    if len(value) >= 1 << (_FRAME_BYTES * 8):  # pragma: no cover - infeasible
        raise ProvenanceError("provenance frame is too large")
    digest.update(len(value).to_bytes(_FRAME_BYTES, "big"))
    digest.update(value)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "FROZEN_CONFIG_FILES",
    "SCORING_SOURCE_FILES",
    "ProvenanceError",
    "content_links_for",
    "frozen_config_sha256",
    "scoring_source_sha256",
]
