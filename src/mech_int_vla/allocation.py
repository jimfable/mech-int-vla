"""Fail-closed allocation audits and content-addressed receipts.

Allocation is protocol state, not a statistic.  This module proves that callers
supplied exactly the episodes frozen by an :class:`~mech_int_vla.manifest.Manifest`
and, later, exactly one score sidecar for each valid reset.  It never discovers
directories and never uses success or any other outcome to choose membership.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from .artifacts import RolloutArtifact
from .config import ProtocolConfig, ProtocolConfigError, SplitName, TaskSpec
from .failure_events import ArtifactIdentity
from .manifest import EpisodeSpec, Manifest, reconstruct_episode_manifest
from .provenance import (
    ProvenanceError,
    frozen_config_sha256,
    scoring_source_sha256,
)
from .scoring import LoadedScoringSidecar

ALLOCATION_SCHEMA_VERSION = 1
MAX_ALLOCATION_JSON_BYTES = 16 * 1024 * 1024
_SHA256_LENGTH = 64
_EXPECTED_EPISODES = {
    SplitName.DISCOVERY: 40,
    SplitName.CALIBRATION: 160,
    SplitName.LOCKED_TEST: 160,
}


class AllocationError(ValueError):
    """Raised when a manifest, artifact allocation, or receipt is inconsistent."""


@dataclass(frozen=True)
class AllocationSourceIdentity:
    """Frozen source identity shared by every raw artifact in one allocation."""

    split: SplitName
    task: TaskSpec
    policy_revision: str
    base_vlm_revision: str
    code_commit: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.split, SplitName):
            raise AllocationError("source split must be a SplitName")
        _validate_task(self.task, "source.task")
        for name in ("policy_revision", "base_vlm_revision", "code_commit"):
            _nonempty_string(getattr(self, name), f"source.{name}")
        _sha256(self.manifest_sha256, "source.manifest_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split.value,
            "task": _task_dict(self.task),
            "policy_revision": self.policy_revision,
            "base_vlm_revision": self.base_vlm_revision,
            "code_commit": self.code_commit,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class ScoringArtifactIdentity:
    """Physical score identity plus all content links that determined it."""

    episode_id: str
    metadata_sha256: str
    primitives_sha256: str
    raw_metadata_sha256: str
    raw_trajectory_sha256: str
    probe_sha256: str
    config_sha256: str
    code_sha256: str

    def __post_init__(self) -> None:
        _safe_name(self.episode_id, "score.episode_id")
        for name in (
            "metadata_sha256",
            "primitives_sha256",
            "raw_metadata_sha256",
            "raw_trajectory_sha256",
            "probe_sha256",
            "config_sha256",
            "code_sha256",
        ):
            _sha256(getattr(self, name), f"score.{name}")

    def to_dict(self) -> dict[str, str]:
        return {
            "episode_id": self.episode_id,
            "metadata_sha256": self.metadata_sha256,
            "primitives_sha256": self.primitives_sha256,
            "raw_metadata_sha256": self.raw_metadata_sha256,
            "raw_trajectory_sha256": self.raw_trajectory_sha256,
            "probe_sha256": self.probe_sha256,
            "config_sha256": self.config_sha256,
            "code_sha256": self.code_sha256,
        }


@dataclass(frozen=True)
class RolloutAllocationReceipt:
    """Proof that raw artifacts exactly cover a frozen episode manifest."""

    manifest: Manifest
    source: AllocationSourceIdentity
    valid_artifacts: tuple[ArtifactIdentity, ...]
    invalid_artifacts: tuple[ArtifactIdentity, ...]

    def __post_init__(self) -> None:
        _validate_manifest(self.manifest)
        if not isinstance(self.source, AllocationSourceIdentity):
            raise AllocationError("source must be an AllocationSourceIdentity")
        if self.source.manifest_sha256 != _manifest_sha256(self.manifest):
            raise AllocationError("source manifest hash differs from manifest")
        if (
            self.source.split is not self.manifest.split
            or self.source.task != self.manifest.task
        ):
            raise AllocationError("source split/task differs from manifest")
        manifest_policy = {item.policy_revision for item in self.manifest.episodes}
        manifest_code = {item.code_commit for item in self.manifest.episodes}
        if manifest_policy != {self.source.policy_revision}:
            raise AllocationError("source policy revision differs from manifest")
        if manifest_code != {self.source.code_commit}:
            raise AllocationError("source code commit differs from manifest")
        valid = _identity_tuple(self.valid_artifacts, "valid_artifacts")
        invalid = _identity_tuple(self.invalid_artifacts, "invalid_artifacts")
        all_identities = valid + invalid
        actual_ids = tuple(item.episode_id for item in all_identities)
        if len(actual_ids) != len(set(actual_ids)):
            raise AllocationError("raw identities contain duplicate episode IDs")
        expected_ids = tuple(
            sorted(episode.episode_id for episode in self.manifest.episodes)
        )
        if tuple(sorted(actual_ids)) != expected_ids:
            raise AllocationError(
                "raw identities do not exactly cover manifest episodes"
            )
        object.__setattr__(self, "valid_artifacts", valid)
        object.__setattr__(self, "invalid_artifacts", invalid)

    @property
    def attempted_count(self) -> int:
        return len(self.manifest.episodes)

    @property
    def valid_episode_ids(self) -> tuple[str, ...]:
        return tuple(item.episode_id for item in self.valid_artifacts)

    @property
    def invalid_episode_ids(self) -> tuple[str, ...]:
        return tuple(item.episode_id for item in self.invalid_artifacts)

    @property
    def invalid_fraction(self) -> float:
        return len(self.invalid_artifacts) / self.attempted_count

    @property
    def raw_artifacts(self) -> tuple[ArtifactIdentity, ...]:
        return tuple(
            sorted(
                self.valid_artifacts + self.invalid_artifacts,
                key=lambda item: item.episode_id,
            )
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": ALLOCATION_SCHEMA_VERSION,
            "kind": "rollout_allocation",
            "manifest": _plain(self.manifest.to_dict()),
            "source": self.source.to_dict(),
            "allocation": {
                "membership_rule": "manifest_and_valid_reset_only",
                "outcome_conditioned": False,
                "attempted_count": self.attempted_count,
                "valid_count": len(self.valid_artifacts),
                "invalid_count": len(self.invalid_artifacts),
                "invalid_fraction": self.invalid_fraction,
                "valid_artifacts": [item.to_dict() for item in self.valid_artifacts],
                "invalid_artifacts": [
                    item.to_dict() for item in self.invalid_artifacts
                ],
            },
        }

    def canonical_json(self) -> bytes:
        return _canonical(self.to_metadata())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()


@dataclass(frozen=True)
class ScoreAllocationReceipt:
    """Proof that score sidecars exactly cover valid raw episode identities."""

    rollout: RolloutAllocationReceipt
    probe_sha256: str
    config_sha256: str
    code_sha256: str
    score_artifacts: tuple[ScoringArtifactIdentity, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rollout, RolloutAllocationReceipt):
            raise AllocationError("rollout must be a RolloutAllocationReceipt")
        for name in ("probe_sha256", "config_sha256", "code_sha256"):
            _sha256(getattr(self, name), name)
        try:
            scores = tuple(self.score_artifacts)
        except TypeError as exc:
            raise AllocationError("score_artifacts must be an iterable") from exc
        if not all(isinstance(item, ScoringArtifactIdentity) for item in scores):
            raise AllocationError("score_artifacts must contain score identities")
        if scores != tuple(sorted(scores, key=lambda item: item.episode_id)):
            raise AllocationError("score_artifacts must be sorted by episode ID")
        ids = tuple(item.episode_id for item in scores)
        if len(ids) != len(set(ids)):
            raise AllocationError("score identities contain duplicate episode IDs")
        if ids != self.rollout.valid_episode_ids:
            raise AllocationError("scores do not exactly cover valid raw episode IDs")
        raw = {item.episode_id: item for item in self.rollout.valid_artifacts}
        for item in scores:
            expected = raw[item.episode_id]
            if (
                item.raw_metadata_sha256 != expected.metadata_sha256
                or item.raw_trajectory_sha256 != expected.trajectory_sha256
                or item.probe_sha256 != self.probe_sha256
                or item.config_sha256 != self.config_sha256
                or item.code_sha256 != self.code_sha256
            ):
                raise AllocationError("score identity has inconsistent content links")
        object.__setattr__(self, "score_artifacts", scores)

    @property
    def scored_episode_ids(self) -> tuple[str, ...]:
        return tuple(item.episode_id for item in self.score_artifacts)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": ALLOCATION_SCHEMA_VERSION,
            "kind": "score_allocation",
            "rollout_receipt": self.rollout.to_metadata(),
            "rollout_receipt_sha256": self.rollout.sha256,
            "source": {
                "probe_sha256": self.probe_sha256,
                "config_sha256": self.config_sha256,
                "code_sha256": self.code_sha256,
            },
            "allocation": {
                "membership_rule": "valid_raw_episode_ids_only",
                "outcome_conditioned": False,
                "scored_count": len(self.score_artifacts),
                "score_artifacts": [item.to_dict() for item in self.score_artifacts],
            },
        }

    def canonical_json(self) -> bytes:
        return _canonical(self.to_metadata())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()


AllocationReceipt: TypeAlias = RolloutAllocationReceipt | ScoreAllocationReceipt


def audit_rollout_allocation(
    manifest: Manifest,
    raw_artifacts: Sequence[RolloutArtifact],
    *,
    protocol: ProtocolConfig,
    repo_root: str | Path,
) -> RolloutAllocationReceipt:
    """Prove exact raw coverage and partition only on the valid-reset bit."""

    _validate_manifest_against_protocol(manifest, protocol, repo_root)
    try:
        artifacts = tuple(raw_artifacts)
    except TypeError as exc:
        raise AllocationError("raw_artifacts must be an iterable") from exc
    if not all(isinstance(item, RolloutArtifact) for item in artifacts):
        raise AllocationError("raw_artifacts must contain RolloutArtifact values")
    expected = {episode.episode_id: episode for episode in manifest.episodes}
    observed_ids = tuple(item.episode_id for item in artifacts)
    duplicates = sorted({item for item in observed_ids if observed_ids.count(item) > 1})
    if duplicates:
        raise AllocationError(f"duplicate raw episode IDs: {duplicates}")
    missing = sorted(set(expected) - set(observed_ids))
    extra = sorted(set(observed_ids) - set(expected))
    if missing or extra:
        raise AllocationError(
            f"raw episode set differs: missing={missing}, extra={extra}"
        )

    by_id = {item.episode_id: item for item in artifacts}
    base_vlm_revisions: set[str] = set()
    valid: list[ArtifactIdentity] = []
    invalid: list[ArtifactIdentity] = []
    for episode_id in sorted(expected):
        episode = expected[episode_id]
        raw = by_id[episode_id]
        _audit_raw_metadata(manifest, episode, raw)
        try:
            base_vlm = raw.metadata["model"]["base_vlm_revision"]
            valid_reset = raw.metadata["validity"]["valid"]
        except (KeyError, TypeError) as exc:
            raise AllocationError(
                f"{episode_id}: missing model/validity metadata"
            ) from exc
        base_vlm_revisions.add(
            _nonempty_string(base_vlm, f"{episode_id}.base_vlm_revision")
        )
        if type(valid_reset) is not bool:
            raise AllocationError(f"{episode_id}: valid-reset flag must be boolean")
        try:
            identity = ArtifactIdentity(
                episode_id=episode_id,
                metadata_sha256=raw.hashes.metadata_sha256,
                trajectory_sha256=raw.hashes.trajectory_sha256,
            )
        except (AttributeError, ValueError) as exc:
            raise AllocationError(
                f"{episode_id}: invalid raw content identity"
            ) from exc
        (valid if valid_reset else invalid).append(identity)
    if len(base_vlm_revisions) != 1:
        raise AllocationError("raw artifacts have mixed base-VLM revisions")

    first = manifest.episodes[0]
    return RolloutAllocationReceipt(
        manifest=manifest,
        source=AllocationSourceIdentity(
            split=manifest.split,
            task=manifest.task,
            policy_revision=first.policy_revision,
            base_vlm_revision=next(iter(base_vlm_revisions)),
            code_commit=first.code_commit,
            manifest_sha256=_manifest_sha256(manifest),
        ),
        valid_artifacts=tuple(valid),
        invalid_artifacts=tuple(invalid),
    )


def audit_score_allocation(
    receipt: RolloutAllocationReceipt,
    sidecars: Sequence[LoadedScoringSidecar],
    bound_probe: object,
    *,
    protocol: ProtocolConfig,
    repo_root: str | Path,
) -> ScoreAllocationReceipt:
    """Prove exact score coverage of valid raws and every content link."""

    if not isinstance(receipt, RolloutAllocationReceipt):
        raise AllocationError("receipt must be a RolloutAllocationReceipt")
    from .probe_artifacts import (
        BoundProbeArtifact,
        BoundProbeError,
        validate_bound_probe_artifact,
    )

    if not isinstance(bound_probe, BoundProbeArtifact):
        raise AllocationError("probe must be a validated BoundProbeArtifact")
    try:
        validate_bound_probe_artifact(
            bound_probe, protocol=protocol, repo_root=repo_root
        )
    except BoundProbeError as exc:
        raise AllocationError(f"bound probe is invalid: {exc}") from exc
    calibration = bound_probe.rollout
    if receipt.source.split is SplitName.CALIBRATION:
        if receipt.sha256 != calibration.sha256:
            raise AllocationError(
                "Calibration score allocation differs from bound probe allocation"
            )
    elif (
        receipt.source.task != calibration.source.task
        or receipt.source.policy_revision != calibration.source.policy_revision
        or receipt.source.base_vlm_revision != calibration.source.base_vlm_revision
    ):
        raise AllocationError("score allocation differs from bound Calibration inputs")
    probe_sha = bound_probe.sha256
    expected_config_hash, expected_code_hash = _trusted_scoring_sources(repo_root)
    try:
        scores = tuple(sidecars)
    except TypeError as exc:
        raise AllocationError("sidecars must be an iterable") from exc
    if not all(isinstance(item, LoadedScoringSidecar) for item in scores):
        raise AllocationError("sidecars must contain LoadedScoringSidecar values")
    ids: list[str] = []
    for sidecar in scores:
        try:
            ids.append(_safe_name(sidecar.metadata["episode_id"], "score.episode_id"))
        except (KeyError, TypeError) as exc:
            raise AllocationError("score sidecar is missing episode_id") from exc
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise AllocationError(f"duplicate score episode IDs: {duplicates}")
    expected_ids = set(receipt.valid_episode_ids)
    missing = sorted(expected_ids - set(ids))
    extra = sorted(set(ids) - expected_ids)
    if missing or extra:
        raise AllocationError(
            f"score episode set differs: missing={missing}, extra={extra}"
        )
    if not scores:
        raise AllocationError(
            "cannot establish score source identities without valid scores"
        )

    raw = {item.episode_id: item for item in receipt.valid_artifacts}
    identities: list[ScoringArtifactIdentity] = []
    config_hashes: set[str] = set()
    code_hashes: set[str] = set()
    for sidecar in scores:
        episode_id = str(sidecar.metadata["episode_id"])
        if sidecar.metadata.get("split") != receipt.source.split.value:
            raise AllocationError(
                f"{episode_id}: score split differs from rollout split"
            )
        links = sidecar.metadata.get("links")
        if not isinstance(links, Mapping):
            raise AllocationError(f"{episode_id}: score links must be a mapping")
        expected_raw = raw[episode_id]
        required = {
            "raw_metadata_sha256",
            "raw_trajectory_sha256",
            "probe_sha256",
            "config_sha256",
            "code_sha256",
        }
        if set(links) != required:
            raise AllocationError(f"{episode_id}: score link keys differ")
        expected_links = {
            "raw_metadata_sha256": expected_raw.metadata_sha256,
            "raw_trajectory_sha256": expected_raw.trajectory_sha256,
            "probe_sha256": probe_sha,
        }
        if any(links[name] != value for name, value in expected_links.items()):
            raise AllocationError(f"{episode_id}: raw/probe score links differ")
        config_hash = _sha256(links["config_sha256"], f"{episode_id}.config_sha256")
        code_hash = _sha256(links["code_sha256"], f"{episode_id}.code_sha256")
        if config_hash != expected_config_hash or code_hash != expected_code_hash:
            raise AllocationError(
                f"{episode_id}: config/code score links differ from repository"
            )
        config_hashes.add(config_hash)
        code_hashes.add(code_hash)
        metadata_hash = _sha256(
            sidecar.metadata_sha256, f"{episode_id}.metadata_sha256"
        )
        primitives_hash = _sha256(
            sidecar.primitives_sha256, f"{episode_id}.primitives_sha256"
        )
        try:
            declared_primitive = sidecar.metadata["files"]["primitives_sha256"]
        except (KeyError, TypeError) as exc:
            raise AllocationError(
                f"{episode_id}: missing primitives declaration"
            ) from exc
        if declared_primitive != primitives_hash:
            raise AllocationError(
                f"{episode_id}: primitives identity differs from metadata"
            )
        identities.append(
            ScoringArtifactIdentity(
                episode_id=episode_id,
                metadata_sha256=metadata_hash,
                primitives_sha256=primitives_hash,
                raw_metadata_sha256=expected_raw.metadata_sha256,
                raw_trajectory_sha256=expected_raw.trajectory_sha256,
                probe_sha256=probe_sha,
                config_sha256=config_hash,
                code_sha256=code_hash,
            )
        )
    if len(config_hashes) != 1 or len(code_hashes) != 1:
        raise AllocationError("score sidecars have mixed config/code source hashes")
    return ScoreAllocationReceipt(
        rollout=receipt,
        probe_sha256=probe_sha,
        config_sha256=next(iter(config_hashes)),
        code_sha256=next(iter(code_hashes)),
        score_artifacts=tuple(sorted(identities, key=lambda item: item.episode_id)),
    )


def revalidate_rollout_receipt(
    receipt: RolloutAllocationReceipt,
    *,
    protocol: ProtocolConfig,
    repo_root: str | Path,
) -> RolloutAllocationReceipt:
    """Revalidate a raw-allocation receipt at a downstream trust boundary.

    This public boundary is used by artifacts derived after rollout collection.
    It regenerates the complete manifest from the frozen protocol and reconstructs
    the receipt so downstream modules do not need private allocation parsers or
    caller-asserted topology hashes.
    """

    if not isinstance(receipt, RolloutAllocationReceipt):
        raise AllocationError("receipt must be a RolloutAllocationReceipt")
    _revalidate_receipt(receipt, protocol, repo_root)
    return RolloutAllocationReceipt(
        manifest=receipt.manifest,
        source=receipt.source,
        valid_artifacts=receipt.valid_artifacts,
        invalid_artifacts=receipt.invalid_artifacts,
    )


def revalidate_score_receipt(
    receipt: ScoreAllocationReceipt,
    bound_probe: object,
    *,
    protocol: ProtocolConfig,
    repo_root: str | Path,
) -> ScoreAllocationReceipt:
    """Revalidate a score allocation and its exact bound-probe context.

    A score receipt is not a sufficient downstream capability by itself: its
    probe digest could otherwise be paired with an unrelated object, and its
    rollout could be replaced with a compatible-looking subset.  This boundary
    therefore validates the current scoring sources, the trusted bound probe,
    and the target-to-Calibration relationship before reconstructing the
    immutable receipt.
    """

    if not isinstance(receipt, ScoreAllocationReceipt):
        raise AllocationError("receipt must be a ScoreAllocationReceipt")
    from .probe_artifacts import (
        BoundProbeArtifact,
        BoundProbeError,
        validate_bound_probe_artifact,
    )

    if not isinstance(bound_probe, BoundProbeArtifact):
        raise AllocationError("probe must be a validated BoundProbeArtifact")
    try:
        validated_probe = validate_bound_probe_artifact(
            bound_probe, protocol=protocol, repo_root=repo_root
        )
    except (BoundProbeError, TypeError, ValueError) as exc:
        raise AllocationError(f"bound probe is invalid: {exc}") from exc

    _revalidate_receipt(receipt, protocol, repo_root)
    if receipt.probe_sha256 != validated_probe.sha256:
        raise AllocationError("score receipt probe digest differs from bound probe")

    calibration = revalidate_rollout_receipt(
        validated_probe.rollout,
        protocol=protocol,
        repo_root=repo_root,
    )
    target = receipt.rollout
    if target.source.split is SplitName.CALIBRATION:
        if target.sha256 != calibration.sha256:
            raise AllocationError(
                "Calibration score receipt differs from bound probe allocation"
            )
    elif target.source.split is SplitName.LOCKED_TEST:
        if (
            target.source.task != calibration.source.task
            or target.source.policy_revision != calibration.source.policy_revision
            or target.source.base_vlm_revision != calibration.source.base_vlm_revision
        ):
            raise AllocationError(
                "Locked Test score receipt is incompatible with bound Calibration"
            )
    else:
        raise AllocationError(
            "feature score receipt must target Calibration or Locked Test"
        )

    return ScoreAllocationReceipt(
        rollout=target,
        probe_sha256=receipt.probe_sha256,
        config_sha256=receipt.config_sha256,
        code_sha256=receipt.code_sha256,
        score_artifacts=receipt.score_artifacts,
    )


def rollout_receipt_from_metadata(value: Any) -> RolloutAllocationReceipt:
    """Strictly reconstruct the intrinsic fields of a raw allocation receipt.

    Call :func:`revalidate_rollout_receipt` immediately afterward when crossing a
    repository/protocol trust boundary.
    """

    return _rollout_receipt_from_metadata(value)


def write_episode_manifest(
    manifest: Manifest,
    output_root: str | Path,
    *,
    protocol: ProtocolConfig,
    repo_root: str | Path,
) -> Path:
    """Atomically publish a canonical, content-addressed episode manifest."""

    _validate_manifest_against_protocol(manifest, protocol, repo_root)
    return _write_payload(
        _canonical(_plain(manifest.to_dict())), output_root, "manifest.json"
    )


def load_episode_manifest(
    path: str | Path,
    *,
    protocol: ProtocolConfig,
    repo_root: str | Path,
    expected_sha256: str | None = None,
) -> Manifest:
    """Load a canonical episode manifest with exact typed reconstruction."""

    payload = _read_payload(path, "manifest.json", expected_sha256)
    manifest = _manifest_from_metadata(payload)
    _validate_manifest_against_protocol(manifest, protocol, repo_root)
    return manifest


def write_allocation_receipt(
    receipt: AllocationReceipt,
    output_root: str | Path,
    *,
    protocol: ProtocolConfig,
    repo_root: str | Path,
) -> Path:
    """Atomically publish a canonical allocation receipt."""

    if not isinstance(receipt, (RolloutAllocationReceipt, ScoreAllocationReceipt)):
        raise AllocationError("receipt has an unsupported type")
    _revalidate_receipt(receipt, protocol, repo_root)
    return _write_payload(receipt.canonical_json(), output_root, "receipt.json")


def load_allocation_receipt(
    path: str | Path,
    *,
    protocol: ProtocolConfig,
    repo_root: str | Path,
    expected_sha256: str | None = None,
) -> AllocationReceipt:
    """Load and exactly reconstruct either allocation receipt kind."""

    payload, canonical = _read_payload_with_bytes(path, "receipt.json", expected_sha256)
    if not isinstance(payload, dict):
        raise AllocationError("receipt root must be a JSON object")
    kind = payload.get("kind")
    if kind == "rollout_allocation":
        result: AllocationReceipt = _rollout_receipt_from_metadata(payload)
    elif kind == "score_allocation":
        result = _score_receipt_from_metadata(payload)
    else:
        raise AllocationError("receipt kind is unsupported")
    _revalidate_receipt(result, protocol, repo_root)
    if result.canonical_json() != canonical:
        raise AllocationError(
            "receipt JSON is not canonical or exactly reconstructable"
        )
    return result


# Short, unsurprising aliases for callers that treat the episode manifest as the
# protocol manifest rather than an allocation receipt.
write_manifest = write_episode_manifest
load_manifest = load_episode_manifest


def _audit_raw_metadata(
    manifest: Manifest, episode: EpisodeSpec, raw: RolloutArtifact
) -> None:
    where = episode.episode_id
    metadata = raw.metadata
    try:
        observed_episode = metadata["episode"]
        observed_task = metadata["task"]
        observed_condition = metadata["condition"]
        model = metadata["model"]
        task_language = metadata["task_language"]
    except (KeyError, TypeError) as exc:
        raise AllocationError(f"{where}: missing allocation metadata") from exc
    if _plain(observed_episode) != _plain(episode.to_dict()):
        raise AllocationError(f"{where}: episode metadata differs from manifest")
    if _plain(observed_task) != _task_dict(manifest.task):
        raise AllocationError(f"{where}: task metadata differs from manifest")
    if task_language != manifest.task.language:
        raise AllocationError(f"{where}: task language differs from manifest")
    expected_condition = {
        "name": episode.condition_name,
        "family": episode.condition_family,
        "index": episode.condition_index,
        "parameters": _plain(episode.condition_parameters),
    }
    if _plain(observed_condition) != expected_condition:
        raise AllocationError(f"{where}: condition metadata differs from manifest")
    if (
        not isinstance(model, Mapping)
        or model.get("policy_revision") != episode.policy_revision
    ):
        raise AllocationError(f"{where}: policy revision differs from manifest")


def _validate_manifest(manifest: Any) -> Manifest:
    if not isinstance(manifest, Manifest):
        raise AllocationError("manifest must be a Manifest")
    if type(manifest.schema_version) is not int or manifest.schema_version != 1:
        raise AllocationError("manifest schema version must be 1")
    if not isinstance(manifest.split, SplitName):
        raise AllocationError("manifest split must be a SplitName")
    _validate_task(manifest.task, "manifest.task")
    if type(manifest.episodes) is not tuple:
        raise AllocationError("manifest episodes must be an immutable tuple")
    episodes = manifest.episodes
    expected_count = _EXPECTED_EPISODES[manifest.split]
    if len(episodes) != expected_count:
        raise AllocationError(
            f"{manifest.split.value} manifest must contain exactly {expected_count} episodes"
        )
    if not all(isinstance(item, EpisodeSpec) for item in episodes):
        raise AllocationError("manifest episodes must contain EpisodeSpec values")
    ids = tuple(item.episode_id for item in episodes)
    if len(ids) != len(set(ids)):
        raise AllocationError("manifest contains duplicate episode IDs")
    if len({item.reset_seed for item in episodes}) != len(episodes):
        raise AllocationError("manifest contains duplicate reset seeds")
    policies = {item.policy_revision for item in episodes}
    commits = {item.code_commit for item in episodes}
    for episode in episodes:
        if episode.split is not manifest.split:
            raise AllocationError(f"{episode.episode_id}: split differs from manifest")
        if (episode.suite, episode.task_id, episode.task_rank) != (
            manifest.task.suite,
            manifest.task.task_id,
            manifest.task.rank,
        ):
            raise AllocationError(f"{episode.episode_id}: task differs from manifest")
        _nonempty_string(episode.policy_revision, "episode.policy_revision")
        _nonempty_string(episode.code_commit, "episode.code_commit")
    if len(policies) != 1 or len(commits) != 1:
        raise AllocationError("manifest has mixed policy/code source identities")
    try:
        _canonical(_plain(manifest.to_dict()))
    except (TypeError, ValueError) as exc:
        raise AllocationError("manifest is not finite canonical JSON") from exc
    return manifest


def _validate_manifest_against_protocol(
    manifest: Any,
    protocol: ProtocolConfig,
    repo_root: str | Path,
) -> Manifest:
    """Regenerate and compare one manifest at every trust boundary."""

    validated = _validate_manifest(manifest)
    if not isinstance(protocol, ProtocolConfig):
        raise AllocationError("protocol must be a ProtocolConfig")
    _existing_real_directory(repo_root, "repo_root")
    policy_revision = validated.episodes[0].policy_revision
    code_commit = validated.episodes[0].code_commit
    try:
        expected = reconstruct_episode_manifest(
            validated.split,
            validated.task,
            protocol,
            policy_revision=policy_revision,
            code_commit=code_commit,
        )
    except ProtocolConfigError as exc:
        raise AllocationError(f"manifest protocol regeneration failed: {exc}") from exc
    if _canonical(_plain(validated.to_dict())) != _canonical(
        _plain(expected.to_dict())
    ):
        raise AllocationError(
            "manifest differs from the exact protocol-generated episode allocation"
        )
    return validated


def _trusted_scoring_sources(repo_root: str | Path) -> tuple[str, str]:
    """Compute config/source identities instead of trusting score declarations."""

    try:
        return (
            frozen_config_sha256(repo_root),
            scoring_source_sha256(repo_root),
        )
    except ProvenanceError as exc:
        raise AllocationError(f"could not derive trusted score sources: {exc}") from exc


def _revalidate_receipt(
    receipt: AllocationReceipt,
    protocol: ProtocolConfig,
    repo_root: str | Path,
) -> None:
    """Re-run all constructible invariants immediately before write/after load."""

    rollout = (
        receipt.rollout if isinstance(receipt, ScoreAllocationReceipt) else receipt
    )
    _validate_manifest_against_protocol(rollout.manifest, protocol, repo_root)
    RolloutAllocationReceipt(
        manifest=rollout.manifest,
        source=rollout.source,
        valid_artifacts=rollout.valid_artifacts,
        invalid_artifacts=rollout.invalid_artifacts,
    )
    if isinstance(receipt, ScoreAllocationReceipt):
        expected_config, expected_code = _trusted_scoring_sources(repo_root)
        if (
            receipt.config_sha256 != expected_config
            or receipt.code_sha256 != expected_code
        ):
            raise AllocationError(
                "score receipt config/code sources differ from repository"
            )
        ScoreAllocationReceipt(
            rollout=receipt.rollout,
            probe_sha256=receipt.probe_sha256,
            config_sha256=receipt.config_sha256,
            code_sha256=receipt.code_sha256,
            score_artifacts=receipt.score_artifacts,
        )


def _identity_tuple(value: Any, where: str) -> tuple[ArtifactIdentity, ...]:
    try:
        result = tuple(value)
    except TypeError as exc:
        raise AllocationError(f"{where} must be an iterable") from exc
    if not all(isinstance(item, ArtifactIdentity) for item in result):
        raise AllocationError(f"{where} must contain ArtifactIdentity values")
    if result != tuple(sorted(result, key=lambda item: item.episode_id)):
        raise AllocationError(f"{where} must be sorted by episode ID")
    return result


def _task_dict(task: TaskSpec) -> dict[str, Any]:
    return {
        "rank": task.rank,
        "suite": task.suite,
        "task_id": task.task_id,
        "language": task.language,
        "primary_object": task.primary_object,
        "planar_symmetry_order": task.planar_symmetry_order,
    }


def _validate_task(task: Any, where: str) -> TaskSpec:
    if not isinstance(task, TaskSpec):
        raise AllocationError(f"{where} must be a TaskSpec")
    if (
        type(task.rank) is not int
        or task.rank < 1
        or type(task.task_id) is not int
        or task.task_id < 0
    ):
        raise AllocationError(f"{where} rank/task ID is invalid")
    if type(task.planar_symmetry_order) is not int or task.planar_symmetry_order < 1:
        raise AllocationError(f"{where} symmetry order is invalid")
    for name in ("suite", "language", "primary_object"):
        _nonempty_string(getattr(task, name), f"{where}.{name}")
    return task


def _manifest_from_metadata(value: Any) -> Manifest:
    metadata = _mapping(value, "manifest")
    _exact_keys(metadata, {"schema_version", "split", "task", "episodes"}, "manifest")
    task_metadata = _mapping(metadata["task"], "manifest.task")
    _exact_keys(
        task_metadata,
        {
            "rank",
            "suite",
            "task_id",
            "language",
            "primary_object",
            "planar_symmetry_order",
        },
        "manifest.task",
    )
    task = TaskSpec(
        rank=_integer(task_metadata["rank"], "manifest.task.rank"),
        suite=_nonempty_string(task_metadata["suite"], "manifest.task.suite"),
        task_id=_integer(task_metadata["task_id"], "manifest.task.task_id"),
        language=_nonempty_string(task_metadata["language"], "manifest.task.language"),
        primary_object=_nonempty_string(
            task_metadata["primary_object"], "manifest.task.primary_object"
        ),
        planar_symmetry_order=_integer(
            task_metadata["planar_symmetry_order"],
            "manifest.task.planar_symmetry_order",
        ),
    )
    split = _split(metadata["split"], "manifest.split")
    raw_episodes = _list(metadata["episodes"], "manifest.episodes")
    episodes = tuple(
        _episode_from_metadata(item, index) for index, item in enumerate(raw_episodes)
    )
    result = Manifest(
        schema_version=_integer(metadata["schema_version"], "manifest.schema_version"),
        split=split,
        task=task,
        episodes=episodes,
    )
    _validate_manifest(result)
    if _canonical(_plain(result.to_dict())) != _canonical(metadata):
        raise AllocationError("manifest is not exactly reconstructable")
    return result


def _episode_from_metadata(value: Any, index: int) -> EpisodeSpec:
    where = f"manifest.episodes[{index}]"
    metadata = _mapping(value, where)
    expected = {
        "episode_id",
        "suite",
        "task_id",
        "task_rank",
        "split",
        "base_init_state_id",
        "condition_index",
        "condition_name",
        "condition_family",
        "condition_parameters",
        "reset_seed",
        "inference_seed",
        "policy_revision",
        "code_commit",
    }
    _exact_keys(metadata, expected, where)
    parameters = _mapping(
        metadata["condition_parameters"], f"{where}.condition_parameters"
    )
    result = EpisodeSpec(
        suite=_nonempty_string(metadata["suite"], f"{where}.suite"),
        task_id=_integer(metadata["task_id"], f"{where}.task_id"),
        task_rank=_integer(metadata["task_rank"], f"{where}.task_rank"),
        split=_split(metadata["split"], f"{where}.split"),
        base_init_state_id=_integer(
            metadata["base_init_state_id"], f"{where}.base_init_state_id"
        ),
        condition_index=_integer(
            metadata["condition_index"], f"{where}.condition_index"
        ),
        condition_name=_nonempty_string(
            metadata["condition_name"], f"{where}.condition_name"
        ),
        condition_family=_nonempty_string(
            metadata["condition_family"], f"{where}.condition_family"
        ),
        condition_parameters=parameters,
        reset_seed=_integer(metadata["reset_seed"], f"{where}.reset_seed"),
        inference_seed=_integer(metadata["inference_seed"], f"{where}.inference_seed"),
        policy_revision=_nonempty_string(
            metadata["policy_revision"], f"{where}.policy_revision"
        ),
        code_commit=_nonempty_string(metadata["code_commit"], f"{where}.code_commit"),
    )
    if metadata["episode_id"] != result.episode_id:
        raise AllocationError(f"{where}.episode_id is not derived from episode fields")
    return result


def _rollout_receipt_from_metadata(value: Any) -> RolloutAllocationReceipt:
    metadata = _mapping(value, "receipt")
    _exact_keys(
        metadata,
        {"schema_version", "kind", "manifest", "source", "allocation"},
        "receipt",
    )
    if (
        metadata["schema_version"] != ALLOCATION_SCHEMA_VERSION
        or metadata["kind"] != "rollout_allocation"
    ):
        raise AllocationError("unsupported rollout receipt schema/kind")
    manifest = _manifest_from_metadata(metadata["manifest"])
    source_data = _mapping(metadata["source"], "receipt.source")
    _exact_keys(
        source_data,
        {
            "split",
            "task",
            "policy_revision",
            "base_vlm_revision",
            "code_commit",
            "manifest_sha256",
        },
        "receipt.source",
    )
    source_task = _task_from_metadata(source_data["task"], "receipt.source.task")
    source = AllocationSourceIdentity(
        split=_split(source_data["split"], "receipt.source.split"),
        task=source_task,
        policy_revision=_nonempty_string(
            source_data["policy_revision"], "receipt.source.policy_revision"
        ),
        base_vlm_revision=_nonempty_string(
            source_data["base_vlm_revision"], "receipt.source.base_vlm_revision"
        ),
        code_commit=_nonempty_string(
            source_data["code_commit"], "receipt.source.code_commit"
        ),
        manifest_sha256=_sha256(
            source_data["manifest_sha256"], "receipt.source.manifest_sha256"
        ),
    )
    allocation = _mapping(metadata["allocation"], "receipt.allocation")
    _exact_keys(
        allocation,
        {
            "membership_rule",
            "outcome_conditioned",
            "attempted_count",
            "valid_count",
            "invalid_count",
            "invalid_fraction",
            "valid_artifacts",
            "invalid_artifacts",
        },
        "receipt.allocation",
    )
    valid = tuple(
        _artifact_identity(item, "valid_artifacts")
        for item in _list(allocation["valid_artifacts"], "valid_artifacts")
    )
    invalid = tuple(
        _artifact_identity(item, "invalid_artifacts")
        for item in _list(allocation["invalid_artifacts"], "invalid_artifacts")
    )
    result = RolloutAllocationReceipt(manifest, source, valid, invalid)
    if result.to_metadata() != metadata:
        raise AllocationError("rollout receipt derived fields are inconsistent")
    return result


def _score_receipt_from_metadata(value: Any) -> ScoreAllocationReceipt:
    metadata = _mapping(value, "receipt")
    _exact_keys(
        metadata,
        {
            "schema_version",
            "kind",
            "rollout_receipt",
            "rollout_receipt_sha256",
            "source",
            "allocation",
        },
        "receipt",
    )
    if (
        metadata["schema_version"] != ALLOCATION_SCHEMA_VERSION
        or metadata["kind"] != "score_allocation"
    ):
        raise AllocationError("unsupported score receipt schema/kind")
    rollout = _rollout_receipt_from_metadata(metadata["rollout_receipt"])
    if metadata["rollout_receipt_sha256"] != rollout.sha256:
        raise AllocationError("nested rollout receipt hash differs")
    source = _mapping(metadata["source"], "receipt.source")
    _exact_keys(
        source, {"probe_sha256", "config_sha256", "code_sha256"}, "receipt.source"
    )
    allocation = _mapping(metadata["allocation"], "receipt.allocation")
    _exact_keys(
        allocation,
        {"membership_rule", "outcome_conditioned", "scored_count", "score_artifacts"},
        "receipt.allocation",
    )
    scores = tuple(
        _scoring_identity(item)
        for item in _list(allocation["score_artifacts"], "score_artifacts")
    )
    result = ScoreAllocationReceipt(
        rollout=rollout,
        probe_sha256=_sha256(source["probe_sha256"], "source.probe_sha256"),
        config_sha256=_sha256(source["config_sha256"], "source.config_sha256"),
        code_sha256=_sha256(source["code_sha256"], "source.code_sha256"),
        score_artifacts=scores,
    )
    if result.to_metadata() != metadata:
        raise AllocationError("score receipt derived fields are inconsistent")
    return result


def _artifact_identity(value: Any, where: str) -> ArtifactIdentity:
    metadata = _mapping(value, where)
    _exact_keys(metadata, {"episode_id", "metadata_sha256", "trajectory_sha256"}, where)
    try:
        return ArtifactIdentity(
            episode_id=_safe_name(metadata["episode_id"], f"{where}.episode_id"),
            metadata_sha256=_sha256(
                metadata["metadata_sha256"], f"{where}.metadata_sha256"
            ),
            trajectory_sha256=_sha256(
                metadata["trajectory_sha256"], f"{where}.trajectory_sha256"
            ),
        )
    except ValueError as exc:
        raise AllocationError(f"{where}: invalid artifact identity") from exc


def _scoring_identity(value: Any) -> ScoringArtifactIdentity:
    metadata = _mapping(value, "score_artifact")
    names = set(ScoringArtifactIdentity.__dataclass_fields__)
    _exact_keys(metadata, names, "score_artifact")
    return ScoringArtifactIdentity(**{name: metadata[name] for name in names})


def _task_from_metadata(value: Any, where: str) -> TaskSpec:
    metadata = _mapping(value, where)
    _exact_keys(
        metadata,
        {
            "rank",
            "suite",
            "task_id",
            "language",
            "primary_object",
            "planar_symmetry_order",
        },
        where,
    )
    return TaskSpec(
        rank=_integer(metadata["rank"], f"{where}.rank"),
        suite=_nonempty_string(metadata["suite"], f"{where}.suite"),
        task_id=_integer(metadata["task_id"], f"{where}.task_id"),
        language=_nonempty_string(metadata["language"], f"{where}.language"),
        primary_object=_nonempty_string(
            metadata["primary_object"], f"{where}.primary_object"
        ),
        planar_symmetry_order=_integer(
            metadata["planar_symmetry_order"], f"{where}.planar_symmetry_order"
        ),
    )


def _write_payload(canonical: bytes, output_root: str | Path, filename: str) -> Path:
    # The digest lock serializes cooperating writers.  Path-based checks cannot
    # make publication race-free against a hostile external filesystem actor;
    # callers must place artifact roots on trusted storage.
    root = _safe_root(output_root)
    digest = hashlib.sha256(canonical).hexdigest()
    destination = root / digest
    if _lexists(destination):
        raise FileExistsError(
            f"refusing to overwrite allocation artifact {destination}"
        )
    lock = root / f".{digest}.publish.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            "another writer is publishing this allocation artifact"
        ) from exc
    os.close(descriptor)
    staging: Path | None = None
    try:
        _fsync(root)
        staging = Path(tempfile.mkdtemp(prefix=f".{digest}.tmp-", dir=root))
        target = staging / filename
        with target.open("xb") as stream:
            stream.write(canonical)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync(staging)
        if _lexists(destination):
            raise FileExistsError(
                f"refusing to overwrite allocation artifact {destination}"
            )
        os.rename(staging, destination)
        staging = None
        _fsync(root)
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        lock.unlink(missing_ok=True)
        _fsync(root)
    return destination


def _read_payload(path: str | Path, filename: str, expected_sha256: str | None) -> Any:
    return _read_payload_with_bytes(path, filename, expected_sha256)[0]


def _read_payload_with_bytes(
    path: str | Path, filename: str, expected_sha256: str | None
) -> tuple[Any, bytes]:
    if expected_sha256 is not None:
        _sha256(expected_sha256, "expected_sha256")
    directory = _absolute(path)
    if _has_symlink_component(directory) or not directory.is_dir():
        raise AllocationError("allocation artifact path must be a real directory")
    entries = list(os.scandir(directory))
    if len(entries) != 1 or entries[0].name != filename:
        raise AllocationError(
            f"allocation artifact must contain exactly one {filename}"
        )
    entry = entries[0]
    entry_stat = entry.stat(follow_symlinks=False)
    if entry.is_symlink() or not stat.S_ISREG(entry_stat.st_mode):
        raise AllocationError(f"{filename} must be a regular file")
    if entry_stat.st_size > MAX_ALLOCATION_JSON_BYTES:
        raise AllocationError(
            f"{filename} exceeds the {MAX_ALLOCATION_JSON_BYTES}-byte limit"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(entry.path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AllocationError(f"{filename} must be a regular file")
        if before.st_size > MAX_ALLOCATION_JSON_BYTES:
            raise AllocationError(
                f"{filename} exceeds the {MAX_ALLOCATION_JSON_BYTES}-byte limit"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            canonical = stream.read(MAX_ALLOCATION_JSON_BYTES + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(canonical) > MAX_ALLOCATION_JSON_BYTES:
        raise AllocationError(
            f"{filename} exceeds the {MAX_ALLOCATION_JSON_BYTES}-byte limit"
        )
    if _stat_identity(before) != _stat_identity(after):
        raise AllocationError("allocation artifact changed while being read")
    final_entries = list(os.scandir(directory))
    if len(final_entries) != 1 or final_entries[0].name != filename:
        raise AllocationError("allocation artifact layout changed while being read")
    final_entry = final_entries[0]
    final_stat = final_entry.stat(follow_symlinks=False)
    if (
        final_entry.is_symlink()
        or not stat.S_ISREG(final_stat.st_mode)
        or _stat_identity(final_stat) != _stat_identity(after)
    ):
        raise AllocationError("allocation artifact changed while being read")
    digest = hashlib.sha256(canonical).hexdigest()
    if directory.name != digest:
        raise AllocationError("allocation directory name differs from content hash")
    if expected_sha256 is not None and digest != expected_sha256:
        raise AllocationError("allocation artifact differs from expected SHA-256")
    try:
        payload = json.loads(
            canonical.decode("utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=_reject_constant,
        )
    except AllocationError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise AllocationError(
            f"allocation artifact is not strict UTF-8 JSON: {exc}"
        ) from exc
    try:
        reconstructed = _canonical(payload)
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise AllocationError(
            f"allocation artifact cannot be canonicalized: {exc}"
        ) from exc
    if reconstructed != canonical:
        raise AllocationError("allocation artifact JSON is not canonical")
    return payload, canonical


def _safe_root(value: str | Path) -> Path:
    root = _absolute(value)
    if any(part in {"configs", "locks"} for part in root.parts):
        raise AllocationError(
            "allocation artifacts may not be written into config/lock paths"
        )
    if _has_symlink_component(root):
        raise AllocationError("output path may not contain a symlink component")
    if root.exists() and not root.is_dir():
        raise AllocationError("output root must be a directory")
    root.mkdir(parents=True, exist_ok=True)
    if _has_symlink_component(root):
        raise AllocationError("output path may not contain a symlink component")
    return root


def _existing_real_directory(value: str | Path, where: str) -> Path:
    try:
        result = _absolute(value)
    except (TypeError, ValueError, OSError) as exc:
        raise AllocationError(f"{where} must be a filesystem path") from exc
    if _has_symlink_component(result) or not result.is_dir():
        raise AllocationError(f"{where} must be a real existing directory")
    return result


def _absolute(value: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(value).expanduser())))


def _has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _manifest_sha256(manifest: Manifest) -> str:
    return hashlib.sha256(_canonical(_plain(manifest.to_dict()))).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AllocationError(f"{where} must be a JSON object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise AllocationError(f"{where} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        raise AllocationError(
            f"{where} keys differ: missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _integer(value: Any, where: str) -> int:
    if type(value) is not int:
        raise AllocationError(f"{where} must be an integer")
    return value


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise AllocationError(f"{where} must be a nonempty string")
    return value


def _safe_name(value: Any, where: str) -> str:
    result = _nonempty_string(value, where)
    if Path(result).name != result:
        raise AllocationError(f"{where} must be a safe name")
    return result


def _sha256(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AllocationError(f"{where} must be a lowercase SHA-256")
    return value


def _split(value: Any, where: str) -> SplitName:
    try:
        return SplitName(value)
    except (TypeError, ValueError) as exc:
        raise AllocationError(f"{where} is not a known split") from exc


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AllocationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise AllocationError(f"non-finite JSON constant {value!r}")


__all__ = [
    "ALLOCATION_SCHEMA_VERSION",
    "MAX_ALLOCATION_JSON_BYTES",
    "AllocationError",
    "AllocationReceipt",
    "AllocationSourceIdentity",
    "RolloutAllocationReceipt",
    "ScoreAllocationReceipt",
    "ScoringArtifactIdentity",
    "audit_rollout_allocation",
    "audit_score_allocation",
    "load_allocation_receipt",
    "load_episode_manifest",
    "load_manifest",
    "revalidate_rollout_receipt",
    "revalidate_score_receipt",
    "rollout_receipt_from_metadata",
    "write_allocation_receipt",
    "write_episode_manifest",
    "write_manifest",
]
