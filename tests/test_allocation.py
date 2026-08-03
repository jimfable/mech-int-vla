from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

import mech_int_vla.allocation as allocation_module
from mech_int_vla.allocation import (
    MAX_ALLOCATION_JSON_BYTES,
    AllocationError,
    RolloutAllocationReceipt,
    ScoreAllocationReceipt,
    audit_rollout_allocation,
    audit_score_allocation,
    load_allocation_receipt,
    load_episode_manifest,
    revalidate_score_receipt,
    write_allocation_receipt,
    write_episode_manifest,
)
from mech_int_vla.artifacts import ArtifactHashes, RolloutArtifact
from mech_int_vla.config import SplitName, load_protocol_config
from mech_int_vla.manifest import generate_episode_manifest
from mech_int_vla.probes import (
    DEFAULT_CANDIDATE_PREFERENCE,
    ProbeSamples,
    circular_targets,
    select_and_fit_circular_probe,
)
from mech_int_vla.provenance import frozen_config_sha256, scoring_source_sha256
from mech_int_vla.scoring import LoadedScoringSidecar

ROOT = Path(__file__).parents[1]
PROTOCOL = load_protocol_config(ROOT / "configs")
POLICY = "1" * 40
CODE = "2" * 40
BASE = "3" * 40
CONFIG_HASH = frozen_config_sha256(ROOT)
CODE_HASH = scoring_source_sha256(ROOT)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@pytest.fixture(scope="module")
def protocol():
    return PROTOCOL


@pytest.fixture(scope="module")
def manifest(protocol):
    return generate_episode_manifest(
        SplitName.DISCOVERY,
        protocol.task_order.tasks[0],
        protocol,
        policy_revision=POLICY,
        code_commit=CODE,
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _deep_replace(
    value: Mapping[str, Any], path: tuple[str, ...], replacement: Any
) -> dict[str, Any]:
    result = _plain(value)
    target = result
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    return result


def _raw(
    episode,
    task,
    *,
    valid: bool = True,
    success: bool = False,
    metadata_override: dict[str, Any] | None = None,
    character: str | None = None,
) -> RolloutArtifact:
    metadata = {
        "schema_version": 1,
        "episode": episode.to_dict(),
        "task_language": task.language,
        "task": {
            "rank": task.rank,
            "suite": task.suite,
            "task_id": task.task_id,
            "language": task.language,
            "primary_object": task.primary_object,
            "planar_symmetry_order": task.planar_symmetry_order,
        },
        "condition": {
            "name": episode.condition_name,
            "family": episode.condition_family,
            "index": episode.condition_index,
            "parameters": dict(episode.condition_parameters),
        },
        "model": {
            "policy_revision": episode.policy_revision,
            "base_vlm_revision": BASE,
        },
        "validity": {"valid": valid},
        # Allocation must never read these labels when choosing membership.
        "outcome": {"success": success, "status": "arbitrary"},
    }
    if metadata_override is not None:
        metadata = metadata_override
    marker = character or episode.episode_id
    return RolloutArtifact(
        path=Path("/") / episode.episode_id,
        metadata=metadata,
        arrays={},
        hashes=ArtifactHashes(_digest("meta-" + marker), _digest("traj-" + marker)),
    )


def _raws(manifest, *, invalid_ids: set[str] | None = None):
    invalid_ids = invalid_ids or set()
    return tuple(
        _raw(
            episode,
            manifest.task,
            valid=episode.episode_id not in invalid_ids,
            success=index % 2 == 0,
        )
        for index, episode in enumerate(manifest.episodes)
    )


def _audit_rollout(manifest, raws):
    return audit_rollout_allocation(
        manifest,
        raws,
        protocol=PROTOCOL,
        repo_root=ROOT,
    )


def _audit_scores(rollout, sidecars, probe):
    if not hasattr(probe, "sha256"):
        return audit_score_allocation(
            rollout,
            sidecars,
            probe,
            protocol=PROTOCOL,
            repo_root=ROOT,
        )

    class FakeBoundProbe:
        def __init__(self):
            self.probe = probe
            self.rollout = rollout
            self.sha256 = probe.sha256()

    bound = FakeBoundProbe()
    with (
        patch("mech_int_vla.probe_artifacts.BoundProbeArtifact", FakeBoundProbe),
        patch(
            "mech_int_vla.probe_artifacts.validate_bound_probe_artifact",
            return_value=bound,
        ),
    ):
        return audit_score_allocation(
            rollout,
            sidecars,
            bound,
            protocol=PROTOCOL,
            repo_root=ROOT,
        )


def _protected_manifest(protocol, split: SplitName, monkeypatch):
    receipt = SimpleNamespace(head_commit=CODE)
    monkeypatch.setattr(
        "mech_int_vla.manifest.assert_calibration_ready",
        lambda *args, **kwargs: receipt,
    )
    monkeypatch.setattr(
        "mech_int_vla.manifest.assert_locked_test_ready",
        lambda *args, **kwargs: receipt,
    )
    return generate_episode_manifest(
        split,
        protocol.task_order.tasks[0],
        protocol,
        policy_revision=POLICY,
        code_commit=CODE,
        repo_root=ROOT,
    )


@pytest.fixture(scope="module")
def probe(protocol):
    groups = np.repeat(np.arange(5, dtype=np.int64), 4)
    theta = np.linspace(-1.0, 1.0, groups.size, dtype=np.float64)
    samples = ProbeSamples.from_arrays(
        theta_rel=theta,
        base_init_state_id=groups,
        episode_id=np.asarray([f"probe-{index // 4}" for index in range(groups.size)]),
        symmetry_order=2,
    )
    features = circular_targets(theta, symmetry_order=2)
    return select_and_fit_circular_probe(
        {name: features.copy() for name in DEFAULT_CANDIDATE_PREFERENCE},
        samples,
        selection_config=protocol.split.calibration_selection,
    ).artifact


def _sidecar(identity, *, probe_sha: str, **changes: Any) -> LoadedScoringSidecar:
    links = {
        "raw_metadata_sha256": identity.metadata_sha256,
        "raw_trajectory_sha256": identity.trajectory_sha256,
        "probe_sha256": probe_sha,
        "config_sha256": CONFIG_HASH,
        "code_sha256": CODE_HASH,
    }
    links.update(changes.pop("links", {}))
    primitive = changes.pop(
        "primitives_sha256", _digest("score-array-" + identity.episode_id)
    )
    metadata = {
        "episode_id": identity.episode_id,
        "split": "discovery",
        "links": links,
        "files": {"primitives_sha256": primitive},
    }
    metadata.update(changes)
    return LoadedScoringSidecar(
        path=Path("/scores") / identity.episode_id,
        metadata=metadata,
        arrays={},
        metadata_sha256=_digest("score-meta-" + identity.episode_id),
        primitives_sha256=primitive,
    )


def test_manifest_round_trip_is_canonical_content_addressed_and_exact(
    manifest, protocol, tmp_path: Path
) -> None:
    path = write_episode_manifest(
        manifest, tmp_path / "manifests", protocol=protocol, repo_root=ROOT
    )
    assert path.name == manifest.sha256
    assert (path / "manifest.json").read_bytes() == json.dumps(
        manifest.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    loaded = load_episode_manifest(
        path,
        protocol=protocol,
        repo_root=ROOT,
        expected_sha256=manifest.sha256,
    )
    assert loaded == manifest
    assert loaded.to_dict() == manifest.to_dict()
    with pytest.raises(FileExistsError):
        write_episode_manifest(
            manifest, tmp_path / "manifests", protocol=protocol, repo_root=ROOT
        )


def test_manifest_loader_refuses_noncanonical_tamper_layout_and_symlink(
    manifest, protocol, tmp_path: Path
) -> None:
    path = write_episode_manifest(
        manifest, tmp_path / "manifests", protocol=protocol, repo_root=ROOT
    )
    canonical = (path / "manifest.json").read_bytes()
    (path / "manifest.json").write_bytes(canonical + b"\n")
    with pytest.raises(AllocationError):
        load_episode_manifest(path, protocol=protocol, repo_root=ROOT)

    other = write_episode_manifest(
        manifest, tmp_path / "other", protocol=protocol, repo_root=ROOT
    )
    (other / "unexpected").write_text("x")
    with pytest.raises(AllocationError, match="exactly one"):
        load_episode_manifest(other, protocol=protocol, repo_root=ROOT)

    link = tmp_path / "manifest-link"
    link.symlink_to(other, target_is_directory=True)
    with pytest.raises(AllocationError, match="real directory"):
        load_episode_manifest(link, protocol=protocol, repo_root=ROOT)
    with pytest.raises(AllocationError, match="config/lock"):
        write_episode_manifest(
            manifest,
            tmp_path / "configs" / "receipts",
            protocol=protocol,
            repo_root=ROOT,
        )


@pytest.mark.parametrize(
    ("split", "first_init", "condition_names"),
    [
        (
            SplitName.CALIBRATION,
            10,
            (
                "iid",
                "yaw_neg_45",
                "yaw_neg_30",
                "yaw_neg_15",
                "yaw_pos_15",
                "yaw_pos_30",
                "yaw_pos_45",
                "planar_axis_3cm",
            ),
        ),
        (
            SplitName.LOCKED_TEST,
            30,
            (
                "iid",
                "yaw_neg_37_5",
                "yaw_neg_22_5",
                "yaw_pos_22_5",
                "yaw_pos_37_5",
                "planar_diagonal_4_5cm",
                "camera_yaw_neg_5",
                "camera_yaw_pos_5",
            ),
        ),
    ],
)
def test_protected_manifests_require_exact_160_cell_protocol_topology(
    protocol,
    monkeypatch,
    tmp_path: Path,
    split: SplitName,
    first_init: int,
    condition_names: tuple[str, ...],
) -> None:
    protected = _protected_manifest(protocol, split, monkeypatch)
    expected_pairs = tuple(
        (init_id, condition_index)
        for init_id in range(first_init, first_init + 20)
        for condition_index in range(8)
    )

    assert len(protected.episodes) == 160
    assert (
        tuple(
            (item.base_init_state_id, item.condition_index)
            for item in protected.episodes
        )
        == expected_pairs
    )
    assert tuple(item.condition_name for item in protected.episodes[:8]) == (
        condition_names
    )
    assert _audit_rollout(protected, _raws(protected)).attempted_count == 160
    persisted = write_episode_manifest(
        protected,
        tmp_path / split.value,
        protocol=protocol,
        repo_root=ROOT,
    )
    assert (
        load_episode_manifest(
            persisted,
            protocol=protocol,
            repo_root=ROOT,
            expected_sha256=protected.sha256,
        ).to_dict()
        == protected.to_dict()
    )

    wrong_init = replace(
        protected.episodes[0],
        base_init_state_id=0,
        reset_seed=999_999,
    )
    malformed = replace(protected, episodes=(wrong_init,) + protected.episodes[1:])
    with pytest.raises(AllocationError, match="exact protocol-generated"):
        _audit_rollout(malformed, _raws(malformed))

    reordered = replace(
        protected,
        episodes=(protected.episodes[1], protected.episodes[0])
        + protected.episodes[2:],
    )
    with pytest.raises(AllocationError, match="exact protocol-generated"):
        _audit_rollout(reordered, _raws(reordered))


def test_discovery_manifest_requires_exact_generated_seed_and_order(manifest) -> None:
    wrong_seed = replace(manifest.episodes[0], reset_seed=999_999)
    malformed = replace(manifest, episodes=(wrong_seed,) + manifest.episodes[1:])
    with pytest.raises(AllocationError, match="exact protocol-generated"):
        _audit_rollout(malformed, _raws(malformed))

    reordered = replace(
        manifest,
        episodes=(manifest.episodes[1], manifest.episodes[0]) + manifest.episodes[2:],
    )
    with pytest.raises(AllocationError, match="exact protocol-generated"):
        _audit_rollout(reordered, _raws(reordered))


def test_manifest_requires_immutable_episodes_and_writers_revalidate(
    manifest, protocol, tmp_path: Path
) -> None:
    mutable = replace(manifest, episodes=list(manifest.episodes))
    with pytest.raises(AllocationError, match="immutable tuple"):
        _audit_rollout(mutable, _raws(mutable))
    with pytest.raises(AllocationError, match="immutable tuple"):
        write_episode_manifest(
            mutable,
            tmp_path / "mutable-manifest",
            protocol=protocol,
            repo_root=ROOT,
        )

    independent = replace(manifest, episodes=tuple(manifest.episodes))
    receipt = _audit_rollout(independent, _raws(independent))
    object.__setattr__(receipt.manifest, "episodes", list(receipt.manifest.episodes))
    with pytest.raises(AllocationError, match="immutable tuple"):
        write_allocation_receipt(
            receipt,
            tmp_path / "mutated-receipt",
            protocol=protocol,
            repo_root=ROOT,
        )


def test_manifest_boundaries_require_an_actual_protocol_config(
    manifest, tmp_path: Path
) -> None:
    with pytest.raises(AllocationError, match="ProtocolConfig"):
        write_episode_manifest(
            manifest,
            tmp_path / "wrong-protocol",
            protocol=object(),  # type: ignore[arg-type]
            repo_root=ROOT,
        )


def test_manifest_loader_caps_payload_before_reading(protocol, tmp_path: Path) -> None:
    artifact = tmp_path / ("0" * 64)
    artifact.mkdir()
    (artifact / "manifest.json").write_bytes(b" " * (MAX_ALLOCATION_JSON_BYTES + 1))
    with pytest.raises(AllocationError, match="exceeds"):
        load_episode_manifest(artifact, protocol=protocol, repo_root=ROOT)


def test_manifest_loader_rechecks_exact_layout_after_read(
    manifest, protocol, monkeypatch, tmp_path: Path
) -> None:
    artifact = write_episode_manifest(
        manifest,
        tmp_path / "layout-race",
        protocol=protocol,
        repo_root=ROOT,
    )
    real_scandir = allocation_module.os.scandir
    calls = 0

    def raced_scandir(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            (Path(path) / "unexpected").write_text("race", encoding="utf-8")
        return real_scandir(path)

    monkeypatch.setattr(allocation_module.os, "scandir", raced_scandir)
    with pytest.raises(AllocationError, match="layout changed"):
        load_episode_manifest(artifact, protocol=protocol, repo_root=ROOT)


def test_rollout_audit_proves_exact_sorted_validity_only_membership(manifest) -> None:
    invalid_ids = {manifest.episodes[3].episode_id, manifest.episodes[20].episode_id}
    raws = list(_raws(manifest, invalid_ids=invalid_ids))
    receipt = _audit_rollout(manifest, tuple(reversed(raws)))

    assert receipt.attempted_count == 40
    assert receipt.invalid_episode_ids == tuple(sorted(invalid_ids))
    assert receipt.valid_episode_ids == tuple(
        sorted({item.episode_id for item in manifest.episodes} - invalid_ids)
    )
    assert receipt.invalid_fraction == 2 / 40
    assert receipt.source.policy_revision == POLICY
    assert receipt.source.code_commit == CODE
    assert receipt.source.base_vlm_revision == BASE
    assert receipt.source.manifest_sha256 == manifest.sha256
    assert receipt.to_metadata()["allocation"]["outcome_conditioned"] is False

    # Flipping every outcome cannot change the valid/invalid partition or hash.
    flipped = tuple(
        _raw(
            episode,
            manifest.task,
            valid=episode.episode_id not in invalid_ids,
            success=index % 2 != 0,
        )
        for index, episode in enumerate(manifest.episodes)
    )
    assert _audit_rollout(manifest, flipped).sha256 == receipt.sha256


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate"])
def test_rollout_audit_rejects_nonexact_episode_sets(manifest, fault: str) -> None:
    raws = list(_raws(manifest))
    if fault == "missing":
        raws.pop()
    elif fault == "duplicate":
        raws[-1] = raws[0]
    else:
        metadata = _plain(raws[-1].metadata)
        metadata["episode"]["episode_id"] = "unexpected"
        raws[-1] = RolloutArtifact(Path("/unexpected"), metadata, {}, raws[-1].hashes)
    with pytest.raises(AllocationError):
        _audit_rollout(manifest, raws)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("episode", "split"), "calibration"),
        (("episode", "task_id"), 999),
        (("episode", "policy_revision"), "9" * 40),
        (("episode", "code_commit"), "8" * 40),
        (("episode", "base_init_state_id"), 999),
        (("episode", "condition_name"), "wrong"),
        (("episode", "condition_parameters"), {"wrong": True}),
        (("episode", "reset_seed"), 999),
        (("episode", "inference_seed"), 999),
        (("task", "primary_object"), "wrong"),
        (("model", "policy_revision"), "7" * 40),
    ],
)
def test_rollout_audit_rejects_every_manifest_metadata_mismatch(
    manifest, path: tuple[str, ...], replacement: Any
) -> None:
    raws = list(_raws(manifest))
    first = raws[0]
    metadata = _deep_replace(first.metadata, path, replacement)
    raws[0] = _raw(manifest.episodes[0], manifest.task, metadata_override=metadata)
    with pytest.raises(AllocationError):
        _audit_rollout(manifest, raws)


def test_rollout_audit_rejects_mixed_base_invalid_hash_and_wrong_manifest_count(
    manifest,
) -> None:
    raws = list(_raws(manifest))
    metadata = _deep_replace(raws[0].metadata, ("model", "base_vlm_revision"), "9" * 40)
    raws[0] = _raw(manifest.episodes[0], manifest.task, metadata_override=metadata)
    with pytest.raises(AllocationError, match="mixed base-VLM"):
        _audit_rollout(manifest, raws)

    bad_hash = replace(raws[1], hashes=ArtifactHashes("X" * 64, "a" * 64))
    raws = list(_raws(manifest))
    raws[1] = bad_hash
    with pytest.raises(AllocationError, match="content identity"):
        _audit_rollout(manifest, raws)

    short = replace(manifest, episodes=manifest.episodes[:-1])
    with pytest.raises(AllocationError, match="exactly 40"):
        _audit_rollout(short, raws[:-1])


def test_score_audit_exact_links_sorted_receipt_and_round_trip(
    manifest, probe, protocol, tmp_path: Path
) -> None:
    invalid_id = manifest.episodes[0].episode_id
    rollout = _audit_rollout(manifest, _raws(manifest, invalid_ids={invalid_id}))
    sidecars = tuple(
        _sidecar(identity, probe_sha=probe.sha256())
        for identity in reversed(rollout.valid_artifacts)
    )
    receipt = _audit_scores(rollout, sidecars, probe)

    assert isinstance(receipt, ScoreAllocationReceipt)
    assert receipt.scored_episode_ids == rollout.valid_episode_ids
    assert invalid_id not in receipt.scored_episode_ids
    assert receipt.config_sha256 == CONFIG_HASH
    assert receipt.code_sha256 == CODE_HASH
    assert receipt.to_metadata()["allocation"]["outcome_conditioned"] is False
    path = write_allocation_receipt(
        receipt,
        tmp_path / "receipts",
        protocol=protocol,
        repo_root=ROOT,
    )
    assert path.name == receipt.sha256
    assert (
        load_allocation_receipt(
            path,
            protocol=protocol,
            repo_root=ROOT,
            expected_sha256=receipt.sha256,
        )
        == receipt
    )

    rollout_path = write_allocation_receipt(
        rollout,
        tmp_path / "rollout-receipts",
        protocol=protocol,
        repo_root=ROOT,
    )
    loaded_rollout = load_allocation_receipt(
        rollout_path, protocol=protocol, repo_root=ROOT
    )
    assert isinstance(loaded_rollout, RolloutAllocationReceipt)
    assert loaded_rollout == rollout


@pytest.mark.parametrize("source", ["config_sha256", "code_sha256"])
def test_score_receipt_context_revalidator_rejects_stale_repository_source(
    protocol, probe, monkeypatch, source: str
) -> None:
    manifest = _protected_manifest(protocol, SplitName.CALIBRATION, monkeypatch)
    rollout = _audit_rollout(manifest, _raws(manifest))
    sidecars = tuple(
        _sidecar(identity, probe_sha=probe.sha256(), split="calibration")
        for identity in rollout.valid_artifacts
    )
    receipt = _audit_scores(rollout, sidecars, probe)

    class FakeBoundProbe:
        def __init__(self):
            self.probe = probe
            self.rollout = rollout
            self.sha256 = probe.sha256()

    bound = FakeBoundProbe()
    stale_sha = _digest(f"stale-{source}")
    stale_scores = tuple(
        replace(identity, **{source: stale_sha}) for identity in receipt.score_artifacts
    )
    stale = replace(receipt, **{source: stale_sha, "score_artifacts": stale_scores})
    with (
        patch("mech_int_vla.probe_artifacts.BoundProbeArtifact", FakeBoundProbe),
        patch(
            "mech_int_vla.probe_artifacts.validate_bound_probe_artifact",
            return_value=bound,
        ),
        pytest.raises(AllocationError, match="differ from repository"),
    ):
        revalidate_score_receipt(
            stale,
            bound,
            protocol=protocol,
            repo_root=ROOT,
        )


def test_score_receipt_context_revalidator_binds_exact_calibration(
    protocol, probe, monkeypatch
) -> None:
    manifest = _protected_manifest(protocol, SplitName.CALIBRATION, monkeypatch)
    rollout = _audit_rollout(manifest, _raws(manifest))
    sidecars = tuple(
        _sidecar(identity, probe_sha=probe.sha256(), split="calibration")
        for identity in rollout.valid_artifacts
    )
    receipt = _audit_scores(rollout, sidecars, probe)
    first = rollout.valid_artifacts[0]
    other_rollout = RolloutAllocationReceipt(
        manifest=rollout.manifest,
        source=rollout.source,
        valid_artifacts=rollout.valid_artifacts[1:],
        invalid_artifacts=tuple(
            sorted(
                (*rollout.invalid_artifacts, first), key=lambda item: item.episode_id
            )
        ),
    )

    class FakeBoundProbe:
        def __init__(self):
            self.probe = probe
            self.rollout = other_rollout
            self.sha256 = probe.sha256()

    bound = FakeBoundProbe()
    wrong_probe_sha = _digest("wrong-bound-probe")
    wrong_probe_receipt = replace(
        receipt,
        probe_sha256=wrong_probe_sha,
        score_artifacts=tuple(
            replace(identity, probe_sha256=wrong_probe_sha)
            for identity in receipt.score_artifacts
        ),
    )
    with (
        patch("mech_int_vla.probe_artifacts.BoundProbeArtifact", FakeBoundProbe),
        patch(
            "mech_int_vla.probe_artifacts.validate_bound_probe_artifact",
            return_value=bound,
        ),
        pytest.raises(AllocationError, match="probe digest differs"),
    ):
        revalidate_score_receipt(
            wrong_probe_receipt,
            bound,
            protocol=protocol,
            repo_root=ROOT,
        )
    with (
        patch("mech_int_vla.probe_artifacts.BoundProbeArtifact", FakeBoundProbe),
        patch(
            "mech_int_vla.probe_artifacts.validate_bound_probe_artifact",
            return_value=bound,
        ),
        pytest.raises(AllocationError, match="bound probe allocation"),
    ):
        revalidate_score_receipt(
            receipt,
            bound,
            protocol=protocol,
            repo_root=ROOT,
        )


@pytest.mark.parametrize("fault", ["missing", "extra_invalid", "duplicate"])
def test_score_audit_rejects_missing_extra_invalid_and_duplicate(
    manifest, probe, fault: str
) -> None:
    invalid_id = manifest.episodes[0].episode_id
    rollout = _audit_rollout(manifest, _raws(manifest, invalid_ids={invalid_id}))
    sidecars = [
        _sidecar(identity, probe_sha=probe.sha256())
        for identity in rollout.valid_artifacts
    ]
    if fault == "missing":
        sidecars.pop()
    elif fault == "duplicate":
        sidecars[-1] = sidecars[0]
    else:
        invalid_raw = next(
            item for item in rollout.invalid_artifacts if item.episode_id == invalid_id
        )
        sidecars.append(_sidecar(invalid_raw, probe_sha=probe.sha256()))
    with pytest.raises(AllocationError):
        _audit_scores(rollout, sidecars, probe)


@pytest.mark.parametrize(
    ("fault", "value"),
    [
        ("raw_metadata_sha256", "a" * 64),
        ("raw_trajectory_sha256", "b" * 64),
        ("probe_sha256", "c" * 64),
        ("config_sha256", "d" * 64),
        ("code_sha256", "e" * 64),
    ],
)
def test_score_audit_rejects_wrong_or_mixed_content_links(
    manifest, probe, fault: str, value: str
) -> None:
    rollout = _audit_rollout(manifest, _raws(manifest))
    sidecars = [
        _sidecar(identity, probe_sha=probe.sha256())
        for identity in rollout.valid_artifacts
    ]
    first_identity = rollout.valid_artifacts[0]
    sidecars[0] = _sidecar(
        first_identity,
        probe_sha=probe.sha256(),
        links={fault: value},
    )
    with pytest.raises(AllocationError):
        _audit_scores(rollout, sidecars, probe)


def test_score_audit_rejects_uniform_bogus_config_and_code_sources(
    manifest, probe
) -> None:
    rollout = _audit_rollout(manifest, _raws(manifest))
    sidecars = tuple(
        _sidecar(
            identity,
            probe_sha=probe.sha256(),
            links={
                "config_sha256": "d" * 64,
                "code_sha256": "e" * 64,
            },
        )
        for identity in rollout.valid_artifacts
    )
    with pytest.raises(AllocationError, match="differ from repository"):
        _audit_scores(rollout, sidecars, probe)


def test_score_audit_rejects_split_primitive_probe_type_and_zero_valid(
    manifest, probe
) -> None:
    rollout = _audit_rollout(manifest, _raws(manifest))
    sidecars = [
        _sidecar(identity, probe_sha=probe.sha256())
        for identity in rollout.valid_artifacts
    ]
    sidecars[0] = _sidecar(
        rollout.valid_artifacts[0], probe_sha=probe.sha256(), split="calibration"
    )
    with pytest.raises(AllocationError, match="split"):
        _audit_scores(rollout, sidecars, probe)

    sidecars = [
        _sidecar(identity, probe_sha=probe.sha256())
        for identity in rollout.valid_artifacts
    ]
    first = sidecars[0]
    sidecars[0] = replace(first, primitives_sha256="f" * 64)
    with pytest.raises(AllocationError, match="primitives"):
        _audit_scores(rollout, sidecars, probe)
    with pytest.raises(AllocationError, match="BoundProbeArtifact"):
        _audit_scores(rollout, sidecars, object())

    all_invalid = _audit_rollout(
        manifest,
        _raws(manifest, invalid_ids={item.episode_id for item in manifest.episodes}),
    )
    with pytest.raises(AllocationError, match="without valid scores"):
        _audit_scores(all_invalid, (), probe)


def test_receipt_loader_refuses_tamper_and_inconsistent_derived_fields(
    manifest, protocol, tmp_path: Path
) -> None:
    receipt = _audit_rollout(manifest, _raws(manifest))
    path = write_allocation_receipt(
        receipt,
        tmp_path / "receipts",
        protocol=protocol,
        repo_root=ROOT,
    )
    metadata = json.loads((path / "receipt.json").read_text())
    metadata["allocation"]["attempted_count"] = 39
    tampered = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    new_hash = hashlib.sha256(tampered).hexdigest()
    new_path = tmp_path / "tampered" / new_hash
    new_path.mkdir(parents=True)
    (new_path / "receipt.json").write_bytes(tampered)
    with pytest.raises(AllocationError, match="derived fields"):
        load_allocation_receipt(new_path, protocol=protocol, repo_root=ROOT)

    metadata = receipt.to_metadata()
    metadata["source"]["policy_revision"] = "9" * 40
    tampered = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    new_hash = hashlib.sha256(tampered).hexdigest()
    new_path = tmp_path / "source-tampered" / new_hash
    new_path.mkdir(parents=True)
    (new_path / "receipt.json").write_bytes(tampered)
    with pytest.raises(AllocationError, match="policy revision"):
        load_allocation_receipt(new_path, protocol=protocol, repo_root=ROOT)
