from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

import mech_int_vla.probe_artifacts as probe_artifacts_module
from mech_int_vla.allocation import (
    AllocationSourceIdentity,
    RolloutAllocationReceipt,
)
from mech_int_vla.artifacts import (
    ACTIVATION_CANDIDATES,
    CohortManifest,
    ProbeCohort,
    probe_cohort_array_sha256,
)
from mech_int_vla.config import (
    CalibrationSelectionConfig,
    SplitName,
    load_protocol_config,
)
from mech_int_vla.failure_events import ArtifactIdentity
from mech_int_vla.manifest import generate_episode_manifest
from mech_int_vla.probe_artifacts import (
    BoundProbeArtifact,
    BoundProbeError,
    bind_probe_artifact,
    load_bound_probe_artifact,
    validate_bound_probe_artifact,
    write_bound_probe_artifact,
)
from mech_int_vla.probes import (
    DEFAULT_CANDIDATE_PREFERENCE,
    FROZEN_RIDGE_ALPHA_GRID,
    ProbeSamples,
    select_and_fit_circular_probe,
)

POLICY = "1" * 40
BASE_VLM = "2" * 40
CODE = "3" * 40
ROOT = Path(__file__).parents[1]
PROTOCOL = load_protocol_config(ROOT / "configs")
TASK = PROTOCOL.task_order.tasks[0]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _manifest(split: SplitName = SplitName.CALIBRATION):
    receipt = type("Receipt", (), {"head_commit": CODE})()
    with (
        patch("mech_int_vla.manifest.assert_calibration_ready", return_value=receipt),
        patch("mech_int_vla.manifest.assert_locked_test_ready", return_value=receipt),
    ):
        return generate_episode_manifest(
            split,
            TASK,
            PROTOCOL,
            policy_revision=POLICY,
            code_commit=CODE,
            repo_root=ROOT,
        )


def _receipt(
    *,
    split: SplitName = SplitName.CALIBRATION,
    invalid_indexes: tuple[int, ...] = (7,),
) -> RolloutAllocationReceipt:
    manifest = _manifest(split)
    identities = tuple(
        sorted(
            (
                ArtifactIdentity(
                    episode.episode_id,
                    _digest("metadata-" + episode.episode_id),
                    _digest("trajectory-" + episode.episode_id),
                )
                for episode in manifest.episodes
            ),
            key=lambda item: item.episode_id,
        )
    )
    invalid_ids = {manifest.episodes[index].episode_id for index in invalid_indexes}
    return RolloutAllocationReceipt(
        manifest=manifest,
        source=AllocationSourceIdentity(
            split=split,
            task=TASK,
            policy_revision=POLICY,
            base_vlm_revision=BASE_VLM,
            code_commit=CODE,
            manifest_sha256=manifest.sha256,
        ),
        valid_artifacts=tuple(
            item for item in identities if item.episode_id not in invalid_ids
        ),
        invalid_artifacts=tuple(
            item for item in identities if item.episode_id in invalid_ids
        ),
    )


def _selection_config() -> CalibrationSelectionConfig:
    return CalibrationSelectionConfig(
        representation_candidates=DEFAULT_CANDIDATE_PREFERENCE,
        ridge_alpha_candidates=FROZEN_RIDGE_ALPHA_GRID,
        patch_strength_candidates=(0.0,),
        predictor_candidates=MappingProxyType({}),
    )


def _cohort(receipt: RolloutAllocationReceipt) -> ProbeCohort:
    valid_ids = receipt.valid_episode_ids
    by_id = {item.episode_id: item for item in receipt.manifest.episodes}
    row_count = len(valid_ids)
    theta = np.linspace(-1.0, 1.0, row_count, dtype=np.float64)
    base_ids = np.asarray(
        [by_id[episode_id].base_init_state_id for episode_id in valid_ids],
        dtype=np.int64,
    )
    samples = ProbeSamples.from_arrays(
        theta_rel=theta,
        base_init_state_id=base_ids,
        episode_id=np.asarray(valid_ids, dtype=np.str_),
        symmetry_order=TASK.planar_symmetry_order,
    )
    scaled = TASK.planar_symmetry_order * theta
    features = np.column_stack((np.cos(scaled), np.sin(scaled))).astype(np.float32)
    activations = MappingProxyType(
        {candidate: features.copy() for candidate in ACTIVATION_CANDIDATES}
    )
    control_step = np.zeros(row_count, dtype=np.int64)
    failure_label = np.asarray(
        [index % 2 == 0 for index in range(row_count)], dtype=np.bool_
    )
    episode_ids = np.asarray(valid_ids, dtype=np.str_)
    training_content = {
        "row_count": row_count,
        "episode_id_sha256": probe_cohort_array_sha256(episode_ids),
        "base_init_state_id_sha256": probe_cohort_array_sha256(base_ids),
        "control_step_sha256": probe_cohort_array_sha256(control_step),
        "theta_rel_sha256": probe_cohort_array_sha256(theta),
        "failure_label_sha256": probe_cohort_array_sha256(failure_label),
        "activation_features": {
            candidate: {
                "shape": list(activations[candidate].shape),
                "dtype": activations[candidate].dtype.str,
                "logical_sha256": probe_cohort_array_sha256(activations[candidate]),
            }
            for candidate in ACTIVATION_CANDIDATES
        },
    }
    payload = {
        "schema_version": 1,
        "kind": "probe_cohort",
        "split": receipt.source.split.value,
        "task": {
            "suite": TASK.suite,
            "task_id": TASK.task_id,
            "task_rank": TASK.rank,
            "language": TASK.language,
            "primary_object": TASK.primary_object,
            "planar_symmetry_order": TASK.planar_symmetry_order,
        },
        "model": {
            "policy_revision": receipt.source.policy_revision,
            "base_vlm_revision": receipt.source.base_vlm_revision,
            "code_commit": receipt.source.code_commit,
        },
        "selection": {
            "kind": "valid_pre_action_control_step_stride",
            "stride": 5,
            "outcome_conditioned": False,
        },
        "activation_candidates": list(ACTIVATION_CANDIDATES),
        "training_content": training_content,
        "episodes": [item.to_dict() for item in receipt.raw_artifacts],
        "invalid_reset_episode_ids": list(receipt.invalid_episode_ids),
    }
    return ProbeCohort(
        samples=samples,
        activation_features=activations,
        control_step=control_step,
        failure_label=failure_label,
        valid_episode_ids=valid_ids,
        invalid_reset_episode_ids=receipt.invalid_episode_ids,
        manifest=CohortManifest(payload),
    )


@pytest.fixture(scope="module")
def live_inputs():
    receipt = _receipt()
    cohort = _cohort(receipt)
    probe = select_and_fit_circular_probe(
        cohort.activation_features,
        cohort.samples,
        selection_config=_selection_config(),
    ).artifact
    return probe, receipt, cohort


def _bind(probe, receipt, cohort):
    with patch(
        "mech_int_vla.manifest.assert_calibration_ready",
        return_value=type("Receipt", (), {"head_commit": CODE})(),
    ):
        return bind_probe_artifact(
            probe,
            receipt,
            cohort,
            protocol=PROTOCOL,
            repo_root=ROOT,
        )


def _write(artifact, output_root: Path):
    with patch(
        "mech_int_vla.manifest.assert_calibration_ready",
        return_value=type("Receipt", (), {"head_commit": CODE})(),
    ):
        return write_bound_probe_artifact(
            artifact,
            output_root,
            protocol=PROTOCOL,
            repo_root=ROOT,
        )


def _load(path: Path, *, expected_sha256: str):
    with patch(
        "mech_int_vla.manifest.assert_calibration_ready",
        return_value=type("Receipt", (), {"head_commit": CODE})(),
    ):
        return load_bound_probe_artifact(
            path,
            protocol=PROTOCOL,
            repo_root=ROOT,
            expected_sha256=expected_sha256,
        )


def _write_modified(root: Path, metadata: dict[str, Any]) -> Path:
    canonical = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    directory = root / hashlib.sha256(canonical).hexdigest()
    directory.mkdir(parents=True)
    (directory / "bound_probe.json").write_bytes(canonical)
    return directory


def test_binding_derives_complete_calibration_identity_and_training_inputs(
    live_inputs,
) -> None:
    probe, receipt, cohort = live_inputs
    artifact = _bind(probe, receipt, cohort)
    metadata = artifact.to_metadata()

    assert artifact.numerical_probe_sha256 == probe.sha256()
    assert artifact.rollout_receipt_sha256 == receipt.sha256
    assert artifact.cohort_manifest_sha256 == cohort.manifest_sha256
    assert metadata["calibration_inputs"]["episode_manifest_sha256"] == (
        receipt.manifest.sha256
    )
    assert metadata["calibration_inputs"]["raw_artifacts"] == [
        item.to_dict() for item in receipt.raw_artifacts
    ]
    assert metadata["training"]["rows"] == cohort.samples.n_rows
    assert metadata["training"]["episodes"] == len(receipt.valid_episode_ids)
    assert metadata["training"]["base_init_state_ids"] == list(range(10, 30))
    assert metadata["training"]["symmetry_order"] == 2
    assert metadata["training"]["selected_candidate"] == probe.candidate
    assert [
        item["candidate"] for item in metadata["training"]["candidate_features"]
    ] == list(ACTIVATION_CANDIDATES)
    assert {item["width"] for item in metadata["training"]["candidate_features"]} == {2}
    assert (
        metadata["calibration_inputs"]["configuration_identity"][
            "episode_manifest_sha256"
        ]
        == receipt.manifest.sha256
    )
    assert (
        len(
            metadata["calibration_inputs"]["configuration_identity"][
                "full_config_sha256"
            ]
        )
        == 64
    )
    assert (
        len(
            metadata["calibration_inputs"]["configuration_identity"][
                "scoring_source_sha256"
            ]
        )
        == 64
    )
    assert metadata["training"]["derivation"] == ("exact_refit_matches_numerical_probe")
    assert len(metadata["training"]["row_alignment"]) == len(receipt.valid_episode_ids)


def test_direct_hash_assertion_construction_is_not_an_api() -> None:
    with pytest.raises(TypeError, match="cannot be constructed directly"):
        BoundProbeArtifact()


def test_round_trip_is_canonical_content_addressed_and_no_overwrite(
    live_inputs, tmp_path: Path
) -> None:
    artifact = _bind(*live_inputs)
    path = _write(artifact, tmp_path / "bound")
    assert path.name == artifact.sha256
    assert (path / "bound_probe.json").read_bytes() == artifact.canonical_json()
    loaded = _load(path, expected_sha256=artifact.sha256)
    assert loaded.to_metadata() == artifact.to_metadata()
    assert loaded.canonical_json() == artifact.canonical_json()
    assert loaded.probe.sha256() == artifact.probe.sha256()
    assert loaded.rollout.sha256 == artifact.rollout.sha256
    assert (
        validate_bound_probe_artifact(loaded, protocol=PROTOCOL, repo_root=ROOT)
        is loaded
    )
    assert _write(loaded, tmp_path / "republished") == (
        tmp_path / "republished" / artifact.sha256
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        _write(artifact, tmp_path / "bound")


def test_feature_and_row_changes_are_rejected_against_cohort_content(
    live_inputs,
) -> None:
    probe, receipt, cohort = live_inputs
    _bind(probe, receipt, cohort)
    features = dict(cohort.activation_features)
    features[ACTIVATION_CANDIDATES[-1]] = features[ACTIVATION_CANDIDATES[-1]].copy()
    features[ACTIVATION_CANDIDATES[-1]][0, 0] += np.float32(0.25)
    changed_features = replace(cohort, activation_features=MappingProxyType(features))
    with pytest.raises(BoundProbeError, match="content hashes"):
        _bind(probe, receipt, changed_features)

    steps = cohort.control_step.copy()
    steps[0] = 5
    changed_rows = replace(cohort, control_step=steps)
    with pytest.raises(BoundProbeError, match="content hashes"):
        _bind(probe, receipt, changed_rows)


@pytest.mark.parametrize(
    "fault",
    ["valid_ids", "raw_identity", "model_source", "base_id", "candidate_width"],
)
def test_builder_rejects_cross_object_mismatches(live_inputs, fault: str) -> None:
    probe, receipt, cohort = live_inputs
    if fault == "valid_ids":
        cohort = replace(cohort, valid_episode_ids=cohort.valid_episode_ids[1:])
    elif fault == "raw_identity":
        payload = _plain(cohort.manifest.payload)
        payload["episodes"][0]["trajectory_sha256"] = "a" * 64
        cohort = replace(cohort, manifest=CohortManifest(payload))
    elif fault == "model_source":
        payload = _plain(cohort.manifest.payload)
        payload["model"]["base_vlm_revision"] = "wrong"
        cohort = replace(cohort, manifest=CohortManifest(payload))
    elif fault == "base_id":
        groups = cohort.base_init_state_id.copy()
        groups[0] += 1
        samples = ProbeSamples.from_arrays(
            theta_rel=cohort.theta_rel,
            base_init_state_id=groups,
            episode_id=cohort.episode_id,
            symmetry_order=cohort.samples.symmetry_order,
        )
        cohort = replace(cohort, samples=samples)
    else:
        features = dict(cohort.activation_features)
        selected = probe.candidate
        features[selected] = np.column_stack(
            (features[selected], np.ones(features[selected].shape[0], dtype=np.float32))
        )
        cohort = replace(cohort, activation_features=MappingProxyType(features))
    with pytest.raises(BoundProbeError):
        _bind(probe, receipt, cohort)


def test_builder_rejects_noncalibration_and_probe_count_mismatch(live_inputs) -> None:
    probe, _, _ = live_inputs
    discovery = _receipt(split=SplitName.LOCKED_TEST, invalid_indexes=())
    with pytest.raises(BoundProbeError, match="calibration"):
        _bind(probe, discovery, _cohort(discovery))

    _, receipt, cohort = live_inputs
    mismatched = replace(probe, training_rows=probe.training_rows + 1)
    with pytest.raises(BoundProbeError, match="row count"):
        _bind(mismatched, receipt, cohort)


def test_builder_requires_exact_frozen_cohort_stride(live_inputs) -> None:
    probe, receipt, cohort = live_inputs
    payload = _plain(cohort.manifest.payload)
    payload["selection"]["stride"] = 10
    wrong_stride = replace(cohort, manifest=CohortManifest(payload))
    with pytest.raises(BoundProbeError, match="row-selection rule"):
        _bind(probe, receipt, wrong_stride)


def test_builder_rejects_unrelated_probe_and_fabricated_topology(live_inputs) -> None:
    _, receipt, cohort = live_inputs
    rng = np.random.default_rng(42)
    unrelated_features = {
        candidate: rng.normal(size=matrix.shape).astype(np.float32)
        for candidate, matrix in cohort.activation_features.items()
    }
    unrelated = select_and_fit_circular_probe(
        unrelated_features,
        cohort.samples,
        selection_config=PROTOCOL.split.calibration_selection,
    ).artifact
    with pytest.raises(BoundProbeError, match="deterministic refit"):
        _bind(unrelated, receipt, cohort)

    wrong_episode = replace(
        receipt.manifest.episodes[0],
        condition_name="caller_invented_condition",
    )
    malformed_manifest = replace(
        receipt.manifest,
        episodes=(wrong_episode,) + receipt.manifest.episodes[1:],
    )
    malformed = RolloutAllocationReceipt(
        manifest=malformed_manifest,
        source=replace(receipt.source, manifest_sha256=malformed_manifest.sha256),
        valid_artifacts=receipt.valid_artifacts,
        invalid_artifacts=receipt.invalid_artifacts,
    )
    with pytest.raises(BoundProbeError, match="revalidate Calibration inputs"):
        _bind(live_inputs[0], malformed, cohort)


def test_untrusted_parser_artifact_cannot_cross_public_boundaries(
    live_inputs, tmp_path: Path
) -> None:
    probe, receipt, cohort = live_inputs
    honest = _bind(probe, receipt, cohort)
    rng = np.random.default_rng(43)
    unrelated = select_and_fit_circular_probe(
        {
            candidate: rng.normal(size=matrix.shape).astype(np.float32)
            for candidate, matrix in cohort.activation_features.items()
        },
        cohort.samples,
        selection_config=PROTOCOL.split.calibration_selection,
    ).artifact
    forged = BoundProbeArtifact._validated(
        probe=unrelated,
        rollout=honest.rollout,
        cohort_manifest=honest.cohort_manifest,
        rows=honest.rows,
        candidate_features=honest.candidate_features,
        theta_rel_sha256=honest.theta_rel_sha256,
        failure_label_sha256=honest.failure_label_sha256,
        config_sha256=honest.config_sha256,
        code_sha256=honest.code_sha256,
    )
    with pytest.raises(BoundProbeError, match="untrusted"):
        validate_bound_probe_artifact(forged, protocol=PROTOCOL, repo_root=ROOT)
    with pytest.raises(BoundProbeError, match="untrusted"):
        _write(forged, tmp_path / "untrusted")

    path = _write_modified(tmp_path / "forged", forged.to_metadata())
    with pytest.raises(BoundProbeError, match="expected_sha256"):
        _load(path, expected_sha256=honest.sha256)


def test_loader_requires_explicit_expected_sha256(live_inputs, tmp_path: Path) -> None:
    artifact = _bind(*live_inputs)
    path = _write(artifact, tmp_path / "bound")
    with pytest.raises(TypeError, match="expected_sha256"):
        load_bound_probe_artifact(  # type: ignore[call-arg]
            path, protocol=PROTOCOL, repo_root=ROOT
        )


def test_loader_rejects_derived_hash_tamper_even_with_matching_directory(
    live_inputs, tmp_path: Path
) -> None:
    metadata = _bind(*live_inputs).to_metadata()
    metadata["training"]["row_alignment_sha256"] = "0" * 64
    path = _write_modified(tmp_path, metadata)
    with pytest.raises(BoundProbeError, match="inconsistent derived"):
        _load(path, expected_sha256=path.name)


def test_loader_rejects_noncanonical_layout_expected_hash_and_symlinks(
    live_inputs, tmp_path: Path
) -> None:
    artifact = _bind(*live_inputs)
    path = _write(artifact, tmp_path / "bound")
    canonical = (path / "bound_probe.json").read_bytes()
    (path / "bound_probe.json").write_bytes(canonical + b"\n")
    with pytest.raises(BoundProbeError, match="content hash"):
        _load(path, expected_sha256=artifact.sha256)

    other = _write(artifact, tmp_path / "other")
    (other / "unexpected").write_text("x")
    with pytest.raises(BoundProbeError, match="exactly one"):
        _load(other, expected_sha256=artifact.sha256)

    clean = _write(artifact, tmp_path / "clean")
    with pytest.raises(BoundProbeError, match="expected_sha256"):
        _load(clean, expected_sha256="f" * 64)
    link = tmp_path / "bound-link"
    link.symlink_to(clean, target_is_directory=True)
    with pytest.raises(BoundProbeError, match="real directory"):
        _load(link, expected_sha256=artifact.sha256)

    file_link_root = tmp_path / "file-link"
    file_link_root.mkdir()
    file_link_dir = file_link_root / artifact.sha256
    file_link_dir.mkdir()
    (file_link_dir / "bound_probe.json").symlink_to(clean / "bound_probe.json")
    with pytest.raises(BoundProbeError, match="regular file"):
        _load(file_link_dir, expected_sha256=artifact.sha256)


def test_loader_refuses_oversize_before_parsing(tmp_path: Path) -> None:
    directory = tmp_path / ("0" * 64)
    directory.mkdir()
    path = directory / "bound_probe.json"
    with path.open("wb") as stream:
        stream.truncate(16 * 1024 * 1024 + 1)
    with pytest.raises(BoundProbeError, match="size limit"):
        _load(directory, expected_sha256="0" * 64)


def test_writer_refuses_protected_and_symlink_roots(
    live_inputs, tmp_path: Path
) -> None:
    artifact = _bind(*live_inputs)
    with pytest.raises(BoundProbeError, match="config/lock"):
        _write(artifact, tmp_path / "configs" / "bound")
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(BoundProbeError, match="symlink"):
        _write(artifact, link / "bound")


def test_racing_file_creation_is_never_overwritten(
    live_inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _bind(*live_inputs)
    real_link = probe_artifacts_module.os.link

    def race(source, destination, *, follow_symlinks=True):
        Path(destination).write_text("racing writer")
        return real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(probe_artifacts_module.os, "link", race)
    root = tmp_path / "race"
    with pytest.raises(FileExistsError):
        _write(artifact, root)
    assert (root / artifact.sha256 / "bound_probe.json").read_text() == (
        "racing writer"
    )


def test_ordinary_publication_failure_cleans_own_destination_for_retry(
    live_inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _bind(*live_inputs)
    real_link = probe_artifacts_module.os.link

    def fail_link(*args, **kwargs):
        raise OSError("synthetic ordinary link failure")

    monkeypatch.setattr(probe_artifacts_module.os, "link", fail_link)
    root = tmp_path / "retry"
    with pytest.raises(OSError, match="ordinary link failure"):
        _write(artifact, root)
    assert not (root / artifact.sha256).exists()

    monkeypatch.setattr(probe_artifacts_module.os, "link", real_link)
    assert _write(artifact, root) == root / artifact.sha256


def _plain(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, dict):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_plain(nested) for nested in value]
    return value
