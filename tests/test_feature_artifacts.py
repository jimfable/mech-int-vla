from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mech_int_vla import feature_artifacts
from mech_int_vla.feature_artifacts import (
    FeatureArtifactError,
    load_feature_cohort,
    load_feature_reference_bundle,
    write_feature_cohort,
    write_feature_reference_bundle,
)
from mech_int_vla.feature_pipeline import (
    CohortIdentity,
    EpisodeSourceHashes,
    FeatureCohort,
    FeatureReferenceBundle,
    FeatureStateRecord,
    TaskIdentity,
)
from mech_int_vla.features import (
    COVERAGE_VECTOR_NAMES,
    M0_FEATURE_NAMES,
    M1_FEATURE_NAMES,
    M2_EXPERT_FEATURE_NAMES,
    ActionScale,
    CoverageState,
    FeatureHierarchy,
    NamedFeatureRow,
    ProbeNormState,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _task() -> TaskIdentity:
    return TaskIdentity(1, "libero_10", 5, "place the black book", "black_book", 1)


def _cohort_identity() -> CohortIdentity:
    return CohortIdentity(
        "pinned-policy",
        "pinned-vlm",
        "c" * 40,
        _sha("config"),
        _sha("code"),
    )


def _source(episode_id: str) -> EpisodeSourceHashes:
    return EpisodeSourceHashes(
        episode_id,
        _sha(f"{episode_id}:raw-metadata"),
        _sha(f"{episode_id}:raw-trajectory"),
        _sha(f"{episode_id}:score-metadata"),
        _sha(f"{episode_id}:score-primitives"),
    )


@pytest.fixture
def reference_bundle() -> FeatureReferenceBundle:
    episode_ids = ("calibration-ep-a", "calibration-ep-b")
    coverage = []
    norms = []
    for index, episode_id in enumerate(episode_ids):
        vector = np.linspace(index, index + 1, len(COVERAGE_VECTOR_NAMES))
        if index == 1:
            vector[3] = np.nan
        coverage.append(
            CoverageState(
                episode_id,
                10 + index,
                index * 5,
                "calibration",
                "pregrasp" if index == 0 else "transport",
                index == 0,
                vector,
            )
        )
        norms.append(
            ProbeNormState(
                episode_id,
                10 + index,
                index * 5,
                "calibration",
                "pregrasp" if index == 0 else "transport",
                index == 0,
                1.25 + index,
            )
        )
    return FeatureReferenceBundle(
        action_scale=ActionScale(
            np.linspace(0.5, 1.1, 7), np.zeros(7, dtype=np.bool_), episode_ids
        ),
        coverage_states=tuple(coverage),
        probe_norm_states=tuple(norms),
        probe_sha256=_sha("probe"),
        selected_candidate="vlm_context",
        task_identity=_task(),
        cohort_identity=_cohort_identity(),
        source_hashes=tuple(_source(episode_id) for episode_id in episode_ids),
    )


def _hierarchy(offset: float) -> FeatureHierarchy:
    m0_values = np.arange(len(M0_FEATURE_NAMES), dtype=np.float64) + offset
    if offset:
        m0_values[2] = np.nan
    m1_values = np.concatenate(
        (m0_values, np.arange(len(M1_FEATURE_NAMES) - len(m0_values)) + 20 + offset)
    )
    m2_values = np.concatenate(
        (m1_values, np.arange(len(M2_EXPERT_FEATURE_NAMES) - len(m1_values)) + 80)
    )
    return FeatureHierarchy(
        NamedFeatureRow(
            M0_FEATURE_NAMES,
            m0_values,
            {"component": "M0", "nested": {"offset": offset, "tags": ["a", "b"]}},
        ),
        NamedFeatureRow(
            M1_FEATURE_NAMES,
            m1_values,
            {"component": "M1_full", "parent": _sha(f"m0:{offset}")},
        ),
        NamedFeatureRow(
            M2_EXPERT_FEATURE_NAMES,
            m2_values,
            {"component": "M2_full", "unicode": "orientation-θ"},
        ),
    )


@pytest.fixture
def feature_cohort(reference_bundle: FeatureReferenceBundle) -> FeatureCohort:
    records = []
    for index, episode_id in enumerate(("calibration-ep-a", "calibration-ep-b")):
        records.append(
            FeatureStateRecord(
                episode_id=episode_id,
                base_init_state_id=10 + index,
                split="calibration",
                control_step=index * 5,
                terminal_failure_label=index == 1,
                phase="pregrasp" if index == 0 else "transport",
                hierarchy=_hierarchy(float(index)),
                source_hashes=_source(episode_id),
            )
        )
    return FeatureCohort(
        split="calibration",
        records=tuple(records),
        m0_names=M0_FEATURE_NAMES,
        m1_names=M1_FEATURE_NAMES,
        m2_names=M2_EXPERT_FEATURE_NAMES,
        m0_matrix=np.stack([record.m0.values for record in records]),
        m1_matrix=np.stack([record.m1.values for record in records]),
        m2_matrix=np.stack([record.m2.values for record in records]),
        probe_sha256=reference_bundle.probe_sha256,
        reference_bundle_sha256=reference_bundle.metadata_sha256,
        task_identity=_task(),
        cohort_identity=_cohort_identity(),
    )


def _same_arrays(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left, right, equal_nan=True))


def _rehash_directory(path: Path) -> Path:
    digest = feature_artifacts._directory_digest(
        (path / "metadata.json").read_bytes(), (path / "arrays.npz").read_bytes()
    )
    destination = path.parent / digest
    path.rename(destination)
    return destination


def _replace_arrays(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    array_bytes = feature_artifacts._deterministic_npz_bytes(arrays)
    (path / "arrays.npz").write_bytes(array_bytes)
    metadata = json.loads((path / "metadata.json").read_bytes())
    metadata["arrays"]["sha256"] = hashlib.sha256(array_bytes).hexdigest()
    (path / "metadata.json").write_bytes(
        feature_artifacts._canonical_json_bytes(metadata)
    )
    return _rehash_directory(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / "arrays.npz", allow_pickle=False) as loaded:
        return {name: np.array(loaded[name], copy=True) for name in loaded.files}


def test_reference_round_trip_is_deterministic_complete_and_immutable(
    tmp_path: Path, reference_bundle: FeatureReferenceBundle
) -> None:
    first = write_feature_reference_bundle(reference_bundle, tmp_path / "first")
    second = write_feature_reference_bundle(reference_bundle, tmp_path / "second")
    assert first.name == second.name
    for name in ("metadata.json", "arrays.npz"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    loaded = load_feature_reference_bundle(first, expected_sha256=first.name)
    assert loaded.to_metadata() == reference_bundle.to_metadata()
    assert _same_arrays(
        loaded.action_scale.values, reference_bundle.action_scale.values
    )
    assert np.array_equal(
        loaded.action_scale.replaced_by_one,
        reference_bundle.action_scale.replaced_by_one,
    )
    for actual, expected in zip(
        loaded.coverage_states, reference_bundle.coverage_states, strict=True
    ):
        assert (
            actual.episode_id,
            actual.base_init_state_id,
            actual.control_step,
            actual.split,
            actual.phase,
            actual.success,
        ) == (
            expected.episode_id,
            expected.base_init_state_id,
            expected.control_step,
            expected.split,
            expected.phase,
            expected.success,
        )
        assert _same_arrays(actual.vector, expected.vector)
        with pytest.raises(ValueError):
            actual.vector[0] = 99.0


def test_cohort_round_trip_preserves_matrices_rows_hierarchy_and_provenance(
    tmp_path: Path, feature_cohort: FeatureCohort
) -> None:
    first = write_feature_cohort(feature_cohort, tmp_path / "first")
    second = write_feature_cohort(feature_cohort, tmp_path / "second")
    assert first.name == second.name
    assert (first / "metadata.json").read_bytes() == (
        second / "metadata.json"
    ).read_bytes()
    assert (first / "arrays.npz").read_bytes() == (second / "arrays.npz").read_bytes()
    loaded = load_feature_cohort(first, expected_sha256=first.name)
    assert loaded.to_metadata() == feature_cohort.to_metadata()
    for name in ("m0_matrix", "m1_matrix", "m2_matrix"):
        assert _same_arrays(getattr(loaded, name), getattr(feature_cohort, name))
        assert not getattr(loaded, name).flags.writeable
    for actual, expected in zip(loaded.records, feature_cohort.records, strict=True):
        assert actual.source_hashes == expected.source_hashes
        assert dict(actual.m0.metadata) == dict(expected.m0.metadata)
        assert dict(actual.m1.metadata) == dict(expected.m1.metadata)
        assert dict(actual.m2.metadata) == dict(expected.m2.metadata)
        assert actual.hierarchy.metadata_sha256 == expected.hierarchy.metadata_sha256


@pytest.mark.parametrize("protected", ["configs", "locks"])
def test_writers_refuse_protected_and_existing_destinations(
    tmp_path: Path,
    reference_bundle: FeatureReferenceBundle,
    protected: str,
) -> None:
    with pytest.raises(FeatureArtifactError, match="config or lock"):
        write_feature_reference_bundle(
            reference_bundle, tmp_path / "ignored" / ".." / protected / "features"
        )
    root = tmp_path / f"safe-{protected}"
    first = write_feature_reference_bundle(reference_bundle, root)
    marker = first / "metadata.json"
    before = marker.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        write_feature_reference_bundle(reference_bundle, root)
    assert marker.read_bytes() == before


def test_writers_and_loaders_reject_symlink_paths_and_files(
    tmp_path: Path, reference_bundle: FeatureReferenceBundle
) -> None:
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(FeatureArtifactError, match="symlink"):
        write_feature_reference_bundle(reference_bundle, linked_root)

    real_artifact = write_feature_reference_bundle(reference_bundle, tmp_path / "real")
    linked_artifact = tmp_path / "linked-artifact"
    linked_artifact.symlink_to(real_artifact, target_is_directory=True)
    with pytest.raises(FeatureArtifactError, match="symlink"):
        load_feature_reference_bundle(linked_artifact)

    file_link_artifact = write_feature_reference_bundle(
        reference_bundle, tmp_path / "file-link"
    )
    metadata_path = file_link_artifact / "metadata.json"
    external = tmp_path / "external.json"
    external.write_bytes(metadata_path.read_bytes())
    metadata_path.unlink()
    metadata_path.symlink_to(external)
    with pytest.raises(FeatureArtifactError, match="symlink"):
        load_feature_reference_bundle(file_link_artifact)


def test_loader_rejects_noncanonical_duplicate_and_extra_files(
    tmp_path: Path, reference_bundle: FeatureReferenceBundle
) -> None:
    noncanonical = write_feature_reference_bundle(
        reference_bundle, tmp_path / "noncanonical"
    )
    metadata_path = noncanonical / "metadata.json"
    metadata_path.write_bytes(metadata_path.read_bytes() + b"\n")
    with pytest.raises(FeatureArtifactError, match="canonical encoding"):
        load_feature_reference_bundle(noncanonical)

    duplicate = write_feature_reference_bundle(reference_bundle, tmp_path / "duplicate")
    metadata_path = duplicate / "metadata.json"
    metadata_path.write_bytes(b'{"schema_version":1,' + metadata_path.read_bytes()[1:])
    with pytest.raises(FeatureArtifactError, match="duplicate key"):
        load_feature_reference_bundle(duplicate)

    extra = write_feature_reference_bundle(reference_bundle, tmp_path / "extra")
    (extra / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(FeatureArtifactError, match="exactly"):
        load_feature_reference_bundle(extra)


def test_loader_rejects_physical_logical_dtype_and_shape_tampering(
    tmp_path: Path, feature_cohort: FeatureCohort
) -> None:
    physical = write_feature_cohort(feature_cohort, tmp_path / "physical")
    (physical / "arrays.npz").write_bytes(
        (physical / "arrays.npz").read_bytes() + b"tamper"
    )
    physical = _rehash_directory(physical)
    with pytest.raises(FeatureArtifactError, match="arrays.npz SHA-256"):
        load_feature_cohort(physical)

    logical = write_feature_cohort(feature_cohort, tmp_path / "logical")
    logical_arrays = _load_npz(logical)
    logical_arrays["m0_matrix"][0, 0] += 0.25
    logical = _replace_arrays(logical, logical_arrays)
    with pytest.raises(FeatureArtifactError, match="logical SHA-256"):
        load_feature_cohort(logical)

    dtype = write_feature_cohort(feature_cohort, tmp_path / "dtype")
    dtype_arrays = _load_npz(dtype)
    dtype_arrays["m0_matrix"] = dtype_arrays["m0_matrix"].astype(np.float32)
    dtype = _replace_arrays(dtype, dtype_arrays)
    with pytest.raises(FeatureArtifactError, match="dtype differs"):
        load_feature_cohort(dtype)

    shape = write_feature_cohort(feature_cohort, tmp_path / "shape")
    shape_arrays = _load_npz(shape)
    shape_arrays["m1_matrix"] = shape_arrays["m1_matrix"][:, :-1]
    shape = _replace_arrays(shape, shape_arrays)
    with pytest.raises(FeatureArtifactError, match="shape differs"):
        load_feature_cohort(shape)


def test_loader_rejects_wrong_digest_kind_schema_and_unknown_metadata(
    tmp_path: Path,
    reference_bundle: FeatureReferenceBundle,
    feature_cohort: FeatureCohort,
) -> None:
    reference = write_feature_reference_bundle(reference_bundle, tmp_path / "reference")
    with pytest.raises(FeatureArtifactError, match="expected_sha256"):
        load_feature_reference_bundle(reference, expected_sha256="0" * 64)
    with pytest.raises(FeatureArtifactError, match="lowercase SHA-256"):
        load_feature_reference_bundle(reference, expected_sha256=reference.name.upper())
    with pytest.raises(FeatureArtifactError, match="not a feature cohort"):
        load_feature_cohort(reference)

    schema = write_feature_cohort(feature_cohort, tmp_path / "schema")
    metadata = json.loads((schema / "metadata.json").read_bytes())
    metadata["schema_version"] = 2
    (schema / "metadata.json").write_bytes(
        feature_artifacts._canonical_json_bytes(metadata)
    )
    schema = _rehash_directory(schema)
    with pytest.raises(FeatureArtifactError, match="schema_version"):
        load_feature_cohort(schema)

    unknown = write_feature_cohort(feature_cohort, tmp_path / "unknown")
    metadata = json.loads((unknown / "metadata.json").read_bytes())
    metadata["unexpected"] = True
    (unknown / "metadata.json").write_bytes(
        feature_artifacts._canonical_json_bytes(metadata)
    )
    unknown = _rehash_directory(unknown)
    with pytest.raises(FeatureArtifactError, match="unexpected"):
        load_feature_cohort(unknown)


def test_failed_publish_removes_staging_and_lock(
    tmp_path: Path,
    reference_bundle: FeatureReferenceBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "crash"

    def fail_rename(_source: Path, _destination: Path) -> None:
        raise RuntimeError("injected rename failure")

    monkeypatch.setattr(feature_artifacts.os, "rename", fail_rename)
    with pytest.raises(RuntimeError, match="injected rename failure"):
        write_feature_reference_bundle(reference_bundle, root)
    assert root.is_dir()
    assert list(root.iterdir()) == []


def test_npz_with_object_payload_is_never_unpickled(
    tmp_path: Path, feature_cohort: FeatureCohort
) -> None:
    path = write_feature_cohort(feature_cohort, tmp_path / "object")
    arrays = _load_npz(path)
    arrays["m0_matrix"] = np.asarray([[object()]], dtype=object)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    array_bytes = buffer.getvalue()
    (path / "arrays.npz").write_bytes(array_bytes)
    metadata = json.loads((path / "metadata.json").read_bytes())
    metadata["arrays"]["sha256"] = hashlib.sha256(array_bytes).hexdigest()
    (path / "metadata.json").write_bytes(
        feature_artifacts._canonical_json_bytes(metadata)
    )
    path = _rehash_directory(path)
    with pytest.raises(FeatureArtifactError, match="without pickle"):
        load_feature_cohort(path)


def test_nan_payload_bits_do_not_change_canonical_array_bytes() -> None:
    first = np.asarray([0x7FF8000000000001], dtype=np.uint64).view(np.float64)
    second = np.asarray([0x7FF8000000000042], dtype=np.uint64).view(np.float64)
    assert np.isnan(first[0]) and np.isnan(second[0])
    assert feature_artifacts._deterministic_npz_bytes(
        {"value": first}
    ) == feature_artifacts._deterministic_npz_bytes({"value": second})


def test_reference_with_noncanonical_nan_payload_round_trips(
    tmp_path: Path, reference_bundle: FeatureReferenceBundle
) -> None:
    state = reference_bundle.coverage_states[1]
    vector = np.array(state.vector, copy=True)
    vector[3] = np.asarray([0x7FF8000000000042], dtype=np.uint64).view(np.float64)[0]
    replaced_state = CoverageState(
        state.episode_id,
        state.base_init_state_id,
        state.control_step,
        state.split,
        state.phase,
        state.success,
        vector,
    )
    bundle = replace(
        reference_bundle,
        coverage_states=(reference_bundle.coverage_states[0], replaced_state),
    )

    path = write_feature_reference_bundle(bundle, tmp_path / "nan-payload")
    loaded = load_feature_reference_bundle(path)

    assert loaded.to_metadata() == bundle.to_metadata()
    assert np.array_equal(
        loaded.coverage_states[1].vector,
        bundle.coverage_states[1].vector,
        equal_nan=True,
    )


def test_loader_rejects_oversized_compressed_member_before_materializing(
    tmp_path: Path, reference_bundle: FeatureReferenceBundle
) -> None:
    path = write_feature_reference_bundle(reference_bundle, tmp_path / "zip-bomb")
    arrays = _load_npz(path)
    arrays["coverage_vectors"] = np.zeros(
        (100_000, len(COVERAGE_VECTOR_NAMES)), dtype=np.float64
    )
    path = _replace_arrays(path, arrays)

    with pytest.raises(FeatureArtifactError, match="compressed member exceeds"):
        load_feature_reference_bundle(path)
