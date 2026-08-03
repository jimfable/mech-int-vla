from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mech_int_vla.config import load_protocol_config
from mech_int_vla.probes import (
    DEFAULT_CANDIDATE_PREFERENCE,
    AlphaCVResult,
    CandidateCVResult,
    CenteredCircularRidge,
    ProbeArtifact,
    ProbeError,
    ProbeSamples,
    circular_targets,
    cross_validate_circular_probe,
    episode_equal_weights,
    evaluate_mean_prediction_baseline,
    evaluate_proprioception_only_baseline,
    evaluate_random_label_baseline,
    evaluate_time_only_baseline,
    fit_centered_ridge,
    load_probe_artifact,
    make_group_folds,
    select_and_fit_circular_probe,
    select_candidate_one_standard_error,
    symmetry_aware_circular_error,
    write_probe_artifact,
)

ROOT = Path(__file__).parents[1]


def _samples(*, symmetry_order: int = 2) -> ProbeSamples:
    # Two episodes and three rows per base-init ID.
    groups = np.repeat(np.arange(10, 20), 6)
    episodes = np.array(
        [
            f"init{group}-episode{episode}"
            for group in range(10, 20)
            for episode in range(2)
            for _ in range(3)
        ]
    )
    theta = np.linspace(-1.2, 1.2, groups.size)
    return ProbeSamples.from_arrays(
        theta_rel=theta,
        base_init_state_id=groups,
        episode_id=episodes,
        symmetry_order=symmetry_order,
    )


def _candidate_result(name: str, mean: float, se: float) -> CandidateCVResult:
    delta = se * math.sqrt(2.0)
    folds = tuple(mean + multiple * delta for multiple in (-2, -1, 0, 1, 2))
    actual_mean = float(np.mean(folds))
    actual_se = float(np.std(folds, ddof=1) / math.sqrt(len(folds)))
    alpha = AlphaCVResult(
        alpha=0.1,
        fold_mae_rad=folds,
        mean_mae_rad=actual_mean,
        standard_error_rad=actual_se,
    )
    return CandidateCVResult(
        candidate=name,
        alpha_results=(alpha,),
        selected_alpha=0.1,
        mean_mae_rad=actual_mean,
        standard_error_rad=actual_se,
    )


@pytest.fixture(scope="module")
def fitted_artifact() -> ProbeArtifact:
    protocol = load_protocol_config(ROOT / "configs")
    samples = _samples(symmetry_order=2)
    features = circular_targets(samples.theta_rel, symmetry_order=2)
    candidates = {name: features.copy() for name in DEFAULT_CANDIDATE_PREFERENCE}
    return select_and_fit_circular_probe(
        candidates,
        samples,
        selection_config=protocol.split.calibration_selection,
    ).artifact


def test_episode_equal_weights_ignore_row_count() -> None:
    weights = episode_equal_weights(["a", "b", "b", "b"])

    assert weights.tolist() == pytest.approx([1.0, 1 / 3, 1 / 3, 1 / 3])
    assert weights[0] == pytest.approx(weights[1:].sum())


def test_group_folds_hold_out_every_group_and_row_once() -> None:
    samples = _samples()
    folds = make_group_folds(samples.base_init_state_id)

    assert len(folds) == 5
    assert [fold.test_groups for fold in folds] == [
        (10, 15),
        (11, 16),
        (12, 17),
        (13, 18),
        (14, 19),
    ]
    heldout_rows = np.concatenate([fold.test_rows for fold in folds])
    assert sorted(heldout_rows.tolist()) == list(range(samples.n_rows))
    for fold in folds:
        train_groups = set(samples.base_init_state_id[fold.train_rows])
        assert train_groups.isdisjoint(fold.test_groups)


def test_symmetry_aware_error_wraps_and_respects_object_symmetry() -> None:
    error = symmetry_aware_circular_error(
        np.array([math.pi - 0.1, math.pi / 2, 0.0]),
        np.array([-math.pi + 0.1, 0.0, math.pi]),
        symmetry_order=2,
    )

    assert error == pytest.approx([0.2, math.pi / 2, 0.0])


def test_centered_ridge_predicts_with_unit_normalization_only_at_prediction() -> None:
    samples = _samples(symmetry_order=1)
    target = circular_targets(samples.theta_rel, symmetry_order=1)
    features = np.column_stack((target, np.linspace(0.0, 1.0, samples.n_rows)))
    features += np.array([7.0, -4.0, 2.0])

    model = fit_centered_ridge(features, samples, alpha=100.0)
    raw = model.predict_raw(features)
    unit = model.predict_unit(features)

    assert model.coefficient.shape == (2, 3)
    assert np.linalg.norm(raw, axis=1) != pytest.approx(np.ones(samples.n_rows))
    assert np.linalg.norm(unit, axis=1) == pytest.approx(np.ones(samples.n_rows))
    assert np.isfinite(model.predict_angle(features)).all()


def test_cross_validation_uses_complete_alpha_grid_and_reports_fold_se() -> None:
    samples = _samples(symmetry_order=1)
    features = circular_targets(samples.theta_rel, symmetry_order=1)
    result = cross_validate_circular_probe(
        "synthetic",
        features,
        samples,
        alpha_grid=(0.0001, 0.1, 100.0),
    )

    assert [entry.alpha for entry in result.alpha_results] == [0.0001, 0.1, 100.0]
    assert all(len(entry.fold_mae_rad) == 5 for entry in result.alpha_results)
    assert all(entry.standard_error_rad >= 0.0 for entry in result.alpha_results)
    assert result.selected_alpha == 0.0001
    assert result.mean_mae_rad < 0.01


def test_one_standard_error_selection_uses_exact_frozen_preference() -> None:
    means = {
        "vlm_context": (0.205, 0.01),
        "early_expert_t1_0": (0.202, 0.01),
        "early_expert_t0_5": (0.201, 0.01),
        "late_expert_t1_0": (0.200, 0.01),
        "late_expert_t0_5": (0.199, 0.01),
    }
    results = tuple(
        _candidate_result(name, *means[name])
        for name in reversed(DEFAULT_CANDIDATE_PREFERENCE)
    )

    selected, threshold, eligible = select_candidate_one_standard_error(results)

    assert threshold == pytest.approx(0.209)
    assert eligible == DEFAULT_CANDIDATE_PREFERENCE
    assert selected.candidate == "vlm_context"


def test_full_selection_fits_final_probe_and_has_stable_hash_ready_metadata() -> None:
    protocol = load_protocol_config(ROOT / "configs")
    samples = _samples(symmetry_order=2)
    features = circular_targets(samples.theta_rel, symmetry_order=2)
    candidates = {name: features.copy() for name in DEFAULT_CANDIDATE_PREFERENCE}

    first = select_and_fit_circular_probe(
        candidates,
        samples,
        selection_config=protocol.split.calibration_selection,
    )
    second = select_and_fit_circular_probe(
        candidates,
        samples,
        selection_config=protocol.split.calibration_selection,
    )

    assert first.artifact.candidate == "vlm_context"
    assert first.eligible_candidates == DEFAULT_CANDIDATE_PREFERENCE
    assert first.artifact.model.alpha in {
        0.0001,
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    }
    assert first.artifact.model.coefficient.shape == (2, 2)
    assert first.artifact.sha256() == second.artifact.sha256()
    assert len(first.artifact.sha256()) == 64
    metadata = json.loads(first.artifact.canonical_json())
    assert metadata["cv"]["group"] == "base_init_state_id"
    assert metadata["training"]["episodes"] == 20
    assert metadata["selection"]["candidate_preference"] == list(
        DEFAULT_CANDIDATE_PREFERENCE
    )
    assert np.allclose(
        metadata["parameters"]["coefficient"], first.artifact.model.coefficient
    )


def test_required_baseline_interfaces_share_grouped_circular_evaluation() -> None:
    protocol = load_protocol_config(ROOT / "configs")
    samples = _samples(symmetry_order=1)
    target = circular_targets(samples.theta_rel, symmetry_order=1)
    time = np.linspace(0.0, 1.0, samples.n_rows)

    time_result = evaluate_time_only_baseline(
        time, samples, selection_config=protocol.split.calibration_selection
    )
    proprio_result = evaluate_proprioception_only_baseline(
        target, samples, selection_config=protocol.split.calibration_selection
    )
    random_result = evaluate_random_label_baseline(
        target,
        samples,
        randomized_theta_rel=samples.theta_rel[::-1],
        selection_config=protocol.split.calibration_selection,
    )
    mean_result = evaluate_mean_prediction_baseline(samples)

    assert time_result.candidate == "time_only"
    assert proprio_result.candidate == "proprioception_only"
    assert random_result.candidate == "random_label"
    assert len(mean_result.fold_mae_rad) == 5
    assert all(
        np.isfinite(result.mean_mae_rad)
        for result in (time_result, proprio_result, random_result, mean_result)
    )


def test_rejects_leaky_episode_groups_and_non_frozen_candidate_order() -> None:
    with pytest.raises(ProbeError, match="positive integer"):
        ProbeSamples.from_arrays(
            theta_rel=[0.0],
            base_init_state_id=[10],
            episode_id=["episode"],
            symmetry_order=1.5,
        )

    with pytest.raises(ProbeError, match="more than one base init"):
        ProbeSamples.from_arrays(
            theta_rel=[0.0, 0.1],
            base_init_state_id=[10, 11],
            episode_id=["same", "same"],
            symmetry_order=1,
        )

    protocol = load_protocol_config(ROOT / "configs")
    samples = _samples()
    features = circular_targets(samples.theta_rel, symmetry_order=2)
    with pytest.raises(ProbeError, match="five frozen candidates"):
        select_and_fit_circular_probe(
            {"vlm_context": features},
            samples,
            selection_config=protocol.split.calibration_selection,
        )


def test_probe_artifact_round_trip_is_byte_stable_and_executable(
    tmp_path: Path, fitted_artifact: ProbeArtifact
) -> None:
    first_path = write_probe_artifact(fitted_artifact, tmp_path / "first")
    second_path = write_probe_artifact(fitted_artifact, tmp_path / "second")

    assert first_path.name == fitted_artifact.sha256()
    assert (first_path / "probe.json").read_bytes() == fitted_artifact.canonical_json()
    assert (first_path / "probe.json").read_bytes() == (
        second_path / "probe.json"
    ).read_bytes()

    loaded = load_probe_artifact(first_path, expected_sha256=fitted_artifact.sha256())
    features = np.array([[1.0, 0.0], [0.0, 1.0], [-0.5, 0.25]])
    assert loaded.canonical_json() == fitted_artifact.canonical_json()
    assert loaded.sha256() == fitted_artifact.sha256()
    assert loaded.model.predict_raw(features) == pytest.approx(
        fitted_artifact.model.predict_raw(features)
    )
    assert loaded.model.predict_angle(features) == pytest.approx(
        fitted_artifact.model.predict_angle(features)
    )


def test_probe_model_and_loaded_arrays_are_defensive_read_only_copies(
    tmp_path: Path, fitted_artifact: ProbeArtifact
) -> None:
    center = np.array([1.0, 2.0])
    coefficient = np.eye(2)
    model = CenteredCircularRidge(
        alpha=0.1,
        symmetry_order=2,
        feature_center=center,
        target_center=np.array([0.0, 0.0]),
        coefficient=coefficient,
    )
    center[0] = 100.0
    coefficient[0, 0] = 100.0
    assert model.feature_center.tolist() == [1.0, 2.0]
    assert model.coefficient.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError):
        model.feature_center[0] = 3.0
    with pytest.raises(ValueError):
        model.coefficient[0, 0] = 3.0

    loaded = load_probe_artifact(write_probe_artifact(fitted_artifact, tmp_path))
    assert not loaded.model.feature_center.flags.writeable
    assert not loaded.model.target_center.flags.writeable
    assert not loaded.model.coefficient.flags.writeable
    with pytest.raises(ValueError):
        loaded.model.coefficient[0, 0] = 3.0


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"alpha": 0.0}, "positive"),
        ({"symmetry_order": 0}, "positive integer"),
        ({"feature_center": [[1.0, 2.0]]}, "rank 1"),
        ({"target_center": [0.0, 1.0, 2.0]}, "shape"),
        ({"coefficient": [[1.0], [2.0]]}, "feature_center width"),
        ({"coefficient": [[1.0, np.nan], [0.0, 1.0]]}, "finite"),
    ],
)
def test_centered_probe_rejects_malformed_parameters(
    changes: dict[str, object], match: str
) -> None:
    parameters: dict[str, object] = {
        "alpha": 0.1,
        "symmetry_order": 2,
        "feature_center": [0.0, 0.0],
        "target_center": [0.0, 0.0],
        "coefficient": [[1.0, 0.0], [0.0, 1.0]],
    }
    parameters.update(changes)
    with pytest.raises(ProbeError, match=match):
        CenteredCircularRidge(**parameters)  # type: ignore[arg-type]


def test_cv_results_and_artifact_consistency_fail_closed(
    fitted_artifact: ProbeArtifact,
) -> None:
    alpha_result = fitted_artifact.candidate_results[0].alpha_results[0]
    with pytest.raises(ProbeError, match="inconsistent with fold"):
        replace(alpha_result, mean_mae_rad=alpha_result.mean_mae_rad + 0.01)
    with pytest.raises(ProbeError, match="finite nonnegative"):
        replace(alpha_result, standard_error_rad=float("nan"))

    candidate_result = fitted_artifact.candidate_results[0]
    with pytest.raises(ProbeError, match="selected_alpha"):
        replace(candidate_result, selected_alpha=123.0)
    nonselected = next(
        result
        for result in candidate_result.alpha_results
        if result.alpha != candidate_result.selected_alpha
    )
    with pytest.raises(ProbeError, match="empirical-best"):
        replace(
            candidate_result,
            selected_alpha=nonselected.alpha,
            mean_mae_rad=nonselected.mean_mae_rad,
            standard_error_rad=nonselected.standard_error_rad,
        )
    with pytest.raises(ProbeError, match="selected alpha result"):
        replace(candidate_result, mean_mae_rad=candidate_result.mean_mae_rad + 0.01)

    with pytest.raises(ProbeError, match="frozen ridge grid"):
        replace(fitted_artifact, alpha_grid=tuple(reversed(fitted_artifact.alpha_grid)))
    with pytest.raises(ProbeError, match="not in candidate_preference"):
        replace(fitted_artifact, candidate="not-a-frozen-candidate")
    with pytest.raises(ProbeError, match="frozen order"):
        replace(
            fitted_artifact,
            candidate_preference=tuple(reversed(fitted_artifact.candidate_preference)),
        )
    with pytest.raises(ProbeError, match="deterministic grouped CV"):
        replace(
            fitted_artifact,
            fold_test_groups=tuple(reversed(fitted_artifact.fold_test_groups)),
        )
    with pytest.raises(ProbeError, match="training counts"):
        replace(fitted_artifact, training_rows=1)
    with pytest.raises(ProbeError, match="model alpha"):
        replace(
            fitted_artifact,
            model=replace(fitted_artifact.model, alpha=100.0),
        )


def test_probe_writer_refuses_overwrite_protected_and_symlink_paths(
    tmp_path: Path, fitted_artifact: ProbeArtifact
) -> None:
    output_root = tmp_path / "artifacts"
    path = write_probe_artifact(fitted_artifact, output_root)
    marker = path / "probe.json"
    original = marker.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        write_probe_artifact(fitted_artifact, output_root)
    assert marker.read_bytes() == original

    with pytest.raises(ProbeError, match="lock/config"):
        write_probe_artifact(
            fitted_artifact, tmp_path / "ignored" / ".." / "configs" / "probes"
        )

    real_root = tmp_path / "real-root"
    real_root.mkdir()
    symlink_root = tmp_path / "linked-root"
    symlink_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ProbeError, match="symlink"):
        write_probe_artifact(fitted_artifact, symlink_root)


def test_probe_loader_rejects_duplicate_unknown_missing_and_noncanonical_json(
    tmp_path: Path, fitted_artifact: ProbeArtifact
) -> None:
    duplicate_path = write_probe_artifact(fitted_artifact, tmp_path / "duplicate")
    duplicate_file = duplicate_path / "probe.json"
    duplicate_file.write_bytes(
        b'{"schema_version":1,' + fitted_artifact.canonical_json()[1:]
    )
    with pytest.raises(ProbeError, match="duplicate key"):
        load_probe_artifact(duplicate_path)

    unknown_path = write_probe_artifact(fitted_artifact, tmp_path / "unknown")
    unknown_file = unknown_path / "probe.json"
    unknown = json.loads(unknown_file.read_bytes())
    unknown["unexpected"] = True
    unknown_file.write_bytes(
        json.dumps(
            unknown,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    with pytest.raises(ProbeError, match="unexpected"):
        load_probe_artifact(unknown_path)

    missing_path = write_probe_artifact(fitted_artifact, tmp_path / "missing")
    missing_file = missing_path / "probe.json"
    missing = json.loads(missing_file.read_bytes())
    del missing["metric"]
    missing_file.write_bytes(
        json.dumps(missing, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    with pytest.raises(ProbeError, match="missing"):
        load_probe_artifact(missing_path)

    noncanonical_path = write_probe_artifact(fitted_artifact, tmp_path / "noncanonical")
    noncanonical_file = noncanonical_path / "probe.json"
    noncanonical_file.write_bytes(noncanonical_file.read_bytes() + b"\n")
    with pytest.raises(ProbeError, match="canonical encoding"):
        load_probe_artifact(noncanonical_path)


def test_probe_loader_rejects_hash_schema_layout_and_symlink_tampering(
    tmp_path: Path, fitted_artifact: ProbeArtifact
) -> None:
    hash_path = write_probe_artifact(fitted_artifact, tmp_path / "hash")
    hash_file = hash_path / "probe.json"
    changed = json.loads(hash_file.read_bytes())
    changed["parameters"]["coefficient"][0][0] += 0.125
    hash_file.write_bytes(
        json.dumps(
            changed,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    with pytest.raises(ProbeError, match="directory does not match.*SHA-256"):
        load_probe_artifact(hash_path)

    schema_path = write_probe_artifact(fitted_artifact, tmp_path / "schema")
    schema_file = schema_path / "probe.json"
    schema = json.loads(schema_file.read_bytes())
    schema["schema_version"] = 2
    schema_file.write_bytes(
        json.dumps(schema, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    with pytest.raises(ProbeError, match="schema"):
        load_probe_artifact(schema_path)

    layout_path = write_probe_artifact(fitted_artifact, tmp_path / "layout")
    (layout_path / "extra").write_bytes(b"unexpected")
    with pytest.raises(ProbeError, match="exactly one"):
        load_probe_artifact(layout_path)

    real_path = write_probe_artifact(fitted_artifact, tmp_path / "real")
    linked_path = tmp_path / "linked-artifact"
    linked_path.symlink_to(real_path, target_is_directory=True)
    with pytest.raises(ProbeError, match="symlink"):
        load_probe_artifact(linked_path)

    file_link_path = write_probe_artifact(fitted_artifact, tmp_path / "file-link")
    probe_file = file_link_path / "probe.json"
    external_file = tmp_path / "external-probe.json"
    external_file.write_bytes(probe_file.read_bytes())
    probe_file.unlink()
    probe_file.symlink_to(external_file)
    with pytest.raises(ProbeError, match="regular file.*symlink"):
        load_probe_artifact(file_link_path)

    with pytest.raises(ProbeError, match="lowercase hexadecimal"):
        load_probe_artifact(real_path, expected_sha256=fitted_artifact.sha256().upper())
    with pytest.raises(ProbeError, match="expected_sha256"):
        load_probe_artifact(real_path, expected_sha256="0" * 64)
