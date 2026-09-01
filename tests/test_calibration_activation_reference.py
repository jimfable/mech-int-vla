from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "calibration_activation_reference_under_test",
    ROOT / "ops" / "build_calibration_activation_reference.py",
)
assert SPEC is not None and SPEC.loader is not None
reference_ops = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference_ops
SPEC.loader.exec_module(reference_ops)


def _sha(character: str) -> str:
    return character * 64


def _sample() -> tuple[dict, dict[str, np.ndarray]]:
    rows_per_episode = 1500
    rows = rows_per_episode * 2
    arrays = {
        "activation_vectors": np.arange(rows * 2, dtype="<f8").reshape(rows, 2),
        "episode_index": np.repeat(np.asarray([0, 1], dtype="<i2"), rows_per_episode),
        "base_init_state_id": np.repeat(
            np.asarray([10, 11], dtype="<i4"), rows_per_episode
        ),
        "control_step": np.tile(
            np.arange(rows_per_episode, dtype="<i4") * 5, 2
        ),
        "natural_query_index": np.asarray(
            np.random.Generator(
                np.random.PCG64(reference_ops.NATURAL_QUERY_SEED)
            ).choice(
                rows, size=reference_ops.NATURAL_QUERY_COUNT, replace=False
            ),
            dtype="<i4",
        ),
        "natural_five_nn_distance": np.full(
            reference_ops.NATURAL_QUERY_COUNT,
            reference_ops.FROZEN_NATURAL_95TH_PERCENTILE,
            dtype="<f8",
        ),
    }
    episodes = []
    for index, episode_id in enumerate(("calibration-a", "calibration-b")):
        episodes.append(
            {
                "episode_index": index,
                "episode_id": episode_id,
                "base_init_state_id": 10 + index,
                "row_start": index * rows_per_episode,
                "row_stop": (index + 1) * rows_per_episode,
                "raw_metadata_sha256": _sha("a"),
                "raw_trajectory_sha256": _sha("b"),
                "score_metadata_sha256": _sha("c"),
                "score_primitives_sha256": _sha("d"),
            }
        )
    source_names = (
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
    )
    metadata = {
        "schema_version": 1,
        "format": reference_ops.ARTIFACT_FORMAT,
        "kind": reference_ops.ARTIFACT_KIND,
        "split": "calibration",
        "selection": {
            "source": "rescore_factual_original_selected_activation",
            "selected_candidate": "early_expert_t1_0",
            "cadence_control_steps": 5,
            "outcome_conditioned": False,
            "labels_used": False,
            "refit_performed": False,
            "transformation": "float64_arithmetic_mean_over_all_frozen_original_draws",
            "natural_draws": 8,
        },
        "geometry": {
            "method": "mean_euclidean_distance_to_5_nearest_leave_self_out",
            "query_seed": reference_ops.NATURAL_QUERY_SEED,
            "query_count": reference_ops.NATURAL_QUERY_COUNT,
            "percentile": reference_ops.NATURAL_PERCENTILE,
            "natural_95th_percentile": reference_ops.FROZEN_NATURAL_95TH_PERCENTILE,
        },
        "counts": {"episodes": 2, "rows": rows, "width": 2},
        "source": {
            **{name: _sha("e") for name in source_names},
            "raw_bytes_reverified_locally": False,
            "raw_hash_links_verified_against_frozen_cohort": True,
        },
        "episodes": episodes,
    }
    return metadata, arrays


def test_content_addressed_round_trip_is_exact_and_resumable(tmp_path: Path) -> None:
    metadata, arrays = _sample()

    first = reference_ops.publish_activation_reference(metadata, arrays, tmp_path)
    second = reference_ops.publish_activation_reference(metadata, arrays, tmp_path)
    loaded = reference_ops.load_activation_reference(first, expected_sha256=first.name)

    assert first == second
    assert loaded.sha256 == first.name
    assert loaded.metadata["selection"]["labels_used"] is False
    assert loaded.metadata["selection"]["refit_performed"] is False
    assert np.array_equal(
        loaded.arrays["activation_vectors"], arrays["activation_vectors"]
    )
    assert loaded.arrays["activation_vectors"].flags.writeable is False


def test_loader_rejects_extra_topology(tmp_path: Path) -> None:
    metadata, arrays = _sample()
    artifact = reference_ops.publish_activation_reference(metadata, arrays, tmp_path)
    (artifact / "unexpected").write_bytes(b"tamper")

    with pytest.raises(reference_ops.ActivationReferenceError, match="topology"):
        reference_ops.load_activation_reference(artifact, expected_sha256=artifact.name)


def test_loader_rejects_tampered_array_bytes(tmp_path: Path) -> None:
    metadata, arrays = _sample()
    artifact = reference_ops.publish_activation_reference(metadata, arrays, tmp_path)
    with (artifact / "arrays.npz").open("ab") as stream:
        stream.write(b"tamper")

    with pytest.raises(reference_ops.ActivationReferenceError, match="content hash"):
        reference_ops.load_activation_reference(artifact, expected_sha256=artifact.name)


def test_selection_contract_rejects_labels_or_refit() -> None:
    metadata, arrays = _sample()
    metadata["selection"]["labels_used"] = True

    with pytest.raises(
        reference_ops.ActivationReferenceError, match="selection protocol"
    ):
        reference_ops._artifact_payload(metadata, arrays)

    metadata["selection"]["labels_used"] = False
    metadata["selection"]["refit_performed"] = True
    with pytest.raises(
        reference_ops.ActivationReferenceError, match="selection protocol"
    ):
        reference_ops._artifact_payload(metadata, arrays)


def test_off_cadence_rows_are_rejected() -> None:
    metadata, arrays = _sample()
    arrays["control_step"][1] = 4

    with pytest.raises(reference_ops.ActivationReferenceError, match="cadence"):
        reference_ops._artifact_payload(metadata, arrays)


def test_geometry_indices_and_percentile_are_frozen() -> None:
    metadata, arrays = _sample()
    arrays["natural_query_index"][0] = arrays["natural_query_index"][1]
    with pytest.raises(reference_ops.ActivationReferenceError, match="geometry"):
        reference_ops._artifact_payload(metadata, arrays)

    metadata, arrays = _sample()
    arrays["natural_five_nn_distance"] += 1.0
    with pytest.raises(
        reference_ops.ActivationReferenceError, match="geometry protocol"
    ):
        reference_ops._artifact_payload(metadata, arrays)


def test_split_topology_rejects_extra_and_symlink_entries(tmp_path: Path) -> None:
    split = tmp_path / "calibration"
    split.mkdir()
    (split / "episode-a").mkdir()
    (split / "episode-b").mkdir()
    reference_ops._exact_split_directories(
        tmp_path, "calibration", {"episode-a", "episode-b"}, "synthetic"
    )

    (split / ".episode-a.tmp-stale").mkdir()
    with pytest.raises(reference_ops.ActivationReferenceError, match="topology"):
        reference_ops._exact_split_directories(
            tmp_path, "calibration", {"episode-a", "episode-b"}, "synthetic"
        )
    (split / ".episode-a.tmp-stale").rmdir()
    (split / "episode-b").rmdir()
    (split / "episode-b").symlink_to(split / "episode-a", target_is_directory=True)
    with pytest.raises(
        reference_ops.ActivationReferenceError, match="real directories"
    ):
        reference_ops._exact_split_directories(
            tmp_path, "calibration", {"episode-a", "episode-b"}, "synthetic"
        )


def test_output_root_rejects_ambiguous_prior_artifact(tmp_path: Path) -> None:
    metadata, arrays = _sample()
    (tmp_path / ("f" * 64)).mkdir()

    with pytest.raises(reference_ops.ActivationReferenceError, match="ambiguous"):
        reference_ops.publish_activation_reference(metadata, arrays, tmp_path)


def test_output_root_allows_valid_immutable_predecessor(tmp_path: Path) -> None:
    metadata, arrays = _sample()
    first = reference_ops.publish_activation_reference(metadata, arrays, tmp_path)
    changed = {name: np.array(value, copy=True) for name, value in arrays.items()}
    changed["activation_vectors"][0, 0] += 1.0

    second = reference_ops.publish_activation_reference(metadata, changed, tmp_path)

    assert first != second
    assert first.is_dir() and second.is_dir()


def test_loader_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    metadata, arrays = _sample()
    real_root = tmp_path / "real"
    artifact = reference_ops.publish_activation_reference(metadata, arrays, real_root)
    alias = tmp_path / "alias"
    alias.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(reference_ops.ActivationReferenceError, match="symlink"):
        reference_ops.load_activation_reference(alias / artifact.name)


def test_builder_contains_no_fit_or_label_selection_path() -> None:
    source = Path(reference_ops.__file__).read_text(encoding="utf-8")

    assert "fit_failure_predictors" not in source
    assert "select_and_fit" not in source
    assert 'metadata["outcome"]' not in source
    assert "failure_label" not in source


def test_natural_vectors_average_all_frozen_original_draws() -> None:
    draws = np.arange(2 * 8 * 3, dtype="<f4").reshape(2, 8, 3)

    observed = reference_ops._natural_activation_vectors(
        draws, state_count=2, expected_width=3
    )

    assert observed.dtype == np.dtype("<f8")
    assert np.array_equal(observed, draws.astype("<f8").mean(axis=1))
    with pytest.raises(reference_ops.ActivationReferenceError, match="activations"):
        reference_ops._natural_activation_vectors(
            draws[:, :7], state_count=2, expected_width=3
        )


def test_source_link_and_row_alignment_tampering_fail_closed() -> None:
    expected = {"raw_metadata_sha256": _sha("a")}
    with pytest.raises(reference_ops.ActivationReferenceError, match="links"):
        reference_ops._require_exact_mapping(
            {"raw_metadata_sha256": _sha("b")}, expected, "links"
        )

    rows = (("calibration-a", 10, 0), ("calibration-a", 10, 5))
    with pytest.raises(reference_ops.ActivationReferenceError, match="rows differ"):
        reference_ops._require_row_alignment(
            rows,
            rows,
            (("calibration-a", 10, 0),),
            rows,
        )


def test_cli_help_bootstraps_src_without_pythonpath() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "ops" / "build_calibration_activation_reference.py"), "--help"],
        cwd="/",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _run_synthetic_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tamper_links: bool = False,
):
    root = tmp_path / "repo"
    (root / "locks").mkdir(parents=True)
    score_root = root / "scores"
    episode_ids = ("calibration-a", "calibration-b")
    for episode_id in episode_ids:
        (score_root / "calibration" / episode_id).mkdir(parents=True)

    manifest_payload = {"kind": "synthetic-calibration-manifest"}
    manifest_file = root / "artifacts" / "manifest.json"
    manifest_file.parent.mkdir()
    manifest_file.write_bytes(reference_ops._canonical(manifest_payload))
    bound_directory = root / "artifacts" / "bound" / ("b" * 64)
    bound_directory.mkdir(parents=True)
    bound_file = bound_directory / "bound_probe.json"
    bound_file.write_bytes(b"synthetic-bound")
    reference_directory = root / "artifacts" / "reference" / ("f" * 64)
    reference_directory.mkdir(parents=True)
    reference_metadata_file = reference_directory / "metadata.json"
    reference_arrays_file = reference_directory / "arrays.npz"
    reference_metadata_file.write_bytes(b"synthetic-reference-metadata")
    reference_arrays_file.write_bytes(b"synthetic-reference-arrays")

    artifact_files = {
        "calibration_manifest": manifest_file,
        "bound_probe": bound_file,
        "feature_reference_metadata": reference_metadata_file,
        "feature_reference_arrays": reference_arrays_file,
    }
    freeze = {
        "artifact_hashes": {
            name: {
                "path": str(path.relative_to(root)),
                "sha256": reference_ops._sha256_file(path),
            }
            for name, path in artifact_files.items()
        }
    }
    freeze_path = root / "locks" / "calibration_frozen.json"
    freeze_path.write_bytes(reference_ops._canonical(freeze))

    rows_per_episode = 1500
    steps = np.arange(rows_per_episode, dtype="<i4") * 5
    row_objects = []
    coverage = []
    specifications = []
    raw_identities = []
    reference_sources = []
    sidecars = {}
    bound_sha = reference_ops._sha256_file(bound_file)
    manifest_sha = reference_ops._sha256_file(manifest_file)
    config_sha, code_sha = _sha("1"), _sha("2")
    for episode_index, episode_id in enumerate(episode_ids):
        base_id = 10 + episode_index
        specifications.append(
            SimpleNamespace(
                episode_id=episode_id,
                base_init_state_id=base_id,
                to_dict=lambda episode_id=episode_id, base_id=base_id: {
                    "episode_id": episode_id,
                    "base_init_state_id": base_id,
                },
            )
        )
        raw_metadata_sha = _sha(str(3 + episode_index))
        raw_trajectory_sha = _sha(str(5 + episode_index))
        score_metadata_sha = _sha(str(7 + episode_index))
        score_primitives_sha = _sha(str(9 - episode_index))
        raw_identities.append(
            SimpleNamespace(
                episode_id=episode_id,
                metadata_sha256=raw_metadata_sha,
                trajectory_sha256=raw_trajectory_sha,
            )
        )
        reference_sources.append(
            SimpleNamespace(
                episode_id=episode_id,
                raw_metadata_sha256=raw_metadata_sha,
                raw_trajectory_sha256=raw_trajectory_sha,
                score_metadata_sha256=score_metadata_sha,
                score_primitives_sha256=score_primitives_sha,
            )
        )
        for step in steps:
            row = SimpleNamespace(
                episode_id=episode_id,
                base_init_state_id=base_id,
                control_step=int(step),
            )
            row_objects.append(row)
            coverage.append(row)
        draws = np.arange(
            rows_per_episode * reference_ops.ORIGINAL_DRAWS * 2, dtype="<f4"
        ).reshape(rows_per_episode, reference_ops.ORIGINAL_DRAWS, 2)
        draws += episode_index * 100_000
        links = {
            "raw_metadata_sha256": raw_metadata_sha,
            "raw_trajectory_sha256": raw_trajectory_sha,
            "probe_sha256": bound_sha,
            "config_sha256": config_sha,
            "code_sha256": code_sha,
        }
        if tamper_links and episode_index == 0:
            links["raw_metadata_sha256"] = _sha("a")
        sidecars[episode_id] = SimpleNamespace(
            metadata={
                "split": "calibration",
                "links": links,
                "capture": {"selected_activation_width": 2},
            },
            arrays={"control_step": steps, "original_activation": draws},
            metadata_sha256=score_metadata_sha,
            primitives_sha256=score_primitives_sha,
        )

    manifest = SimpleNamespace(
        episodes=tuple(specifications), to_dict=lambda: manifest_payload
    )
    rollout = SimpleNamespace(
        source=SimpleNamespace(
            split=reference_ops.SplitName.CALIBRATION,
            manifest_sha256=manifest_sha,
        ),
        invalid_episode_ids=(),
        valid_episode_ids=episode_ids,
        valid_artifacts=tuple(raw_identities),
        manifest=manifest,
        sha256=_sha("c"),
    )
    bound = SimpleNamespace(
        rollout=rollout,
        probe=SimpleNamespace(candidate="early_expert_t1_0"),
        sha256=bound_sha,
        candidate_features=(
            SimpleNamespace(
                candidate="early_expert_t1_0", rows=rows_per_episode * 2, width=2
            ),
        ),
        rows=tuple(row_objects),
    )
    reference = SimpleNamespace(
        probe_sha256=bound_sha,
        selected_candidate="early_expert_t1_0",
        source_hashes=tuple(reference_sources),
        coverage_states=tuple(coverage),
        probe_norm_states=tuple(coverage),
    )
    monkeypatch.setattr(reference_ops, "EXPECTED_EPISODES", 2)
    monkeypatch.setattr(
        reference_ops, "load_protocol_config", lambda _path: SimpleNamespace()
    )
    monkeypatch.setattr(
        reference_ops, "load_bound_probe_artifact", lambda *_args, **_kwargs: bound
    )
    monkeypatch.setattr(
        reference_ops,
        "load_feature_reference_bundle",
        lambda *_args, **_kwargs: reference,
    )
    monkeypatch.setattr(
        reference_ops,
        "load_scoring_sidecar",
        lambda path, **_kwargs: sidecars[path.name],
    )
    monkeypatch.setattr(reference_ops, "frozen_config_sha256", lambda _root: config_sha)
    monkeypatch.setattr(reference_ops, "scoring_source_sha256", lambda _root: code_sha)
    monkeypatch.setattr(
        reference_ops,
        "ScoreAllocationReceipt",
        lambda **_kwargs: SimpleNamespace(sha256=_sha("d")),
    )
    query = np.arange(reference_ops.NATURAL_QUERY_COUNT, dtype="<i4")
    distance = np.full(
        reference_ops.NATURAL_QUERY_COUNT,
        reference_ops.FROZEN_NATURAL_95TH_PERCENTILE,
        dtype="<f8",
    )
    monkeypatch.setattr(
        reference_ops, "_frozen_natural_geometry", lambda _matrix: (query, distance)
    )
    arguments = SimpleNamespace(
        repo_root=root,
        calibration_freeze=freeze_path,
        manifest=manifest_file,
        bound_probe=bound_directory,
        feature_reference=reference_directory,
        score_root=score_root,
    )
    return reference_ops._build_from_frozen_sources(arguments), sidecars


def test_synthetic_builder_path_aggregates_all_draws_and_binds_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (metadata, arrays), sidecars = _run_synthetic_builder(tmp_path, monkeypatch)

    expected = np.concatenate(
        [
            sidecars[episode_id]
            .arrays["original_activation"]
            .astype("<f8")
            .mean(axis=1)
            for episode_id in ("calibration-a", "calibration-b")
        ]
    )
    assert np.array_equal(arrays["activation_vectors"], expected)
    assert metadata["counts"] == {"episodes": 2, "rows": 3000, "width": 2}
    assert metadata["source"]["raw_bytes_reverified_locally"] is False
    assert metadata["source"]["raw_hash_links_verified_against_frozen_cohort"] is True


def test_synthetic_builder_path_rejects_tampered_sidecar_source_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(reference_ops.ActivationReferenceError, match="source links"):
        _run_synthetic_builder(tmp_path, monkeypatch, tamper_links=True)
