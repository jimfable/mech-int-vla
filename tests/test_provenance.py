from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from mech_int_vla import provenance
from mech_int_vla.failure_events import ArtifactIdentity
from mech_int_vla.probes import (
    DEFAULT_CANDIDATE_PREFERENCE,
    FROZEN_RIDGE_ALPHA_GRID,
    AlphaCVResult,
    CandidateCVResult,
    CenteredCircularRidge,
    ProbeArtifact,
)
from mech_int_vla.provenance import (
    FROZEN_CONFIG_FILES,
    SCORING_SOURCE_FILES,
    ProvenanceError,
    content_links_for,
    frozen_config_sha256,
    scoring_source_sha256,
)


def _write_allowlists(root: Path) -> None:
    for index, relative in enumerate(
        (*FROZEN_CONFIG_FILES, *SCORING_SOURCE_FILES), start=1
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(f"{index}:{relative}\n".encode())


def _probe() -> ProbeArtifact:
    folds = (0.0,) * 5
    alpha_results = tuple(
        AlphaCVResult(
            alpha=alpha,
            fold_mae_rad=folds,
            mean_mae_rad=0.0,
            standard_error_rad=0.0,
        )
        for alpha in FROZEN_RIDGE_ALPHA_GRID
    )
    candidate_results = tuple(
        CandidateCVResult(
            candidate=name,
            alpha_results=alpha_results,
            selected_alpha=FROZEN_RIDGE_ALPHA_GRID[0],
            mean_mae_rad=0.0,
            standard_error_rad=0.0,
        )
        for name in DEFAULT_CANDIDATE_PREFERENCE
    )
    return ProbeArtifact(
        model=CenteredCircularRidge(
            alpha=FROZEN_RIDGE_ALPHA_GRID[0],
            symmetry_order=2,
            feature_center=np.asarray([0.0]),
            target_center=np.asarray([1.0, 0.0]),
            coefficient=np.asarray([[1.0], [0.0]]),
        ),
        candidate=DEFAULT_CANDIDATE_PREFERENCE[0],
        alpha_grid=FROZEN_RIDGE_ALPHA_GRID,
        candidate_preference=DEFAULT_CANDIDATE_PREFERENCE,
        candidate_results=candidate_results,
        one_standard_error_threshold_rad=0.0,
        fold_test_groups=((0,), (1,), (2,), (3,), (4,)),
        training_rows=10,
        training_episodes=5,
        training_base_init_state_ids=(0, 1, 2, 3, 4),
    )


def _independent_hash(root: Path, paths: tuple[str, ...], domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(len(domain).to_bytes(8, "big"))
    digest.update(domain)
    digest.update(len(paths).to_bytes(8, "big"))
    for relative in paths:
        encoded = relative.encode()
        contents = (root / relative).read_bytes()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def test_hashes_are_deterministic_framed_and_domain_separated(tmp_path: Path) -> None:
    _write_allowlists(tmp_path)

    config_first = frozen_config_sha256(tmp_path)
    source_first = scoring_source_sha256(tmp_path)

    assert config_first == frozen_config_sha256(tmp_path)
    assert source_first == scoring_source_sha256(tmp_path)
    assert config_first == _independent_hash(
        tmp_path, FROZEN_CONFIG_FILES, b"mech-int-vla:frozen-config:v1"
    )
    assert source_first == _independent_hash(
        tmp_path, SCORING_SOURCE_FILES, b"mech-int-vla:scoring-source:v1"
    )
    assert config_first != source_first
    assert len(config_first) == len(source_first) == 64


def test_mutation_changes_only_the_owning_allowlist_hash(tmp_path: Path) -> None:
    _write_allowlists(tmp_path)
    original_config = frozen_config_sha256(tmp_path)
    original_source = scoring_source_sha256(tmp_path)

    (tmp_path / FROZEN_CONFIG_FILES[0]).write_bytes(b"mutated config")
    assert frozen_config_sha256(tmp_path) != original_config
    assert scoring_source_sha256(tmp_path) == original_source

    (tmp_path / SCORING_SOURCE_FILES[0]).write_bytes(b"mutated source")
    assert scoring_source_sha256(tmp_path) != original_source


@pytest.mark.parametrize("kind", ["config", "source"])
def test_missing_allowlisted_file_fails_closed(tmp_path: Path, kind: str) -> None:
    _write_allowlists(tmp_path)
    if kind == "config":
        (tmp_path / FROZEN_CONFIG_FILES[0]).unlink()
        call = frozen_config_sha256
    else:
        (tmp_path / SCORING_SOURCE_FILES[0]).unlink()
        call = scoring_source_sha256

    with pytest.raises(ProvenanceError, match="allowlisted file"):
        call(tmp_path)


def test_symlinked_file_and_directory_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_allowlists(root)
    external = tmp_path / "external"
    external.write_bytes(b"outside")
    config_path = root / FROZEN_CONFIG_FILES[0]
    config_path.unlink()
    config_path.symlink_to(external)
    with pytest.raises(ProvenanceError, match="symlink"):
        frozen_config_sha256(root)

    config_path.unlink()
    config_path.write_bytes(b"restored")
    source_directory = root / "src"
    external_directory = tmp_path / "external-source"
    os.rename(source_directory, external_directory)
    source_directory.symlink_to(external_directory, target_is_directory=True)
    with pytest.raises(ProvenanceError, match="symlink component"):
        scoring_source_sha256(root)


def test_symlinked_repo_root_fails_closed(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    _write_allowlists(real_root)
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ProvenanceError, match="repo_root.*symlink"):
        frozen_config_sha256(linked_root)


def test_allowlist_repo_root_escape_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_allowlists(tmp_path)
    (tmp_path.parent / "outside").write_bytes(b"must not be read")
    monkeypatch.setattr(provenance, "FROZEN_CONFIG_FILES", ("../outside",))

    with pytest.raises(ProvenanceError, match="escapes"):
        frozen_config_sha256(tmp_path)


def test_content_links_compute_and_bind_every_exact_input(tmp_path: Path) -> None:
    _write_allowlists(tmp_path)
    raw = ArtifactIdentity(
        episode_id="calibration-task1-init10-iid",
        metadata_sha256="1" * 64,
        trajectory_sha256="2" * 64,
    )
    probe = _probe()

    links = content_links_for(raw, probe, tmp_path)

    assert links.raw_metadata_sha256 == raw.metadata_sha256
    assert links.raw_trajectory_sha256 == raw.trajectory_sha256
    assert links.probe_sha256 == probe.sha256()
    assert links.config_sha256 == frozen_config_sha256(tmp_path)
    assert links.code_sha256 == scoring_source_sha256(tmp_path)
    assert all(
        len(value) == 64 and value == value.lower()
        for value in links.as_dict().values()
    )


def test_content_links_reject_unvalidated_raw_or_probe(tmp_path: Path) -> None:
    _write_allowlists(tmp_path)
    raw = ArtifactIdentity("episode", "1" * 64, "2" * 64)
    probe = _probe()

    with pytest.raises(ProvenanceError, match="raw"):
        content_links_for(object(), probe, tmp_path)  # type: ignore[arg-type]
    with pytest.raises(ProvenanceError, match="probe"):
        content_links_for(raw, object(), tmp_path)  # type: ignore[arg-type]
