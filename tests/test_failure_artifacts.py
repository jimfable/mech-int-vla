from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mech_int_vla import failure_artifacts as persistence
from mech_int_vla.failure_artifacts import (
    FailureArtifactError,
    FailureEventArtifact,
    create_failure_event_artifact,
    create_failure_event_artifacts,
    load_failure_event_artifact,
    load_failure_freeze,
    write_failure_event_artifact,
    write_failure_freeze,
)
from mech_int_vla.failure_events import (
    AnnotationStatus,
    ArtifactIdentity,
    DiscoveryBoundsDerivation,
    DiscoveryEpisodeProvenance,
    FailureEventFreezeManifest,
    FailureEventResult,
    FailureEventType,
    ReachableBounds,
    TaskIdentity,
)


def _digest(character: str) -> str:
    return character * 64


def _identity(episode_id: str, character: str) -> ArtifactIdentity:
    trajectory_character = format((int(character, 16) + 1) % 16, "x")
    return ArtifactIdentity(
        episode_id=episode_id,
        metadata_sha256=_digest(character),
        trajectory_sha256=_digest(trajectory_character),
    )


@pytest.fixture
def freeze() -> FailureEventFreezeManifest:
    identities = (
        _identity("a-success", "a"),
        _identity("b-failed", "b"),
        _identity("c-invalid", "c"),
    )
    provenance = (
        DiscoveryEpisodeProvenance(
            artifact=identities[0],
            valid_reset=True,
            success=True,
            validity_reasons=(),
            frame_zero_included=True,
            successful_path_included=True,
        ),
        DiscoveryEpisodeProvenance(
            artifact=identities[1],
            valid_reset=True,
            success=False,
            validity_reasons=(),
            frame_zero_included=True,
            successful_path_included=False,
        ),
        DiscoveryEpisodeProvenance(
            artifact=identities[2],
            valid_reset=False,
            success=False,
            validity_reasons=("reset_pose_out_of_tolerance",),
            frame_zero_included=False,
            successful_path_included=False,
        ),
    )
    bounds = DiscoveryBoundsDerivation(
        raw_bounds=ReachableBounds((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        expanded_bounds=ReachableBounds((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05)),
        margin_m=0.05,
        expected_artifacts=identities,
        provenance=provenance,
    )
    annotations = (
        FailureEventResult(
            "a-success",
            True,
            True,
            AnnotationStatus.SUCCESS_NO_EVENT,
            None,
            None,
            None,
        ),
        FailureEventResult(
            "b-failed",
            True,
            False,
            AnnotationStatus.ANNOTATED,
            FailureEventType.TERMINAL_HORIZON,
            520,
            520,
        ),
        FailureEventResult(
            "c-invalid",
            False,
            False,
            AnnotationStatus.EXCLUDED_INVALID_RESET,
            None,
            None,
            None,
        ),
    )
    return FailureEventFreezeManifest(
        task=TaskIdentity(
            "libero_10",
            5,
            1,
            "pick up the black book and place it in the back compartment",
            "black_book",
            2,
        ),
        bounds=bounds,
        primary_placement_predicate_keys=("0:In:black_book_1:back",),
        annotations=annotations,
        video_audit_episode_ids=tuple(item.episode_id for item in identities),
        implementation_commit="d" * 40,
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_freeze_roundtrip_preserves_exact_hashes_annotations_and_video_audit(
    tmp_path: Path, freeze: FailureEventFreezeManifest
) -> None:
    path = write_failure_freeze(freeze, tmp_path / "freezes")
    assert path.name == freeze.sha256
    assert (path / "failure-freeze.json").read_bytes() == freeze.canonical_json()

    loaded = load_failure_freeze(path, expected_sha256=freeze.sha256)
    assert loaded == freeze
    assert loaded.sha256 == freeze.sha256
    assert loaded.bounds.expected_artifacts == freeze.bounds.expected_artifacts
    assert loaded.annotations == freeze.annotations
    assert loaded.video_audit_episode_ids == freeze.video_audit_episode_ids


def test_per_episode_records_are_canonical_freeze_bound_and_roundtrip(
    tmp_path: Path, freeze: FailureEventFreezeManifest
) -> None:
    records = create_failure_event_artifacts(freeze)
    assert tuple(item.annotation.episode_id for item in records) == (
        "a-success",
        "b-failed",
        "c-invalid",
    )
    assert all(item.freeze_sha256 == freeze.sha256 for item in records)

    for record, source, annotation in zip(
        records,
        freeze.bounds.expected_artifacts,
        freeze.annotations,
        strict=True,
    ):
        assert record.source_artifact == source
        assert record.annotation == annotation
        path = write_failure_event_artifact(record, tmp_path / "events", freeze=freeze)
        assert path.name == record.sha256
        loaded = load_failure_event_artifact(
            path, expected_sha256=record.sha256, freeze=freeze
        )
        assert loaded == record

    with pytest.raises(FailureArtifactError, match="exact artifact.*video audit"):
        create_failure_event_artifact(freeze, "not-present")


def test_record_container_rejects_cross_episode_and_cross_freeze_links(
    tmp_path: Path, freeze: FailureEventFreezeManifest
) -> None:
    first = create_failure_event_artifact(freeze, "a-success")
    with pytest.raises(FailureArtifactError, match="episode IDs differ"):
        FailureEventArtifact(
            source_artifact=freeze.bounds.expected_artifacts[1],
            annotation=first.annotation,
            freeze_sha256=freeze.sha256,
        )

    path = write_failure_event_artifact(first, tmp_path / "records", freeze=freeze)
    other_freeze = replace(freeze, implementation_commit="e" * 40)
    with pytest.raises(FailureArtifactError, match="differs from the supplied freeze"):
        load_failure_event_artifact(path, freeze=other_freeze)


@pytest.mark.parametrize("writer_kind", ["freeze", "event"])
def test_writers_refuse_overwrite_protected_and_symlink_paths(
    tmp_path: Path,
    freeze: FailureEventFreezeManifest,
    writer_kind: str,
) -> None:
    value = (
        freeze
        if writer_kind == "freeze"
        else create_failure_event_artifact(freeze, "a-success")
    )

    def writer(item: object, root: Path) -> Path:
        if writer_kind == "freeze":
            return write_failure_freeze(item, root)  # type: ignore[arg-type]
        return write_failure_event_artifact(  # type: ignore[arg-type]
            item, root, freeze=freeze
        )

    output_root = tmp_path / writer_kind
    path = writer(value, output_root)
    original = next(path.iterdir()).read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        writer(value, output_root)
    assert next(path.iterdir()).read_bytes() == original

    with pytest.raises(FailureArtifactError, match="lock/config"):
        writer(value, tmp_path / "discarded" / ".." / "configs" / writer_kind)

    real_root = tmp_path / f"real-{writer_kind}"
    real_root.mkdir()
    linked_root = tmp_path / f"linked-{writer_kind}"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(FailureArtifactError, match="symlink"):
        writer(value, linked_root)


def test_publication_failure_removes_staging_and_lock(
    tmp_path: Path,
    freeze: FailureEventFreezeManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fsync = persistence._fsync_directory
    calls = 0

    def fail_after_staging(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected publication crash")
        real_fsync(path)

    monkeypatch.setattr(persistence, "_fsync_directory", fail_after_staging)
    output_root = tmp_path / "crash"
    with pytest.raises(RuntimeError, match="injected publication crash"):
        write_failure_freeze(freeze, output_root)
    assert output_root.is_dir()
    assert list(output_root.iterdir()) == []


def test_freeze_loader_rejects_duplicate_unknown_missing_schema_and_noncanonical(
    tmp_path: Path, freeze: FailureEventFreezeManifest
) -> None:
    duplicate = write_failure_freeze(freeze, tmp_path / "duplicate")
    duplicate_file = duplicate / "failure-freeze.json"
    duplicate_file.write_bytes(b'{"schema_version":1,' + freeze.canonical_json()[1:])
    with pytest.raises(FailureArtifactError, match="duplicate key"):
        load_failure_freeze(duplicate)

    unknown = write_failure_freeze(freeze, tmp_path / "unknown")
    unknown_file = unknown / "failure-freeze.json"
    unknown_data = json.loads(unknown_file.read_bytes())
    unknown_data["unexpected"] = True
    unknown_file.write_bytes(_canonical(unknown_data))
    with pytest.raises(FailureArtifactError, match="unexpected"):
        load_failure_freeze(unknown)

    missing = write_failure_freeze(freeze, tmp_path / "missing")
    missing_file = missing / "failure-freeze.json"
    missing_data = json.loads(missing_file.read_bytes())
    del missing_data["video_audit_episode_ids"]
    missing_file.write_bytes(_canonical(missing_data))
    with pytest.raises(FailureArtifactError, match="missing"):
        load_failure_freeze(missing)

    schema = write_failure_freeze(freeze, tmp_path / "schema")
    schema_file = schema / "failure-freeze.json"
    schema_data = json.loads(schema_file.read_bytes())
    schema_data["schema_version"] = 2
    schema_file.write_bytes(_canonical(schema_data))
    with pytest.raises(FailureArtifactError, match="schema_version"):
        load_failure_freeze(schema)

    noncanonical = write_failure_freeze(freeze, tmp_path / "noncanonical")
    noncanonical_file = noncanonical / "failure-freeze.json"
    noncanonical_file.write_bytes(noncanonical_file.read_bytes() + b"\n")
    with pytest.raises(FailureArtifactError, match="canonical encoding"):
        load_failure_freeze(noncanonical)


def test_freeze_loader_rejects_hash_protocol_and_coverage_tampering(
    tmp_path: Path, freeze: FailureEventFreezeManifest
) -> None:
    changed_hash = write_failure_freeze(freeze, tmp_path / "hash")
    changed_hash_file = changed_hash / "failure-freeze.json"
    changed_hash_data = json.loads(changed_hash_file.read_bytes())
    changed_hash_data["bounds"]["expected_artifacts"][0]["metadata_sha256"] = _digest(
        "f"
    )
    changed_hash_file.write_bytes(_canonical(changed_hash_data))
    with pytest.raises(FailureArtifactError, match="provenance hashes"):
        load_failure_freeze(changed_hash)

    protocol = write_failure_freeze(freeze, tmp_path / "protocol")
    protocol_file = protocol / "failure-freeze.json"
    protocol_data = json.loads(protocol_file.read_bytes())
    protocol_data["protocol"]["workspace_exit"]["margin_m"] = 0.06
    protocol_file.write_bytes(_canonical(protocol_data))
    with pytest.raises(FailureArtifactError, match="protocol metadata has drifted"):
        load_failure_freeze(protocol)

    audit = write_failure_freeze(freeze, tmp_path / "audit")
    audit_file = audit / "failure-freeze.json"
    audit_data = json.loads(audit_file.read_bytes())
    audit_data["video_audit_episode_ids"].pop()
    audit_file.write_bytes(_canonical(audit_data))
    with pytest.raises(FailureArtifactError, match="video audit IDs"):
        load_failure_freeze(audit)


def test_loaders_reject_layout_symlink_file_symlink_and_digest_tampering(
    tmp_path: Path, freeze: FailureEventFreezeManifest
) -> None:
    layout = write_failure_freeze(freeze, tmp_path / "layout")
    (layout / "extra").write_bytes(b"unexpected")
    with pytest.raises(FailureArtifactError, match="exactly one"):
        load_failure_freeze(layout)

    real = write_failure_freeze(freeze, tmp_path / "real")
    linked = tmp_path / "linked-artifact"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(FailureArtifactError, match="symlink"):
        load_failure_freeze(linked)

    file_link = write_failure_freeze(freeze, tmp_path / "file-link")
    freeze_file = file_link / "failure-freeze.json"
    external = tmp_path / "external.json"
    external.write_bytes(freeze_file.read_bytes())
    freeze_file.unlink()
    freeze_file.symlink_to(external)
    with pytest.raises(FailureArtifactError, match="regular file.*symlink"):
        load_failure_freeze(file_link)

    hash_path = write_failure_freeze(freeze, tmp_path / "digest")
    hash_file = hash_path / "failure-freeze.json"
    hash_file.write_bytes(hash_file.read_bytes().replace(b"black_book", b"black-book"))
    with pytest.raises(FailureArtifactError, match="SHA-256"):
        load_failure_freeze(hash_path)

    with pytest.raises(FailureArtifactError, match="lowercase hexadecimal"):
        load_failure_freeze(real, expected_sha256=freeze.sha256.upper())
    with pytest.raises(FailureArtifactError, match="expected_sha256"):
        load_failure_freeze(real, expected_sha256="0" * 64)


def test_event_loader_rejects_schema_layout_noncanonical_and_freeze_tampering(
    tmp_path: Path, freeze: FailureEventFreezeManifest
) -> None:
    record = create_failure_event_artifact(freeze, "b-failed")

    schema = write_failure_event_artifact(record, tmp_path / "schema", freeze=freeze)
    schema_file = schema / "failure-event.json"
    schema_data = json.loads(schema_file.read_bytes())
    schema_data["schema_version"] = 2
    schema_file.write_bytes(_canonical(schema_data))
    with pytest.raises(FailureArtifactError, match="schema_version"):
        load_failure_event_artifact(schema, freeze=freeze)

    noncanonical = write_failure_event_artifact(
        record, tmp_path / "noncanonical", freeze=freeze
    )
    event_file = noncanonical / "failure-event.json"
    event_file.write_bytes(event_file.read_bytes() + b"\n")
    with pytest.raises(FailureArtifactError, match="canonical encoding"):
        load_failure_event_artifact(noncanonical, freeze=freeze)

    wrong_freeze_record = replace(record, freeze_sha256=_digest("e"))
    with pytest.raises(FailureArtifactError, match="differs from the freeze"):
        write_failure_event_artifact(
            wrong_freeze_record, tmp_path / "tampered", freeze=freeze
        )
    tampered = persistence._write_artifact(
        canonical=wrong_freeze_record.canonical_json(),
        output_root=tmp_path / "tampered-load",
        filename="failure-event.json",
        description="failure-event artifact",
    )
    with pytest.raises(FailureArtifactError, match="supplied freeze"):
        load_failure_event_artifact(tampered, freeze=freeze)

    layout = write_failure_event_artifact(record, tmp_path / "layout", freeze=freeze)
    (layout / "unexpected").mkdir()
    with pytest.raises(FailureArtifactError, match="exactly one"):
        load_failure_event_artifact(layout, freeze=freeze)


def test_freeze_native_types_are_canonical_and_numpy_steps_are_normalized(
    freeze: FailureEventFreezeManifest,
) -> None:
    with pytest.raises(ValueError, match="version or tag"):
        replace(freeze, schema_version=True)
    with pytest.raises(ValueError, match="annotations"):
        replace(freeze, annotations=list(freeze.annotations))  # type: ignore[arg-type]

    event = replace(
        freeze.annotations[1],
        onset_step=np.int64(520),
        confirmation_step=np.int64(520),
    )
    assert type(event.onset_step) is int
    assert type(event.confirmation_step) is int
    assert (
        b'"onset_step":520'
        in FailureEventArtifact(
            freeze.bounds.expected_artifacts[1], event, freeze.sha256
        ).canonical_json()
    )


def test_freeze_loader_wraps_huge_finite_json_number(
    tmp_path: Path, freeze: FailureEventFreezeManifest
) -> None:
    path = write_failure_freeze(freeze, tmp_path / "huge-number")
    metadata_path = path / "failure-freeze.json"
    metadata = json.loads(metadata_path.read_bytes())
    metadata["bounds"]["margin_m"] = 10**400
    metadata_path.write_bytes(_canonical(metadata))

    with pytest.raises(FailureArtifactError, match="finite number"):
        load_failure_freeze(path)
