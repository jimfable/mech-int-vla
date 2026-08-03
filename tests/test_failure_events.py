from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mech_int_vla.artifacts import ArtifactHashes, RolloutArtifact
from mech_int_vla.failure_events import (
    AnnotationStatus,
    ArtifactIdentity,
    DiscoveryEpisode,
    FailureEventError,
    FailureEventTrace,
    FailureEventType,
    ReachableBounds,
    TaskIdentity,
    annotate_failure_event,
    artifact_identity_from_rollout,
    create_freeze_manifest,
    derive_discovery_bounds,
    discovery_episode_from_artifact,
    failure_event_trace_from_artifact,
)


def digest(character: str) -> str:
    return character * 64


def identity(episode_id: str, character: str = "a") -> ArtifactIdentity:
    return ArtifactIdentity(episode_id, digest(character), digest(character))


def trace(
    episode_id: str = "episode",
    *,
    actions: int = 10,
    valid: bool = True,
    success: bool = False,
    object_position: np.ndarray | None = None,
    eef_position: np.ndarray | None = None,
    contact: np.ndarray | None = None,
    grasped: np.ndarray | None = None,
    phase: np.ndarray | None = None,
    gripper: np.ndarray | None = None,
) -> FailureEventTrace:
    if not valid:
        actions = 0
    frame_count = actions + 1
    obj = (
        np.tile(np.asarray([0.0, 0.0, 0.1]), (frame_count, 1))
        if object_position is None
        else np.asarray(object_position, dtype=np.float64)
    )
    eef = np.zeros((frame_count, 3)) if eef_position is None else eef_position
    contact_array = (
        np.zeros(frame_count, dtype=np.bool_) if contact is None else contact
    )
    grasped_array = (
        np.zeros(frame_count, dtype=np.bool_) if grasped is None else grasped
    )
    phase_array = (
        np.full(frame_count, "pregrasp", dtype="U16") if phase is None else phase
    )
    action_array = np.zeros((actions, 7), dtype=np.float64)
    action_array[:, 6] = -1.0
    if gripper is not None:
        action_array[:, 6] = gripper
    terminated = np.zeros(actions, dtype=np.bool_)
    truncated = np.zeros(actions, dtype=np.bool_)
    if actions:
        if success or actions < 520:
            terminated[-1] = True
        else:
            truncated[-1] = True
    task_success = np.zeros(frame_count, dtype=np.bool_)
    task_success[-1] = success
    return FailureEventTrace(
        episode_id=episode_id,
        valid_reset=valid,
        success=success,
        validity_reasons=() if valid else ("invalid_reset",),
        actions=action_array,
        frame_control_step=np.arange(frame_count, dtype=np.int32),
        frame_eef_position=eef,
        frame_object_position=obj,
        frame_contact=contact_array,
        frame_grasped=grasped_array,
        frame_task_success=task_success,
        frame_phase=phase_array,
        terminated=terminated,
        truncated=truncated,
    )


BOUNDS = ReachableBounds((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))


def rollout_artifact(
    item: FailureEventTrace,
    *,
    metadata_digest: str = "d",
    trajectory_digest: str = "e",
    validity_reasons: tuple[str, ...] | None = None,
) -> RolloutArtifact:
    """Build the same frozen container returned after loader validation."""

    arrays = {
        "actions": item.actions.copy(),
        "frame_control_step": item.frame_control_step.copy(),
        "frame_eef_position": item.frame_eef_position.copy(),
        "frame_primary_object_position": item.frame_object_position.copy(),
        "frame_primary_gripper_contact": item.frame_contact.copy(),
        "frame_primary_grasped": item.frame_grasped.copy(),
        "frame_task_success": item.frame_task_success.copy(),
        "frame_phase": item.frame_phase.copy(),
        "terminated": item.terminated.copy(),
        "truncated": item.truncated.copy(),
    }
    reasons = item.validity_reasons if validity_reasons is None else validity_reasons
    return RolloutArtifact(
        path=Path("/synthetic-rollouts") / item.episode_id,
        metadata={
            "episode": {"episode_id": item.episode_id},
            "validity": {"valid": item.valid_reset, "reasons": list(reasons)},
            "outcome": {"success": item.success},
        },
        arrays=arrays,
        hashes=ArtifactHashes(
            metadata_sha256=digest(metadata_digest),
            trajectory_sha256=digest(trajectory_digest),
        ),
    )


@pytest.mark.parametrize(
    ("item", "reasons"),
    [
        (trace("failed", actions=3), ()),
        (trace("success", actions=3, success=True), ()),
        (
            trace("invalid", valid=False),
            ("reset_pose_out_of_tolerance", "settle_velocity_out_of_tolerance"),
        ),
    ],
    ids=("valid-failed", "valid-success", "invalid-reset"),
)
def test_rollout_conversion_preserves_validated_arrays_flags_reasons_and_hashes(
    item: FailureEventTrace, reasons: tuple[str, ...]
) -> None:
    artifact = rollout_artifact(item, validity_reasons=reasons)

    artifact_identity = artifact_identity_from_rollout(artifact)
    converted = failure_event_trace_from_artifact(artifact)
    episode = discovery_episode_from_artifact(artifact)

    assert artifact_identity.episode_id == item.episode_id
    assert artifact_identity.metadata_sha256 == digest("d")
    assert artifact_identity.trajectory_sha256 == digest("e")
    assert converted.valid_reset is item.valid_reset
    assert converted.success is item.success
    assert converted.validity_reasons == reasons
    for converted_name, artifact_name in (
        ("actions", "actions"),
        ("frame_control_step", "frame_control_step"),
        ("frame_eef_position", "frame_eef_position"),
        ("frame_object_position", "frame_primary_object_position"),
        ("frame_contact", "frame_primary_gripper_contact"),
        ("frame_grasped", "frame_primary_grasped"),
        ("frame_task_success", "frame_task_success"),
        ("frame_phase", "frame_phase"),
        ("terminated", "terminated"),
        ("truncated", "truncated"),
    ):
        assert np.array_equal(
            getattr(converted, converted_name), artifact.arrays[artifact_name]
        )
        assert not getattr(converted, converted_name).flags.writeable
    assert episode.artifact == artifact_identity
    assert episode.trace.episode_id == converted.episode_id
    assert episode.trace.valid_reset is converted.valid_reset
    assert episode.trace.success is converted.success
    assert episode.trace.validity_reasons == converted.validity_reasons
    assert np.array_equal(episode.trace.actions, converted.actions)

    artifact.arrays["frame_eef_position"][:] = 99.0
    assert not np.any(converted.frame_eef_position == 99.0)


def test_rollout_conversion_rejects_nonvalidated_container() -> None:
    with pytest.raises(FailureEventError, match="validated RolloutArtifact"):
        artifact_identity_from_rollout(object())  # type: ignore[arg-type]
    with pytest.raises(FailureEventError, match="validated RolloutArtifact"):
        failure_event_trace_from_artifact(object())  # type: ignore[arg-type]
    with pytest.raises(FailureEventError, match="validated RolloutArtifact"):
        discovery_episode_from_artifact(object())  # type: ignore[arg-type]


def test_trace_is_defensively_copied_readonly_and_fails_on_schema_drift() -> None:
    positions = np.zeros((11, 3))
    item = trace(object_position=positions)
    positions[:] = 99.0
    assert np.all(item.frame_object_position == 0.0)
    assert not item.actions.flags.writeable
    assert not item.frame_object_position.flags.writeable
    with pytest.raises(FailureEventError, match="0..len"):
        FailureEventTrace(
            **{
                **item.__dict__,
                "frame_control_step": np.asarray([0] * 11, dtype=np.int64),
            }
        )
    with pytest.raises(FailureEventError, match="finite"):
        bad = np.zeros((11, 3))
        bad[2, 1] = np.nan
        trace(object_position=bad)


def test_discovery_bounds_have_exact_coverage_and_exclude_failed_wandering() -> None:
    successful_positions = np.asarray(
        [[0.0, 0.0, 0.1], [1.0, 0.0, 0.2], [0.5, 0.0, 0.3]]
    )
    failed_positions = np.asarray(
        [[2.0, 0.0, 0.1], [100.0, 100.0, 100.0], [200.0, 200.0, 200.0]]
    )
    episodes = [
        DiscoveryEpisode(
            identity("success", "a"),
            trace(
                "success", actions=2, success=True, object_position=successful_positions
            ),
        ),
        DiscoveryEpisode(
            identity("failure", "b"),
            trace("failure", actions=2, object_position=failed_positions),
        ),
        DiscoveryEpisode(
            identity("invalid", "c"),
            trace(
                "invalid",
                valid=False,
                object_position=np.asarray([[999.0, 999.0, 999.0]]),
            ),
        ),
    ]
    expected = [episode.artifact for episode in episodes]
    result = derive_discovery_bounds(
        list(reversed(episodes)), expected_artifacts=list(reversed(expected))
    )
    assert result.raw_bounds.lower_xyz == (0.0, 0.0, 0.1)
    assert result.raw_bounds.upper_xyz == (2.0, 0.0, 0.3)
    assert result.expanded_bounds.lower_xyz == (-0.05, -0.05, 0.05)
    assert result.expanded_bounds.upper_xyz == (2.05, 0.05, 0.35)
    provenance = {item.artifact.episode_id: item for item in result.provenance}
    assert provenance["failure"].frame_zero_included
    assert not provenance["failure"].successful_path_included
    assert not provenance["invalid"].frame_zero_included

    with pytest.raises(FailureEventError, match="coverage mismatch"):
        derive_discovery_bounds(episodes[:-1], expected_artifacts=expected)
    wrong = list(episodes)
    wrong[0] = DiscoveryEpisode(identity("success", "d"), wrong[0].trace)
    with pytest.raises(FailureEventError, match="hash mismatch"):
        derive_discovery_bounds(wrong, expected_artifacts=expected)


def missed_trace(
    *, net: float = 0.005, contact_step: int | None = None
) -> FailureEventTrace:
    eef = np.zeros((11, 3))
    eef[1:, 0] = np.linspace(0.0, net, 10)
    obj = np.zeros((11, 3))
    contact = np.zeros(11, dtype=np.bool_)
    if contact_step is not None:
        contact[contact_step] = True
    gripper = np.full(10, -1.0)
    gripper[0] = 1.0
    return trace(
        eef_position=eef, object_position=obj, contact=contact, gripper=gripper
    )


def test_missed_grasp_exact_window_threshold_and_first_close_only() -> None:
    result = annotate_failure_event(missed_trace(), BOUNDS)
    assert result.event_type is FailureEventType.MISSED_GRASP
    assert (result.onset_step, result.confirmation_step) == (1, 10)

    below = np.nextafter(0.005, 0.0)
    assert (
        annotate_failure_event(missed_trace(net=below), BOUNDS).status
        is AnnotationStatus.UNANNOTATABLE_EARLY_TERMINAL
    )
    assert (
        annotate_failure_event(missed_trace(contact_step=10), BOUNDS).event_type is None
    )

    gripper = np.full(12, -1.0)
    gripper[[0, 2]] = 1.0
    eef = np.zeros((13, 3))
    eef[3:13, 0] = np.linspace(0.0, 0.01, 10)
    first_close_fails = trace(actions=12, eef_position=eef, gripper=gripper)
    assert annotate_failure_event(first_close_fails, BOUNDS).event_type is None


def test_incomplete_missed_grasp_window_does_not_qualify() -> None:
    gripper = np.full(10, -1.0)
    gripper[1] = 1.0
    eef = np.zeros((11, 3))
    eef[:, 0] = np.linspace(0.0, 0.1, 11)
    result = annotate_failure_event(trace(eef_position=eef, gripper=gripper), BOUNDS)
    assert result.status is AnnotationStatus.UNANNOTATABLE_EARLY_TERMINAL


def test_drop_uses_inclusive_height_dwell_and_recontact_restarts_loss() -> None:
    contact = np.asarray([False, True, False, False, False, False, False, False])
    grasped = np.asarray([False, True, False, False, False, False, False, False])
    obj = np.tile(np.asarray([0.0, 0.0, 0.2]), (8, 1))
    obj[0, 2] = 0.1
    obj[4:7, 2] = 0.105
    dropped = trace(actions=7, contact=contact, grasped=grasped, object_position=obj)
    result = annotate_failure_event(dropped, BOUNDS)
    assert result.event_type is FailureEventType.DROPPED_OBJECT
    assert (result.onset_step, result.confirmation_step) == (2, 6)

    contact = np.asarray([False, True, False, True, False, False, False, False])
    grasped = np.asarray([False, True, False, True, False, False, False, False])
    obj = np.tile(np.asarray([0.0, 0.0, 0.2]), (8, 1))
    obj[0, 2] = 0.1
    obj[5:8, 2] = 0.105
    reacquired = trace(actions=7, contact=contact, grasped=grasped, object_position=obj)
    restarted = annotate_failure_event(reacquired, BOUNDS)
    assert restarted.event_type is FailureEventType.DROPPED_OBJECT
    assert (restarted.onset_step, restarted.confirmation_step) == (4, 7)


def test_drop_rejects_above_threshold_placed_and_missing_prior_grasp() -> None:
    contact = np.asarray([True, False, False, False, False])
    grasped = np.asarray([True, False, False, False, False])
    obj = np.tile(np.asarray([0.0, 0.0, 0.1]), (5, 1))
    obj[1:, 2] = 0.1051
    above = trace(actions=4, contact=contact, grasped=grasped, object_position=obj)
    assert annotate_failure_event(above, BOUNDS).event_type is None

    obj[1:, 2] = 0.1
    phase = np.full(5, "pregrasp", dtype="U16")
    phase[1:4] = "placed"
    placed = trace(
        actions=4, contact=contact, grasped=grasped, object_position=obj, phase=phase
    )
    assert annotate_failure_event(placed, BOUNDS).event_type is None
    no_grasp = trace(
        actions=4,
        contact=contact,
        grasped=np.zeros(5, dtype=np.bool_),
        object_position=obj,
    )
    assert annotate_failure_event(no_grasp, BOUNDS).event_type is None


def test_workspace_exit_is_closed_bound_and_final_suffix_after_reentry() -> None:
    obj = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [1.2, 0.0, 0.0],
        ]
    )
    result = annotate_failure_event(trace(actions=5, object_position=obj), BOUNDS)
    assert result.event_type is FailureEventType.IRRECOVERABLE_WORKSPACE_EXIT
    assert (result.onset_step, result.confirmation_step) == (4, 5)

    all_outside = np.tile(np.asarray([2.0, 0.0, 0.0]), (4, 1))
    onset_zero = annotate_failure_event(
        trace(actions=3, object_position=all_outside), BOUNDS
    )
    assert onset_zero.onset_step == 0


def test_event_ties_follow_textual_precedence() -> None:
    item = missed_trace()
    obj = item.frame_object_position.copy()
    obj[1:, 0] = 2.0
    eef = np.zeros((11, 3))
    eef[1:, 0] = 2.0 + np.linspace(0.0, 0.006, 10)
    contact = item.frame_contact.copy()
    grasped = item.frame_grasped.copy()
    contact[0] = True
    grasped[0] = True
    tied = trace(
        eef_position=eef,
        object_position=obj,
        contact=contact,
        grasped=grasped,
        gripper=item.actions[:, 6],
    )
    result = annotate_failure_event(tied, BOUNDS)
    assert result.event_type is FailureEventType.MISSED_GRASP
    assert result.onset_step == 1


def test_success_invalid_and_terminal_fallback_cannot_leak_events() -> None:
    successful = missed_trace()
    success_trace = trace(
        success=True,
        eef_position=successful.frame_eef_position,
        gripper=successful.actions[:, 6],
    )
    assert (
        annotate_failure_event(success_trace, BOUNDS).status
        is AnnotationStatus.SUCCESS_NO_EVENT
    )
    invalid = trace(valid=False)
    assert (
        annotate_failure_event(invalid, BOUNDS).status
        is AnnotationStatus.EXCLUDED_INVALID_RESET
    )
    fallback = annotate_failure_event(trace(actions=520), BOUNDS)
    assert fallback.event_type is FailureEventType.TERMINAL_HORIZON
    assert fallback.onset_step == fallback.confirmation_step == 520
    early = annotate_failure_event(trace(actions=519), BOUNDS)
    assert early.status is AnnotationStatus.UNANNOTATABLE_EARLY_TERMINAL


def test_freeze_manifest_is_canonical_finite_and_exactly_covered() -> None:
    episodes = [
        DiscoveryEpisode(identity("a", "a"), trace("a", actions=1, success=True)),
        DiscoveryEpisode(identity("b", "b"), trace("b", actions=1)),
    ]
    bounds = derive_discovery_bounds(
        episodes, expected_artifacts=[item.artifact for item in episodes]
    )
    annotations = [
        annotate_failure_event(item.trace, bounds.expanded_bounds) for item in episodes
    ]
    task_id = TaskIdentity("libero_10", 5, 1, "pick up the book", "black_book", 2)
    first = create_freeze_manifest(
        task=task_id,
        bounds=bounds,
        primary_placement_predicate_keys=("0:In:black_book_1:back",),
        annotations=list(reversed(annotations)),
        video_audit_episode_ids=("b", "a"),
        implementation_commit="c" * 40,
    )
    second = create_freeze_manifest(
        task=task_id,
        bounds=bounds,
        primary_placement_predicate_keys=("0:In:black_book_1:back",),
        annotations=annotations,
        video_audit_episode_ids=("a", "b"),
        implementation_commit="c" * 40,
    )
    assert first.canonical_json() == second.canonical_json()
    assert first.sha256 == second.sha256
    payload = json.loads(first.canonical_json())
    assert (
        payload["protocol"]["dropped_object"]["proxy"] == "initial-support-height proxy"
    )
    assert payload["bounds"]["provenance"][1]["success"] is False

    with pytest.raises(FailureEventError, match="video audit IDs must exactly cover"):
        create_freeze_manifest(
            task=task_id,
            bounds=bounds,
            primary_placement_predicate_keys=("0:In:black_book_1:back",),
            annotations=annotations,
            video_audit_episode_ids=("a",),
            implementation_commit="c" * 40,
        )

    with pytest.raises(FailureEventError, match="exactly cover"):
        create_freeze_manifest(
            task=task_id,
            bounds=bounds,
            primary_placement_predicate_keys=("0:In:black_book_1:back",),
            annotations=annotations[:1],
            video_audit_episode_ids=("a",),
            implementation_commit="c" * 40,
        )
