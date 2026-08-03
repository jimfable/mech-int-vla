from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from pathlib import Path

import pytest

from mech_int_vla.artifacts import ArtifactHashes, RolloutArtifact
from mech_int_vla.config import SplitName, load_protocol_config
from mech_int_vla.failure_events import ArtifactIdentity
from mech_int_vla.manifest import Manifest, generate_episode_manifest
from mech_int_vla.reality_gate import (
    ORIENTATION_CADENCE,
    ORIENTATION_WEIGHTING,
    DiscoveryCellResult,
    OrientationState,
    RealityGateError,
    RealityGateLockReceipt,
    RealityGateReceipt,
    TaskDiscoveryAttempt,
    TaskGateDecision,
    ValidityRetryEvidence,
    _DISCOVERY_ATTEMPT_FACTORY_TOKEN,
    _DISCOVERY_CELL_FACTORY_TOKEN,
    _ORIENTATION_ROLLOUT_FACTORY_TOKEN,
    _evaluate_orientation_eligibility,
    decide_reality_gate,
    evaluate_orientation_eligibility,
    evaluate_task_attempt,
    finalize_reality_gate,
    orientation_eligibility_from_metadata,
    reality_gate_receipt_from_metadata,
    variable_fallback_order,
)

ROOT = Path(__file__).parents[1]
POLICY = "31d453f7edd78c839a8bbc39744a292686daf0de"
COMMIT = "b491dc76641efe3a5c5d7eef6bb87af13d85f10b"


@pytest.fixture(scope="module")
def protocol():
    return load_protocol_config(ROOT / "configs")


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def identity(episode_id: str, suffix: str = "") -> ArtifactIdentity:
    return ArtifactIdentity(
        episode_id=episode_id,
        metadata_sha256=digest(f"metadata:{episode_id}:{suffix}"),
        trajectory_sha256=digest(f"trajectory:{episode_id}:{suffix}"),
    )


def dummy_rollout(episode_id: str, suffix: str = "") -> RolloutArtifact:
    return RolloutArtifact(
        path=Path("/synthetic") / episode_id,
        metadata={"episode": {"episode_id": episode_id}},
        arrays={},
        hashes=ArtifactHashes(
            metadata_sha256=digest(f"metadata:{episode_id}:{suffix}"),
            trajectory_sha256=digest(f"trajectory:{episode_id}:{suffix}"),
        ),
    )


def executed_rollouts_for_cells(
    cells: tuple[DiscoveryCellResult, ...] | list[DiscoveryCellResult],
) -> tuple[RolloutArtifact, ...]:
    return tuple(dummy_rollout(cell.episode_id) for cell in cells if cell.executed)


def discovery_manifest(
    protocol,
    rank: int = 1,
    *,
    policy_revision: str = POLICY,
    code_commit: str = COMMIT,
) -> Manifest:
    return generate_episode_manifest(
        SplitName.DISCOVERY,
        protocol.task_order.tasks[rank - 1],
        protocol,
        policy_revision=policy_revision,
        code_commit=code_commit,
    )


def task_attempt(
    manifest: Manifest,
    *,
    iid_successes: int,
    perturbation_failures: int = 6,
    perturbation_invalid: int = 0,
    run_perturbations: bool | None = None,
    reverse: bool = False,
    hash_suffix: str = "",
    retry_agrees_on_invalidity: bool = True,
    retry_agrees_on_reasons: bool = True,
) -> TaskDiscoveryAttempt:
    if run_perturbations is None:
        run_perturbations = iid_successes >= 6
    iid_seen = 0
    perturb_seen = 0
    cells: list[DiscoveryCellResult] = []
    for episode in manifest.episodes:
        if episode.condition_family == "iid":
            success = iid_seen < iid_successes
            iid_seen += 1
            cells.append(
                DiscoveryCellResult(
                    episode.episode_id,
                    True,
                    identity(episode.episode_id, hash_suffix),
                    True,
                    success,
                    ValidityRetryEvidence(False),
                    _factory_token=_DISCOVERY_CELL_FACTORY_TOKEN,
                )
            )
            continue
        assert episode.condition_family == "object_yaw"
        if not run_perturbations:
            cells.append(
                DiscoveryCellResult(
                    episode.episode_id,
                    False,
                    None,
                    None,
                    None,
                    None,
                    _factory_token=_DISCOVERY_CELL_FACTORY_TOKEN,
                )
            )
            continue
        valid = perturb_seen >= perturbation_invalid
        valid_position = perturb_seen - perturbation_invalid
        success = bool(valid and valid_position >= perturbation_failures)
        perturb_seen += 1
        cells.append(
            DiscoveryCellResult(
                episode.episode_id,
                True,
                identity(episode.episode_id, hash_suffix),
                valid,
                success,
                (
                    ValidityRetryEvidence(False)
                    if valid
                    else ValidityRetryEvidence(
                        True,
                        same_reset_seed_and_condition=True,
                        agrees_on_invalidity=retry_agrees_on_invalidity,
                        agrees_on_reasons=retry_agrees_on_reasons,
                    )
                ),
                _factory_token=_DISCOVERY_CELL_FACTORY_TOKEN,
            )
        )
    executed_rollouts = tuple(
        dummy_rollout(cell.episode_id, hash_suffix)
        for cell in (tuple(reversed(cells)) if reverse else tuple(cells))
        if cell.executed
    )
    return TaskDiscoveryAttempt(
        manifest,
        tuple(reversed(cells)) if reverse else tuple(cells),
        executed_rollouts,
        _factory_token=_DISCOVERY_ATTEMPT_FACTORY_TOKEN,
    )


@pytest.mark.parametrize("failures", [6, 24])
def test_inclusive_dynamic_failure_rate_thresholds_pass(
    protocol, failures: int
) -> None:
    decision = evaluate_task_attempt(
        protocol,
        task_attempt(
            discovery_manifest(protocol),
            iid_successes=6,
            perturbation_failures=failures,
        ),
    )

    assert decision.reproduction.passes
    assert decision.reproduction.success_interval.rate == 0.6
    assert decision.reproduction.success_interval.lower < 0.6
    assert decision.dynamic_range is not None
    assert decision.dynamic_range.failure_interval.rate == failures / 30
    assert decision.dynamic_range.passes
    assert decision.passes


def test_reproduction_point_threshold_and_iid_first_short_circuit(protocol) -> None:
    manifest = discovery_manifest(protocol)
    failed = evaluate_task_attempt(
        protocol,
        task_attempt(manifest, iid_successes=5, run_perturbations=False),
    )
    assert not failed.reproduction.passes
    assert failed.dynamic_range is None
    assert not failed.passes
    assert len(failed.raw_artifacts) == 10

    represented_run = task_attempt(
        manifest,
        iid_successes=5,
        run_perturbations=True,
    )
    with pytest.raises(RealityGateError, match="reproduction-failing"):
        evaluate_task_attempt(protocol, represented_run)


def test_exact_valid_fraction_threshold_and_failure_rate_denominator(protocol) -> None:
    manifest = discovery_manifest(protocol)
    exact = evaluate_task_attempt(
        protocol,
        task_attempt(
            manifest,
            iid_successes=10,
            perturbation_invalid=3,
            perturbation_failures=6,
        ),
    )
    assert exact.dynamic_range is not None
    assert exact.dynamic_range.validity_interval.rate == 0.9
    assert exact.dynamic_range.failure_interval.total == 27
    assert exact.dynamic_range.failure_interval.successes == 6
    assert exact.dynamic_range.passes

    below = evaluate_task_attempt(
        protocol,
        task_attempt(
            manifest,
            iid_successes=10,
            perturbation_invalid=4,
            perturbation_failures=6,
        ),
    )
    assert below.dynamic_range is not None
    assert not below.dynamic_range.passes_validity
    assert not below.passes

    with pytest.raises(RealityGateError, match="undefined with zero valid"):
        evaluate_task_attempt(
            protocol,
            task_attempt(
                manifest,
                iid_successes=10,
                perturbation_invalid=30,
                perturbation_failures=0,
            ),
        )


def test_validity_retry_contract_matches_raw_metadata_and_retains_disagreement(
    protocol,
) -> None:
    manifest = discovery_manifest(protocol)
    attempt = task_attempt(
        manifest,
        iid_successes=10,
        perturbation_invalid=1,
        retry_agrees_on_invalidity=False,
        retry_agrees_on_reasons=False,
    )
    invalid = next(cell for cell in attempt.cells if cell.valid is False)
    assert invalid.validity_retry is not None
    assert invalid.validity_retry.to_dict() == {
        "performed": True,
        "same_reset_seed_and_condition": True,
        "agrees_on_invalidity": False,
        "agrees_on_reasons": False,
    }
    decision = evaluate_task_attempt(protocol, attempt)
    invalid_metadata = next(
        cell for cell in decision.to_dict()["cells"] if cell["valid"] is False
    )
    assert invalid_metadata["validity_retry"] == invalid.validity_retry.to_dict()

    valid = next(cell for cell in attempt.cells if cell.valid is True)
    assert valid.validity_retry is not None
    assert valid.validity_retry.to_dict() == {"performed": False}

    disagreeing = decide_reality_gate(protocol, (attempt,))
    agreeing = decide_reality_gate(
        protocol,
        (
            task_attempt(
                manifest,
                iid_successes=10,
                perturbation_invalid=1,
                retry_agrees_on_invalidity=True,
                retry_agrees_on_reasons=True,
            ),
        ),
    )
    assert disagreeing.sha256 != agreeing.sha256


def test_validity_retry_contract_fails_closed_on_missing_or_wrong_evidence() -> None:
    raw = identity("episode")
    with pytest.raises(RealityGateError, match="requires validated"):
        DiscoveryCellResult(
            "episode", True, raw, True, False, None,
            _factory_token=_DISCOVERY_CELL_FACTORY_TOKEN,
        )
    with pytest.raises(RealityGateError, match="valid initial reset"):
        DiscoveryCellResult(
            "episode",
            True,
            raw,
            True,
            False,
            ValidityRetryEvidence(True, True, True, True),
            _factory_token=_DISCOVERY_CELL_FACTORY_TOKEN,
        )
    with pytest.raises(RealityGateError, match="invalid initial reset"):
        DiscoveryCellResult(
            "episode", True, raw, False, False, ValidityRetryEvidence(False),
            _factory_token=_DISCOVERY_CELL_FACTORY_TOKEN,
        )
    with pytest.raises(RealityGateError, match="same reset seed"):
        ValidityRetryEvidence(True, False, True, True)
    with pytest.raises(RealityGateError, match="both agreement"):
        ValidityRetryEvidence(True, True, None, True)
    with pytest.raises(RealityGateError, match="cannot carry retry fields"):
        ValidityRetryEvidence(False, None, False, None)


@pytest.mark.parametrize("failures", [5, 25])
def test_outside_dynamic_failure_bounds_fails(protocol, failures: int) -> None:
    decision = evaluate_task_attempt(
        protocol,
        task_attempt(
            discovery_manifest(protocol),
            iid_successes=10,
            perturbation_failures=failures,
        ),
    )
    assert decision.dynamic_range is not None
    assert decision.dynamic_range.passes_validity
    assert not decision.dynamic_range.passes_failure_range
    assert not decision.passes


def test_reproduction_pass_requires_all_30_perturbation_results(protocol) -> None:
    attempt = task_attempt(discovery_manifest(protocol), iid_successes=6)
    cells = list(attempt.cells)
    yaw_index = next(
        index for index, cell in enumerate(cells) if "cell0" not in cell.episode_id
    )
    cell = cells[yaw_index]
    cells[yaw_index] = DiscoveryCellResult(
        cell.episode_id,
        False,
        None,
        None,
        None,
        None,
        _factory_token=_DISCOVERY_CELL_FACTORY_TOKEN,
    )
    with pytest.raises(RealityGateError, match="all thirty"):
        evaluate_task_attempt(
            protocol,
            TaskDiscoveryAttempt(
                attempt.manifest,
                tuple(cells),
                executed_rollouts_for_cells(cells),
                _factory_token=_DISCOVERY_ATTEMPT_FACTORY_TOKEN,
            ),
        )


def test_exact_manifest_result_coverage_and_order_are_validated(protocol) -> None:
    attempt = task_attempt(discovery_manifest(protocol), iid_successes=6)
    duplicate = list(attempt.cells)
    duplicate[-1] = duplicate[0]
    with pytest.raises(RealityGateError, match="unique"):
        evaluate_task_attempt(
            protocol,
            TaskDiscoveryAttempt(
                attempt.manifest,
                tuple(duplicate),
                executed_rollouts_for_cells(duplicate),
                _factory_token=_DISCOVERY_ATTEMPT_FACTORY_TOKEN,
            ),
        )

    malformed = list(attempt.cells)
    malformed[-1] = DiscoveryCellResult(
        "unmanifested-episode",
        True,
        identity("unmanifested-episode"),
        malformed[-1].valid,
        malformed[-1].success,
        malformed[-1].validity_retry,
        _factory_token=_DISCOVERY_CELL_FACTORY_TOKEN,
    )
    with pytest.raises(RealityGateError, match="coverage mismatch"):
        evaluate_task_attempt(
            protocol,
            TaskDiscoveryAttempt(
                attempt.manifest,
                tuple(malformed),
                executed_rollouts_for_cells(malformed),
                _factory_token=_DISCOVERY_ATTEMPT_FACTORY_TOKEN,
            ),
        )

    reordered_manifest = Manifest(
        schema_version=attempt.manifest.schema_version,
        split=attempt.manifest.split,
        task=attempt.manifest.task,
        episodes=tuple(reversed(attempt.manifest.episodes)),
    )
    with pytest.raises(RealityGateError, match="deterministic assigned-yaw"):
        evaluate_task_attempt(
            protocol,
            TaskDiscoveryAttempt(
                reordered_manifest,
                attempt.cells,
                attempt.executed_rollouts,
                _factory_token=_DISCOVERY_ATTEMPT_FACTORY_TOKEN,
            ),
        )


def test_first_complete_pass_selects_and_retains_ordered_attempts(protocol) -> None:
    first = task_attempt(
        discovery_manifest(protocol, 1),
        iid_successes=5,
        run_perturbations=False,
    )
    second = task_attempt(discovery_manifest(protocol, 2), iid_successes=6)
    receipt = decide_reality_gate(protocol, (first, second))

    assert receipt.selected_task == protocol.task_order.tasks[1]
    assert [item.task.rank for item in receipt.attempts] == [1, 2]
    assert receipt.attempts[0].dynamic_range is None
    assert receipt.attempts[1].passes
    metadata = receipt.to_dict()
    assert metadata["all_attempts_retained"] is True
    assert metadata["decision_basis"] == "point_estimates"
    assert metadata["wilson_intervals_role"] == "descriptive_only"
    assert len(metadata["attempts"][0]["cells"]) == 40
    first_raw = metadata["attempts"][0]["cells"][0]["artifact"]
    assert first_raw["metadata_sha256"] == first.cells[0].artifact.metadata_sha256
    assert metadata["attempts"][1]["manifest_sha256"] == second.manifest.sha256
    assert metadata["policy_revision"] == POLICY
    assert metadata["code_commit"] == COMMIT


@pytest.mark.parametrize(
    "mutate_tasks",
    [
        lambda tasks: tasks[:2],
        lambda tasks: (replace(tasks[0], task_id=999), *tasks[1:]),
        lambda tasks: (tasks[1], tasks[0], tasks[2]),
    ],
)
def test_exact_amended_three_task_shortlist_is_hard_validated(
    protocol, mutate_tasks
) -> None:
    changed = replace(
        protocol,
        task_order=replace(
            protocol.task_order,
            tasks=tuple(mutate_tasks(protocol.task_order.tasks)),
        ),
    )
    attempt = task_attempt(discovery_manifest(protocol), iid_successes=6)
    with pytest.raises(RealityGateError, match="exact amended three-task"):
        evaluate_task_attempt(changed, attempt)

    receipt = decide_reality_gate(
        protocol,
        (task_attempt(discovery_manifest(protocol), iid_successes=6),),
    )
    with pytest.raises(RealityGateError, match="produced only by decide_reality_gate"):
        replace(receipt, task_order=changed.task_order.tasks)


@pytest.mark.parametrize(
    ("second_policy", "second_commit"),
    [("b" * 40, COMMIT), (POLICY, "b" * 40)],
)
def test_task_attempts_require_identical_policy_and_code_provenance(
    protocol, second_policy: str, second_commit: str
) -> None:
    first = task_attempt(
        discovery_manifest(protocol, 1),
        iid_successes=5,
        run_perturbations=False,
    )
    second = task_attempt(
        discovery_manifest(
            protocol,
            2,
            policy_revision=second_policy,
            code_commit=second_commit,
        ),
        iid_successes=6,
    )
    with pytest.raises(RealityGateError, match="identical policy revision"):
        decide_reality_gate(protocol, (first, second))


def test_dynamic_failure_advances_to_next_ordered_task(protocol) -> None:
    first = task_attempt(
        discovery_manifest(protocol, 1),
        iid_successes=10,
        perturbation_failures=5,
    )
    second = task_attempt(discovery_manifest(protocol, 2), iid_successes=6)
    receipt = decide_reality_gate(protocol, (first, second))

    assert receipt.attempts[0].reproduction.passes
    assert receipt.attempts[0].dynamic_range is not None
    assert not receipt.attempts[0].dynamic_range.passes
    assert receipt.selected_task == protocol.task_order.tasks[1]


def test_task_selection_rejects_skips_partial_failure_and_work_after_pass(
    protocol,
) -> None:
    failed_first = task_attempt(
        discovery_manifest(protocol, 1),
        iid_successes=5,
        run_perturbations=False,
    )
    passed_first = task_attempt(discovery_manifest(protocol, 1), iid_successes=6)
    passed_second = task_attempt(discovery_manifest(protocol, 2), iid_successes=6)

    with pytest.raises(RealityGateError, match="unresolved"):
        decide_reality_gate(protocol, (failed_first,))
    with pytest.raises(RealityGateError, match="ordered prefix"):
        decide_reality_gate(protocol, (passed_second,))
    with pytest.raises(RealityGateError, match="after the first pass"):
        decide_reality_gate(protocol, (passed_first, passed_second))


def test_all_ordered_failures_produce_no_selection(protocol) -> None:
    attempts = tuple(
        task_attempt(
            discovery_manifest(protocol, rank),
            iid_successes=5,
            run_perturbations=False,
        )
        for rank in (1, 2, 3)
    )
    receipt = decide_reality_gate(protocol, attempts)
    assert receipt.selected_task is None
    assert len(receipt.attempts) == 3


def test_receipt_is_canonical_and_binds_raw_hashes(protocol) -> None:
    manifest = discovery_manifest(protocol)
    ordered = decide_reality_gate(
        protocol,
        (task_attempt(manifest, iid_successes=6),),
    )
    reversed_input = decide_reality_gate(
        protocol,
        (task_attempt(manifest, iid_successes=6, reverse=True),),
    )
    changed_hash = decide_reality_gate(
        protocol,
        (task_attempt(manifest, iid_successes=6, hash_suffix="changed"),),
    )

    assert ordered.canonical_json() == reversed_input.canonical_json()
    assert ordered.sha256 == reversed_input.sha256
    assert ordered.sha256 != changed_hash.sha256
    assert hashlib.sha256(ordered.canonical_json()).hexdigest() == ordered.sha256


def orientation_sources() -> tuple[ArtifactIdentity, ...]:
    return (identity("orientation-episode"),)


def orientation_states(values: list[float | None]) -> tuple[OrientationState, ...]:
    return tuple(
        OrientationState("orientation-episode", step, value)
        for step, value in enumerate(values)
    )


def terminal_steps(
    sources: tuple[ArtifactIdentity, ...],
    terminal_control_step: int,
) -> tuple[tuple[ArtifactIdentity, int], ...]:
    return tuple((artifact, terminal_control_step) for artifact in sources)


def orientation_result(
    values: list[float | None],
    *,
    symmetry_order: int = 1,
    unit_tests: bool = True,
):
    sources = orientation_sources()
    return _evaluate_orientation_eligibility(
        sources,
        orientation_states(values),
        terminal_control_steps=terminal_steps(sources, len(values) - 1),
        symmetry_order=symmetry_order,
        extraction_unit_tests_passed=unit_tests,
        weighting=ORIENTATION_WEIGHTING,
        cadence=ORIENTATION_CADENCE,
        _factory_token=_ORIENTATION_ROLLOUT_FACTORY_TOKEN,
    )


def selected_task_evidence(protocol, *, spread: bool = True):
    gate = decide_reality_gate(
        protocol,
        (task_attempt(discovery_manifest(protocol), iid_successes=6),),
    )
    artifacts = tuple(
        sorted(gate.attempts[-1].raw_artifacts, key=lambda item: item.episode_id)
    )
    states = tuple(
        OrientationState(
            artifact.episode_id,
            step,
            0.6 if spread and index % 2 else 0.0,
        )
        for index, artifact in enumerate(artifacts)
        for step in range(2)
    )
    orientation = _evaluate_orientation_eligibility(
        artifacts,
        states,
        terminal_control_steps=terminal_steps(artifacts, 1),
        symmetry_order=gate.selected_task.planar_symmetry_order,
        extraction_unit_tests_passed=True,
        weighting=ORIENTATION_WEIGHTING,
        cadence=ORIENTATION_CADENCE,
        _factory_token=_ORIENTATION_ROLLOUT_FACTORY_TOKEN,
    )
    return gate, artifacts, orientation


def test_final_lock_receipt_binds_gate_raw_set_and_orientation_evidence(
    protocol,
) -> None:
    gate, artifacts, orientation = selected_task_evidence(protocol)
    lock = finalize_reality_gate(gate, orientation)

    assert isinstance(lock, RealityGateLockReceipt)
    assert lock.selected_variable == "orientation"
    assert lock.selected_task_artifacts == artifacts
    metadata = lock.to_dict()
    assert metadata["task_gate_receipt_sha256"] == gate.sha256
    assert metadata["task_gate_receipt"] == gate.to_dict()
    assert metadata["orientation_eligibility_sha256"] == orientation.sha256
    assert metadata["orientation_eligibility"]["state_sha256"] == (
        orientation.state_sha256
    )
    assert len(metadata["selected_task_artifacts"]) == 40
    assert hashlib.sha256(lock.canonical_json()).hexdigest() == lock.sha256

    changed_states = tuple(
        OrientationState(
            artifact.episode_id,
            step,
            0.7 if index % 2 else 0.0,
        )
        for index, artifact in enumerate(artifacts)
        for step in range(2)
    )
    changed_orientation = _evaluate_orientation_eligibility(
        artifacts,
        changed_states,
        terminal_control_steps=terminal_steps(artifacts, 1),
        symmetry_order=gate.selected_task.planar_symmetry_order,
        extraction_unit_tests_passed=True,
        weighting=ORIENTATION_WEIGHTING,
        cadence=ORIENTATION_CADENCE,
        _factory_token=_ORIENTATION_ROLLOUT_FACTORY_TOKEN,
    )
    assert finalize_reality_gate(gate, changed_orientation).sha256 != lock.sha256


def test_finalization_rejects_subset_or_unrelated_orientation_sources(protocol) -> None:
    gate, artifacts, _ = selected_task_evidence(protocol)
    subset = artifacts[:2]
    subset_orientation = _evaluate_orientation_eligibility(
        subset,
        (
            OrientationState(subset[0].episode_id, 0, 0.0),
            OrientationState(subset[1].episode_id, 0, 0.6),
        ),
        terminal_control_steps=terminal_steps(subset, 0),
        symmetry_order=gate.selected_task.planar_symmetry_order,
        extraction_unit_tests_passed=True,
        weighting=ORIENTATION_WEIGHTING,
        cadence=ORIENTATION_CADENCE,
        _factory_token=_ORIENTATION_ROLLOUT_FACTORY_TOKEN,
    )
    with pytest.raises(RealityGateError, match="exact selected-task 40-artifact"):
        finalize_reality_gate(gate, subset_orientation)

    changed_sources = tuple(
        identity(artifact.episode_id, "unrelated") for artifact in artifacts
    )
    changed_states = tuple(
        OrientationState(
            artifact.episode_id,
            0,
            0.6 if index % 2 else 0.0,
        )
        for index, artifact in enumerate(changed_sources)
    )
    changed_orientation = _evaluate_orientation_eligibility(
        changed_sources,
        changed_states,
        terminal_control_steps=terminal_steps(changed_sources, 0),
        symmetry_order=gate.selected_task.planar_symmetry_order,
        extraction_unit_tests_passed=True,
        weighting=ORIENTATION_WEIGHTING,
        cadence=ORIENTATION_CADENCE,
        _factory_token=_ORIENTATION_ROLLOUT_FACTORY_TOKEN,
    )
    with pytest.raises(RealityGateError, match="exact selected-task 40-artifact"):
        finalize_reality_gate(gate, changed_orientation)


def test_finalization_fails_closed_without_task_or_eligible_orientation(
    protocol,
) -> None:
    no_task = decide_reality_gate(
        protocol,
        tuple(
            task_attempt(
                discovery_manifest(protocol, rank),
                iid_successes=5,
                run_perturbations=False,
            )
            for rank in (1, 2, 3)
        ),
    )
    with pytest.raises(RealityGateError, match="without a selected task"):
        finalize_reality_gate(no_task, orientation_result([0.0, 0.6]))

    gate, _, ineligible = selected_task_evidence(protocol, spread=False)
    assert not ineligible.eligible
    with pytest.raises(RealityGateError, match="not prospectively operationalized"):
        finalize_reality_gate(gate, ineligible)


def test_final_lock_cannot_select_unoperationalized_fallback(protocol) -> None:
    gate, _, orientation = selected_task_evidence(protocol)
    lock = finalize_reality_gate(gate, orientation)
    with pytest.raises(RealityGateError, match="only eligible orientation"):
        replace(lock, selected_variable="contact")


def test_orientation_requires_unit_tests_and_at_least_90_percent_finite() -> None:
    spread = [math.radians(value) for value in range(0, 90, 10)]
    exact_ninety = orientation_result([*spread, None])
    assert exact_ninety.finite_states == 9
    assert exact_ninety.total_states == 10
    assert exact_ninety.finite_fraction == 0.9
    assert exact_ninety.eligible

    too_few = orientation_result([*spread[:8], None, math.nan])
    assert too_few.finite_fraction == 0.8
    assert not too_few.eligible

    tests_failed = orientation_result([*spread, None], unit_tests=False)
    assert not tests_failed.eligible


def test_orientation_physical_sd_threshold_is_inclusive() -> None:
    threshold = math.radians(15.0)
    half_separation = math.acos(math.exp(-(threshold**2) / 2.0))
    exact = orientation_result([-half_separation, half_separation])
    below = orientation_result([-0.99 * half_separation, 0.99 * half_separation])

    assert exact.physical_circular_sd_rad == pytest.approx(threshold, abs=1e-15)
    assert exact.eligible
    assert below.physical_circular_sd_rad < threshold
    assert not below.eligible


def test_orientation_symmetry_and_wrapping_are_physical_angle_aware() -> None:
    asymmetric = orientation_result([0.0, math.pi], symmetry_order=1)
    symmetric = orientation_result([0.0, math.pi], symmetry_order=2)
    assert asymmetric.eligible
    assert symmetric.physical_circular_sd_rad == pytest.approx(0.0, abs=1e-8)
    assert not symmetric.eligible

    base = orientation_result([0.0, 0.4, 0.8], symmetry_order=2)
    wrapped = orientation_result(
        [2 * math.pi, 0.4 - math.pi, 0.8 + 3 * math.pi],
        symmetry_order=2,
    )
    assert wrapped.resultant_length == pytest.approx(base.resultant_length, abs=1e-15)
    assert wrapped.physical_circular_sd_rad == pytest.approx(
        base.physical_circular_sd_rad, abs=1e-15
    )


def test_orientation_all_nonfinite_is_ineligible_without_nonfinite_metadata() -> None:
    result = orientation_result([None, math.nan, math.inf, -math.inf])
    assert result.finite_states == 0
    assert result.resultant_length is None
    assert result.physical_circular_sd_rad is None
    assert not result.eligible
    assert b"NaN" not in result.canonical_json()
    assert b"Infinity" not in result.canonical_json()


def test_orientation_fails_closed_on_ambiguous_weighting_or_cadence() -> None:
    sources = orientation_sources()
    states = orientation_states([0.0, 0.5])
    with pytest.raises(RealityGateError, match="weighting"):
        evaluate_orientation_eligibility(
            sources,
            states,
            terminal_control_steps=terminal_steps(sources, 1),
            symmetry_order=1,
            extraction_unit_tests_passed=True,
            weighting="episode_equal",
            cadence=ORIENTATION_CADENCE,
        )
    with pytest.raises(RealityGateError, match="cadence"):
        evaluate_orientation_eligibility(
            sources,
            states,
            terminal_control_steps=terminal_steps(sources, 1),
            symmetry_order=1,
            extraction_unit_tests_passed=True,
            weighting=ORIENTATION_WEIGHTING,
            cadence="every_five_steps",
        )
    missing_step = (
        OrientationState("orientation-episode", 0, 0.0),
        OrientationState("orientation-episode", 2, 0.5),
    )
    with pytest.raises(RealityGateError, match="complete cadence"):
        evaluate_orientation_eligibility(
            sources,
            missing_step,
            terminal_control_steps=terminal_steps(sources, 2),
            symmetry_order=1,
            extraction_unit_tests_passed=True,
            weighting=ORIENTATION_WEIGHTING,
            cadence=ORIENTATION_CADENCE,
        )


def test_orientation_provenance_hash_is_order_independent_and_source_bound() -> None:
    sources = (
        identity("a"),
        identity("b"),
    )
    states = (
        OrientationState("a", 0, 0.0),
        OrientationState("b", 0, 0.7),
    )
    first = evaluate_orientation_eligibility(
        sources,
        states,
        terminal_control_steps=terminal_steps(sources, 0),
        symmetry_order=1,
        extraction_unit_tests_passed=True,
        weighting=ORIENTATION_WEIGHTING,
        cadence=ORIENTATION_CADENCE,
    )
    second = evaluate_orientation_eligibility(
        tuple(reversed(sources)),
        tuple(reversed(states)),
        terminal_control_steps=tuple(reversed(terminal_steps(sources, 0))),
        symmetry_order=1,
        extraction_unit_tests_passed=True,
        weighting=ORIENTATION_WEIGHTING,
        cadence=ORIENTATION_CADENCE,
    )
    assert first.canonical_json() == second.canonical_json()
    assert first.sha256 == second.sha256
    assert [item.episode_id for item in first.source_artifacts] == ["a", "b"]


def test_variable_fallback_helper_returns_only_the_frozen_order() -> None:
    order = variable_fallback_order()
    assert order == ("orientation", "planar_position", "contact")
    assert isinstance(order, tuple)


def test_receipt_and_orientation_metadata_rehydrate_fail_closed(protocol) -> None:
    gate = decide_reality_gate(
        protocol,
        (task_attempt(discovery_manifest(protocol), iid_successes=6),),
    )
    parsed_gate = reality_gate_receipt_from_metadata(gate.to_dict(), protocol)
    assert parsed_gate.sha256 == gate.sha256

    tampered_gate = gate.to_dict()
    tampered_gate["attempts"][0]["passes"] = not tampered_gate["attempts"][0]["passes"]
    with pytest.raises(RealityGateError):
        reality_gate_receipt_from_metadata(tampered_gate, protocol)

    _, _, orientation = selected_task_evidence(protocol)
    parsed_orientation = orientation_eligibility_from_metadata(orientation.to_dict())
    assert parsed_orientation.sha256 == orientation.sha256
    tampered_orientation = orientation.to_dict()
    tampered_orientation["finite_states"] -= 1
    with pytest.raises(RealityGateError):
        orientation_eligibility_from_metadata(tampered_orientation)
