"""Outcome-blind utilities for preregistered causal subspace patching.

The functions in this module operate on frozen activations, matching metadata,
and action arrays.  They do not load rollout outcomes or depend on the simulator.
All boundary comparisons follow Section 10 of ``PREREG.md``: confirmatory angle
differences are inclusive at 30 and 90 degrees, matched controls are strictly
below 5 degrees, matching tolerances are inclusive, and claim thresholds that
say "exceeds" are strict.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

YAW_ACTION_INDEX = 5
ACTION_SUMMARY_STEPS = 10
PAIR_LIMIT_PER_SEED = 20
CONFIRMATORY_PAIR_MINIMUM = 30
RANDOM_CONTROL_COUNT = 1_000
PATCH_ALPHA_GRID = (0.25, 0.5, 1.0)


class CausalAnalysisError(ValueError):
    """Raised when causal-analysis inputs cannot support the frozen protocol."""


@dataclass(frozen=True)
class CandidateState:
    """Immutable state metadata used for donor/recipient matching.

    ``non_primary_predicates`` must contain *only* the task-specific predicates
    not involving the primary orientation variable.  Storing predicates as a
    canonical tuple makes equality and serialization independent of dict order.
    Positions are three-dimensional metric coordinates and angles are radians.
    """

    candidate_id: str
    base_init_id: Hashable
    phase: str
    contact: bool
    gripper_opening: float
    eef_position: tuple[float, float, float]
    object_position: tuple[float, float, float]
    normalized_time: float
    non_primary_predicates: tuple[tuple[str, bool], ...]
    orientation_rad: float
    symmetry_order: int

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise CausalAnalysisError("candidate_id must be a non-empty string")
        try:
            hash(self.base_init_id)
        except TypeError as exc:
            raise CausalAnalysisError("base_init_id must be hashable") from exc
        if self.base_init_id is None or isinstance(self.base_init_id, bool):
            raise CausalAnalysisError("base_init_id must identify a real cluster")
        if not isinstance(self.phase, str) or not self.phase:
            raise CausalAnalysisError("phase must be a non-empty string")
        if not isinstance(self.contact, bool):
            raise CausalAnalysisError("contact must be boolean")
        _finite_scalar(self.gripper_opening, "gripper_opening")
        if not isinstance(self.eef_position, tuple) or not isinstance(
            self.object_position, tuple
        ):
            raise CausalAnalysisError("positions must be immutable tuples")
        _position(self.eef_position, "eef_position")
        _position(self.object_position, "object_position")
        normalized_time = _finite_scalar(self.normalized_time, "normalized_time")
        if not 0.0 <= normalized_time <= 1.0:
            raise CausalAnalysisError("normalized_time must lie in [0, 1]")
        _finite_scalar(self.orientation_rad, "orientation_rad")
        if (
            isinstance(self.symmetry_order, bool)
            or not isinstance(self.symmetry_order, Integral)
            or self.symmetry_order < 1
        ):
            raise CausalAnalysisError("symmetry_order must be a positive integer")

        if not isinstance(self.non_primary_predicates, tuple):
            raise CausalAnalysisError(
                "non_primary_predicates must be an immutable tuple"
            )
        names: list[str] = []
        for item in self.non_primary_predicates:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or not isinstance(item[1], bool)
            ):
                raise CausalAnalysisError(
                    "non_primary_predicates must be (non-empty str, bool) pairs"
                )
            names.append(item[0])
        if names != sorted(names) or len(names) != len(set(names)):
            raise CausalAnalysisError(
                "non_primary_predicates must have unique names in sorted order"
            )

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        base_init_id: Hashable,
        phase: str,
        contact: bool,
        gripper_opening: float,
        eef_position: Sequence[float],
        object_position: Sequence[float],
        normalized_time: float,
        non_primary_predicates: Mapping[str, bool],
        orientation_rad: float,
        symmetry_order: int,
    ) -> CandidateState:
        """Build a state while defensively copying mutable caller inputs."""

        predicates = tuple(sorted(non_primary_predicates.items()))
        return cls(
            candidate_id=candidate_id,
            base_init_id=base_init_id,
            phase=phase,
            contact=contact,
            gripper_opening=float(gripper_opening),
            eef_position=tuple(float(value) for value in eef_position),  # type: ignore[arg-type]
            object_position=tuple(float(value) for value in object_position),  # type: ignore[arg-type]
            normalized_time=float(normalized_time),
            non_primary_predicates=predicates,
            orientation_rad=float(orientation_rad),
            symmetry_order=symmetry_order,
        )


@dataclass(frozen=True)
class PairEligibility:
    eligible: bool
    orientation_difference_deg: float
    standardized_matching_distance: float
    reasons: tuple[str, ...]
    mode: str


@dataclass(frozen=True)
class SelectedPair:
    seed: int
    recipient_id: str
    donor_id: str
    standardized_matching_distance: float
    orientation_difference_deg: float


@dataclass(frozen=True)
class PairSelection:
    seed: int
    mode: str
    pairs: tuple[SelectedPair, ...]
    candidate_count: int
    possible_edge_count: int
    eligible_edge_count: int
    eligible_edges_not_selected: int
    unmatched_candidate_ids: tuple[str, ...]
    exclusion_reason_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ThreeSeedPairSelection:
    seeds: tuple[int, int, int]
    selections: tuple[PairSelection, PairSelection, PairSelection]
    attempted_pairs: int
    maximum_pairs: int = 60


@dataclass(frozen=True)
class RandomSubspaceControl:
    """One random projector and its probe-shift-norm-matched direction."""

    projector: NDArray[np.float64]
    matched_shift: NDArray[np.float64]
    alpha: float
    raw_projected_norm: float
    target_projected_norm: float
    matched_patch_norm: float


@dataclass(frozen=True)
class RandomSubspaceShiftControl:
    """Memory-efficient production representation of one random control."""

    orthonormal_basis: NDArray[np.float64]
    matched_shift: NDArray[np.float64]
    alpha: float
    raw_projected_norm: float
    target_projected_norm: float
    matched_patch_norm: float


@dataclass(frozen=True)
class ActionEffectSummary:
    target_effect: float
    natural_target_effect: float
    donor_aligned_target_effect: float
    standardized_mean_change: tuple[float, ...]
    off_target_change: float
    off_target_ratio: float
    temporal_yaw_dot_product: float
    sign_correct: bool
    steps: int = ACTION_SUMMARY_STEPS
    yaw_index: int = YAW_ACTION_INDEX


@dataclass(frozen=True)
class RandomControlComparison:
    observed_effect: float
    percentile: float
    percentile_95_threshold: float
    exceeds_95th_percentile: bool
    controls: int


@dataclass(frozen=True)
class OffManifoldCheck:
    patched_five_nn_distance: float
    natural_95th_percentile: float
    off_manifold: bool


@dataclass(frozen=True)
class RateInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    replicates: int
    clusters: int
    seed: int


@dataclass(frozen=True)
class ConfirmatoryCausalResult:
    status: str
    valid_pairs: int
    median_target_effect: float
    random_95th_percentile: float
    off_target_ratio: float
    sign_correctness: float
    sign_interval: RateInterval | None
    positive_seed_count: int
    supporting_layer_count: int
    random_control_passes: bool
    specificity_passes: bool
    sign_passes: bool
    seed_stability_passes: bool
    layer_support_passes: bool
    succeeds: bool
    reasons: tuple[str, ...]


def orthogonal_probe_projector(coefficient: ArrayLike) -> NDArray[np.float64]:
    """Return the orthogonal projector onto a 2-by-d probe's row space.

    ``pinv`` implements the frozen formula, while the confirmatory path fails
    closed unless both circular-probe rows define a numerically rank-two space.
    """

    rows = np.asarray(coefficient, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] != 2 or rows.shape[1] < 1:
        raise CausalAnalysisError("coefficient must have shape (2, d) with d >= 1")
    if not np.isfinite(rows).all():
        raise CausalAnalysisError("coefficient must contain only finite values")
    rank = int(np.linalg.matrix_rank(rows))
    if rank != 2:
        raise CausalAnalysisError(
            f"coefficient row space must have numerical rank exactly 2, got {rank}"
        )
    projector = rows.T @ np.linalg.pinv(rows @ rows.T) @ rows
    projector = (projector + projector.T) / 2.0
    return _readonly(projector)


def probe_patch_shift(
    recipient: ArrayLike,
    donor: ArrayLike,
    coefficient: ArrayLike,
    *,
    alpha: float,
) -> NDArray[np.float64]:
    """Return ``alpha * P @ (donor - recipient)`` for the frozen probe space."""

    recipient_array = _vector(recipient, "recipient")
    donor_array = _vector(donor, "donor")
    if donor_array.shape != recipient_array.shape:
        raise CausalAnalysisError("recipient and donor must have identical shape")
    alpha_value = _finite_scalar(alpha, "alpha")
    if alpha_value < 0.0:
        raise CausalAnalysisError("alpha must be nonnegative")
    projector = orthogonal_probe_projector(coefficient)
    if projector.shape[0] != recipient_array.size:
        raise CausalAnalysisError("coefficient feature dimension does not match states")
    return _readonly(alpha_value * (projector @ (donor_array - recipient_array)))


def patch_activation(
    recipient: ArrayLike,
    donor: ArrayLike,
    coefficient: ArrayLike,
    *,
    alpha: float,
) -> NDArray[np.float64]:
    """Return a patched activation vector without mutating either input."""

    recipient_array = _vector(recipient, "recipient")
    return _readonly(
        recipient_array
        + probe_patch_shift(recipient_array, donor, coefficient, alpha=alpha)
    )


def iter_norm_matched_random_subspaces(
    coefficient: ArrayLike,
    activation_difference: ArrayLike,
    *,
    seed: int,
    alpha: float,
    count: int = RANDOM_CONTROL_COUNT,
) -> Iterator[RandomSubspaceControl]:
    """Yield deterministic random 2-D controls matched to probe-shift norm.

    Each Gaussian row space is orthonormalized, producing an actual rank-two
    projector.  Its projected activation difference is then rescaled to the norm
    of the selected ``alpha * P @ difference`` patch.  The projector itself is
    retained for auditing; the rescaled ``matched_shift`` is the complete control
    intervention shift and therefore must not be multiplied by alpha again.

    This streaming form is the production path.  At hidden width 720, retaining
    all 1,000 dense 720-by-720 projectors would require more than four GiB per
    pair even though inference consumes one shift at a time.
    """

    rows = np.asarray(coefficient, dtype=np.float64)
    projector = orthogonal_probe_projector(rows)
    difference = _vector(activation_difference, "activation_difference")
    if difference.size != projector.shape[0]:
        raise CausalAnalysisError("activation difference has the wrong dimension")
    seed_value = _seed(seed)
    alpha_value = _finite_scalar(alpha, "alpha")
    if alpha_value not in PATCH_ALPHA_GRID:
        raise CausalAnalysisError(
            f"alpha must be one of the frozen values {PATCH_ALPHA_GRID}"
        )
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise CausalAnalysisError("count must be a positive integer")
    if difference.size < 2:
        raise CausalAnalysisError("random 2-D subspaces require dimension at least two")

    target_projected_norm = float(np.linalg.norm(projector @ difference))
    matched_patch_norm = alpha_value * target_projected_norm
    rng = np.random.default_rng(seed_value)
    for _ in range(int(count)):
        gaussian = rng.standard_normal((difference.size, 2))
        basis, _ = np.linalg.qr(gaussian, mode="reduced")
        random_projector = basis @ basis.T
        raw_shift = random_projector @ difference
        raw_norm = float(np.linalg.norm(raw_shift))
        if raw_norm == 0.0:
            raise CausalAnalysisError("random projection unexpectedly has zero norm")
        if matched_patch_norm == 0.0:
            matched = np.zeros_like(raw_shift)
        else:
            matched = raw_shift * (matched_patch_norm / raw_norm)
        yield RandomSubspaceControl(
            projector=_readonly(random_projector),
            matched_shift=_readonly(matched),
            alpha=alpha_value,
            raw_projected_norm=raw_norm,
            target_projected_norm=target_projected_norm,
            matched_patch_norm=matched_patch_norm,
        )


def iter_norm_matched_random_shifts(
    coefficient: ArrayLike,
    activation_difference: ArrayLike,
    *,
    seed: int,
    alpha: float,
    count: int = RANDOM_CONTROL_COUNT,
) -> Iterator[RandomSubspaceShiftControl]:
    """Yield norm-matched random shifts without dense ``d``-by-``d`` projectors.

    For an orthonormal basis ``Q``, ``(Q Q.T) difference`` is evaluated as
    ``Q @ (Q.T @ difference)``.  This is algebraically identical but reduces
    each registered 720-wide control from quadratic to linear storage and work.
    The two basis columns are retained so every row space remains reproducible
    and auditable from the pairing seed.
    """

    rows = np.asarray(coefficient, dtype=np.float64)
    projector = orthogonal_probe_projector(rows)
    difference = _vector(activation_difference, "activation_difference")
    if difference.size != projector.shape[0]:
        raise CausalAnalysisError("activation difference has the wrong dimension")
    seed_value = _seed(seed)
    alpha_value = _finite_scalar(alpha, "alpha")
    if alpha_value not in PATCH_ALPHA_GRID:
        raise CausalAnalysisError(
            f"alpha must be one of the frozen values {PATCH_ALPHA_GRID}"
        )
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise CausalAnalysisError("count must be a positive integer")
    if difference.size < 2:
        raise CausalAnalysisError("random 2-D subspaces require dimension at least two")
    target_projected_norm = float(np.linalg.norm(projector @ difference))
    matched_patch_norm = alpha_value * target_projected_norm
    rng = np.random.default_rng(seed_value)
    for _ in range(int(count)):
        gaussian = rng.standard_normal((difference.size, 2))
        basis, _ = np.linalg.qr(gaussian, mode="reduced")
        raw_shift = basis @ (basis.T @ difference)
        raw_norm = float(np.linalg.norm(raw_shift))
        if raw_norm == 0.0:
            raise CausalAnalysisError("random projection unexpectedly has zero norm")
        matched = (
            np.zeros_like(raw_shift)
            if matched_patch_norm == 0.0
            else raw_shift * (matched_patch_norm / raw_norm)
        )
        yield RandomSubspaceShiftControl(
            orthonormal_basis=_readonly(basis),
            matched_shift=_readonly(matched),
            alpha=alpha_value,
            raw_projected_norm=raw_norm,
            target_projected_norm=target_projected_norm,
            matched_patch_norm=matched_patch_norm,
        )


def norm_matched_random_subspaces(
    coefficient: ArrayLike,
    activation_difference: ArrayLike,
    *,
    seed: int,
    alpha: float,
    count: int = RANDOM_CONTROL_COUNT,
) -> tuple[RandomSubspaceControl, ...]:
    """Return a materialized tuple of random controls for small/test callers.

    Production callers executing the registered 1,000-control design should use
    :func:`iter_norm_matched_random_subspaces` so only one dense projector exists
    at a time.
    """

    return tuple(
        iter_norm_matched_random_subspaces(
            coefficient,
            activation_difference,
            seed=seed,
            alpha=alpha,
            count=count,
        )
    )


def symmetry_aware_orientation_difference(
    left_rad: float, right_rad: float, *, symmetry_order: int
) -> float:
    """Return the shortest orientation difference in radians modulo symmetry."""

    left = _finite_scalar(left_rad, "left_rad")
    right = _finite_scalar(right_rad, "right_rad")
    if (
        isinstance(symmetry_order, bool)
        or not isinstance(symmetry_order, Integral)
        or symmetry_order < 1
    ):
        raise CausalAnalysisError("symmetry_order must be a positive integer")
    scaled = int(symmetry_order) * (left - right)
    wrapped = math.atan2(math.sin(scaled), math.cos(scaled))
    return abs(wrapped) / int(symmetry_order)


def pair_eligibility(
    recipient: CandidateState,
    donor: CandidateState,
    *,
    mode: str = "confirmatory",
) -> PairEligibility:
    """Check every frozen matching constraint and return all exclusion reasons."""

    if mode not in {"confirmatory", "matched_control"}:
        raise CausalAnalysisError("mode must be 'confirmatory' or 'matched_control'")
    if recipient.candidate_id == donor.candidate_id:
        return PairEligibility(False, 0.0, math.inf, ("same_candidate",), mode)

    reasons: list[str] = []
    if recipient.symmetry_order != donor.symmetry_order:
        reasons.append("symmetry_order")
        orientation_deg = math.inf
    else:
        orientation_deg = math.degrees(
            symmetry_aware_orientation_difference(
                recipient.orientation_rad,
                donor.orientation_rad,
                symmetry_order=recipient.symmetry_order,
            )
        )
    if recipient.phase != donor.phase:
        reasons.append("phase")
    if recipient.contact != donor.contact:
        reasons.append("contact")
    gripper_difference = abs(recipient.gripper_opening - donor.gripper_opening)
    eef_difference = _euclidean(recipient.eef_position, donor.eef_position)
    object_difference = _euclidean(recipient.object_position, donor.object_position)
    time_difference = abs(recipient.normalized_time - donor.normalized_time)
    if _strictly_greater(gripper_difference, 0.01):
        reasons.append("gripper_opening")
    if _strictly_greater(eef_difference, 0.02):
        reasons.append("eef_position")
    if _strictly_greater(object_difference, 0.02):
        reasons.append("object_position")
    if _strictly_greater(time_difference, 0.10):
        reasons.append("normalized_time")
    if recipient.non_primary_predicates != donor.non_primary_predicates:
        reasons.append("non_primary_predicates")
    if math.isfinite(orientation_deg):
        at_least_30 = orientation_deg > 30.0 or math.isclose(
            orientation_deg, 30.0, rel_tol=1e-12, abs_tol=1e-12
        )
        at_most_90 = not _strictly_greater(orientation_deg, 90.0)
        if mode == "confirmatory" and not (at_least_30 and at_most_90):
            reasons.append("confirmatory_orientation")
        if mode == "matched_control" and (
            orientation_deg > 5.0
            or math.isclose(orientation_deg, 5.0, rel_tol=1e-12, abs_tol=1e-12)
        ):
            reasons.append("control_orientation")

    distance = math.sqrt(
        (gripper_difference / 0.01) ** 2
        + (eef_difference / 0.02) ** 2
        + (object_difference / 0.02) ** 2
        + (time_difference / 0.10) ** 2
    )
    return PairEligibility(
        eligible=not reasons,
        orientation_difference_deg=orientation_deg,
        standardized_matching_distance=distance,
        reasons=tuple(reasons),
        mode=mode,
    )


def select_pairs(
    candidates: Sequence[CandidateState],
    *,
    seed: int,
    mode: str = "confirmatory",
    max_pairs: int = PAIR_LIMIT_PER_SEED,
) -> PairSelection:
    """Select a seeded, input-order-independent greedy matching.

    The seed defines a stable recipient traversal using SHA-256.  For each unused
    recipient, the nearest eligible unused donor is selected, breaking equal
    distances by canonical donor ID.  A candidate cannot appear in either role
    more than once within a seed.  This gives the seed a genuine pairing role while
    preserving the preregistered nearest-match preference.
    """

    seed_value = _seed(seed)
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, Integral):
        raise CausalAnalysisError("max_pairs must be an integer")
    if not 1 <= max_pairs <= PAIR_LIMIT_PER_SEED:
        raise CausalAnalysisError(f"max_pairs must lie in [1, {PAIR_LIMIT_PER_SEED}]")
    ordered = sorted(candidates, key=lambda state: state.candidate_id)
    ids = [state.candidate_id for state in ordered]
    if len(ids) != len(set(ids)):
        raise CausalAnalysisError("candidate_id values must be unique")

    possible_edges = len(ordered) * (len(ordered) - 1) // 2
    reason_counts: Counter[str] = Counter()
    eligible_edge_count = 0
    eligibility: dict[tuple[str, str], PairEligibility] = {}
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            result = pair_eligibility(left, right, mode=mode)
            eligibility[(left.candidate_id, right.candidate_id)] = result
            if result.eligible:
                eligible_edge_count += 1
            else:
                reason_counts.update(result.reasons)

    traversal = sorted(
        ordered,
        key=lambda state: (
            _seeded_key(seed_value, state.candidate_id),
            state.candidate_id,
        ),
    )
    unused = {state.candidate_id for state in ordered}
    selected: list[SelectedPair] = []
    for recipient in traversal:
        if len(selected) >= max_pairs:
            break
        if recipient.candidate_id not in unused:
            continue
        donor_options: list[tuple[float, str, CandidateState, PairEligibility]] = []
        for donor in ordered:
            if (
                donor.candidate_id == recipient.candidate_id
                or donor.candidate_id not in unused
            ):
                continue
            edge_key = tuple(sorted((recipient.candidate_id, donor.candidate_id)))
            result = eligibility[edge_key]
            if result.eligible:
                donor_options.append(
                    (
                        result.standardized_matching_distance,
                        donor.candidate_id,
                        donor,
                        result,
                    )
                )
        if not donor_options:
            continue
        _, _, donor, result = min(donor_options, key=lambda item: (item[0], item[1]))
        selected.append(
            SelectedPair(
                seed=seed_value,
                recipient_id=recipient.candidate_id,
                donor_id=donor.candidate_id,
                standardized_matching_distance=result.standardized_matching_distance,
                orientation_difference_deg=result.orientation_difference_deg,
            )
        )
        unused.remove(recipient.candidate_id)
        unused.remove(donor.candidate_id)

    selected_ids = {
        candidate_id
        for pair in selected
        for candidate_id in (pair.recipient_id, pair.donor_id)
    }
    return PairSelection(
        seed=seed_value,
        mode=mode,
        pairs=tuple(selected),
        candidate_count=len(ordered),
        possible_edge_count=possible_edges,
        eligible_edge_count=eligible_edge_count,
        eligible_edges_not_selected=eligible_edge_count - len(selected),
        unmatched_candidate_ids=tuple(sorted(set(ids) - selected_ids)),
        exclusion_reason_counts=tuple(sorted(reason_counts.items())),
    )


def select_pairs_for_three_seeds(
    candidates: Sequence[CandidateState],
    *,
    seeds: Sequence[int],
    mode: str = "confirmatory",
) -> ThreeSeedPairSelection:
    """Attempt at most 20 pairs for each of exactly three explicit seeds."""

    if len(seeds) != 3:
        raise CausalAnalysisError("confirmatory selection requires exactly three seeds")
    normalized = tuple(_seed(seed) for seed in seeds)
    if len(set(normalized)) != 3:
        raise CausalAnalysisError("the three pairing seeds must be distinct")
    selections = tuple(
        select_pairs(candidates, seed=seed, mode=mode) for seed in normalized
    )
    return ThreeSeedPairSelection(
        seeds=normalized,  # type: ignore[arg-type]
        selections=selections,  # type: ignore[arg-type]
        attempted_pairs=sum(len(selection.pairs) for selection in selections),
    )


def summarize_action_effect(
    recipient_actions: ArrayLike,
    donor_actions: ArrayLike,
    patched_actions: ArrayLike,
    *,
    action_scale: ArrayLike,
    yaw_index: int = YAW_ACTION_INDEX,
) -> ActionEffectSummary:
    """Summarize the first ten unnormalized actions for one patch.

    Specificity uses the L2 norm of the mean standardized non-yaw shift divided by
    the absolute mean standardized yaw shift.  The complete mean vector is retained
    to make alternative descriptive aggregations possible without rerunning policy
    inference.
    """

    recipient = _actions(recipient_actions, "recipient_actions")
    donor = _actions(donor_actions, "donor_actions")
    patched = _actions(patched_actions, "patched_actions")
    if not (recipient.shape == donor.shape == patched.shape):
        raise CausalAnalysisError("all action arrays must have identical shape")
    if recipient.shape[0] < ACTION_SUMMARY_STEPS:
        raise CausalAnalysisError("action arrays must contain at least ten actions")
    if isinstance(yaw_index, bool) or not isinstance(yaw_index, Integral):
        raise CausalAnalysisError("yaw_index must be an integer")
    yaw = int(yaw_index)
    if yaw < 0 or yaw >= recipient.shape[1]:
        raise CausalAnalysisError("yaw_index is outside the action dimension")
    scale = _vector(action_scale, "action_scale")
    if scale.size != recipient.shape[1] or np.any(scale <= 0.0):
        raise CausalAnalysisError("action_scale must be positive and match action dim")

    selected = slice(0, ACTION_SUMMARY_STEPS)
    standardized_change = np.mean(
        (patched[selected] - recipient[selected]) / scale,
        axis=0,
    )
    natural_change = np.mean(
        (donor[selected] - recipient[selected]) / scale,
        axis=0,
    )
    target_effect = float(standardized_change[yaw])
    natural_target_effect = float(natural_change[yaw])
    donor_alignment = float(np.sign(natural_target_effect))
    donor_aligned_target_effect = target_effect * donor_alignment
    temporal_yaw_dot_product = float(
        np.mean(
            (patched[selected, yaw] - recipient[selected, yaw])
            * (donor[selected, yaw] - recipient[selected, yaw])
        )
    )
    off_target = float(np.linalg.norm(np.delete(standardized_change, yaw)))
    ratio = math.inf if target_effect == 0.0 else off_target / abs(target_effect)
    return ActionEffectSummary(
        target_effect=target_effect,
        natural_target_effect=natural_target_effect,
        donor_aligned_target_effect=donor_aligned_target_effect,
        standardized_mean_change=tuple(float(value) for value in standardized_change),
        off_target_change=off_target,
        off_target_ratio=ratio,
        temporal_yaw_dot_product=temporal_yaw_dot_product,
        sign_correct=temporal_yaw_dot_product > 0.0,
        yaw_index=yaw,
    )


def compare_random_controls(
    observed_effect: float, random_effects: ArrayLike
) -> RandomControlComparison:
    """Compare an observed median effect with the empirical random distribution."""

    observed = _finite_scalar(observed_effect, "observed_effect")
    controls = _vector(random_effects, "random_effects")
    threshold = float(np.percentile(controls, 95.0))
    percentile = 100.0 * float(
        (
            np.count_nonzero(controls < observed)
            + 0.5 * np.count_nonzero(controls == observed)
        )
        / controls.size
    )
    return RandomControlComparison(
        observed_effect=observed,
        percentile=percentile,
        percentile_95_threshold=threshold,
        exceeds_95th_percentile=_strictly_greater(observed, threshold),
        controls=int(controls.size),
    )


def five_nearest_neighbor_distance(query: ArrayLike, reference: ArrayLike) -> float:
    """Return the mean Euclidean distance to the five nearest reference rows."""

    vector = _vector(query, "query")
    matrix = _matrix(reference, "reference")
    if matrix.shape[1] != vector.size:
        raise CausalAnalysisError("query and reference feature dimensions differ")
    if matrix.shape[0] < 5:
        raise CausalAnalysisError("five-nearest-neighbor distance needs five rows")
    distances = np.linalg.norm(matrix - vector, axis=1)
    return float(np.mean(np.partition(distances, 4)[:5]))


def off_manifold_flag(
    *, patched_five_nn_distance: float, natural_95th_percentile: float
) -> OffManifoldCheck:
    """Flag a patch only when its 5-NN distance strictly exceeds the natural 95th."""

    patched = _finite_scalar(patched_five_nn_distance, "patched_five_nn_distance")
    threshold = _finite_scalar(natural_95th_percentile, "natural_95th_percentile")
    if patched < 0.0 or threshold < 0.0:
        raise CausalAnalysisError("nearest-neighbor distances must be nonnegative")
    return OffManifoldCheck(patched, threshold, _strictly_greater(patched, threshold))


def cluster_bootstrap_rate_interval(
    sign_correct: Sequence[bool] | NDArray[np.bool_],
    base_init_ids: Sequence[Hashable],
    *,
    seed: int,
    replicates: int = 10_000,
    confidence: float = 0.90,
) -> RateInterval:
    """Percentile interval for a rate, resampling whole base-init clusters."""

    signs = np.asarray(sign_correct)
    if signs.ndim != 1 or signs.size == 0 or signs.dtype.kind != "b":
        raise CausalAnalysisError("sign_correct must be a non-empty boolean vector")
    if len(base_init_ids) != signs.size:
        raise CausalAnalysisError("base_init_ids must match sign_correct length")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, Integral)
        or replicates < 1
    ):
        raise CausalAnalysisError("replicates must be a positive integer")
    confidence_value = _finite_scalar(confidence, "confidence")
    if not 0.0 < confidence_value < 1.0:
        raise CausalAnalysisError("confidence must lie strictly between zero and one")

    clusters: dict[tuple[str, str], list[bool]] = {}
    for flag, cluster in zip(signs.tolist(), base_init_ids, strict=True):
        try:
            hash(cluster)
        except TypeError as exc:
            raise CausalAnalysisError("base_init_ids must be hashable") from exc
        if cluster is None or isinstance(cluster, bool):
            raise CausalAnalysisError("base_init_ids must identify real clusters")
        clusters.setdefault(_canonical_hashable(cluster), []).append(bool(flag))
    ordered = sorted(clusters)
    if len(ordered) < 2:
        raise CausalAnalysisError("cluster bootstrap requires at least two clusters")

    rng = np.random.default_rng(_seed(seed))
    bootstrap = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        sampled = rng.integers(0, len(ordered), size=len(ordered))
        selected = [
            value
            for cluster_index in sampled
            for value in clusters[ordered[cluster_index]]
        ]
        bootstrap[index] = float(np.mean(selected))
    tail = (1.0 - confidence_value) / 2.0
    lower, upper = np.percentile(bootstrap, [100.0 * tail, 100.0 * (1.0 - tail)])
    return RateInterval(
        estimate=float(np.mean(signs)),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence_value,
        replicates=int(replicates),
        clusters=len(ordered),
        seed=_seed(seed),
    )


def evaluate_confirmatory_causal_claim(
    donor_aligned_effects: ArrayLike,
    sign_correct: Sequence[bool] | NDArray[np.bool_],
    base_init_ids: Sequence[Hashable],
    pairing_seeds: Sequence[int],
    random_control_effects: ArrayLike,
    pair_off_target_ratios: ArrayLike,
    supporting_layer_effects: Mapping[str, float],
    *,
    expected_pairing_seeds: Sequence[int],
    bootstrap_seed: int,
    bootstrap_replicates: int = 10_000,
) -> ConfirmatoryCausalResult:
    """Apply the exact preregistered confirmatory success criteria.

    ``donor_aligned_effects`` are yaw effects multiplied by the sign of the
    natural donor-minus-recipient yaw difference.  This puts differently oriented
    pairs on the common preregistered expected-sign axis before seed and random
    control summaries are computed.

    This decision intentionally does not add an unregistered Locked-Test
    off-manifold-rate or matched-control threshold.  Those are required reported
    comparisons, while the final success sentence in Section 10 defines no cutoff
    for either.  Calibration's <=5% off-manifold constraint belongs to alpha
    selection and must already be frozen before this function is called.
    """

    effect_array = np.asarray(donor_aligned_effects, dtype=np.float64)
    if effect_array.ndim != 1 or not np.isfinite(effect_array).all():
        raise CausalAnalysisError("donor_aligned_effects must be a finite vector")
    sign_array = np.asarray(sign_correct)
    if sign_array.ndim != 1 or sign_array.dtype.kind != "b":
        raise CausalAnalysisError("sign_correct must be a boolean vector")
    if len(base_init_ids) != effect_array.size or sign_array.size != effect_array.size:
        raise CausalAnalysisError("effect, sign, and cluster arrays must align")
    if len(pairing_seeds) != effect_array.size:
        raise CausalAnalysisError("pairing_seeds must align with effects")
    seed_values = tuple(_seed(seed) for seed in pairing_seeds)
    expected_seeds = tuple(_seed(seed) for seed in expected_pairing_seeds)
    if len(expected_seeds) != 3 or len(set(expected_seeds)) != 3:
        raise CausalAnalysisError(
            "expected_pairing_seeds must contain three distinct seeds"
        )
    if not set(seed_values).issubset(expected_seeds):
        raise CausalAnalysisError("a row uses an undeclared pairing seed")
    ratios = np.asarray(pair_off_target_ratios, dtype=np.float64)
    if (
        ratios.ndim != 1
        or ratios.size != effect_array.size
        or np.isnan(ratios).any()
        or np.any(ratios < 0.0)
    ):
        raise CausalAnalysisError("off-target ratios must be nonnegative and align")
    if set(supporting_layer_effects) != {"vlm_context", "early_expert", "late_expert"}:
        raise CausalAnalysisError(
            "supporting_layer_effects must contain the three preregistered locations"
        )
    layer_effects = {
        name: _finite_scalar(value, f"supporting_layer_effects[{name!r}]")
        for name, value in supporting_layer_effects.items()
    }

    random_effect_array = _vector(random_control_effects, "random_control_effects")
    if random_effect_array.size != RANDOM_CONTROL_COUNT:
        raise CausalAnalysisError(
            f"confirmatory analysis requires exactly {RANDOM_CONTROL_COUNT} "
            "random controls"
        )
    valid_pairs = int(effect_array.size)
    median_effect = float(np.median(effect_array)) if valid_pairs else math.nan
    median_ratio = float(np.median(ratios)) if valid_pairs else math.inf
    random_threshold = float(np.percentile(random_effect_array, 95.0))
    random_comparison = (
        compare_random_controls(median_effect, random_effect_array)
        if valid_pairs
        else None
    )
    row_seeds = np.asarray(seed_values)
    positive_seeds = sum(
        bool(np.any(row_seeds == seed))
        and float(np.median(effect_array[row_seeds == seed])) > 0.0
        for seed in expected_seeds
    )
    supporting_layers = sum(value > 0.0 for value in layer_effects.values())
    reasons: list[str] = []

    if valid_pairs < CONFIRMATORY_PAIR_MINIMUM:
        reasons.append("fewer_than_30_valid_pairs")
        return ConfirmatoryCausalResult(
            status="inconclusive",
            valid_pairs=valid_pairs,
            median_target_effect=median_effect,
            random_95th_percentile=random_threshold,
            off_target_ratio=median_ratio,
            sign_correctness=float(np.mean(sign_array))
            if sign_array.size
            else math.nan,
            sign_interval=None,
            positive_seed_count=positive_seeds,
            supporting_layer_count=supporting_layers,
            random_control_passes=False,
            specificity_passes=False,
            sign_passes=False,
            seed_stability_passes=False,
            layer_support_passes=False,
            succeeds=False,
            reasons=tuple(reasons),
        )

    sign_interval = cluster_bootstrap_rate_interval(
        sign_array,
        base_init_ids,
        seed=bootstrap_seed,
        replicates=bootstrap_replicates,
        confidence=0.90,
    )
    if random_comparison is None:  # guarded by the >=30 branch, for type narrowing
        raise AssertionError("nonempty confirmatory effects lost their comparison")
    random_passes = random_comparison.exceeds_95th_percentile
    specificity_passes = median_ratio <= 0.25
    sign_passes = sign_interval.estimate > 0.50 and sign_interval.lower > 0.50
    seed_passes = positive_seeds >= 2
    layer_passes = supporting_layers >= 2
    checks = (
        (random_passes, "median_effect_not_above_random_95th"),
        (specificity_passes, "off_target_ratio_above_25_percent"),
        (sign_passes, "sign_correctness_interval_not_above_50_percent"),
        (seed_passes, "effect_direction_not_supported_by_two_seeds"),
        (layer_passes, "effect_direction_not_supported_by_two_layers"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    succeeds = all(passed for passed, _ in checks)
    return ConfirmatoryCausalResult(
        status="success" if succeeds else "does_not_meet_criteria",
        valid_pairs=valid_pairs,
        median_target_effect=median_effect,
        random_95th_percentile=random_comparison.percentile_95_threshold,
        off_target_ratio=median_ratio,
        sign_correctness=sign_interval.estimate,
        sign_interval=sign_interval,
        positive_seed_count=positive_seeds,
        supporting_layer_count=supporting_layers,
        random_control_passes=random_passes,
        specificity_passes=specificity_passes,
        sign_passes=sign_passes,
        seed_stability_passes=seed_passes,
        layer_support_passes=layer_passes,
        succeeds=succeeds,
        reasons=tuple(reasons),
    )


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CausalAnalysisError(f"{name} must be finite")
    return result


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise CausalAnalysisError(f"{name} must be a non-empty finite vector")
    return array


def _matrix(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[0] == 0
        or array.shape[1] == 0
        or not np.isfinite(array).all()
    ):
        raise CausalAnalysisError(f"{name} must be a non-empty finite matrix")
    return array


def _actions(values: ArrayLike, name: str) -> NDArray[np.float64]:
    return _matrix(values, name)


def _position(values: Sequence[float], name: str) -> None:
    if len(values) != 3 or not all(math.isfinite(float(value)) for value in values):
        raise CausalAnalysisError(f"{name} must contain exactly three finite values")


def _euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right)))


def _readonly(array: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.asarray(array, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise CausalAnalysisError("seed must be a nonnegative integer")
    return int(value)


def _seeded_key(seed: int, candidate_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{candidate_id}".encode()).digest()


def _canonical_hashable(value: Hashable) -> tuple[str, str]:
    return type(value).__qualname__, repr(value)


def _strictly_greater(left: float, right: float) -> bool:
    """Numerically stable strict comparison at decimal protocol boundaries."""

    return left > right and not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
