from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mech_int_vla import scoring
from mech_int_vla.config import load_perturbations
from mech_int_vla.scoring import (
    FROZEN_TRANSFORMS,
    CallCost,
    ContentLinks,
    FactualReplay,
    ReplayFrame,
    ReplayTransition,
    ScoredCall,
    ScoringError,
    ScoringTransform,
    ScoringValidationError,
    SimulatorSnapshot,
    TransformValidity,
    load_scoring_sidecar,
    score_replay_to_sidecar,
    scoring_transforms_from_conditions,
    validate_frozen_transform_order,
)


def _frame(step: int, *, image_delta: int = 0, state_delta: float = 0.0) -> ReplayFrame:
    first = np.full((2, 2, 3), 10 + step + image_delta, dtype=np.uint8)
    second = np.full((2, 2, 3), 20 + step + image_delta, dtype=np.uint8)
    return ReplayFrame(
        control_step=step,
        camera_images={
            "agentview_image": first,
            "robot0_eye_in_hand_image": second,
        },
        low_dimensional={
            "policy_state": np.asarray(
                [step + state_delta, step / 10], dtype=np.float64
            ),
            "contact": np.asarray(step % 2 == 0, dtype=np.bool_),
        },
        success=False,
    )


def _replay(*, steps: int = 6) -> FactualReplay:
    frames = tuple(_frame(step) for step in range(steps + 1))
    actions = np.stack(
        [np.full(7, step / 100, dtype=np.float32) for step in range(steps)]
    )
    terminated = np.zeros(steps, dtype=np.bool_)
    truncated = np.zeros(steps, dtype=np.bool_)
    truncated[-1] = True
    return FactualReplay(
        episode_id="calibration-task1-init10-cell0",
        split="calibration",
        frames=frames,
        actions=actions,
        terminated=terminated,
        truncated=truncated,
        terminal_status="truncated",
        success=False,
    )


def _links() -> ContentLinks:
    return ContentLinks(
        raw_metadata_sha256="1" * 64,
        raw_trajectory_sha256="2" * 64,
        probe_sha256="3" * 64,
        config_sha256="4" * 64,
        code_sha256="5" * 64,
    )


class FakeAdapter:
    def __init__(
        self,
        replay: FactualReplay,
        *,
        unavailable: set[str] | None = None,
        reset_mismatch: bool = False,
        mutate_rng: bool = False,
        broken_restore: bool = False,
    ) -> None:
        self.replay = replay
        self.unavailable = unavailable or set()
        self.reset_mismatch = reset_mismatch
        self.mutate_rng = mutate_rng
        self.broken_restore = broken_restore
        self.index = 0
        self.transform: ScoringTransform | None = None
        self.state = np.asarray([0.0, 1.0], dtype=np.float64)
        self.camera_positions = np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64)
        self.camera_quaternions = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        self.queue: list[float] = []
        self.call_log: list[tuple[str, int, int, float | None]] = []

    def reset_replay(self) -> ReplayFrame:
        self.index = 0
        self.transform = None
        if self.reset_mismatch:
            return _frame(0, image_delta=1)
        return _frame(0)

    def step_replay(self, action: np.ndarray) -> ReplayTransition:
        assert np.array_equal(action, self.replay.actions[self.index])
        self.index += 1
        self.state[0] = self.index
        return ReplayTransition(
            frame=_frame(self.index),
            terminated=bool(self.replay.terminated[self.index - 1]),
            truncated=bool(self.replay.truncated[self.index - 1]),
        )

    def capture_snapshot(self) -> SimulatorSnapshot:
        return SimulatorSnapshot(
            state=self.state,
            camera_positions=self.camera_positions,
            camera_quaternions=self.camera_quaternions,
        )

    def restore_snapshot(self, snapshot: SimulatorSnapshot) -> None:
        self.state = snapshot.state.copy()
        self.camera_positions = snapshot.camera_positions.copy()
        self.camera_quaternions = snapshot.camera_quaternions.copy()
        if self.broken_restore:
            self.state[0] += 0.5
        self.transform = None

    def apply_transform(self, transform: ScoringTransform) -> None:
        self.transform = transform
        if transform.family == "camera_render":
            self.camera_positions[0, 0] += transform.value
        elif transform.family == "object_pose":
            self.state[1] += transform.value

    def transformed_validity(self) -> TransformValidity:
        assert self.transform is not None
        available = self.transform.name not in self.unavailable
        return TransformValidity(
            finite=available, penetration_ok=available, workspace_ok=available
        )

    def current_frame(self) -> ReplayFrame:
        assert self.transform is not None
        if self.transform.family == "camera_render":
            return _frame(self.index, image_delta=int(self.transform.value))
        return _frame(self.index, state_delta=self.transform.value)

    def process_observation(self, frame: ReplayFrame) -> tuple[int, int, float, str]:
        pixel = int(frame.camera_images["agentview_image"][0, 0, 0])
        state = float(frame.low_dimensional["policy_state"][0])
        context = (
            "transformed"
            if pixel != 10 + frame.control_step or state != frame.control_step
            else "factual"
        )
        return frame.control_step, pixel, state, context

    def noise_for_seed(self, seed: int) -> np.ndarray:
        return np.asarray([seed], dtype=np.int64)

    def predict_action_chunk(
        self,
        processed_observation: tuple[int, int, float, str],
        *,
        noise: np.ndarray,
        intervention_degrees: float | None,
    ) -> ScoredCall:
        if self.mutate_rng:
            random.random()
        step, pixel, state, context = processed_observation
        base = (int(noise[0]) % 997) / 997 + pixel / 1000 + state / 100
        if intervention_degrees is not None:
            base += intervention_degrees / 100
        actions = np.full((12, 7), base, dtype=np.float32)
        activation = np.asarray(
            [step, pixel / 255, (int(noise[0]) % 101) / 101], dtype=np.float32
        )
        self.call_log.append((context, step, id(noise), intervention_degrees))
        return ScoredCall(
            actions=actions,
            activation=activation,
            cost=CallCost(
                cuda_event_ms=1.25,
                wall_time_ns=100,
                forward_count=1,
                intervention_count=int(intervention_degrees is not None),
                peak_allocated_bytes=200,
                incremental_peak_allocated_bytes=50,
                logical_activation_bytes=activation.nbytes,
                compressed_activation_bytes=8,
            ),
        )

    def intervention_available(self, activation: np.ndarray) -> bool:
        return bool(np.linalg.norm(activation.astype(np.float64)))

    def begin_score_state(self) -> None:
        pass

    def policy_queue_state(self) -> dict[str, list[float]]:
        return {"queue": self.queue.copy()}


def _run(tmp_path: Path, adapter: FakeAdapter, replay: FactualReplay | None = None):
    factual = adapter.replay if replay is None else replay
    return score_replay_to_sidecar(
        adapter,
        factual,
        _links(),
        transforms=FROZEN_TRANSFORMS,
        output_root=tmp_path / "scores",
    )


def test_scoring_replays_and_writes_exact_shapes_and_seed_reuse(tmp_path: Path) -> None:
    replay = _replay()
    adapter = FakeAdapter(replay)
    result = _run(tmp_path, adapter)
    loaded = load_scoring_sidecar(result.path)

    assert result.scored_control_steps == (0, 5)
    assert loaded.arrays["original_actions"].shape == (2, 8, 10, 7)
    assert loaded.arrays["transformed_actions"].shape == (2, 6, 4, 10, 7)
    assert loaded.arrays["original_activation"].shape == (2, 8, 3)
    assert loaded.arrays["transformed_activation"].shape == (2, 6, 4, 3)
    assert loaded.arrays["intervention_minus_actions"].shape == (2, 4, 10, 7)
    assert loaded.arrays["intervention_plus_actions"].shape == (2, 4, 10, 7)
    assert loaded.arrays["original_cost"].shape == (2, 8, 8)
    assert loaded.metadata["links"] == _links().as_dict()
    assert loaded.metadata["capture"]["policy_queue_checked"] is True

    for step in (0, 5):
        calls = [entry for entry in adapter.call_log if entry[1] == step]
        original_ids = [entry[2] for entry in calls[:8]]
        for draw in range(4):
            transformed = [
                entry[2]
                for entry in calls
                if entry[0] != "factual" and entry[3] is None
            ]
            assert all(
                transformed[transform_index * 4 + draw] == original_ids[draw]
                for transform_index in range(6)
            )
            interventions = [entry[2] for entry in calls if entry[3] is not None]
            assert interventions[draw] == original_ids[draw]
            assert interventions[4 + draw] == original_ids[draw]


def test_unavailable_transform_is_nan_and_masked(tmp_path: Path) -> None:
    replay = _replay()
    unavailable = {"object_yaw_pos_15"}
    loaded = load_scoring_sidecar(
        _run(tmp_path, FakeAdapter(replay, unavailable=unavailable)).path
    )
    mask = loaded.arrays["transform_available"]
    assert np.all(mask[:, :5])
    assert not np.any(mask[:, 5])
    assert np.isnan(loaded.arrays["transformed_actions"][:, 5]).all()
    assert np.isnan(loaded.arrays["transformed_activation"][:, 5]).all()
    assert np.isnan(loaded.arrays["transformed_cost"][:, 5]).all()
    assert np.isfinite(loaded.arrays["transformed_actions"][:, :5]).all()


def test_publication_is_byte_deterministic(tmp_path: Path) -> None:
    replay = _replay()
    first = _run(tmp_path / "first", FakeAdapter(replay))
    second = _run(tmp_path / "second", FakeAdapter(replay))
    assert first.sha256 == second.sha256
    assert (first.path / "metadata.json").read_bytes() == (
        second.path / "metadata.json"
    ).read_bytes()
    assert (first.path / "primitives.npz").read_bytes() == (
        second.path / "primitives.npz"
    ).read_bytes()


def test_factual_mismatch_refuses_publication(tmp_path: Path) -> None:
    replay = _replay()
    with pytest.raises(ScoringError, match="not exactly equal"):
        _run(tmp_path, FakeAdapter(replay, reset_mismatch=True))
    assert not (tmp_path / "scores" / replay.split / replay.episode_id).exists()


def test_snapshot_restoration_mismatch_refuses_publication(tmp_path: Path) -> None:
    replay = _replay()
    with pytest.raises(ScoringError, match="restoration was not exact"):
        _run(tmp_path, FakeAdapter(replay, broken_restore=True))
    assert not (tmp_path / "scores" / replay.split / replay.episode_id).exists()


def test_rng_mutation_refuses_publication(tmp_path: Path) -> None:
    replay = _replay()
    before = random.getstate()
    with pytest.raises(ScoringError, match="RNG state changed"):
        _run(tmp_path, FakeAdapter(replay, mutate_rng=True))
    assert random.getstate() == before
    assert not (tmp_path / "scores" / replay.split / replay.episode_id).exists()


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    replay = _replay()
    destination = tmp_path / "scores" / replay.split / replay.episode_id
    destination.mkdir(parents=True)
    marker = destination / "keep"
    marker.write_text("safe", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _run(tmp_path, FakeAdapter(replay))
    assert marker.read_text(encoding="utf-8") == "safe"


@pytest.mark.parametrize("protected_name", ["locks", "configs"])
def test_protocol_and_lock_paths_are_never_score_destinations(
    tmp_path: Path, protected_name: str
) -> None:
    replay = _replay()
    with pytest.raises(ScoringError, match="protocol or lock"):
        score_replay_to_sidecar(
            FakeAdapter(replay),
            replay,
            _links(),
            transforms=FROZEN_TRANSFORMS,
            output_root=tmp_path / "scores" / ".." / protected_name / "scores",
        )


def test_crash_removes_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay = _replay()

    def crash(_arrays: Any) -> bytes:
        raise RuntimeError("injected crash")

    monkeypatch.setattr(scoring, "_deterministic_npz_bytes", crash)
    with pytest.raises(RuntimeError, match="injected crash"):
        _run(tmp_path, FakeAdapter(replay))
    parent = tmp_path / "scores" / replay.split
    assert parent.exists()
    assert list(parent.iterdir()) == []


def test_loader_rejects_hash_tampering(tmp_path: Path) -> None:
    path = _run(tmp_path, FakeAdapter(_replay())).path
    primitive_path = path / "primitives.npz"
    primitive_path.write_bytes(primitive_path.read_bytes() + b"tamper")
    with pytest.raises(ScoringValidationError, match="SHA-256"):
        load_scoring_sidecar(path)


def test_loader_rejects_schema_tampering(tmp_path: Path) -> None:
    path = _run(tmp_path, FakeAdapter(_replay())).path
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["capture"]["state_count"] = 3
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ScoringValidationError, match="expected"):
        load_scoring_sidecar(path)


def test_intervention_availability_is_masked_per_common_draw(tmp_path: Path) -> None:
    class AlternatingAdapter(FakeAdapter):
        checks = 0

        def intervention_available(self, activation: np.ndarray) -> bool:
            del activation
            self.checks += 1
            return self.checks % 2 == 0

    replay = _replay()
    loaded = load_scoring_sidecar(_run(tmp_path, AlternatingAdapter(replay)).path)
    expected = np.asarray([[False, True, False, True]] * 2, dtype=np.bool_)
    assert np.array_equal(loaded.arrays["intervention_available"], expected)
    assert np.isnan(loaded.arrays["intervention_minus_actions"][~expected]).all()
    assert np.isfinite(loaded.arrays["intervention_plus_actions"][expected]).all()


def test_transform_order_and_content_hashes_fail_closed() -> None:
    reordered = (FROZEN_TRANSFORMS[1], FROZEN_TRANSFORMS[0], *FROZEN_TRANSFORMS[2:])
    with pytest.raises(ScoringError, match="frozen"):
        validate_frozen_transform_order(reordered)
    with pytest.raises(ScoringError, match="lowercase SHA-256"):
        ContentLinks(
            raw_metadata_sha256="A" * 64,
            raw_trajectory_sha256="2" * 64,
            probe_sha256="3" * 64,
            config_sha256="4" * 64,
            code_sha256="5" * 64,
        )


def test_actual_frozen_config_converts_to_exact_transform_order() -> None:
    perturbations = load_perturbations(Path("configs/perturbations.yaml"))
    assert (
        scoring_transforms_from_conditions(perturbations.counterfactual_score_grid)
        == FROZEN_TRANSFORMS
    )


def test_brightness_formula_updates_both_uint8_cameras() -> None:
    frame = ReplayFrame(
        control_step=0,
        camera_images={
            "agentview_image": np.asarray([[[1, 2, 255]]], dtype=np.uint8),
            "robot0_eye_in_hand_image": np.asarray([[[3, 4, 250]]], dtype=np.uint8),
        },
        low_dimensional={"state": np.asarray([0.0])},
        success=False,
    )
    bright = scoring._brightness_frame(frame, 1.15)
    assert np.array_equal(
        bright.camera_images["agentview_image"],
        np.asarray([[[1, 2, 255]]], dtype=np.uint8),
    )
    assert np.array_equal(
        bright.camera_images["robot0_eye_in_hand_image"],
        np.asarray([[[3, 5, 255]]], dtype=np.uint8),
    )
