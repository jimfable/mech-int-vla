from __future__ import annotations

import copy
import io
import random
import zlib
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Self

import numpy as np
import pytest
import torch

from mech_int_vla import scoring_runtime
from mech_int_vla.artifacts import ArtifactHashes, RolloutArtifact
from mech_int_vla.config import ConditionSpec
from mech_int_vla.libero_runtime import RawTraceFrame
from mech_int_vla.probes import (
    DEFAULT_CANDIDATE_PREFERENCE,
    FROZEN_RIDGE_ALPHA_GRID,
    AlphaCVResult,
    CandidateCVResult,
    CenteredCircularRidge,
    ProbeArtifact,
)
from mech_int_vla.scoring import (
    FROZEN_TRANSFORMS,
    ContentLinks,
    ScoringTransform,
    SimulatorSnapshot,
    load_scoring_sidecar,
    score_replay_to_sidecar,
)
from mech_int_vla.scoring_runtime import (
    CANDIDATE_TARGETS,
    ScoringRuntimeError,
    SmolVLAScoringAdapter,
    candidate_target,
    factual_replay_from_artifact,
)


def _artifact(*, valid: bool = True, action_count: int = 2) -> RolloutArtifact:
    frame_count = action_count + 1 if action_count else 1
    actions = (
        np.stack(
            [np.full(7, index / 10, dtype=np.float32) for index in range(action_count)]
        )
        if action_count
        else np.empty((0, 7), dtype=np.float32)
    )
    terminated = np.zeros(action_count, dtype=np.bool_)
    truncated = np.zeros(action_count, dtype=np.bool_)
    if action_count:
        truncated[-1] = True
    arrays: dict[str, np.ndarray] = {
        "actions": actions,
        "terminated": terminated,
        "truncated": truncated,
        "frame_control_step": np.arange(frame_count, dtype=np.int32),
        "frame_simulator_time": np.arange(frame_count, dtype=np.float64) / 20,
        "frame_policy_state": np.stack(
            [np.arange(8, dtype=np.float32) + index for index in range(frame_count)]
        ),
        "frame_agentview_image": np.stack(
            [
                np.full((2, 2, 3), 10 + index, dtype=np.uint8)
                for index in range(frame_count)
            ]
        ),
        "frame_robot0_eye_in_hand_image": np.stack(
            [
                np.full((2, 2, 3), 20 + index, dtype=np.uint8)
                for index in range(frame_count)
            ]
        ),
        "frame_eef_position": np.stack(
            [
                np.asarray([index, 1, 2], dtype=np.float64)
                for index in range(frame_count)
            ]
        ),
        "frame_eef_quaternion_xyzw": np.tile(
            np.asarray([0, 0, 0, 1], dtype=np.float64), (frame_count, 1)
        ),
        "frame_primary_object_position": np.stack(
            [
                np.asarray([3, index, 4], dtype=np.float64)
                for index in range(frame_count)
            ]
        ),
        "frame_primary_object_quaternion_wxyz": np.tile(
            np.asarray([1, 0, 0, 0], dtype=np.float64), (frame_count, 1)
        ),
        "frame_gripper_qpos": np.tile(
            np.asarray([0.1, 0.2], dtype=np.float64), (frame_count, 1)
        ),
        "frame_gripper_qvel": np.tile(
            np.asarray([0.0, 0.0], dtype=np.float64), (frame_count, 1)
        ),
        "frame_goal_present": np.ones(frame_count, dtype=np.bool_),
        "frame_goal_position": np.tile(
            np.asarray([5, 6, 7], dtype=np.float64), (frame_count, 1)
        ),
        "frame_goal_quaternion_wxyz": np.tile(
            np.asarray([1, 0, 0, 0], dtype=np.float64), (frame_count, 1)
        ),
        "frame_primary_gripper_contact": np.zeros(frame_count, dtype=np.bool_),
        "frame_primary_grasped": np.zeros(frame_count, dtype=np.bool_),
        "frame_task_success": np.zeros(frame_count, dtype=np.bool_),
        "frame_phase": np.asarray(["pregrasp"] * frame_count, dtype="U16"),
        "frame_task_predicates": np.tile(
            np.asarray([False, True], dtype=np.bool_), (frame_count, 1)
        ),
    }
    for value in arrays.values():
        value.flags.writeable = False
    metadata = {
        "episode": {
            "episode_id": "libero_10-task5-calibration-init10-cell0",
            "suite": "libero_10",
            "task_id": 5,
            "task_rank": 1,
            "split": "calibration",
            "base_init_state_id": 10,
            "condition_index": 0,
            "condition_name": "iid",
            "condition_family": "iid",
            "condition_parameters": {},
            "reset_seed": 10100,
            "policy_revision": "a" * 40,
            "code_commit": "b" * 40,
        },
        "task": {
            "suite": "libero_10",
            "task_id": 5,
            "rank": 1,
            "language": "put the black book on the shelf",
            "primary_object": "black_book",
            "planar_symmetry_order": 2,
        },
        "condition": {"name": "iid", "family": "iid", "index": 0, "parameters": {}},
        "model": {"base_vlm_revision": "c" * 40},
        "execution": {"max_steps": 2, "reset_noop_steps": 10},
        "validity": {"valid": valid},
        "outcome": {
            "status": "truncated" if valid else "invalid_reset",
            "success": False,
        },
        "capture": {"task_predicate_names": ("0:first", "1:second")},
    }
    return RolloutArtifact(
        path=Path("/tmp/fake-artifact"),
        metadata=metadata,
        arrays=MappingProxyType(arrays),
        hashes=ArtifactHashes("1" * 64, "2" * 64),
    )


def _trace(step: int) -> RawTraceFrame:
    raw = {
        "agentview_image": np.full((2, 2, 3), 10 + step, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((2, 2, 3), 20 + step, dtype=np.uint8),
        "robot0_eef_pos": np.asarray([step, 1, 2], dtype=np.float64),
        "robot0_eef_quat": np.asarray([0, 0, 0, 1], dtype=np.float64),
        "robot0_gripper_qpos": np.asarray([0.1, 0.2], dtype=np.float64),
        "robot0_gripper_qvel": np.asarray([0.0, 0.0], dtype=np.float64),
        "extra": np.asarray([99.0]),
    }
    return RawTraceFrame(
        control_step=step,
        simulator_time=step / 20,
        raw_observation=raw,
        policy_state=np.arange(8, dtype=np.float32) + step,
        eef_position=np.asarray([step, 1, 2], dtype=np.float64),
        eef_quaternion_xyzw=np.asarray([0, 0, 0, 1], dtype=np.float64),
        primary_object_position=np.asarray([3, step, 4], dtype=np.float64),
        primary_object_quaternion_wxyz=np.asarray([1, 0, 0, 0], dtype=np.float64),
        goal_position=np.asarray([5, 6, 7], dtype=np.float64),
        goal_quaternion_wxyz=np.asarray([1, 0, 0, 0], dtype=np.float64),
        gripper_qpos=np.asarray([0.1, 0.2], dtype=np.float64),
        gripper_qvel=np.asarray([0.0, 0.0], dtype=np.float64),
        primary_gripper_contact=False,
        primary_grasped=False,
        task_success=False,
        task_predicates={"1:second": True, "0:first": False},
        phase="pregrasp",
    )


class FakeEpisode:
    def __init__(self) -> None:
        self.task = SimpleNamespace(
            suite="libero_10",
            task_id=5,
            rank=1,
            language="put the black book on the shelf",
            primary_object="black_book",
            planar_symmetry_order=2,
        )
        self.execution = SimpleNamespace(
            n_action_steps=1, max_steps=2, reset_noop_steps=10
        )
        self.validity_config = SimpleNamespace(
            **dict(scoring_runtime.FROZEN_VALIDITY_VALUES)
        )
        self.wrapper = object()
        self.primary_object_name = None
        self._has_reset = False
        self.index = 0
        self.formatted: dict[str, Any] | None = None

    def reset(self, *, seed: int, condition: ConditionSpec) -> Any:
        assert seed == 10100
        assert condition.name == "iid"
        random.random()
        np.random.random()
        torch.rand(1)
        self._has_reset = True
        self.primary_object_name = "black_book_1"
        self.index = 0
        return SimpleNamespace(validity=SimpleNamespace(valid=True), frame=_trace(0))

    def step(self, action: np.ndarray) -> Any:
        assert action.shape == (7,)
        self.index += 1
        return SimpleNamespace(
            frame=_trace(self.index),
            terminated=False,
            truncated=self.index == 2,
        )

    def current_raw_trace(self) -> RawTraceFrame:
        return _trace(self.index)

    def format_observation(self, raw: dict[str, Any]) -> dict[str, Any]:
        self.formatted = copy.deepcopy(raw)
        return raw


class FakeInstrumentation:
    def __init__(self, model: Any, selected: str = "vlm_context") -> None:
        self.flow_model = model
        self.capture_steps = frozenset((0, 5))
        self.expected_action_tokens = 50
        self.is_installed = False
        self.records: list[Any] = []
        self.calls: list[Any] = []
        self.selected = selected
        self.active_patch: tuple[str, int | None, torch.Tensor] | None = None
        self.patch_requests: list[tuple[str, int | None, torch.Tensor]] = []

    def clear(self) -> None:
        self.records.clear()
        self.calls.clear()

    def _append_record(self, **kwargs: Any) -> None:
        self.records.append(
            SimpleNamespace(
                location=kwargs["location"],
                denoising_step=kwargs["frame"].denoising_step,
                value=kwargs["value"],
                patched=kwargs["patched"],
            )
        )

    def __enter__(self) -> Self:
        self.is_installed = True
        return self

    def __exit__(self, *_args: object) -> None:
        self.is_installed = False

    @contextmanager
    def patch(self, location: str, shift: torch.Tensor, *, denoising_step: int | None):
        self.active_patch = (location, denoising_step, shift.detach().clone())
        self.patch_requests.append(self.active_patch)
        try:
            yield
        finally:
            self.active_patch = None

    def emit(self, marker: float, noise: torch.Tensor) -> None:
        self.records = []
        for index, (location, step) in enumerate(CANDIDATE_TARGETS.values()):
            value = torch.tensor(
                [marker + float(noise[0, 0, 0]), 2.0 + index, 3.0],
                dtype=torch.float32,
            )
            patched = (
                self.active_patch is not None
                and (
                    location,
                    step,
                )
                == self.active_patch[:2]
            )
            if patched:
                value = value + self.active_patch[2].to(dtype=torch.float32)
            self._append_record(
                location=location,
                frame=SimpleNamespace(denoising_step=step),
                value=value[None],
                patched=patched,
            )
        self.calls = [SimpleNamespace(phase="prefix_cache", denoising_step=None)] + [
            SimpleNamespace(phase="denoising", denoising_step=step)
            for step in range(10)
        ]

    def activations(
        self, location: str, *, denoising_step: int | None
    ) -> tuple[Any, ...]:
        return tuple(
            record
            for record in self.records
            if record.location == location and record.denoising_step == denoising_step
        )


class FakePolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            n_action_steps=1,
            chunk_size=50,
            adapt_to_pi_aloha=False,
            max_action_dim=9,
            device="cpu",
            action_feature=SimpleNamespace(shape=(7,)),
            rtc_config=None,
        )
        self.model = SimpleNamespace(
            config=SimpleNamespace(num_steps=10), training=False
        )
        self.training = False
        self.device = torch.device("cpu")
        self._queues = {"action": deque([], maxlen=1)}
        self.instrumentation: FakeInstrumentation | None = None
        self.private_calls = 0
        self.public_calls = 0

    def _rtc_enabled(self) -> bool:
        return False

    def predict_action_chunk(self, *_args: Any, **_kwargs: Any) -> None:
        self.public_calls += 1
        raise AssertionError("public queue-mutating inference must not be called")

    def _get_action_chunk(
        self, batch: dict[str, Any], noise: torch.Tensor
    ) -> torch.Tensor:
        self.private_calls += 1
        marker = float(batch["marker"].item())
        assert self.instrumentation is not None
        self.instrumentation.emit(marker, noise)
        return noise[:, :, :7] + marker


class FakePolicyRuntime:
    def __init__(self, policy: FakePolicy) -> None:
        self.policy = policy
        self.last_raw: dict[str, Any] | None = None
        self.snapshots = SimpleNamespace(
            lock=SimpleNamespace(
                policy_revision="a" * 40,
                base_vlm_revision="c" * 40,
                policy_num_flow_steps=10,
                policy_chunk_size=50,
                policy_n_action_steps=1,
                lerobot_commit=scoring_runtime.FROZEN_LEROBOT_COMMIT,
                lerobot_python_sha256=(scoring_runtime.FROZEN_LEROBOT_PYTHON_SHA256),
            )
        )
        self.original_state_shape = (6,)
        self.runtime_state_shape = (8,)
        self.normalization_state_shapes = {"mean": (8,), "std": (8,), "count": (1,)}

    def preprocess_observation(
        self, observation: dict[str, Any], *, task: str
    ) -> dict[str, Any]:
        assert task == "put the black book on the shelf"
        self.last_raw = copy.deepcopy(observation)
        return {"marker": torch.tensor(1.0), "nested": {"x": torch.tensor([2.0])}}

    @staticmethod
    def postprocessor(actions: torch.Tensor) -> torch.Tensor:
        return actions * 2


class FakeTimer:
    def __init__(self) -> None:
        self.clock = 0
        self.reset_calls = 0
        self.sync_calls = 0

    def synchronize(self) -> None:
        self.sync_calls += 1

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1

    @staticmethod
    def memory_allocated() -> int:
        return 100

    @staticmethod
    def max_memory_allocated() -> int:
        return 180

    def create_event(self) -> object:
        return object()

    @staticmethod
    def record_event(_event: object) -> None:
        pass

    @staticmethod
    def elapsed_ms(_start: object, _end: object) -> float:
        return 1.5

    def perf_counter_ns(self) -> int:
        self.clock += 100
        return self.clock


def _probe(
    candidate: str = "vlm_context", *, coefficient: np.ndarray | None = None
) -> ProbeArtifact:
    rows = (
        np.asarray([[1.0, 0, 0], [0, 1.0, 0]], dtype=np.float64)
        if coefficient is None
        else coefficient
    )
    model = CenteredCircularRidge(
        alpha=0.1,
        symmetry_order=2,
        feature_center=np.zeros(3, dtype=np.float64),
        target_center=np.zeros(2, dtype=np.float64),
        coefficient=rows,
    )
    target_index = DEFAULT_CANDIDATE_PREFERENCE.index(candidate)
    candidate_results = []
    for index, name in enumerate(DEFAULT_CANDIDATE_PREFERENCE):
        selected_mean = (
            0.1 if index == target_index else (0.4 if index < target_index else 0.2)
        )
        alpha_results = []
        for alpha in FROZEN_RIDGE_ALPHA_GRID:
            score = selected_mean if alpha == 0.1 else selected_mean + 1.0 + alpha
            alpha_results.append(
                AlphaCVResult(
                    alpha=alpha,
                    fold_mae_rad=(score,) * 5,
                    mean_mae_rad=score,
                    standard_error_rad=0.0,
                )
            )
        candidate_results.append(
            CandidateCVResult(
                candidate=name,
                alpha_results=tuple(alpha_results),
                selected_alpha=0.1,
                mean_mae_rad=selected_mean,
                standard_error_rad=0.0,
            )
        )
    return ProbeArtifact(
        model=model,
        candidate=candidate,
        alpha_grid=FROZEN_RIDGE_ALPHA_GRID,
        candidate_preference=DEFAULT_CANDIDATE_PREFERENCE,
        candidate_results=tuple(candidate_results),
        one_standard_error_threshold_rad=0.1,
        fold_test_groups=((0, 5), (1, 6), (2, 7), (3, 8), (4, 9)),
        training_rows=10,
        training_episodes=10,
        training_base_init_state_ids=tuple(range(10)),
    )


def _adapter(
    *,
    candidate: str = "vlm_context",
    probe: Any | None = None,
) -> tuple[
    SmolVLAScoringAdapter, FakeEpisode, FakePolicy, FakeInstrumentation, FakeTimer
]:
    episode = FakeEpisode()
    policy = FakePolicy()
    instrumentation = FakeInstrumentation(policy.model, selected=candidate)
    policy.instrumentation = instrumentation
    timer = FakeTimer()
    adapter = SmolVLAScoringAdapter(
        episode,
        FakePolicyRuntime(policy),
        _artifact(),
        probe or _probe(candidate),
        instrumentation,
        reset_seed=10100,
        original_condition=ConditionSpec("iid", "iid", 0, {}),
        timer_backend=timer,
    )
    return adapter, episode, policy, instrumentation, timer


def test_factual_replay_conversion_covers_exact_numeric_state() -> None:
    artifact = _artifact()
    replay = factual_replay_from_artifact(artifact)
    assert replay.episode_id == artifact.episode_id
    assert replay.terminal_status == "truncated"
    assert replay.actions.dtype == np.float32
    assert len(replay.frames) == 3
    frame = replay.frames[1]
    assert set(frame.low_dimensional) == {
        "simulator_time",
        "policy_state",
        "eef_position",
        "eef_quaternion_xyzw",
        "primary_object_position",
        "primary_object_quaternion_wxyz",
        "gripper_qpos",
        "gripper_qvel",
        "goal_present",
        "goal_position",
        "goal_quaternion_wxyz",
        "primary_gripper_contact",
        "primary_grasped",
        "task_success",
        "phase_one_hot",
        "task_predicate_schema_sha256",
        "task_predicates",
    }
    assert np.array_equal(frame.low_dimensional["task_predicates"], [False, True])
    assert np.array_equal(
        frame.low_dimensional["phase_one_hot"], [True, False, False, False]
    )
    assert not frame.camera_images["agentview_image"].flags.writeable
    assert not replay.actions.flags.writeable


def test_factual_replay_rejects_invalid_or_empty_artifact() -> None:
    with pytest.raises(ScoringRuntimeError, match="valid nonempty"):
        factual_replay_from_artifact(_artifact(valid=False, action_count=0))


@pytest.mark.parametrize(
    ("candidate", "expected"),
    list(CANDIDATE_TARGETS.items()),
)
def test_candidate_mapping(candidate: str, expected: tuple[str, int | None]) -> None:
    assert candidate_target(candidate) == expected


def test_local_noise_is_deterministic_without_global_rng_mutation() -> None:
    adapter, *_ = _adapter()
    before = torch.random.get_rng_state().clone()
    first = adapter.noise_for_seed(123)
    second = adapter.noise_for_seed(123)
    third = adapter.noise_for_seed(124)
    assert first.shape == (1, 50, 9)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    assert torch.equal(before, torch.random.get_rng_state())


def test_private_queue_bypass_capture_cache_and_cost() -> None:
    adapter, _, policy, _, timer = _adapter()
    adapter.begin_score_state()
    queue_before = adapter.policy_queue_state()
    noise = adapter.noise_for_seed(7)
    processed = {"marker": torch.tensor(1.0), "nested": {"x": torch.tensor([2.0])}}
    result = adapter.predict_action_chunk(
        processed, noise=noise, intervention_degrees=None
    )
    assert result.actions.shape == (50, 7)
    assert result.actions.dtype == np.float32
    assert result.activation.shape == (3,)
    assert policy.private_calls == 1
    assert policy.public_calls == 0
    assert adapter.policy_queue_state() == queue_before
    assert timer.reset_calls == 1
    assert timer.sync_calls == 3
    assert result.cost.cuda_event_ms == 1.5
    assert result.cost.wall_time_ns == 100
    assert result.cost.forward_count == 1
    assert result.cost.intervention_count == 0
    assert result.cost.peak_allocated_bytes == 180
    assert result.cost.incremental_peak_allocated_bytes == 80
    assert result.cost.logical_activation_bytes == result.activation.nbytes
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, result.activation, allow_pickle=False)
    assert result.cost.compressed_activation_bytes == len(
        zlib.compress(buffer.getvalue(), 9)
    )


def test_transformed_same_noise_does_not_overwrite_original_patch_source() -> None:
    adapter, _, _, instrumentation, _ = _adapter()
    adapter.begin_score_state()
    noise = adapter.noise_for_seed(8)
    original = adapter.predict_action_chunk(
        {"marker": torch.tensor(1.0)}, noise=noise, intervention_degrees=None
    )
    transformed = adapter.predict_action_chunk(
        {"marker": torch.tensor(50.0)}, noise=noise, intervention_degrees=None
    )
    assert not np.array_equal(original.activation, transformed.activation)
    rows = torch.as_tensor(adapter.probe.model.coefficient.copy())
    source = torch.as_tensor(original.activation, dtype=torch.float64)
    raw = source @ rows.mT
    for degrees in (-10.0, 10.0):
        adapter.predict_action_chunk(
            {"marker": torch.tensor(1.0)},
            noise=noise,
            intervention_degrees=degrees,
        )
        location, step, shift = instrumentation.patch_requests[-1]
        assert (location, step) == ("vlm_context", None)
        moved = (source + shift) @ rows.mT
        angle = 2 * np.deg2rad(degrees)
        expected = torch.as_tensor(
            [
                raw[0] * np.cos(angle) - raw[1] * np.sin(angle),
                raw[0] * np.sin(angle) + raw[1] * np.cos(angle),
            ]
        )
        torch.testing.assert_close(moved, expected)


def test_process_observation_replaces_cameras_and_robot_fields() -> None:
    adapter, episode, *_ = _adapter()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()
    adapter.reset_replay()
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    frame = factual_replay_from_artifact(_artifact()).frames[1]
    processed = adapter.process_observation(frame)
    assert processed["marker"].item() == 1
    assert episode.formatted is not None
    assert np.array_equal(
        episode.formatted["agentview_image"], frame.camera_images["agentview_image"]
    )
    assert np.array_equal(
        episode.formatted["robot0_eef_pos"], frame.low_dimensional["eef_position"]
    )
    assert episode.formatted["extra"].item() == 99


def test_reset_step_and_snapshot_transform_validity_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, episode, *_ = _adapter()
    first = adapter.reset_replay()
    transition = adapter.step_replay(np.zeros(7, dtype=np.float32))
    assert first.control_step == 0
    assert transition.frame.control_step == 1

    state = SimpleNamespace(
        state=np.asarray([1.0], dtype=np.float64),
        camera_positions=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        camera_quaternions=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )
    calls: list[Any] = []
    runtime = SimpleNamespace(
        SimulatorSnapshot=lambda **kwargs: SimpleNamespace(**kwargs),
        capture_simulator_snapshot=lambda wrapper: state,
        restore_simulator_snapshot=lambda wrapper, snapshot: calls.append(
            ("restore", snapshot)
        ),
        apply_condition=lambda wrapper, condition, primary_object_name: calls.append(
            ("apply", condition, primary_object_name)
        ),
        counterfactual_validity_reasons=lambda *args, **kwargs: (
            "counterfactual_primary_object_penetration",
        ),
    )
    monkeypatch.setattr(scoring_runtime, "_libero_runtime_module", lambda: runtime)
    snapshot = adapter.capture_snapshot()
    assert isinstance(snapshot, SimulatorSnapshot)
    adapter.apply_transform(ScoringTransform("object_yaw_pos_15", "object_pose", 15.0))
    validity = adapter.transformed_validity()
    assert validity.finite and not validity.penetration_ok and validity.workspace_ok
    adapter.restore_snapshot(snapshot)
    assert calls[0][0] == "apply"
    assert calls[-1][0] == "restore"
    assert episode.primary_object_name == "black_book_1"


def test_fail_closed_policy_probe_and_noise_invariants() -> None:
    adapter, _, policy, _, _ = _adapter()
    policy.config.adapt_to_pi_aloha = True
    with pytest.raises(ScoringRuntimeError, match="adapt_to_pi_aloha"):
        _adapter_from_parts(policy=policy)

    deficient = _probe(
        coefficient=np.asarray([[1.0, 0, 0], [2.0, 0, 0]], dtype=np.float64)
    )
    with pytest.raises(ScoringRuntimeError, match="rank exactly two"):
        _adapter(probe=deficient)

    invalid_episode = FakeEpisode()
    invalid_episode.validity_config.max_penetration_depth_m = 0.006
    with pytest.raises(ScoringRuntimeError, match="validity config"):
        _adapter_from_parts(policy=FakePolicy(), episode=invalid_episode)

    adapter.begin_score_state()
    with pytest.raises(ScoringRuntimeError, match="noise tensor"):
        adapter.predict_action_chunk(
            {"marker": torch.tensor(1.0)},
            noise=torch.zeros((1, 49, 9)),
            intervention_degrees=None,
        )
    assert not adapter.intervention_available(np.asarray([0.0, 0.0, 5.0], np.float32))
    with pytest.raises(ScoringRuntimeError, match="no cached original"):
        adapter.predict_action_chunk(
            {"marker": torch.tensor(1.0)},
            noise=adapter.noise_for_seed(99),
            intervention_degrees=10.0,
        )


def test_private_inference_refuses_queue_mutation() -> None:
    adapter, _, policy, _, _ = _adapter()
    original = policy._get_action_chunk

    def mutating(batch: dict[str, Any], noise: torch.Tensor) -> torch.Tensor:
        result = original(batch, noise)
        policy._queues["action"].append(torch.zeros(7))
        return result

    policy._get_action_chunk = mutating  # type: ignore[method-assign]
    adapter.begin_score_state()
    with pytest.raises(ScoringRuntimeError, match="mutated policy queues"):
        adapter.predict_action_chunk(
            {"marker": torch.tensor(1.0)},
            noise=adapter.noise_for_seed(11),
            intervention_degrees=None,
        )


def test_adapter_runs_end_to_end_through_atomic_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, *_ = _adapter()
    snapshot = SimpleNamespace(
        state=np.asarray([1.0], dtype=np.float64),
        camera_positions=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        camera_quaternions=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )
    runtime = SimpleNamespace(
        SimulatorSnapshot=lambda **kwargs: SimpleNamespace(**kwargs),
        capture_simulator_snapshot=lambda wrapper: snapshot,
        restore_simulator_snapshot=lambda wrapper, value: None,
        apply_condition=lambda wrapper, condition, primary_object_name: None,
        counterfactual_validity_reasons=lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(scoring_runtime, "_libero_runtime_module", lambda: runtime)
    result = score_replay_to_sidecar(
        adapter,
        adapter.factual,
        ContentLinks(
            "1" * 64,
            "2" * 64,
            adapter.probe.sha256(),
            "4" * 64,
            "5" * 64,
        ),
        transforms=FROZEN_TRANSFORMS,
        output_root=tmp_path,
    )
    loaded = load_scoring_sidecar(result.path)
    assert np.array_equal(loaded.arrays["control_step"], [0])
    assert loaded.arrays["original_actions"].shape == (1, 8, 10, 7)
    assert loaded.arrays["transformed_actions"].shape == (1, 6, 4, 10, 7)


def test_sidecar_refuses_stale_raw_or_probe_content_links(tmp_path: Path) -> None:
    adapter, *_ = _adapter()
    with pytest.raises(ScoringRuntimeError, match="raw metadata hash link"):
        score_replay_to_sidecar(
            adapter,
            adapter.factual,
            ContentLinks(
                "0" * 64,
                "2" * 64,
                adapter.probe.sha256(),
                "4" * 64,
                "5" * 64,
            ),
            transforms=FROZEN_TRANSFORMS,
            output_root=tmp_path,
        )
    with pytest.raises(ScoringRuntimeError, match="probe hash link"):
        score_replay_to_sidecar(
            adapter,
            adapter.factual,
            ContentLinks("1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64),
            transforms=FROZEN_TRANSFORMS,
            output_root=tmp_path,
        )


def _adapter_from_parts(
    *, policy: FakePolicy, episode: FakeEpisode | None = None
) -> SmolVLAScoringAdapter:
    selected_episode = episode or FakeEpisode()
    instrumentation = FakeInstrumentation(policy.model)
    policy.instrumentation = instrumentation
    return SmolVLAScoringAdapter(
        selected_episode,
        FakePolicyRuntime(policy),
        _artifact(),
        _probe(),
        instrumentation,
        reset_seed=10100,
        original_condition=ConditionSpec("iid", "iid", 0, {}),
        timer_backend=FakeTimer(),
    )
