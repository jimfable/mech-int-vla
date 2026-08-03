from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mech_int_vla.config import (
    ConditionSpec,
    PolicyExecutionConfig,
    SplitName,
    TaskSpec,
)
from mech_int_vla.instrumentation import (
    EXPERT_EARLY,
    EXPERT_LATE,
    VLM_CONTEXT,
    ActivationRecord,
    CallPhase,
    CallRecord,
)
from mech_int_vla.libero_runtime import (
    RawTraceFrame,
    ResetResult,
    StepResult,
    ValidityResult,
)
from mech_int_vla.manifest import EpisodeSpec
from mech_int_vla.rollout import (
    ACTIVATION_CANDIDATES,
    FRAME_SCALAR_FEATURE_NAMES,
    RolloutError,
    run_single_episode,
)

POLICY_REVISION = "1" * 40
BASE_REVISION = "2" * 40


def make_frame(step: int, *, success: bool = False) -> RawTraceFrame:
    agent_image = np.full((360, 360, 3), step, dtype=np.uint8)
    wrist_image = np.full((360, 360, 3), step + 10, dtype=np.uint8)
    return RawTraceFrame(
        control_step=step,
        simulator_time=0.05 * step,
        raw_observation={
            "agentview_image": agent_image,
            "robot0_eye_in_hand_image": wrist_image,
        },
        policy_state=np.arange(8, dtype=np.float32) + step,
        eef_position=np.asarray([0.0, 0.0, 1.0]),
        eef_quaternion_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        primary_object_position=np.asarray([0.1 + 0.01 * step, 0.0, 0.9]),
        primary_object_quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        goal_position=np.asarray([0.3, 0.0, 0.9]),
        goal_quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        gripper_qpos=np.asarray([0.02, 0.02]),
        gripper_qvel=np.asarray([0.0, 0.0]),
        primary_gripper_contact=step >= 1,
        primary_grasped=step >= 1 and not success,
        task_success=success,
        task_predicates={"goal_satisfied": success},
        phase="placed" if success else ("grasped" if step >= 1 else "pregrasp"),
    )


class FakeInstrumentation:
    def __init__(self, flow_model: object) -> None:
        self.flow_model = flow_model
        self.is_installed = False
        self.records: tuple[ActivationRecord, ...] = ()
        self.calls: tuple[CallRecord, ...] = ()
        self.run_index = -1
        self.exit_count = 0

    def __enter__(self):
        self.is_installed = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.is_installed = False
        self.exit_count += 1

    def clear(self) -> None:
        self.records = ()
        self.calls = ()

    def emit(self) -> None:
        if not self.is_installed:
            return
        self.run_index += 1
        self.calls = (
            CallRecord(CallPhase.PREFIX_CACHE, self.run_index, None, None),
            *(
                CallRecord(
                    CallPhase.DENOISING,
                    self.run_index,
                    denoising_step=step,
                    flow_time=1.0 - step / 10,
                )
                for step in range(10)
            ),
        )
        records = []
        for candidate_index, (_, (location, step)) in enumerate(
            ACTIVATION_CANDIDATES.items()
        ):
            value = torch.full((1, 4), self.run_index * 10 + candidate_index)
            records.append(
                ActivationRecord(
                    location=location,
                    phase=(
                        CallPhase.PREFIX_CACHE
                        if location == VLM_CONTEXT
                        else CallPhase.DENOISING
                    ),
                    run_index=self.run_index,
                    denoising_step=step,
                    flow_time=None if step is None else 1.0 - step / 10,
                    value=value,
                    norm_input=value.clone(),
                    source_shape=(1, 1 if location == VLM_CONTEXT else 50, 4),
                    token_count=1 if location == VLM_CONTEXT else 50,
                    patched=False,
                )
            )
        self.records = tuple(records)


class FakePolicy:
    def __init__(self, model: object) -> None:
        self.model = model
        self.config = SimpleNamespace(n_action_steps=1)
        self.instrumentation: FakeInstrumentation | None = None
        self.reset_count = 0
        self.select_count = 0
        self.fail = False

    def reset(self) -> None:
        self.reset_count += 1
        self.select_count = 0

    def select_action(self, observation):
        assert observation["processed"] is True
        self.select_count += 1
        if self.fail:
            raise RuntimeError("synthetic inference crash")
        assert self.instrumentation is not None
        self.instrumentation.emit()
        return torch.full((1, 7), float(self.select_count))


class FakePolicyRuntime:
    def __init__(self) -> None:
        model = object()
        self.policy = FakePolicy(model)
        self.instrumentation = FakeInstrumentation(model)
        self.policy.instrumentation = self.instrumentation
        self.postprocessor = lambda action: action
        self.snapshots = SimpleNamespace(
            lock=SimpleNamespace(
                policy_n_action_steps=1,
                policy_revision=POLICY_REVISION,
                base_vlm_revision=BASE_REVISION,
            )
        )
        self.preprocess_calls: list[tuple[dict, str]] = []

    def preprocess_observation(self, observation: dict, *, task: str) -> dict:
        self.preprocess_calls.append((observation, task))
        return {"processed": True}


class FakeEpisode:
    def __init__(
        self,
        *,
        max_steps: int = 3,
        success_after: int | None = 2,
        valid: bool = True,
    ) -> None:
        self.task = TaskSpec(
            rank=1,
            suite="libero_10",
            task_id=5,
            language="pick up the book and place it in the back compartment",
            primary_object="book",
            planar_symmetry_order=1,
        )
        self.execution = PolicyExecutionConfig(
            control_mode="relative",
            max_steps=max_steps,
            reset_noop_steps=10,
            n_action_steps=1,
        )
        self.validity_config = SimpleNamespace(schema="synthetic-frozen-validity")
        self.success_after = success_after
        self.valid = valid
        self.primary_object_name: str | None = None
        self._has_reset = False
        self.control_step = 0
        self.terminal = False
        self.reset_count = 0
        self.actions: list[np.ndarray] = []

    def reset(self, *, seed: int, condition: ConditionSpec) -> ResetResult:
        assert seed == 101000
        assert condition.name == "iid"
        self.reset_count += 1
        self._has_reset = True
        self.primary_object_name = "book_1"
        self.control_step = 0
        self.terminal = False
        validity = ValidityResult(
            valid=self.valid,
            reasons=() if self.valid else ("primary_object_penetration",),
            finite=True,
            deepest_primary_penetration_m=0.0 if self.valid else 0.01,
            linear_speed_m_s=0.0,
            angular_speed_rad_s=0.0,
            in_workspace=True,
            initial_success=False,
        )
        return ResetResult(
            observation={"step": 0},
            frame=make_frame(0),
            validity=validity,
            settle_actions=tuple((0.0,) * 6 + (-1.0,) for _ in range(10)),
        )

    def step(self, action: np.ndarray) -> StepResult:
        if self.terminal:
            raise AssertionError("runner advanced a preserved terminal state")
        self.actions.append(action.copy())
        self.control_step += 1
        success = (
            self.success_after is not None and self.control_step >= self.success_after
        )
        terminated = success
        truncated = self.control_step >= self.execution.max_steps and not terminated
        self.terminal = terminated or truncated
        frame = make_frame(self.control_step, success=success)
        return StepResult(
            observation={"step": self.control_step},
            reward=float(success),
            terminated=terminated,
            truncated=truncated,
            info={"is_success": success},
            frame=frame,
        )


def episode_spec() -> EpisodeSpec:
    return EpisodeSpec(
        suite="libero_10",
        task_id=5,
        task_rank=1,
        split=SplitName.DISCOVERY,
        base_init_state_id=0,
        condition_index=0,
        condition_name="iid",
        condition_family="iid",
        condition_parameters={},
        reset_seed=101000,
        inference_seed=12345,
        policy_revision=POLICY_REVISION,
        code_commit="3" * 40,
    )


def condition() -> ConditionSpec:
    return ConditionSpec(name="iid", family="iid", index=0, parameters={})


def test_successful_episode_records_actions_frames_and_five_candidates(
    tmp_path,
) -> None:
    runtime = FakePolicyRuntime()
    episode = FakeEpisode(success_after=2)
    result = run_single_episode(
        runtime,
        episode,
        runtime.instrumentation,
        episode_spec(),
        condition(),
        validity_retry_factory=lambda: FakeEpisode(valid=False),
        artifact_root=tmp_path / "artifacts" / "raw",
    )

    assert result.status == "success"
    assert result.success
    assert result.control_steps == 2
    assert episode.terminal
    assert episode.reset_count == 1
    assert runtime.policy.reset_count == 1
    assert runtime.instrumentation.exit_count == 1
    assert not runtime.instrumentation.is_installed
    assert len(runtime.preprocess_calls) == 2
    assert [action[0] for action in episode.actions] == [1.0, 2.0]

    metadata = json.loads((result.artifact_path / "metadata.json").read_text())
    assert metadata["outcome"] == {
        "control_steps": 2,
        "reward_sum": 1.0,
        "status": "success",
        "success": True,
        "terminal_state_preserved": True,
        "terminated": True,
        "truncated": False,
    }
    assert metadata["capture"]["activation_candidates"] == list(ACTIVATION_CANDIDATES)
    assert metadata["capture"]["score_stride_steps"] == 5
    assert metadata["capture"]["scored_policy_select_calls"] == 1
    assert metadata["task"]["primary_object"] == "book"
    assert metadata["task"]["planar_symmetry_order"] == 1
    assert metadata["capture"]["frame_scalar_feature_names"] == list(
        FRAME_SCALAR_FEATURE_NAMES
    )
    assert metadata["capture"]["task_predicate_names"] == ["goal_satisfied"]
    assert metadata["capture"]["raw_images_stored"] is True
    assert metadata["capture"]["raw_image_shape"] == [360, 360, 3]

    with np.load(result.artifact_path / "trajectory.npz", allow_pickle=False) as data:
        assert data["actions"].shape == (2, 7)
        assert data["frame_policy_state"].shape == (3, 8)
        assert data["frame_agentview_image"].shape == (3, 360, 360, 3)
        assert data["frame_robot0_eye_in_hand_image"].shape == (3, 360, 360, 3)
        assert data["frame_agentview_image"][:, 0, 0, 0].tolist() == [0, 1, 2]
        assert data["frame_robot0_eye_in_hand_image"][:, 0, 0, 0].tolist() == [
            10,
            11,
            12,
        ]
        assert data["frame_goal_position"].shape == (3, 3)
        assert data["frame_scalar_features"].shape == (
            3,
            len(FRAME_SCALAR_FEATURE_NAMES),
        )
        assert data["frame_task_predicates"].shape == (3, 1)
        assert data["frame_task_success"].tolist() == [False, False, True]
        feature_names = metadata["capture"]["frame_scalar_feature_names"]
        eef_object_sin = feature_names.index("symmetry_eef_object_yaw_sin")
        eef_object_cos = feature_names.index("symmetry_eef_object_yaw_cos")
        assert data["frame_scalar_features"][:, eef_object_sin] == pytest.approx(0.0)
        assert data["frame_scalar_features"][:, eef_object_cos] == pytest.approx(1.0)
        assert data["activation_control_step"].tolist() == [0]
        for candidate_index, candidate in enumerate(ACTIVATION_CANDIDATES):
            values = data[f"activation_{candidate}"]
            assert values.shape == (1, 4)
            assert values[:, 0].tolist() == [float(candidate_index)]
    assert not list(result.artifact_path.parent.glob(".*.tmp-*"))


def test_valid_episode_runs_until_frozen_horizon_truncation(tmp_path) -> None:
    runtime = FakePolicyRuntime()
    episode = FakeEpisode(max_steps=3, success_after=None)
    result = run_single_episode(
        runtime,
        episode,
        runtime.instrumentation,
        episode_spec(),
        condition(),
        validity_retry_factory=lambda: FakeEpisode(valid=False),
        artifact_root=tmp_path / "artifacts" / "raw",
    )

    assert result.status == "truncated"
    assert result.truncated
    assert not result.terminated
    assert result.control_steps == 3
    assert episode.terminal
    with np.load(result.artifact_path / "trajectory.npz", allow_pickle=False) as data:
        assert data["terminated"].tolist() == [False, False, False]
        assert data["truncated"].tolist() == [False, False, True]


def test_activations_are_serialized_only_at_five_step_cadence(tmp_path) -> None:
    runtime = FakePolicyRuntime()
    episode = FakeEpisode(max_steps=6, success_after=None)
    result = run_single_episode(
        runtime,
        episode,
        runtime.instrumentation,
        episode_spec(),
        condition(),
        validity_retry_factory=lambda: FakeEpisode(valid=False),
        artifact_root=tmp_path / "artifacts" / "raw",
    )

    metadata = json.loads((result.artifact_path / "metadata.json").read_text())
    assert metadata["capture"]["policy_select_calls"] == 6
    assert metadata["capture"]["scored_policy_select_calls"] == 2
    assert metadata["capture"]["instrumented_internal_calls"] == 22
    with np.load(result.artifact_path / "trajectory.npz", allow_pickle=False) as data:
        assert data["activation_control_step"].tolist() == [0, 5]
        assert all(
            data[f"activation_{candidate}"].shape == (2, 4)
            for candidate in ACTIVATION_CANDIDATES
        )


def test_invalid_reset_records_zero_action_artifact_without_policy_inference(
    tmp_path,
) -> None:
    runtime = FakePolicyRuntime()
    episode = FakeEpisode(valid=False)
    retry = FakeEpisode(valid=False)
    result = run_single_episode(
        runtime,
        episode,
        runtime.instrumentation,
        episode_spec(),
        condition(),
        validity_retry_factory=lambda: retry,
        artifact_root=tmp_path / "artifacts" / "raw",
    )

    assert result.status == "invalid_reset"
    assert not result.valid_reset
    assert result.control_steps == 0
    assert runtime.policy.reset_count == 0
    assert runtime.policy.select_count == 0
    assert runtime.instrumentation.exit_count == 0
    assert retry.reset_count == 1
    metadata = json.loads((result.artifact_path / "metadata.json").read_text())
    assert metadata["validity"]["reasons"] == ["primary_object_penetration"]
    assert metadata["validity_retry"]["performed"] is True
    assert metadata["validity_retry"]["agrees_on_invalidity"] is True
    assert metadata["validity_retry"]["agrees_on_reasons"] is True
    with np.load(result.artifact_path / "trajectory.npz", allow_pickle=False) as data:
        assert data["actions"].shape == (0, 7)
        assert data["frame_policy_state"].shape == (1, 8)
        assert all(
            data[f"activation_{name}"].shape == (0, 0) for name in ACTIVATION_CANDIDATES
        )


def test_inference_exception_removes_hooks_and_publishes_no_artifact(tmp_path) -> None:
    runtime = FakePolicyRuntime()
    runtime.policy.fail = True
    episode = FakeEpisode()
    artifact_root = tmp_path / "artifacts" / "raw"

    with pytest.raises(RuntimeError, match="synthetic inference crash"):
        run_single_episode(
            runtime,
            episode,
            runtime.instrumentation,
            episode_spec(),
            condition(),
            validity_retry_factory=lambda: FakeEpisode(valid=False),
            artifact_root=artifact_root,
        )

    assert runtime.instrumentation.exit_count == 1
    assert not runtime.instrumentation.is_installed
    assert not (artifact_root / "discovery" / episode_spec().episode_id).exists()


def test_missing_or_malformed_raw_camera_fails_before_artifact_publication(
    tmp_path,
) -> None:
    runtime = FakePolicyRuntime()
    episode = FakeEpisode(success_after=1)
    original_reset = episode.reset

    def reset_without_wrist(*, seed: int, condition: ConditionSpec) -> ResetResult:
        result = original_reset(seed=seed, condition=condition)
        raw = dict(result.frame.raw_observation)
        raw.pop("robot0_eye_in_hand_image")
        frame = RawTraceFrame(
            **{
                **result.frame.__dict__,
                "raw_observation": raw,
            }
        )
        return ResetResult(
            observation=result.observation,
            frame=frame,
            validity=result.validity,
            settle_actions=result.settle_actions,
        )

    episode.reset = reset_without_wrist
    artifact_root = tmp_path / "artifacts" / "raw"
    with pytest.raises(RolloutError, match="missing required camera"):
        run_single_episode(
            runtime,
            episode,
            runtime.instrumentation,
            episode_spec(),
            condition(),
            validity_retry_factory=lambda: FakeEpisode(valid=False),
            artifact_root=artifact_root,
        )
    assert not (artifact_root / "discovery" / episode_spec().episode_id).exists()


def test_protected_artifact_path_fails_before_simulator_reset(tmp_path) -> None:
    runtime = FakePolicyRuntime()
    episode = FakeEpisode()

    with pytest.raises(RolloutError, match="protocol or lock paths"):
        run_single_episode(
            runtime,
            episode,
            runtime.instrumentation,
            episode_spec(),
            condition(),
            validity_retry_factory=lambda: FakeEpisode(valid=False),
            artifact_root=tmp_path / "locks" / "rollouts",
        )

    assert episode.reset_count == 0


def test_runner_rejects_reusing_an_episode_runtime(tmp_path) -> None:
    runtime = FakePolicyRuntime()
    episode = FakeEpisode()
    episode._has_reset = True

    with pytest.raises(RolloutError, match="fresh runtime"):
        run_single_episode(
            runtime,
            episode,
            runtime.instrumentation,
            episode_spec(),
            condition(),
            validity_retry_factory=lambda: FakeEpisode(valid=False),
            artifact_root=tmp_path / "artifacts" / "raw",
        )

    assert episode.reset_count == 0


@pytest.mark.parametrize("wrong_location", [EXPERT_EARLY, EXPERT_LATE])
def test_activation_schema_drift_fails_closed(tmp_path, wrong_location) -> None:
    runtime = FakePolicyRuntime()
    original_emit = runtime.instrumentation.emit

    def emit_wrong_location() -> None:
        original_emit()
        records = list(runtime.instrumentation.records)
        record = records[0]
        records[0] = ActivationRecord(
            location=wrong_location,
            phase=record.phase,
            run_index=record.run_index,
            denoising_step=record.denoising_step,
            flow_time=record.flow_time,
            value=record.value,
            norm_input=record.norm_input,
            source_shape=record.source_shape,
            token_count=record.token_count,
            patched=record.patched,
        )
        runtime.instrumentation.records = tuple(records)

    runtime.instrumentation.emit = emit_wrong_location
    with pytest.raises(RolloutError, match="candidate vlm_context"):
        run_single_episode(
            runtime,
            FakeEpisode(),
            runtime.instrumentation,
            episode_spec(),
            condition(),
            validity_retry_factory=lambda: FakeEpisode(valid=False),
            artifact_root=tmp_path / "artifacts" / "raw",
        )
