from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from mech_int_vla.config import ConditionSpec, load_protocol_config
from mech_int_vla.libero_runtime import (
    CAMERA_NAME_MAPPING,
    LiberoRuntimeError,
    RawLiberoEpisode,
    apply_camera_transform,
    brightness_transform_raw_observation,
    build_raw_libero_env,
    capture_simulator_snapshot,
    check_validity,
    deterministic_phase,
    extract_policy_state,
    resolve_primary_object_name,
    restore_simulator_snapshot,
    temporary_condition,
)

ROOT = Path(__file__).parents[1]


@dataclass
class FakeContact:
    geom1: int
    geom2: int
    dist: float


class FakeModel:
    body_parentid = np.asarray([0, 0])
    geom_bodyid = np.asarray([1, 0])

    def __init__(self) -> None:
        self.cam_pos = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
        self.cam_quat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)

    def camera_name2id(self, name: str) -> int:
        assert name == "agentview"
        return 0

    def geom_name2id(self, name: str) -> int:
        return {"book_geom": 0, "gripper_geom": 1}[name]


class FakeData:
    def __init__(self) -> None:
        self.qpos = {"book_free": np.asarray([0.0, 0.0, 0.92, 1.0, 0.0, 0.0, 0.0])}
        self.qvel = {"book_free": np.zeros(6, dtype=np.float64)}
        self.body_xpos = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.92]])
        self.body_xquat = np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        self.contact: list[FakeContact] = []
        self.ncon = 0
        self.time = 0.0

    def get_joint_qpos(self, name: str) -> np.ndarray:
        return self.qpos[name].copy()

    def set_joint_qpos(self, name: str, value: np.ndarray) -> None:
        self.qpos[name] = np.asarray(value, dtype=np.float64).copy()

    def get_joint_qvel(self, name: str) -> np.ndarray:
        return self.qvel[name].copy()


class FakeSim:
    def __init__(self) -> None:
        self.model = FakeModel()
        self.data = FakeData()

    def forward(self) -> None:
        self.data.body_xpos[1] = self.data.qpos["book_free"][:3]
        self.data.body_xquat[1] = self.data.qpos["book_free"][3:7]


class FakeObjectState:
    object_state_type = "object"

    def get_geom_state(self):
        return {
            "pos": np.asarray([0.2, 0.1, 0.95]),
            "quat": np.asarray([1.0, 0.0, 0.0, 0.0]),
        }


class FakeBackend:
    def __init__(self) -> None:
        self.sim = FakeSim()
        self.objects_dict = {
            "book_1": types.SimpleNamespace(
                name="book_1",
                category_name="black_book",
                joints=["book_free"],
                contact_geoms=["book_geom"],
            )
        }
        self.obj_body_id = {"book_1": 1}
        self.workspace_offset = np.asarray([0.0, 0.0, 0.9])
        self.table_full_size = np.asarray([1.0, 1.2, 0.05])
        self.parsed_problem = {"goal_state": [("in", "book_1", "caddy_region")]}
        self.object_states_dict = {"caddy_region": FakeObjectState()}
        gripper = types.SimpleNamespace(contact_geoms=["gripper_geom"])
        self.robots = [types.SimpleNamespace(gripper=gripper)]
        self.grasped = False
        self.placed = False

    def _get_observations(self):
        return {
            "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eef_pos": np.asarray([0.0, 0.0, 1.0]),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.asarray([0.02, 0.02]),
            "robot0_gripper_qvel": np.asarray([0.0, 0.0]),
        }

    def _eval_predicate(self, state) -> bool:
        assert tuple(state) == ("in", "book_1", "caddy_region")
        return self.placed

    def _check_grasp(self, *, gripper, object_geoms) -> bool:
        assert object_geoms == ["book_geom"]
        return self.grasped


class FakeOffscreen:
    def __init__(self, backend: FakeBackend) -> None:
        self.env = backend
        self.steps = 0
        self.success_after: int | None = None

    def _state(self) -> np.ndarray:
        data = self.env.sim.data
        return np.concatenate(
            [
                np.asarray([data.time]),
                data.qpos["book_free"],
                data.qvel["book_free"],
            ]
        )

    def get_sim_state(self) -> np.ndarray:
        return self._state().copy()

    def set_state(self, state: np.ndarray) -> None:
        state = np.asarray(state)
        self.env.sim.data.time = float(state[0])
        self.env.sim.data.qpos["book_free"] = state[1:8].copy()
        self.env.sim.data.qvel["book_free"] = state[8:14].copy()
        self.env.sim.forward()

    def check_success(self) -> bool:
        return self.env.placed

    def step(self, action: np.ndarray):
        assert np.asarray(action).shape == (7,)
        self.steps += 1
        self.env.sim.data.time += 0.05
        if self.success_after is not None and self.steps >= self.success_after:
            self.env.placed = True
        return self.env._get_observations(), 1.0, self.env.placed, {}


class FakeWrapper:
    def __init__(self) -> None:
        self.backend = FakeBackend()
        self._env = FakeOffscreen(self.backend)
        self.reset_count = 0
        self.task = "fake task"

    def reset(self, seed=None):
        self.reset_count += 1
        return self._format_raw_obs(self.backend._get_observations()), {}

    def _format_raw_obs(self, raw):
        return {
            "pixels": {
                "camera1": raw["agentview_image"],
                "camera2": raw["robot0_eye_in_hand_image"],
            },
            "robot_state": raw,
        }

    def close(self) -> None:
        pass


def test_policy_state_is_exactly_eight_values() -> None:
    raw = FakeBackend()._get_observations()
    state = extract_policy_state(raw)
    assert state.shape == (8,)
    assert np.allclose(state, [0, 0, 1, 0, 0, 0, 0.02, 0.02])


@pytest.mark.parametrize(
    ("requested", "runtime_key"),
    (("black_book", "black_book_1"), ("white_yellow_mug", "white_yellow_mug_1")),
)
def test_pinned_primary_object_categories_resolve_real_bddl_keys(
    requested: str, runtime_key: str
) -> None:
    backend = types.SimpleNamespace(
        objects_dict={
            runtime_key: types.SimpleNamespace(
                name=runtime_key,
                category_name=requested,
            )
        }
    )
    assert resolve_primary_object_name(backend, requested) == runtime_key


def test_phase_precedence_and_two_cm_transport_threshold() -> None:
    start = np.zeros(3)
    assert (
        deterministic_phase(
            placed=False,
            grasped=False,
            object_position=np.ones(3),
            post_settle_start_position=start,
            displacement_threshold_m=0.02,
        )
        == "pregrasp"
    )
    assert (
        deterministic_phase(
            placed=False,
            grasped=True,
            object_position=np.asarray([0.019, 0, 0]),
            post_settle_start_position=start,
            displacement_threshold_m=0.02,
        )
        == "grasped"
    )
    assert (
        deterministic_phase(
            placed=False,
            grasped=True,
            object_position=np.asarray([0.02, 0, 0]),
            post_settle_start_position=start,
            displacement_threshold_m=0.02,
        )
        == "transport"
    )
    assert (
        deterministic_phase(
            placed=True,
            grasped=False,
            object_position=start,
            post_settle_start_position=start,
            displacement_threshold_m=0.02,
        )
        == "placed"
    )


def test_object_and_camera_transforms_restore_exactly() -> None:
    wrapper = FakeWrapper()
    original = capture_simulator_snapshot(wrapper)
    yaw = ConditionSpec("yaw", "object_yaw", parameters={"value": 90})
    with temporary_condition(wrapper, yaw, primary_object_name="book_1"):
        qpos = wrapper.backend.sim.data.qpos["book_free"]
        assert np.allclose(qpos[3:7], [math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    assert np.array_equal(capture_simulator_snapshot(wrapper).state, original.state)

    apply_camera_transform(wrapper, yaw_degrees=90, lateral_m=0.02)
    assert np.allclose(wrapper.backend.sim.model.cam_pos[0], [1.02, 2.0, 3.0])
    assert np.allclose(
        wrapper.backend.sim.model.cam_quat[0],
        [math.sqrt(0.5), 0, 0, math.sqrt(0.5)],
    )
    restore_simulator_snapshot(wrapper, original)
    assert np.array_equal(wrapper.backend.sim.model.cam_pos, original.camera_positions)
    assert np.array_equal(
        wrapper.backend.sim.model.cam_quat, original.camera_quaternions
    )


def test_brightness_transform_uses_both_uint8_cameras_and_rounds() -> None:
    raw = FakeBackend()._get_observations()
    raw["agentview_image"][:] = np.asarray([[[1, 2, 255]]], dtype=np.uint8)
    raw["robot0_eye_in_hand_image"][:] = np.asarray([[[100, 101, 200]]], dtype=np.uint8)
    transformed = brightness_transform_raw_observation(raw, 1.15)
    assert transformed["agentview_image"][0, 0].tolist() == [1, 2, 255]
    assert transformed["robot0_eye_in_hand_image"][0, 0].tolist() == [115, 116, 230]
    assert raw["robot0_eye_in_hand_image"][0, 0].tolist() == [100, 101, 200]


def test_runtime_counterfactual_observation_restores_and_marks_invalid() -> None:
    protocol = load_protocol_config(ROOT / "configs")
    wrapper = FakeWrapper()
    runtime = RawLiberoEpisode(
        wrapper,
        protocol.task_order.tasks[0],
        protocol.split.policy_execution,
        protocol.perturbations.validity,
    )
    reset = runtime.reset(seed=101000, condition=ConditionSpec("iid", "iid"))
    current = runtime.current_raw_trace()
    assert current.control_step == 0
    assert np.array_equal(current.policy_state, reset.frame.policy_state)
    current.raw_observation["agentview_image"][0, 0, 0] = 99
    assert runtime.current_raw_trace().raw_observation["agentview_image"][0, 0, 0] != 99
    original = capture_simulator_snapshot(wrapper)
    yaw = ConditionSpec("yaw", "object_pose", parameters={"yaw": 15})
    with runtime.counterfactual_observation(yaw) as counterfactual:
        assert counterfactual.available
        assert counterfactual.observation is not None
        edited = runtime.current_raw_trace()
        assert not np.array_equal(
            edited.primary_object_quaternion_wxyz,
            reset.frame.primary_object_quaternion_wxyz,
        )
        assert not np.array_equal(
            capture_simulator_snapshot(wrapper).state,
            original.state,
        )
    restored = capture_simulator_snapshot(wrapper)
    assert np.array_equal(restored.state, original.state)
    assert np.array_equal(restored.camera_positions, original.camera_positions)
    assert reset.frame.control_step == 0

    wrapper.backend.sim.data.contact = [FakeContact(0, 1, -0.006)]
    wrapper.backend.sim.data.ncon = 1
    with runtime.counterfactual_observation(yaw) as counterfactual:
        assert not counterfactual.available
        assert counterfactual.observation is None
        assert counterfactual.reasons == ("counterfactual_primary_object_penetration",)

    camera = ConditionSpec(
        "camera",
        "camera_render",
        parameters={"yaw": 3, "lateral_m": 0.0},
    )
    with runtime.counterfactual_observation(camera) as counterfactual:
        assert counterfactual.available


def test_amended_validity_thresholds_are_operational() -> None:
    wrapper = FakeWrapper()
    validity = load_protocol_config(ROOT / "configs").perturbations.validity
    result = check_validity(wrapper, "book_1", validity)
    assert result.valid

    wrapper.backend.sim.data.contact = [FakeContact(0, 1, -0.0051)]
    wrapper.backend.sim.data.ncon = 1
    wrapper.backend.sim.data.qvel["book_free"] = np.asarray([0.051, 0, 0, 0, 0, 0.51])
    result = check_validity(wrapper, "book_1", validity)
    assert not result.valid
    assert set(result.reasons) == {
        "primary_object_penetration",
        "primary_object_linear_instability",
        "primary_object_angular_instability",
    }

    wrapper = FakeWrapper()
    wrapper.backend.sim.data.qpos["book_free"][0] = 0.51
    wrapper.backend.sim.forward()
    wrapper.backend.placed = True
    result = check_validity(wrapper, "book_1", validity)
    assert set(result.reasons) == {
        "primary_object_outside_tabletop_workspace",
        "initial_task_success",
    }


def test_manual_backend_step_preserves_terminal_state() -> None:
    protocol = load_protocol_config(ROOT / "configs")
    wrapper = FakeWrapper()
    runtime = RawLiberoEpisode(
        wrapper,
        protocol.task_order.tasks[0],
        protocol.split.policy_execution,
        protocol.perturbations.validity,
    )
    reset = runtime.reset(seed=101000, condition=ConditionSpec("iid", "iid"))
    assert len(reset.settle_actions) == 10
    assert wrapper._env.steps == 10
    assert reset.frame.policy_state.shape == (8,)
    assert set(reset.observation["pixels"]) == {"camera1", "camera2"}

    wrapper._env.success_after = 11
    step = runtime.step(np.zeros(7, dtype=np.float32))
    assert step.terminated and step.info["is_success"]
    assert wrapper.reset_count == 1
    terminal_state = capture_simulator_snapshot(wrapper).state.copy()
    with pytest.raises(LiberoRuntimeError, match="terminal"):
        runtime.step(np.zeros(7, dtype=np.float32))
    assert np.array_equal(capture_simulator_snapshot(wrapper).state, terminal_state)
    with pytest.raises(LiberoRuntimeError, match="single-use"):
        runtime.reset(seed=101000, condition=ConditionSpec("iid", "iid"))


def test_exact_raw_libero_constructor_is_lazy(monkeypatch) -> None:
    captured = {}

    class FakeLiberoEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.num_steps_wait = kwargs["num_steps_wait"]
            self.camera_name_mapping = kwargs["camera_name_mapping"]

    libero_module = types.ModuleType("lerobot.envs.libero")
    libero_module.LiberoEnv = FakeLiberoEnv
    libero_module._get_suite = lambda name: f"suite:{name}"
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.envs", types.ModuleType("lerobot.envs"))
    monkeypatch.setitem(sys.modules, "lerobot.envs.libero", libero_module)

    task = load_protocol_config(ROOT / "configs").task_order.tasks[0]
    build_raw_libero_env(task, base_init_state_id=7)
    assert captured["task_id"] == 5
    assert captured["episode_index"] == 7
    assert captured["camera_name_mapping"] == CAMERA_NAME_MAPPING
    assert captured["observation_width"] == captured["observation_height"] == 360
    assert captured["num_steps_wait"] == 0
    assert captured["control_mode"] == "relative"
