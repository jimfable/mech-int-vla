"""Exact single-episode LIBERO runtime used by the preregistered study.

This module deliberately bypasses :meth:`lerobot.envs.libero.LiberoEnv.step`.
That wrapper resets immediately after terminal success, which would destroy the
terminal simulator state needed for trace extraction and counterfactuals.
"""

from __future__ import annotations

import contextlib
import copy
import math
import random
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ConditionSpec, PolicyExecutionConfig, TaskSpec, ValidityConfig

CAMERA_NAME_MAPPING = {
    "agentview_image": "camera1",
    "robot0_eye_in_hand_image": "camera2",
}
CAMERA_NAMES = ("agentview_image", "robot0_eye_in_hand_image")
NOOP_ACTION = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
POLICY_STATE_LENGTH = 8


class LiberoRuntimeError(RuntimeError):
    """Raised when the simulator violates a frozen runtime assumption."""


@dataclass(frozen=True)
class WorkspaceBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    table_surface_z: float


@dataclass(frozen=True)
class ValidityResult:
    valid: bool
    reasons: tuple[str, ...]
    finite: bool
    deepest_primary_penetration_m: float
    linear_speed_m_s: float
    angular_speed_rad_s: float
    in_workspace: bool
    initial_success: bool


@dataclass(frozen=True)
class RawTraceFrame:
    control_step: int
    simulator_time: float
    raw_observation: Mapping[str, Any]
    policy_state: np.ndarray
    eef_position: np.ndarray
    eef_quaternion_xyzw: np.ndarray
    primary_object_position: np.ndarray
    primary_object_quaternion_wxyz: np.ndarray
    goal_position: np.ndarray | None
    goal_quaternion_wxyz: np.ndarray | None
    gripper_qpos: np.ndarray
    gripper_qvel: np.ndarray
    primary_gripper_contact: bool
    primary_grasped: bool
    task_success: bool
    task_predicates: Mapping[str, bool]
    phase: str


@dataclass(frozen=True)
class ResetResult:
    observation: Mapping[str, Any]
    frame: RawTraceFrame
    validity: ValidityResult
    settle_actions: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class StepResult:
    observation: Mapping[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    frame: RawTraceFrame


@dataclass(frozen=True)
class SimulatorSnapshot:
    state: np.ndarray
    camera_positions: np.ndarray
    camera_quaternions: np.ndarray


@dataclass(frozen=True)
class CounterfactualObservation:
    available: bool
    reasons: tuple[str, ...]
    raw_observation: Mapping[str, Any] | None
    observation: Mapping[str, Any] | None


def seed_runtime(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (when installed) deterministically."""

    if not 0 <= seed < 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _normalise_name(value: str) -> str:
    return "_".join(value.lower().replace("-", "_").split())


def resolve_primary_object_name(backend: Any, requested: str) -> str:
    """Resolve a semantic config name to the unique LIBERO runtime object key."""

    requested_normalized = _normalise_name(requested)
    matches: list[str] = []
    for name, obj in backend.objects_dict.items():
        candidates = {
            _normalise_name(str(name)),
            _normalise_name(str(getattr(obj, "name", ""))),
            _normalise_name(str(getattr(obj, "category_name", ""))),
        }
        if requested_normalized in candidates or any(
            candidate.startswith(f"{requested_normalized}_") for candidate in candidates
        ):
            matches.append(str(name))
    if len(matches) != 1:
        raise LiberoRuntimeError(
            f"primary object {requested!r} resolved to {matches}; expected exactly one"
        )
    return matches[0]


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    result = np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or norm == 0:
        raise LiberoRuntimeError("cannot normalize invalid quaternion")
    return result / norm


def _world_z_quaternion(yaw_radians: float) -> np.ndarray:
    return np.asarray(
        [math.cos(yaw_radians / 2), 0.0, 0.0, math.sin(yaw_radians / 2)],
        dtype=np.float64,
    )


def _rotation_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_angle_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    # Match the pinned LiberoProcessorStep implementation, including its
    # float32 conversion, w clamp, and lack of hemisphere canonicalization.
    quat = np.asarray(quaternion, dtype=np.float32)
    if not np.isfinite(quat).all():
        raise LiberoRuntimeError("EEF quaternion is invalid")
    w = float(np.clip(quat[3], -1.0, 1.0))
    denominator = math.sqrt(max(0.0, 1.0 - w * w))
    if denominator <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.acos(w)
    return np.asarray(quat[:3] / denominator * angle, dtype=np.float32)


def extract_policy_state(raw_observation: Mapping[str, Any]) -> np.ndarray:
    """Reproduce ``LiberoProcessorStep``'s 8-D state construction exactly."""

    state = np.concatenate(
        [
            np.asarray(raw_observation["robot0_eef_pos"], dtype=np.float32),
            _axis_angle_from_xyzw(raw_observation["robot0_eef_quat"]),
            np.asarray(raw_observation["robot0_gripper_qpos"], dtype=np.float32),
        ]
    )
    if state.shape != (POLICY_STATE_LENGTH,):
        raise LiberoRuntimeError(
            f"LIBERO policy state has shape {state.shape}, expected ({POLICY_STATE_LENGTH},)"
        )
    if not np.isfinite(state).all():
        raise LiberoRuntimeError("LIBERO policy state contains NaN or Inf")
    return state


def build_raw_libero_env(
    task: TaskSpec,
    *,
    base_init_state_id: int,
    episode_length: int = 520,
) -> Any:
    """Construct the exact non-vectorized LeRobot ``LiberoEnv`` for one init."""

    if base_init_state_id < 0:
        raise ValueError("base_init_state_id must be nonnegative")
    try:
        from lerobot.envs.libero import LiberoEnv, _get_suite
    except ImportError as exc:  # pragma: no cover - rollout-host integration
        raise LiberoRuntimeError("LeRobot with the LIBERO extra is required") from exc

    suite = _get_suite(task.suite)
    env = LiberoEnv(
        task_suite=suite,
        task_id=task.task_id,
        task_suite_name=task.suite,
        episode_length=episode_length,
        camera_name=list(CAMERA_NAMES),
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=360,
        observation_height=360,
        init_states=True,
        episode_index=base_init_state_id,
        n_envs=1,
        camera_name_mapping=dict(CAMERA_NAME_MAPPING),
        num_steps_wait=0,
        control_mode="relative",
    )
    if env.num_steps_wait != 0 or env.camera_name_mapping != CAMERA_NAME_MAPPING:
        raise LiberoRuntimeError("raw LIBERO environment construction drifted")
    return env


def _backend(wrapper: Any) -> Any:
    backend = getattr(wrapper, "_env", None)
    if backend is None:
        raise LiberoRuntimeError("LIBERO backend has not been initialized")
    return backend.env


def _offscreen(wrapper: Any) -> Any:
    backend = getattr(wrapper, "_env", None)
    if backend is None:
        raise LiberoRuntimeError("LIBERO backend has not been initialized")
    return backend


def _sim(wrapper: Any) -> Any:
    return _backend(wrapper).sim


def _object(wrapper: Any, object_name: str) -> Any:
    return _backend(wrapper).objects_dict[object_name]


def _free_joint_name(wrapper: Any, object_name: str) -> str:
    joints = tuple(getattr(_object(wrapper, object_name), "joints", ()))
    if not joints:
        raise LiberoRuntimeError(f"primary object {object_name} has no joint")
    joint = str(joints[-1])
    qpos = np.asarray(_sim(wrapper).data.get_joint_qpos(joint))
    if qpos.shape != (7,):
        raise LiberoRuntimeError(
            f"primary object joint {joint} is not free: qpos shape {qpos.shape}"
        )
    return joint


def primary_object_pose(
    wrapper: Any, object_name: str
) -> tuple[np.ndarray, np.ndarray]:
    backend = _backend(wrapper)
    body_id = int(backend.obj_body_id[object_name])
    return (
        np.asarray(backend.sim.data.body_xpos[body_id], dtype=np.float64).copy(),
        np.asarray(backend.sim.data.body_xquat[body_id], dtype=np.float64).copy(),
    )


def primary_object_velocity(
    wrapper: Any, object_name: str
) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.asarray(
        _sim(wrapper).data.get_joint_qvel(_free_joint_name(wrapper, object_name)),
        dtype=np.float64,
    )
    if velocity.shape != (6,):
        raise LiberoRuntimeError(
            f"free-joint velocity has shape {velocity.shape}, expected (6,)"
        )
    return velocity[:3].copy(), velocity[3:].copy()


def apply_object_yaw(wrapper: Any, object_name: str, yaw_degrees: float) -> None:
    """Compose a primary-object rotation about world Z without advancing time."""

    sim = _sim(wrapper)
    joint = _free_joint_name(wrapper, object_name)
    qpos = np.asarray(sim.data.get_joint_qpos(joint), dtype=np.float64).copy()
    qpos[3:7] = _quat_multiply_wxyz(
        _world_z_quaternion(math.radians(float(yaw_degrees))), qpos[3:7]
    )
    sim.data.set_joint_qpos(joint, qpos)
    sim.forward()


def apply_object_translation(
    wrapper: Any, object_name: str, delta_xy_m: tuple[float, float] | list[float]
) -> None:
    sim = _sim(wrapper)
    joint = _free_joint_name(wrapper, object_name)
    qpos = np.asarray(sim.data.get_joint_qpos(joint), dtype=np.float64).copy()
    delta = np.asarray(delta_xy_m, dtype=np.float64)
    if delta.shape != (2,) or not np.isfinite(delta).all():
        raise ValueError("delta_xy_m must contain two finite values")
    qpos[:2] += delta
    sim.data.set_joint_qpos(joint, qpos)
    sim.forward()


def _camera_id(model: Any, camera_name: str) -> int:
    for attribute in ("camera_name2id", "cam_name2id"):
        resolver = getattr(model, attribute, None)
        if callable(resolver):
            return int(resolver(camera_name))
    raise LiberoRuntimeError("MuJoCo model has no camera-name resolver")


def apply_camera_transform(
    wrapper: Any,
    *,
    camera_name: str = "agentview",
    yaw_degrees: float,
    lateral_m: float,
) -> None:
    """Yaw about world Z at the optical center, then translate on local +X."""

    sim = _sim(wrapper)
    camera_id = _camera_id(sim.model, camera_name)
    old_quaternion = np.asarray(sim.model.cam_quat[camera_id], dtype=np.float64).copy()
    local_horizontal_world = _rotation_matrix_wxyz(old_quaternion)[:, 0]
    sim.model.cam_pos[camera_id] = (
        np.asarray(sim.model.cam_pos[camera_id], dtype=np.float64)
        + float(lateral_m) * local_horizontal_world
    )
    sim.model.cam_quat[camera_id] = _quat_multiply_wxyz(
        _world_z_quaternion(math.radians(float(yaw_degrees))), old_quaternion
    )
    sim.forward()


def capture_simulator_snapshot(wrapper: Any) -> SimulatorSnapshot:
    sim = _sim(wrapper)
    return SimulatorSnapshot(
        state=np.asarray(_offscreen(wrapper).get_sim_state(), dtype=np.float64).copy(),
        camera_positions=np.asarray(sim.model.cam_pos, dtype=np.float64).copy(),
        camera_quaternions=np.asarray(sim.model.cam_quat, dtype=np.float64).copy(),
    )


def restore_simulator_snapshot(wrapper: Any, snapshot: SimulatorSnapshot) -> None:
    offscreen = _offscreen(wrapper)
    sim = _sim(wrapper)
    offscreen.set_state(snapshot.state.copy())
    sim.model.cam_pos[...] = snapshot.camera_positions
    sim.model.cam_quat[...] = snapshot.camera_quaternions
    sim.forward()
    restored = np.asarray(offscreen.get_sim_state(), dtype=np.float64)
    if not np.array_equal(restored, snapshot.state, equal_nan=True):
        raise LiberoRuntimeError("simulator state restoration was not exact")


@contextlib.contextmanager
def temporary_condition(
    wrapper: Any,
    condition: ConditionSpec,
    *,
    primary_object_name: str,
) -> Iterator[None]:
    """Apply a render/counterfactual edit and restore model plus dynamic state."""

    snapshot = capture_simulator_snapshot(wrapper)
    try:
        apply_condition(wrapper, condition, primary_object_name=primary_object_name)
        yield
    finally:
        restore_simulator_snapshot(wrapper, snapshot)


def apply_condition(
    wrapper: Any, condition: ConditionSpec, *, primary_object_name: str
) -> None:
    parameters = condition.parameters
    if condition.family == "iid":
        return
    if condition.family in {"object_yaw", "object_pose"}:
        angle = parameters.get("value", parameters.get("yaw"))
        if angle is None:
            raise LiberoRuntimeError(f"{condition.name} has no yaw value")
        apply_object_yaw(wrapper, primary_object_name, float(angle))
        return
    if condition.family == "object_planar_translation":
        if "delta_xy_m" not in parameters:
            raise LiberoRuntimeError(f"{condition.name} has no realized delta_xy_m")
        apply_object_translation(wrapper, primary_object_name, parameters["delta_xy_m"])
        return
    if condition.family in {"agentview_camera_extrinsic", "camera_render"}:
        apply_camera_transform(
            wrapper,
            yaw_degrees=float(parameters["yaw"]),
            lateral_m=float(parameters.get("lateral_m", 0.0)),
        )
        return
    if condition.family == "photometric":
        raise LiberoRuntimeError(
            "photometric transforms operate on pixels, not simulator state"
        )
    raise LiberoRuntimeError(f"unsupported condition family: {condition.family}")


def workspace_bounds(wrapper: Any) -> WorkspaceBounds:
    backend = _backend(wrapper)
    center = np.asarray(backend.workspace_offset, dtype=np.float64)
    size = None
    for name in (
        "kitchen_table_full_size",
        "study_table_full_size",
        "living_room_table_full_size",
        "coffee_table_full_size",
        "table_full_size",
    ):
        candidate = getattr(backend, name, None)
        if candidate is not None:
            size = np.asarray(candidate, dtype=np.float64)
            break
    if center.shape != (3,) or size is None or size.shape != (3,):
        raise LiberoRuntimeError("could not derive tabletop workspace bounds")
    return WorkspaceBounds(
        x_min=float(center[0] - size[0] / 2),
        x_max=float(center[0] + size[0] / 2),
        y_min=float(center[1] - size[1] / 2),
        y_max=float(center[1] + size[1] / 2),
        table_surface_z=float(center[2]),
    )


def _descendant_body_ids(model: Any, root_body_id: int) -> set[int]:
    parents = np.asarray(model.body_parentid, dtype=np.int64)
    descendants = {root_body_id}
    changed = True
    while changed:
        changed = False
        for body_id, parent in enumerate(parents):
            if body_id not in descendants and int(parent) in descendants:
                descendants.add(body_id)
                changed = True
    return descendants


def _primary_geom_ids(wrapper: Any, object_name: str) -> set[int]:
    backend = _backend(wrapper)
    root = int(backend.obj_body_id[object_name])
    bodies = _descendant_body_ids(backend.sim.model, root)
    return {
        geom_id
        for geom_id, body_id in enumerate(np.asarray(backend.sim.model.geom_bodyid))
        if int(body_id) in bodies
    }


def deepest_primary_penetration(wrapper: Any, object_name: str) -> float:
    sim = _sim(wrapper)
    primary_geoms = _primary_geom_ids(wrapper, object_name)
    deepest = 0.0
    for contact in sim.data.contact[: int(sim.data.ncon)]:
        if int(contact.geom1) in primary_geoms or int(contact.geom2) in primary_geoms:
            deepest = max(deepest, max(0.0, -float(contact.dist)))
    return deepest


def _named_geom_ids(model: Any, names: Any) -> set[int]:
    result: set[int] = set()
    resolver = getattr(model, "geom_name2id", None)
    if not callable(resolver):
        return result
    for name in names or ():
        result.add(int(resolver(str(name))))
    return result


def primary_gripper_contact(wrapper: Any, object_name: str) -> bool:
    backend = _backend(wrapper)
    obj = backend.objects_dict[object_name]
    robot = backend.robots[0]
    object_geoms = _named_geom_ids(backend.sim.model, getattr(obj, "contact_geoms", ()))
    gripper_geoms = _named_geom_ids(
        backend.sim.model, getattr(robot.gripper, "contact_geoms", ())
    )
    for contact in backend.sim.data.contact[: int(backend.sim.data.ncon)]:
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair & object_geoms and pair & gripper_geoms:
            return True
    return False


def backend_grasp_predicate(wrapper: Any, object_name: str) -> bool:
    """Evaluate robosuite's two-finger grasp predicate for the primary object."""

    backend = _backend(wrapper)
    obj = backend.objects_dict[object_name]
    robot = backend.robots[0]
    checker = getattr(backend, "_check_grasp", None)
    if callable(checker):
        try:
            return bool(checker(gripper=robot.gripper, object_geoms=obj.contact_geoms))
        except TypeError:
            try:
                return bool(checker(robot.gripper, obj.contact_geoms))
            except TypeError:
                pass
    raise LiberoRuntimeError("LIBERO backend exposes no compatible grasp predicate")


def task_predicates(wrapper: Any) -> dict[str, bool]:
    backend = _backend(wrapper)
    predicates: dict[str, bool] = {}
    for index, state in enumerate(backend.parsed_problem.get("goal_state", ())):
        key = f"{index}:" + ":".join(str(value) for value in state)
        predicates[key] = bool(backend._eval_predicate(state))
    return predicates


def primary_placement_predicate(wrapper: Any, object_name: str) -> bool:
    backend = _backend(wrapper)
    values = []
    for state in backend.parsed_problem.get("goal_state", ()):
        normalized = tuple(str(value) for value in state)
        if (
            normalized
            and normalized[0].lower() in {"in", "on", "stack"}
            and object_name in normalized[1:]
        ):
            values.append(bool(backend._eval_predicate(state)))
    return bool(values) and all(values)


def goal_pose(
    wrapper: Any, object_name: str
) -> tuple[np.ndarray | None, np.ndarray | None]:
    backend = _backend(wrapper)
    for state in backend.parsed_problem.get("goal_state", ()):
        normalized = tuple(str(value) for value in state)
        if not normalized or normalized[0].lower() not in {"in", "on", "stack"}:
            continue
        if object_name not in normalized[1:]:
            continue
        other_names = [name for name in normalized[1:] if name != object_name]
        if not other_names:
            continue
        state_object = backend.object_states_dict.get(other_names[0])
        if state_object is None:
            continue
        pose = state_object.get_geom_state()
        position = np.asarray(pose["pos"], dtype=np.float64).copy()
        quaternion = np.asarray(pose["quat"], dtype=np.float64).copy()
        if getattr(state_object, "object_state_type", None) == "site":
            # SiteObjectState uses robosuite's xyzw transform utility, whereas
            # MuJoCo body_xquat above is wxyz.
            quaternion = quaternion[[3, 0, 1, 2]]
        return position, quaternion
    return None, None


def deterministic_phase(
    *,
    placed: bool,
    grasped: bool,
    object_position: np.ndarray,
    post_settle_start_position: np.ndarray,
    displacement_threshold_m: float,
) -> str:
    """Apply the frozen placed/transport/grasped/pregrasp precedence."""

    if placed:
        return "placed"
    moved = float(
        np.linalg.norm(
            np.asarray(object_position, dtype=np.float64)
            - np.asarray(post_settle_start_position, dtype=np.float64)
        )
    )
    if grasped and moved >= displacement_threshold_m:
        return "transport"
    if grasped:
        return "grasped"
    return "pregrasp"


def check_validity(
    wrapper: Any,
    object_name: str,
    config: ValidityConfig,
) -> ValidityResult:
    sim_state = np.asarray(_offscreen(wrapper).get_sim_state(), dtype=np.float64)
    object_position, object_quaternion = primary_object_pose(wrapper, object_name)
    linear_velocity, angular_velocity = primary_object_velocity(wrapper, object_name)
    camera_positions = np.asarray(_sim(wrapper).model.cam_pos, dtype=np.float64)
    camera_quaternions = np.asarray(_sim(wrapper).model.cam_quat, dtype=np.float64)
    contact_distances = np.asarray(
        [
            float(contact.dist)
            for contact in _sim(wrapper).data.contact[: int(_sim(wrapper).data.ncon)]
        ],
        dtype=np.float64,
    )
    finite = bool(
        np.isfinite(sim_state).all()
        and np.isfinite(object_position).all()
        and np.isfinite(object_quaternion).all()
        and np.isfinite(linear_velocity).all()
        and np.isfinite(angular_velocity).all()
        and np.isfinite(camera_positions).all()
        and np.isfinite(camera_quaternions).all()
        and np.isfinite(contact_distances).all()
    )
    penetration = deepest_primary_penetration(wrapper, object_name)
    linear_speed = float(np.linalg.norm(linear_velocity))
    angular_speed = float(np.linalg.norm(angular_velocity))
    bounds = workspace_bounds(wrapper)
    in_workspace = bool(
        bounds.x_min <= object_position[0] <= bounds.x_max
        and bounds.y_min <= object_position[1] <= bounds.y_max
        and bounds.table_surface_z - config.min_height_below_table_m
        <= object_position[2]
        <= bounds.table_surface_z + config.max_height_above_table_m
    )
    initial_success = bool(_offscreen(wrapper).check_success())
    reasons: list[str] = []
    if config.reject_nan_or_inf and not finite:
        reasons.append("nonfinite_simulator_or_object_value")
    if penetration > config.max_penetration_depth_m:
        reasons.append("primary_object_penetration")
    if linear_speed > config.max_linear_speed_m_s:
        reasons.append("primary_object_linear_instability")
    if angular_speed > config.max_angular_speed_rad_s:
        reasons.append("primary_object_angular_instability")
    if config.reject_workspace_exit and not in_workspace:
        reasons.append("primary_object_outside_tabletop_workspace")
    if config.reject_initial_success and initial_success:
        reasons.append("initial_task_success")
    return ValidityResult(
        valid=not reasons,
        reasons=tuple(reasons),
        finite=finite,
        deepest_primary_penetration_m=penetration,
        linear_speed_m_s=linear_speed,
        angular_speed_rad_s=angular_speed,
        in_workspace=in_workspace,
        initial_success=initial_success,
    )


def counterfactual_validity_reasons(
    wrapper: Any,
    object_name: str,
    config: ValidityConfig,
    *,
    check_object_edit: bool = True,
) -> tuple[str, ...]:
    """Check the frozen non-advancing counterfactual availability constraints."""

    sim = _sim(wrapper)
    state = np.asarray(_offscreen(wrapper).get_sim_state(), dtype=np.float64)
    object_position, object_quaternion = primary_object_pose(wrapper, object_name)
    camera_positions = np.asarray(sim.model.cam_pos, dtype=np.float64)
    camera_quaternions = np.asarray(sim.model.cam_quat, dtype=np.float64)
    contact_distances = np.asarray(
        [float(contact.dist) for contact in sim.data.contact[: int(sim.data.ncon)]],
        dtype=np.float64,
    )
    finite = bool(
        np.isfinite(state).all()
        and np.isfinite(object_position).all()
        and np.isfinite(object_quaternion).all()
        and np.isfinite(camera_positions).all()
        and np.isfinite(camera_quaternions).all()
        and np.isfinite(contact_distances).all()
    )
    bounds = workspace_bounds(wrapper)
    in_workspace = bool(
        bounds.x_min <= object_position[0] <= bounds.x_max
        and bounds.y_min <= object_position[1] <= bounds.y_max
        and bounds.table_surface_z - config.min_height_below_table_m
        <= object_position[2]
        <= bounds.table_surface_z + config.max_height_above_table_m
    )
    reasons: list[str] = []
    if not finite:
        reasons.append("nonfinite_counterfactual_state")
    if check_object_edit:
        if (
            deepest_primary_penetration(wrapper, object_name)
            > config.max_penetration_depth_m
        ):
            reasons.append("counterfactual_primary_object_penetration")
        if not in_workspace:
            reasons.append("counterfactual_primary_object_outside_workspace")
    return tuple(reasons)


def brightness_transform_raw_observation(
    raw_observation: Mapping[str, Any], multiplier: float
) -> dict[str, Any]:
    """Apply the frozen uint8 brightness transform to both policy cameras."""

    value = float(multiplier)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("brightness multiplier must be finite and positive")
    transformed = _copy_observation(raw_observation)
    for key in CAMERA_NAME_MAPPING:
        if key not in transformed:
            raise LiberoRuntimeError(f"raw observation is missing camera {key!r}")
        image = np.asarray(transformed[key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise LiberoRuntimeError(f"camera {key!r} must be an HxWx3 uint8 array")
        transformed[key] = np.rint(
            np.clip(image.astype(np.float32) * value, 0.0, 255.0)
        ).astype(np.uint8)
    return transformed


def _copy_observation(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _copy_observation(item) for key, item in value.items()}
    return copy.deepcopy(value)


_OBSERVABLE_SAMPLING_FIELDS = (
    "_time_since_last_sample",
    "_current_delay",
    "_current_observed_value",
    "_sampled",
)


@contextlib.contextmanager
def _preserved_observable_sampling(backend: Any) -> Iterator[None]:
    """Restore robosuite observable sampling state around a non-advancing read.

    ``_get_observations(force_update=True)`` calls ``_update_observables(force=True)``,
    which advances every observable's ``_time_since_last_sample`` by one model
    timestep *without* advancing the simulation, and refills ``_obs_cache``.  A
    replay performs an arbitrary number of such non-advancing reads per scored
    state, so leaving the timers advanced shifts the sampling phase of the next
    real ``step`` and makes the replayed trajectory drift away from the recorded
    rollout.  Snapshotting and restoring the sampling state keeps a forced read
    observationally pure: it reports the current simulator state and leaves no
    trace behind.
    """

    observables = getattr(backend, "_observables", None)
    if observables is None:
        raise LiberoRuntimeError("LIBERO backend exposes no observables to preserve")
    saved = {
        name: tuple(
            copy.deepcopy(getattr(observable, field))
            for field in _OBSERVABLE_SAMPLING_FIELDS
        )
        for name, observable in observables.items()
    }
    saved_cache = _copy_observation(getattr(backend, "_obs_cache", {}))
    try:
        yield
    finally:
        for name, observable in observables.items():
            for field, value in zip(_OBSERVABLE_SAMPLING_FIELDS, saved[name]):
                setattr(observable, field, value)
        cache = getattr(backend, "_obs_cache", None)
        if cache is not None:
            cache.clear()
            cache.update(saved_cache)


class RawLiberoEpisode:
    """Own one raw LeRobot LIBERO environment and preserve terminal state."""

    def __init__(
        self,
        wrapper: Any,
        task: TaskSpec,
        execution: PolicyExecutionConfig,
        validity: ValidityConfig,
    ) -> None:
        if execution.control_mode != "relative":
            raise LiberoRuntimeError(
                "only the frozen relative control mode is supported"
            )
        if execution.n_action_steps != 1:
            raise LiberoRuntimeError("runtime requires n_action_steps=1")
        if execution.reset_noop_steps != validity.settle_steps_after_edit:
            raise LiberoRuntimeError("settling-step configs disagree")
        self.wrapper = wrapper
        self.task = task
        self.execution = execution
        self.validity_config = validity
        self.primary_object_name: str | None = None
        self.post_settle_start_position: np.ndarray | None = None
        self.control_step = 0
        self.terminal = False
        self._has_reset = False

    @classmethod
    def create(
        cls,
        task: TaskSpec,
        *,
        base_init_state_id: int,
        execution: PolicyExecutionConfig,
        validity: ValidityConfig,
    ) -> RawLiberoEpisode:
        return cls(
            build_raw_libero_env(
                task,
                base_init_state_id=base_init_state_id,
                episode_length=execution.max_steps,
            ),
            task,
            execution,
            validity,
        )

    def _raw_observation(self, *, force_update: bool = False) -> Mapping[str, Any]:
        """Return the backend observation, optionally forcing a fresh sample.

        robosuite serves cached observable values that are refreshed only inside
        ``step``/``reset``.  Any read taken after a direct simulator edit must
        pass ``force_update=True``, otherwise it silently reports the pre-edit
        state and the counterfactual becomes a no-op.  The forced read is wrapped
        so that it leaves the observable sampling state untouched.
        """

        backend = _backend(self.wrapper)
        if not force_update:
            return backend._get_observations()
        with _preserved_observable_sampling(backend):
            return _copy_observation(backend._get_observations(force_update=True))

    def _format_observation(self, raw: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.wrapper._format_raw_obs(raw)

    def format_observation(self, raw: Mapping[str, Any]) -> Mapping[str, Any]:
        """Public defensive formatter used by deterministic replay scoring."""

        return _copy_observation(self._format_observation(_copy_observation(raw)))

    def current_raw_trace(self) -> RawTraceFrame:
        """Capture the current non-advancing raw observation and trace state.

        Replay scoring uses this after a temporary simulator edit.  The returned
        frame owns defensive copies, so callers cannot mutate the live simulator
        or wrapper observation buffers.
        """

        if self.primary_object_name is None or not self._has_reset:
            raise LiberoRuntimeError("episode has not been reset")
        return self._trace_frame(
            _copy_observation(self._raw_observation(force_update=True))
        )

    def photometric_observation(
        self, raw: Mapping[str, Any], *, multiplier: float
    ) -> Mapping[str, Any]:
        """Return a formatted two-camera brightness counterfactual."""

        return self.format_observation(
            brightness_transform_raw_observation(raw, multiplier)
        )

    @contextlib.contextmanager
    def counterfactual_observation(
        self, condition: ConditionSpec
    ) -> Iterator[CounterfactualObservation]:
        """Yield one non-advancing camera/object observation and restore exactly."""

        if self.primary_object_name is None or not self._has_reset:
            raise LiberoRuntimeError("episode has not been reset")
        if self.terminal:
            raise LiberoRuntimeError("cannot score a terminal simulator state")
        if condition.family not in {"camera_render", "object_pose"}:
            raise LiberoRuntimeError(
                "counterfactual observation requires camera_render or object_pose"
            )
        with temporary_condition(
            self.wrapper,
            condition,
            primary_object_name=self.primary_object_name,
        ):
            reasons = counterfactual_validity_reasons(
                self.wrapper,
                self.primary_object_name,
                self.validity_config,
                check_object_edit=condition.family == "object_pose",
            )
            if reasons:
                yield CounterfactualObservation(False, reasons, None, None)
            else:
                raw = _copy_observation(self._raw_observation(force_update=True))
                yield CounterfactualObservation(
                    True,
                    (),
                    raw,
                    self.format_observation(raw),
                )

    def _trace_frame(self, raw: Mapping[str, Any]) -> RawTraceFrame:
        if self.primary_object_name is None or self.post_settle_start_position is None:
            raise LiberoRuntimeError("episode has not been reset")
        object_position, object_quaternion = primary_object_pose(
            self.wrapper, self.primary_object_name
        )
        contact = primary_gripper_contact(self.wrapper, self.primary_object_name)
        grasped = backend_grasp_predicate(self.wrapper, self.primary_object_name)
        placed = primary_placement_predicate(self.wrapper, self.primary_object_name)
        target_position, target_quaternion = goal_pose(
            self.wrapper, self.primary_object_name
        )
        phase = deterministic_phase(
            placed=placed,
            grasped=grasped,
            object_position=object_position,
            post_settle_start_position=self.post_settle_start_position,
            displacement_threshold_m=self.validity_config.phase_min_object_displacement_m,
        )
        sim = _sim(self.wrapper)
        return RawTraceFrame(
            control_step=self.control_step,
            simulator_time=float(sim.data.time),
            raw_observation=_copy_observation(raw),
            policy_state=extract_policy_state(raw),
            eef_position=np.asarray(raw["robot0_eef_pos"], dtype=np.float64).copy(),
            eef_quaternion_xyzw=np.asarray(
                raw["robot0_eef_quat"], dtype=np.float64
            ).copy(),
            primary_object_position=object_position,
            primary_object_quaternion_wxyz=object_quaternion,
            goal_position=target_position,
            goal_quaternion_wxyz=target_quaternion,
            gripper_qpos=np.asarray(
                raw["robot0_gripper_qpos"], dtype=np.float64
            ).copy(),
            gripper_qvel=np.asarray(
                raw["robot0_gripper_qvel"], dtype=np.float64
            ).copy(),
            primary_gripper_contact=contact,
            primary_grasped=grasped,
            task_success=bool(_offscreen(self.wrapper).check_success()),
            task_predicates=task_predicates(self.wrapper),
            phase=phase,
        )

    def reset(self, *, seed: int, condition: ConditionSpec) -> ResetResult:
        if self._has_reset:
            raise LiberoRuntimeError(
                "RawLiberoEpisode is single-use; construct one runtime per condition"
            )
        # LeRobot advances its internal init-state index on every reset. Mark this
        # instance consumed before touching the wrapper so even a failed reset
        # cannot silently shift a later paired condition to the next init state.
        self._has_reset = True
        seed_runtime(seed)
        self.wrapper.reset(seed=seed)
        backend = _backend(self.wrapper)
        self.primary_object_name = resolve_primary_object_name(
            backend, self.task.primary_object
        )
        apply_condition(
            self.wrapper,
            condition,
            primary_object_name=self.primary_object_name,
        )
        settle_actions: list[tuple[float, ...]] = []
        raw: Mapping[str, Any] | None = None
        # Direct backend steps avoid wrapper autoreset and give every condition
        # exactly the same post-edit timing.
        for _ in range(self.validity_config.settle_steps_after_edit):
            raw, _, _, _ = _offscreen(self.wrapper).step(NOOP_ACTION.copy())
            settle_actions.append(tuple(float(value) for value in NOOP_ACTION))
        if raw is None:
            # Unreachable while the protocol mandates settle steps, but a forced
            # read is the only safe default here: this point follows a direct
            # condition edit, so a cached observation would report the pre-edit
            # state.
            raw = self._raw_observation(force_update=True)
        self.post_settle_start_position = primary_object_pose(
            self.wrapper, self.primary_object_name
        )[0]
        self.control_step = 0
        validity = check_validity(
            self.wrapper, self.primary_object_name, self.validity_config
        )
        self.terminal = validity.initial_success
        frame = self._trace_frame(raw)
        return ResetResult(
            observation=self._format_observation(raw),
            frame=frame,
            validity=validity,
            settle_actions=tuple(settle_actions),
        )

    def step(self, action: np.ndarray) -> StepResult:
        if self.terminal:
            raise LiberoRuntimeError(
                "episode is terminal; refusing to advance or reset the preserved state"
            )
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("action must be a finite 7-vector")
        raw, reward, backend_done, info = _offscreen(self.wrapper).step(action)
        self.control_step += 1
        success = bool(_offscreen(self.wrapper).check_success())
        terminated = bool(backend_done or success)
        truncated = self.control_step >= self.execution.max_steps and not terminated
        self.terminal = terminated or truncated
        result_info = dict(info)
        result_info.update(
            {
                "task": getattr(self.wrapper, "task", self.task.language),
                "task_id": self.task.task_id,
                "done": bool(backend_done),
                "is_success": success,
            }
        )
        frame = self._trace_frame(raw)
        return StepResult(
            observation=self._format_observation(raw),
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=result_info,
            frame=frame,
        )

    def close(self) -> None:
        self.wrapper.close()
