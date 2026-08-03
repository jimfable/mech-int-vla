"""Typed loading and validation for the preregistered YAML configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


class ProtocolConfigError(ValueError):
    """Raised when protocol configuration is missing, malformed, or unsupported."""


class SplitName(str, Enum):
    DISCOVERY = "discovery"
    CALIBRATION = "calibration"
    LOCKED_TEST = "locked_test"


@dataclass(frozen=True)
class InitStateRange:
    start: int
    stop_inclusive: int

    def ids(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.stop_inclusive + 1))


@dataclass(frozen=True)
class SplitSpec:
    init_state_ids: InitStateRange
    seed_base: int


@dataclass(frozen=True)
class PolicyExecutionConfig:
    control_mode: str
    max_steps: int
    reset_noop_steps: int
    n_action_steps: int


@dataclass(frozen=True)
class ScoringConfig:
    full_score_stride_steps: int
    primary_steps: tuple[int, ...]


@dataclass(frozen=True)
class BootstrapConfig:
    cluster: str
    replicates: int
    interval: float
    seed: int


@dataclass(frozen=True)
class LockedTestGuardConfig:
    required_file: Path
    required_tag: str
    require_clean_worktree: bool


@dataclass(frozen=True)
class CalibrationGuardConfig:
    required_file: Path
    required_tag: str
    require_clean_worktree: bool


@dataclass(frozen=True)
class PredictorCandidateConfig:
    family: str
    hyperparameter_grid: Mapping[str, tuple[Any, ...]]
    max_iter_min: int | None = None
    max_iter_max: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "hyperparameter_grid", _freeze_mapping(self.hyperparameter_grid)
        )


@dataclass(frozen=True)
class CalibrationSelectionConfig:
    representation_candidates: tuple[str, ...]
    ridge_alpha_candidates: tuple[float, ...]
    patch_strength_candidates: tuple[float, ...]
    predictor_candidates: Mapping[str, PredictorCandidateConfig]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "predictor_candidates", _freeze_mapping(self.predictor_candidates)
        )


@dataclass(frozen=True)
class SplitProtocolConfig:
    version: int
    group_key: tuple[str, ...]
    splits: Mapping[SplitName, SplitSpec]
    seed_formula: str
    policy_execution: PolicyExecutionConfig
    scoring: ScoringConfig
    bootstrap: BootstrapConfig
    calibration_guard: CalibrationGuardConfig
    locked_test_guard: LockedTestGuardConfig
    calibration_selection: CalibrationSelectionConfig

    def __post_init__(self) -> None:
        object.__setattr__(self, "splits", _freeze_mapping(self.splits))

    def seed_for(
        self,
        split: SplitName,
        *,
        task_rank: int,
        base_init_state_id: int,
        condition_index: int,
    ) -> int:
        spec = self.splits[split]
        if base_init_state_id not in spec.init_state_ids.ids():
            raise ProtocolConfigError(
                f"init state {base_init_state_id} is outside split {split.value}"
            )
        if task_rank < 1:
            raise ProtocolConfigError("task_rank must be positive")
        if not 0 <= condition_index <= 9:
            raise ProtocolConfigError("condition_index must be in [0, 9]")
        return (
            spec.seed_base
            + task_rank * 1000
            + base_init_state_id * 10
            + condition_index
        )


@dataclass(frozen=True)
class TaskSpec:
    rank: int
    suite: str
    task_id: int
    language: str
    primary_object: str
    planar_symmetry_order: int


@dataclass(frozen=True)
class GateConfig:
    baseline_iid_episodes: int
    baseline_min_successes: int
    perturbed_episodes: int
    min_valid_fraction: float
    min_failure_rate: float
    max_failure_rate: float


@dataclass(frozen=True)
class TaskOrderConfig:
    version: int
    written: str
    selection_rule: str
    tasks: tuple[TaskSpec, ...]
    gates: GateConfig


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    family: str
    index: int | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))


@dataclass(frozen=True)
class ValidityConfig:
    settle_steps_after_edit: int
    max_invalid_fraction: float
    reject_nan_or_inf: bool
    reject_workspace_exit: bool
    reject_initial_success: bool
    max_penetration_depth_m: float
    max_linear_speed_m_s: float
    max_angular_speed_rad_s: float
    min_height_below_table_m: float
    max_height_above_table_m: float
    phase_min_object_displacement_m: float


@dataclass(frozen=True)
class PerturbationsConfig:
    version: int
    angles_in_degrees: bool
    reality_gate_cells: tuple[ConditionSpec, ...]
    reality_gate_assignment: str
    calibration_rollouts: tuple[ConditionSpec, ...]
    locked_test_rollouts: tuple[ConditionSpec, ...]
    counterfactual_score_grid: tuple[ConditionSpec, ...]
    validity: ValidityConfig


@dataclass(frozen=True)
class ProtocolConfig:
    split: SplitProtocolConfig
    task_order: TaskOrderConfig
    perturbations: PerturbationsConfig


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolConfigError(f"{where} must be a mapping")
    return value


def _sequence(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProtocolConfigError(f"{where} must be a sequence")
    return value


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(nested) for nested in value)
    return value


def _freeze_mapping(value: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(
        {key: _freeze_value(nested) for key, nested in value.items()}
    )


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except OSError as exc:
        raise ProtocolConfigError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ProtocolConfigError(f"invalid YAML in {path}: {exc}") from exc
    return _mapping(loaded, str(path))


def _condition(raw: Any, where: str, *, index_required: bool) -> ConditionSpec:
    data = dict(_mapping(raw, where))
    try:
        name = str(data.pop("name"))
        family = str(data.pop("family"))
    except KeyError as exc:
        raise ProtocolConfigError(f"{where} is missing {exc.args[0]}") from exc
    index_raw = data.pop("index", None)
    if index_required and index_raw is None:
        raise ProtocolConfigError(f"{where} is missing index")
    index = None if index_raw is None else int(index_raw)
    return ConditionSpec(
        name=name,
        family=family,
        index=index,
        parameters=_freeze_mapping(data),
    )


def load_split_protocol(path: Path) -> SplitProtocolConfig:
    data = _load_yaml(path)
    if int(data.get("version", -1)) != 1:
        raise ProtocolConfigError("only split protocol version 1 is supported")
    formula = str(data.get("seed_formula", ""))
    expected_formula = (
        "seed_base + task_rank*1000 + base_init_state_id*10 + condition_index"
    )
    if formula != expected_formula:
        raise ProtocolConfigError(f"unsupported seed_formula: {formula!r}")

    split_data = _mapping(data.get("splits"), "splits")
    splits: dict[SplitName, SplitSpec] = {}
    for split in SplitName:
        raw = _mapping(split_data.get(split.value), f"splits.{split.value}")
        raw_range = _mapping(
            raw.get("init_state_ids"), f"splits.{split.value}.init_state_ids"
        )
        state_range = InitStateRange(
            start=int(raw_range["start"]),
            stop_inclusive=int(raw_range["stop_inclusive"]),
        )
        if state_range.start > state_range.stop_inclusive:
            raise ProtocolConfigError(f"empty init-state range for {split.value}")
        splits[split] = SplitSpec(state_range, int(raw["seed_base"]))

    id_sets = [set(spec.init_state_ids.ids()) for spec in splits.values()]
    if any(id_sets[i] & id_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ProtocolConfigError("split init-state ranges overlap")
    seed_ranges = [
        {
            spec.seed_base + rank * 1000 + init_id * 10 + condition
            for rank in range(1, 4)
            for init_id in spec.init_state_ids.ids()
            for condition in range(10)
        }
        for spec in splits.values()
    ]
    if any(seed_ranges[i] & seed_ranges[j] for i in range(3) for j in range(i + 1, 3)):
        raise ProtocolConfigError("configured reset-seed ranges overlap")

    policy = _mapping(data.get("policy_execution"), "policy_execution")
    scoring = _mapping(data.get("scoring"), "scoring")
    bootstrap = _mapping(data.get("bootstrap"), "bootstrap")
    calibration_guard = _mapping(data.get("calibration_guard"), "calibration_guard")
    guard = _mapping(data.get("locked_test_guard"), "locked_test_guard")
    selection = _mapping(data.get("calibration_selection"), "calibration_selection")
    raw_predictors = _mapping(
        selection.get("predictor_candidates"),
        "calibration_selection.predictor_candidates",
    )
    predictors: dict[str, PredictorCandidateConfig] = {}
    for family, raw_candidate in raw_predictors.items():
        candidate = dict(
            _mapping(
                raw_candidate,
                f"calibration_selection.predictor_candidates.{family}",
            )
        )
        minimum = candidate.pop("max_iter_min", None)
        maximum = candidate.pop("max_iter_max", None)
        predictors[str(family)] = PredictorCandidateConfig(
            family=str(family),
            hyperparameter_grid=MappingProxyType(
                {
                    str(name): tuple(
                        _sequence(
                            values,
                            f"calibration_selection.predictor_candidates.{family}.{name}",
                        )
                    )
                    for name, values in candidate.items()
                }
            ),
            max_iter_min=None if minimum is None else int(minimum),
            max_iter_max=None if maximum is None else int(maximum),
        )
    representations = tuple(
        str(value)
        for value in _sequence(
            selection.get("representation_candidates"),
            "calibration_selection.representation_candidates",
        )
    )
    if len(representations) != 5 or len(set(representations)) != 5:
        raise ProtocolConfigError(
            "calibration_selection must define five unique representation candidates"
        )
    ridge_alphas = tuple(
        float(value)
        for value in _sequence(
            selection.get("ridge_alpha_candidates"),
            "calibration_selection.ridge_alpha_candidates",
        )
    )
    if set(ridge_alphas) != {0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0}:
        raise ProtocolConfigError(
            "ridge_alpha_candidates must match the preregistered search grid"
        )
    patch_strengths = tuple(
        float(value)
        for value in _sequence(
            selection.get("patch_strength_candidates"),
            "calibration_selection.patch_strength_candidates",
        )
    )
    if set(patch_strengths) != {0.25, 0.5, 1.0}:
        raise ProtocolConfigError(
            "patch_strength_candidates must match the preregistered search grid"
        )
    if set(predictors) != {
        "logistic_regression",
        "histogram_gradient_boosting",
    }:
        raise ProtocolConfigError(
            "predictor_candidates must match the two preregistered families"
        )
    logistic = predictors["logistic_regression"]
    if set(logistic.hyperparameter_grid) != {"C"} or {
        float(value) for value in logistic.hyperparameter_grid["C"]
    } != {0.01, 0.1, 1.0, 10.0, 100.0}:
        raise ProtocolConfigError(
            "logistic_regression C values must match the preregistered search grid"
        )
    if logistic.max_iter_min is not None or logistic.max_iter_max is not None:
        raise ProtocolConfigError("logistic_regression must not define max_iter bounds")
    boosting = predictors["histogram_gradient_boosting"]
    expected_boosting = {
        "learning_rate": {0.03, 0.1},
        "max_leaf_nodes": {7.0, 15.0},
        "min_samples_leaf": {10.0, 20.0},
        "l2_regularization": {0.0, 1.0},
    }
    if set(boosting.hyperparameter_grid) != set(expected_boosting) or any(
        {float(value) for value in boosting.hyperparameter_grid[name]} != allowed
        for name, allowed in expected_boosting.items()
    ):
        raise ProtocolConfigError(
            "histogram_gradient_boosting values must match the preregistered search grid"
        )
    if boosting.max_iter_min != 1 or boosting.max_iter_max != 200:
        raise ProtocolConfigError(
            "histogram_gradient_boosting max_iter must be bounded to [1, 200]"
        )
    return SplitProtocolConfig(
        version=1,
        group_key=tuple(
            str(value) for value in _sequence(data.get("group_key"), "group_key")
        ),
        splits=MappingProxyType(splits),
        seed_formula=formula,
        policy_execution=PolicyExecutionConfig(
            control_mode=str(policy["control_mode"]),
            max_steps=int(policy["max_steps"]),
            reset_noop_steps=int(policy["reset_noop_steps"]),
            n_action_steps=int(policy["n_action_steps"]),
        ),
        scoring=ScoringConfig(
            full_score_stride_steps=int(scoring["full_score_stride_steps"]),
            primary_steps=tuple(
                int(step)
                for step in _sequence(scoring["primary_steps"], "scoring.primary_steps")
            ),
        ),
        bootstrap=BootstrapConfig(
            cluster=str(bootstrap["cluster"]),
            replicates=int(bootstrap["replicates"]),
            interval=float(bootstrap["interval"]),
            seed=int(bootstrap["seed"]),
        ),
        calibration_guard=CalibrationGuardConfig(
            required_file=Path(str(calibration_guard["required_file"])),
            required_tag=str(calibration_guard["required_tag"]),
            require_clean_worktree=bool(calibration_guard["require_clean_worktree"]),
        ),
        locked_test_guard=LockedTestGuardConfig(
            required_file=Path(str(guard["required_file"])),
            required_tag=str(guard["required_tag"]),
            require_clean_worktree=bool(guard["require_clean_worktree"]),
        ),
        calibration_selection=CalibrationSelectionConfig(
            representation_candidates=representations,
            ridge_alpha_candidates=ridge_alphas,
            patch_strength_candidates=patch_strengths,
            predictor_candidates=MappingProxyType(predictors),
        ),
    )


def load_task_order(path: Path) -> TaskOrderConfig:
    data = _load_yaml(path)
    if int(data.get("version", -1)) != 1:
        raise ProtocolConfigError("only task-order version 1 is supported")
    tasks = tuple(
        TaskSpec(
            rank=int(raw["rank"]),
            suite=str(raw["suite"]),
            task_id=int(raw["task_id"]),
            language=str(raw["language"]),
            primary_object=str(raw["primary_object"]),
            planar_symmetry_order=int(raw["planar_symmetry_order"]),
        )
        for raw in (
            _mapping(item, f"tasks[{index}]")
            for index, item in enumerate(_sequence(data.get("tasks"), "tasks"))
        )
    )
    ranks = [task.rank for task in tasks]
    if ranks != list(range(1, len(tasks) + 1)):
        raise ProtocolConfigError("task ranks must be ordered and contiguous from 1")
    if len({(task.suite, task.task_id) for task in tasks}) != len(tasks):
        raise ProtocolConfigError("task suite/task_id pairs must be unique")
    gates = _mapping(data.get("gates"), "gates")
    return TaskOrderConfig(
        version=1,
        written=str(data["written"]),
        selection_rule=str(data["selection_rule"]),
        tasks=tasks,
        gates=GateConfig(
            baseline_iid_episodes=int(gates["baseline_iid_episodes"]),
            baseline_min_successes=int(gates["baseline_min_successes"]),
            perturbed_episodes=int(gates["perturbed_episodes"]),
            min_valid_fraction=float(gates["min_valid_fraction"]),
            min_failure_rate=float(gates["min_failure_rate"]),
            max_failure_rate=float(gates["max_failure_rate"]),
        ),
    )


def load_perturbations(path: Path) -> PerturbationsConfig:
    data = _load_yaml(path)
    if int(data.get("version", -1)) != 1:
        raise ProtocolConfigError("only perturbations version 1 is supported")
    reality = _mapping(data.get("reality_gate"), "reality_gate")
    validity = _mapping(data.get("validity"), "validity")
    config = PerturbationsConfig(
        version=1,
        angles_in_degrees=bool(data["angles_in_degrees"]),
        reality_gate_cells=tuple(
            _condition(item, f"reality_gate.cells[{index}]", index_required=False)
            for index, item in enumerate(
                _sequence(reality.get("cells"), "reality_gate.cells")
            )
        ),
        reality_gate_assignment=str(reality["assignment"]),
        calibration_rollouts=tuple(
            _condition(item, f"calibration_rollouts[{index}]", index_required=True)
            for index, item in enumerate(
                _sequence(data.get("calibration_rollouts"), "calibration_rollouts")
            )
        ),
        locked_test_rollouts=tuple(
            _condition(item, f"locked_test_rollouts[{index}]", index_required=True)
            for index, item in enumerate(
                _sequence(data.get("locked_test_rollouts"), "locked_test_rollouts")
            )
        ),
        counterfactual_score_grid=tuple(
            _condition(
                item, f"counterfactual_score_grid[{index}]", index_required=False
            )
            for index, item in enumerate(
                _sequence(
                    data.get("counterfactual_score_grid"), "counterfactual_score_grid"
                )
            )
        ),
        validity=ValidityConfig(
            settle_steps_after_edit=int(validity["settle_steps_after_edit"]),
            max_invalid_fraction=float(validity["max_invalid_fraction"]),
            reject_nan_or_inf=bool(validity["reject_nan_or_inf"]),
            reject_workspace_exit=bool(validity["reject_workspace_exit"]),
            reject_initial_success=bool(validity["reject_initial_success"]),
            max_penetration_depth_m=float(validity["max_penetration_depth_m"]),
            max_linear_speed_m_s=float(validity["max_linear_speed_m_s"]),
            max_angular_speed_rad_s=float(validity["max_angular_speed_rad_s"]),
            min_height_below_table_m=float(validity["min_height_below_table_m"]),
            max_height_above_table_m=float(validity["max_height_above_table_m"]),
            phase_min_object_displacement_m=float(
                validity["phase_min_object_displacement_m"]
            ),
        ),
    )
    thresholds = config.validity
    if thresholds.settle_steps_after_edit != 10:
        raise ProtocolConfigError("validity settling must be exactly ten steps")
    positive_thresholds = (
        thresholds.max_penetration_depth_m,
        thresholds.max_linear_speed_m_s,
        thresholds.max_angular_speed_rad_s,
        thresholds.min_height_below_table_m,
        thresholds.max_height_above_table_m,
        thresholds.phase_min_object_displacement_m,
    )
    if any(value <= 0 for value in positive_thresholds):
        raise ProtocolConfigError("validity thresholds must be positive")
    if (
        config.reality_gate_assignment
        != "balanced_hash_without_replacement_three_cells_per_init_state"
    ):
        raise ProtocolConfigError("unsupported Reality Gate assignment")
    for name, conditions in (
        ("calibration", config.calibration_rollouts),
        ("locked_test", config.locked_test_rollouts),
    ):
        indices = [condition.index for condition in conditions]
        if indices != list(range(len(conditions))):
            raise ProtocolConfigError(
                f"{name} condition indices must be ordered from zero"
            )
    return config


def load_protocol_config(config_dir: str | Path) -> ProtocolConfig:
    """Load and validate all three protocol YAML files from ``config_dir``."""

    root = Path(config_dir)
    return ProtocolConfig(
        split=load_split_protocol(root / "split_protocol.yaml"),
        task_order=load_task_order(root / "task_order.yaml"),
        perturbations=load_perturbations(root / "perturbations.yaml"),
    )
