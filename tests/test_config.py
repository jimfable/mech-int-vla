from __future__ import annotations

from pathlib import Path

import pytest

from mech_int_vla.config import ProtocolConfigError, SplitName, load_protocol_config

ROOT = Path(__file__).parents[1]


def test_loads_preregistered_yaml_into_typed_config() -> None:
    config = load_protocol_config(ROOT / "configs")

    assert config.split.splits[SplitName.DISCOVERY].init_state_ids.ids() == tuple(
        range(10)
    )
    assert [task.task_id for task in config.task_order.tasks] == [5, 2, 9]
    assert [task.language for task in config.task_order.tasks] == [
        "pick up the book and place it in the back compartment of the caddy",
        "turn on the stove and put the moka pot on it",
        "put the yellow and white mug in the microwave and close it",
    ]
    assert len(config.perturbations.calibration_rollouts) == 8
    assert len(config.perturbations.locked_test_rollouts) == 8
    assert config.perturbations.validity.max_penetration_depth_m == 0.005
    assert config.perturbations.validity.max_linear_speed_m_s == 0.05
    assert config.perturbations.validity.max_angular_speed_rad_s == 0.5
    assert config.perturbations.validity.phase_min_object_displacement_m == 0.02
    assert len(config.split.calibration_selection.representation_candidates) == 5
    assert set(config.split.calibration_selection.patch_strength_candidates) == {
        0.25,
        0.5,
        1.0,
    }
    assert (
        config.split.seed_for(
            SplitName.CALIBRATION,
            task_rank=1,
            base_init_state_id=10,
            condition_index=7,
        )
        == 201107
    )


def test_seed_rejects_init_id_from_another_split() -> None:
    config = load_protocol_config(ROOT / "configs")

    with pytest.raises(ProtocolConfigError, match="outside split"):
        config.split.seed_for(
            SplitName.DISCOVERY,
            task_rank=1,
            base_init_state_id=10,
            condition_index=0,
        )


def test_loaded_protocol_is_deeply_immutable() -> None:
    config = load_protocol_config(ROOT / "configs")

    with pytest.raises(TypeError):
        config.perturbations.locked_test_rollouts[1].parameters["value"] = -99
    with pytest.raises(TypeError):
        config.split.splits[SplitName.LOCKED_TEST] = config.split.splits[
            SplitName.DISCOVERY
        ]
