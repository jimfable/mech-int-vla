from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "ops" / "build_calibration_freeze.py"
SPEC = importlib.util.spec_from_file_location("build_calibration_freeze", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_brier_uses_episode_total_one_weights() -> None:
    # The first episode contributes two rows at weight 1/2 each; the second
    # contributes one row at weight 1.  A row mean would incorrectly be 1/3.
    value = MODULE._episode_weighted_brier(
        np.asarray([1.0, 0.0, 1.0]),
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([0.5, 0.5, 1.0]),
    )
    assert value == pytest.approx(0.25)


def test_brier_rejects_misaligned_or_nonpositive_weights() -> None:
    with pytest.raises(RuntimeError, match="aligned"):
        MODULE._episode_weighted_brier(
            np.asarray([0.5]), np.asarray([0.0, 1.0]), np.asarray([1.0])
        )
    with pytest.raises(RuntimeError, match="positively weighted"):
        MODULE._episode_weighted_brier(
            np.asarray([0.5]), np.asarray([0.0]), np.asarray([0.0])
        )
