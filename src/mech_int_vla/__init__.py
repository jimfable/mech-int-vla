"""Executable protocol utilities for the preregistered SmolVLA study.

PyTorch is intentionally an optional dependency.  Import instrumentation APIs
from :mod:`mech_int_vla.instrumentation` only in processes that need model
hooks; importing this package's protocol/manifest surface does not import torch.
"""

from .config import (
    ProtocolConfig,
    ProtocolConfigError,
    SplitName,
    TaskSpec,
    load_protocol_config,
)
from .determinism import hash_order, hash_seed
from .guard import (
    CalibrationGuardError,
    CalibrationReceipt,
    LockedTestGuardError,
    LockedTestReceipt,
    assert_calibration_ready,
    assert_locked_test_ready,
)
from .manifest import EpisodeSpec, Manifest, generate_episode_manifest

__all__ = [
    "CalibrationGuardError",
    "CalibrationReceipt",
    "EpisodeSpec",
    "LockedTestGuardError",
    "LockedTestReceipt",
    "Manifest",
    "ProtocolConfig",
    "ProtocolConfigError",
    "SplitName",
    "TaskSpec",
    "assert_calibration_ready",
    "assert_locked_test_ready",
    "generate_episode_manifest",
    "hash_order",
    "hash_seed",
    "load_protocol_config",
]
