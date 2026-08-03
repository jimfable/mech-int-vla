from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from mech_int_vla.snapshots import (
    BASE_VLM_ALLOW_PATTERNS,
    POLICY_ALLOW_PATTERNS,
    SnapshotError,
    SnapshotPaths,
    load_locked_smolvla,
    load_model_input_lock,
    resolve_snapshot_paths,
)

ROOT = Path(__file__).parents[1]


def lock_for_weights(tmp_path: Path, weights: bytes) -> Path:
    lock_path = tmp_path / "environment.lock"
    contents = (ROOT / "environment.lock").read_text(encoding="utf-8")
    expected = load_model_input_lock(ROOT / "environment.lock").policy_model_sha256
    contents = contents.replace(expected, hashlib.sha256(weights).hexdigest())
    lock_path.write_text(contents, encoding="utf-8")
    return lock_path


def test_environment_lock_has_full_model_revisions() -> None:
    lock = load_model_input_lock(ROOT / "environment.lock")
    assert lock.policy_revision == "31d453f7edd78c839a8bbc39744a292686daf0de"
    assert lock.base_vlm_revision == "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    assert lock.policy_n_action_steps == 1
    assert lock.policy_model_sha256 == (
        "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"
    )


def test_snapshot_resolution_passes_exact_revisions_and_defaults_offline(
    tmp_path: Path, monkeypatch
) -> None:
    weights = b"weights"
    lock_path = lock_for_weights(tmp_path, weights)
    lock = load_model_input_lock(lock_path)
    policy = tmp_path / lock.policy_revision
    base = tmp_path / lock.base_vlm_revision
    policy.mkdir()
    base.mkdir()
    (policy / "config.json").write_text("{}")
    (policy / lock.policy_model_file).write_bytes(weights)
    (base / "config.json").write_text("{}")
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(policy if kwargs["repo_id"] == lock.policy_repo else base)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    paths = resolve_snapshot_paths(lock_path)
    assert paths.policy == policy.resolve()
    assert paths.base_vlm == base.resolve()
    assert [call["revision"] for call in calls] == [
        lock.policy_revision,
        lock.base_vlm_revision,
    ]
    assert all(call["local_files_only"] for call in calls)
    assert tuple(calls[0]["allow_patterns"]) == POLICY_ALLOW_PATTERNS
    assert tuple(calls[1]["allow_patterns"]) == BASE_VLM_ALLOW_PATTERNS
    assert "*.safetensors" not in BASE_VLM_ALLOW_PATTERNS


def test_model_input_lock_rejects_non_lowercase_or_missing_checkpoint_digest(
    tmp_path: Path,
) -> None:
    original = (ROOT / "environment.lock").read_text(encoding="utf-8")
    expected = load_model_input_lock(ROOT / "environment.lock").policy_model_sha256
    for index, replacement in enumerate((expected.upper(), expected[:-1], "")):
        candidate = tmp_path / f"invalid-{index}.lock"
        if replacement:
            contents = original.replace(expected, replacement)
        else:
            contents = original.replace(f'policy_model_sha256 = "{expected}"\n', "")
        candidate.write_text(contents, encoding="utf-8")
        with pytest.raises(SnapshotError, match="policy_model_sha256|environment lock"):
            load_model_input_lock(candidate)


def test_snapshot_resolution_fails_closed_on_checkpoint_byte_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    lock = load_model_input_lock(ROOT / "environment.lock")
    policy = tmp_path / lock.policy_revision
    base = tmp_path / lock.base_vlm_revision
    policy.mkdir()
    base.mkdir()
    (policy / "config.json").write_text("{}", encoding="utf-8")
    (policy / lock.policy_model_file).write_bytes(b"tampered weights")
    (base / "config.json").write_text("{}", encoding="utf-8")

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **kwargs: str(
        policy if kwargs["repo_id"] == lock.policy_repo else base
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    with pytest.raises(SnapshotError, match="checkpoint SHA-256 mismatch"):
        resolve_snapshot_paths(ROOT / "environment.lock")


def test_policy_and_tokenizer_are_both_overridden_to_local_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    lock = load_model_input_lock(ROOT / "environment.lock")
    policy_path = tmp_path / lock.policy_revision
    vlm_path = tmp_path / lock.base_vlm_revision
    policy_path.mkdir()
    vlm_path.mkdir()
    paths = SnapshotPaths(policy_path, vlm_path, lock)
    captured = {}

    class FakeConfig:
        pass

    class FakePreTrainedConfig:
        @classmethod
        def from_pretrained(cls, path, local_files_only):
            captured["config_path"] = path
            captured["config_offline"] = local_files_only
            config = FakeConfig()
            config.input_features = {
                "observation.state": types.SimpleNamespace(shape=(6,))
            }
            config.output_features = {"action": types.SimpleNamespace(shape=(7,))}
            return config

    class FakePolicy:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            captured["policy_path"] = path
            captured["policy_kwargs"] = kwargs
            instance = cls()
            instance.config = kwargs["config"]
            return instance

        def eval(self):
            captured["eval"] = True

    tokenizer_step = types.SimpleNamespace(tokenizer_name=str(vlm_path))
    normalizer_step = types.SimpleNamespace(
        features={"observation.state": types.SimpleNamespace(shape=(8,))},
        _tensor_stats={
            "observation.state": {
                "count": types.SimpleNamespace(shape=(1,)),
                "mean": types.SimpleNamespace(shape=(8,)),
                "std": types.SimpleNamespace(shape=(8,)),
            }
        },
    )

    class FakePreprocessor:
        def __init__(self):
            self.steps = [tokenizer_step, normalizer_step]

        def __call__(self, observation):
            return observation

    def make_pre_post_processors(config, **kwargs):
        captured["processor_config"] = config
        captured["processor_kwargs"] = kwargs
        return FakePreprocessor(), object()

    class FakePipeline:
        def __init__(self, *, steps):
            self.steps = steps

        def __call__(self, observation):
            return observation

    class FakeLiberoProcessorStep:
        pass

    factory = types.ModuleType("lerobot.policies.factory")
    factory.make_pre_post_processors = make_pre_post_processors
    policies_config = types.ModuleType("lerobot.configs.policies")
    policies_config.PreTrainedConfig = FakePreTrainedConfig
    configuration = types.ModuleType("lerobot.policies.smolvla.configuration_smolvla")
    configuration.SmolVLAConfig = FakeConfig
    modeling = types.ModuleType("lerobot.policies.smolvla.modeling_smolvla")
    modeling.SmolVLAPolicy = FakePolicy
    processor = types.ModuleType("lerobot.processor")
    processor.LiberoProcessorStep = FakeLiberoProcessorStep
    processor.PolicyProcessorPipeline = FakePipeline
    envs = types.ModuleType("lerobot.envs")
    env_utils = types.ModuleType("lerobot.envs.utils")
    env_utils.preprocess_observation = lambda observation: {
        "observation.state": np.zeros((1, 8), dtype=np.float32)
    }
    for name in (
        "lerobot",
        "lerobot.configs",
        "lerobot.policies",
        "lerobot.policies.smolvla",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "lerobot.configs.policies", policies_config)
    monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory)
    monkeypatch.setitem(
        sys.modules, "lerobot.policies.smolvla.configuration_smolvla", configuration
    )
    monkeypatch.setitem(
        sys.modules, "lerobot.policies.smolvla.modeling_smolvla", modeling
    )
    monkeypatch.setitem(sys.modules, "lerobot.processor", processor)
    monkeypatch.setitem(sys.modules, "lerobot.envs", envs)
    monkeypatch.setitem(sys.modules, "lerobot.envs.utils", env_utils)

    loaded = load_locked_smolvla(paths, device="cuda")
    config = loaded.policy.config
    assert config.vlm_model_name == str(vlm_path)
    assert config.num_steps == 10
    assert config.chunk_size == 50
    assert config.n_action_steps == 1
    assert config.empty_cameras == 1
    assert config.load_vlm_weights is False
    assert config.input_features["observation.state"].shape == (8,)
    assert captured["policy_kwargs"]["local_files_only"] is True
    assert captured["policy_kwargs"]["strict"] is True
    overrides = captured["processor_kwargs"]["preprocessor_overrides"]
    assert overrides["tokenizer_processor"] == {"tokenizer_name": str(vlm_path)}
    assert overrides["device_processor"] == {"device": "cuda"}
    assert overrides["normalizer_processor"]["features"]["observation.state"].shape == (
        8,
    )
    assert loaded.original_state_shape == (6,)
    assert loaded.runtime_state_shape == (8,)
    assert loaded.normalization_state_shapes == {
        "count": (1,),
        "mean": (8,),
        "std": (8,),
    }
    processed = loaded.preprocess_observation(
        {"placeholder": np.zeros(1)}, task="pick up the book"
    )
    assert processed["observation.state"].shape == (1, 8)
