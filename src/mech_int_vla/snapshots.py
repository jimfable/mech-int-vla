"""Pinned Hugging Face snapshot resolution and offline SmolVLA loading.

Imports of Hugging Face, LeRobot, Transformers, and PyTorch are deliberately kept
inside the functions that need them.  The protocol/configuration package can
therefore be imported on machines without the rollout stack.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SnapshotError(RuntimeError):
    """Raised when an immutable model input cannot be resolved or verified."""


@dataclass(frozen=True)
class ModelInputLock:
    policy_repo: str
    policy_revision: str
    policy_model_file: str
    policy_num_flow_steps: int
    policy_chunk_size: int
    policy_n_action_steps: int
    base_vlm_repo: str
    base_vlm_revision: str


@dataclass(frozen=True)
class SnapshotPaths:
    policy: Path
    base_vlm: Path
    lock: ModelInputLock


@dataclass(frozen=True)
class LoadedPolicyRuntime:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    env_preprocessor: Any
    snapshots: SnapshotPaths
    original_state_shape: tuple[int, ...]
    runtime_state_shape: tuple[int, ...]
    normalization_state_shapes: dict[str, tuple[int, ...]]

    def preprocess_observation(
        self, observation: dict[str, Any], *, task: str
    ) -> dict[str, Any]:
        """Run both official preprocessing stages and assert 8-D state."""

        try:
            import numpy as np
            from lerobot.envs.utils import preprocess_observation
        except ImportError as exc:  # pragma: no cover - rollout-host integration
            raise SnapshotError("LeRobot rollout dependencies are required") from exc

        def add_batch(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value[None, ...]
            if isinstance(value, dict):
                return {key: add_batch(item) for key, item in value.items()}
            return value

        converted = preprocess_observation(add_batch(observation))
        converted["task"] = [task]
        env_processed = self.env_preprocessor(converted)
        state = env_processed.get("observation.state")
        if state is None or tuple(state.shape) != (1, 8):
            shape = None if state is None else tuple(state.shape)
            raise SnapshotError(
                f"LIBERO environment processor emitted state shape {shape}, expected (1, 8)"
            )
        processed = self.preprocessor(env_processed)
        processed_state = processed.get("observation.state")
        if processed_state is None or tuple(processed_state.shape) != (1, 8):
            shape = None if processed_state is None else tuple(processed_state.shape)
            raise SnapshotError(
                f"policy preprocessor emitted state shape {shape}, expected (1, 8)"
            )
        return processed


POLICY_ALLOW_PATTERNS = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "*.safetensors",
)
BASE_VLM_ALLOW_PATTERNS = ("*.json", "*.model", "*.txt")


def load_model_input_lock(path: str | Path) -> ModelInputLock:
    """Read the model-related immutable values from ``environment.lock``."""

    lock_path = Path(path)
    with lock_path.open("rb") as stream:
        raw = tomllib.load(stream)
    required = (
        "policy_repo",
        "policy_revision",
        "policy_model_file",
        "policy_num_flow_steps",
        "policy_chunk_size",
        "policy_n_action_steps",
        "base_vlm_repo",
        "base_vlm_revision",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise SnapshotError(f"environment lock is missing: {', '.join(missing)}")
    lock = ModelInputLock(
        policy_repo=str(raw["policy_repo"]),
        policy_revision=str(raw["policy_revision"]),
        policy_model_file=str(raw["policy_model_file"]),
        policy_num_flow_steps=int(raw["policy_num_flow_steps"]),
        policy_chunk_size=int(raw["policy_chunk_size"]),
        policy_n_action_steps=int(raw["policy_n_action_steps"]),
        base_vlm_repo=str(raw["base_vlm_repo"]),
        base_vlm_revision=str(raw["base_vlm_revision"]),
    )
    for name, revision in (
        ("policy_revision", lock.policy_revision),
        ("base_vlm_revision", lock.base_vlm_revision),
    ):
        if len(revision) != 40 or any(
            char not in "0123456789abcdef" for char in revision
        ):
            raise SnapshotError(
                f"{name} must be a full lowercase 40-character commit SHA"
            )
    return lock


def _snapshot_download(
    repo_id: str,
    revision: str,
    *,
    cache_dir: str | Path | None,
    local_files_only: bool,
    allow_patterns: tuple[str, ...],
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised only on rollout host
        raise SnapshotError(
            "huggingface_hub is required to resolve model snapshots"
        ) from exc

    try:
        resolved = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=None if cache_dir is None else str(cache_dir),
                local_files_only=local_files_only,
                allow_patterns=list(allow_patterns),
            )
        ).resolve()
    except Exception as exc:
        mode = "local cache" if local_files_only else "Hub/cache"
        raise SnapshotError(
            f"could not resolve {repo_id}@{revision} from {mode}: {exc}"
        ) from exc
    # Standard huggingface_hub cache snapshots end in their resolved commit.  A
    # mismatch is a fail-closed guard against a branch/ref being returned.
    if resolved.name != revision:
        raise SnapshotError(
            f"snapshot for {repo_id} resolved to {resolved.name}, expected {revision}"
        )
    return resolved


def resolve_snapshot_paths(
    environment_lock: str | Path,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = True,
) -> SnapshotPaths:
    """Resolve both exact snapshots, defaulting to a network-free lookup."""

    lock = load_model_input_lock(environment_lock)
    policy = _snapshot_download(
        lock.policy_repo,
        lock.policy_revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_patterns=POLICY_ALLOW_PATTERNS,
    )
    base_vlm = _snapshot_download(
        lock.base_vlm_repo,
        lock.base_vlm_revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_patterns=BASE_VLM_ALLOW_PATTERNS,
    )
    required_policy_files = ("config.json", lock.policy_model_file)
    missing = [name for name in required_policy_files if not (policy / name).is_file()]
    if missing:
        raise SnapshotError(f"policy snapshot is missing: {', '.join(missing)}")
    if not (base_vlm / "config.json").is_file():
        raise SnapshotError("base-VLM snapshot is missing config.json")
    return SnapshotPaths(policy=policy, base_vlm=base_vlm, lock=lock)


def load_locked_smolvla(
    snapshots: SnapshotPaths,
    *,
    device: str = "cuda",
) -> LoadedPolicyRuntime:
    """Load model, processor, and tokenizer exclusively from verified local paths.

    The policy checkpoint config contains a Hub name for its VLM.  Both that
    config field and the serialized tokenizer processor are overridden with the
    exact local base-VLM snapshot before either can perform a Hub lookup.
    """

    try:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor import LiberoProcessorStep, PolicyProcessorPipeline
    except ImportError as exc:  # pragma: no cover - rollout-host integration
        raise SnapshotError("LeRobot with the smolvla extra is required") from exc

    lock = snapshots.lock
    config = SmolVLAConfig.from_pretrained(
        snapshots.policy,
        local_files_only=True,
    )
    config.vlm_model_name = str(snapshots.base_vlm)
    # The policy safetensor contains the full VLM state.  Construct architecture
    # from the pinned base config, then load only the complete policy checkpoint.
    config.load_vlm_weights = False
    state_feature = config.input_features.get("observation.state")
    if state_feature is None:
        raise SnapshotError("checkpoint config has no observation.state feature")
    original_state_shape = tuple(state_feature.shape)
    if original_state_shape not in {(6,), (8,)}:
        raise SnapshotError(
            f"checkpoint state shape is {original_state_shape}; "
            "expected stale (6,) or corrected (8,)"
        )
    state_feature.shape = (8,)
    config.device = device
    config.num_steps = lock.policy_num_flow_steps
    config.chunk_size = lock.policy_chunk_size
    config.n_action_steps = lock.policy_n_action_steps
    config.empty_cameras = 1
    state_shape = tuple(state_feature.shape)
    if (
        config.num_steps != 10
        or config.chunk_size != 50
        or config.n_action_steps != 1
        or config.empty_cameras != 1
        or config.load_vlm_weights
        or state_shape != (8,)
    ):
        raise SnapshotError("runtime policy overrides do not match the frozen protocol")

    policy = SmolVLAPolicy.from_pretrained(
        snapshots.policy,
        config=config,
        local_files_only=True,
        strict=True,
    )
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(snapshots.policy),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(snapshots.base_vlm)},
            "device_processor": {"device": device},
            "normalizer_processor": {
                "features": {**config.input_features, **config.output_features}
            },
        },
    )
    tokenizer_steps = [
        step for step in preprocessor.steps if hasattr(step, "tokenizer_name")
    ]
    if len(tokenizer_steps) != 1 or tokenizer_steps[0].tokenizer_name != str(
        snapshots.base_vlm
    ):
        raise SnapshotError(
            "preprocessor tokenizer did not bind to the local VLM snapshot"
        )
    normalizer_steps = [
        step
        for step in preprocessor.steps
        if hasattr(step, "_tensor_stats") and "observation.state" in step._tensor_stats
    ]
    if len(normalizer_steps) != 1:
        raise SnapshotError("expected one preprocessor state normalizer")
    normalizer = normalizer_steps[0]
    if tuple(normalizer.features["observation.state"].shape) != (8,):
        raise SnapshotError("normalizer state feature was not overridden to 8-D")
    normalization_state_shapes = {
        str(name): tuple(tensor.shape)
        for name, tensor in normalizer._tensor_stats["observation.state"].items()
    }
    vector_shapes = {
        name: shape
        for name, shape in normalization_state_shapes.items()
        if name != "count"
    }
    count_shape = normalization_state_shapes.get("count")
    if (
        not {"mean", "std"}.issubset(vector_shapes)
        or any(shape != (8,) for shape in vector_shapes.values())
        or count_shape not in {None, (1,)}
    ):
        raise SnapshotError(
            f"state normalization tensors are not all 8-D: {normalization_state_shapes}"
        )
    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    policy.eval()
    return LoadedPolicyRuntime(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        env_preprocessor=env_preprocessor,
        snapshots=snapshots,
        original_state_shape=original_state_shape,
        runtime_state_shape=state_shape,
        normalization_state_shapes=normalization_state_shapes,
    )
