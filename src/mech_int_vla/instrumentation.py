"""Dependency-light activation capture and patching for SmolVLA.

This module deliberately depends only on PyTorch.  In particular, importing it
does not import LeRobot, Transformers, or instantiate a policy/checkpoint.

The hooks implement the five representation/time candidates preregistered for
the study:

* the final state token entering VLM layer 12 during prefix-cache creation;
* expert residuals entering layers 4 and 12 at denoising steps 0 and 5.

SmolVLA v0.6.0 calls ``vlm_with_expert.forward(...)`` directly, bypassing
``nn.Module.__call__`` and therefore bypassing forward hooks placed on the root
module.  To classify prefix and denoising calls reliably, installation wraps
that one bound method.  The activation and patch operations themselves remain
ordinary, removable PyTorch hooks on the three input-norm modules.

Causal interventions mutate the hooked residual tensor itself under ``no_grad``.
Returning a replacement norm argument is not sufficient: SmolVLA's manually
implemented residual branch retains a reference to the original tensor.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import math
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import Any, Self

import torch
from torch import Tensor, nn

VLM_CONTEXT = "vlm_context"
EXPERT_EARLY = "expert_layer_4"
EXPERT_LATE = "expert_layer_12"

_VALID_LOCATIONS = frozenset((VLM_CONTEXT, EXPERT_EARLY, EXPERT_LATE))
_EXPERT_LOCATIONS = frozenset((EXPERT_EARLY, EXPERT_LATE))


class InstrumentationError(RuntimeError):
    """Base class for instrumentation failures."""


class ActivationShapeError(InstrumentationError):
    """Raised when a hooked tensor violates the preregistered shape contract."""


class CallPhase(str, Enum):
    """Kind of ``vlm_with_expert.forward`` call being observed."""

    PREFIX_CACHE = "prefix_cache"
    DENOISING = "denoising"
    FULL_FORWARD = "full_forward"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CallRecord:
    """Metadata for one call to ``vlm_with_expert.forward``."""

    phase: CallPhase
    run_index: int
    denoising_step: int | None
    flow_time: float | None


@dataclass(frozen=True, slots=True)
class ActivationRecord:
    """One captured candidate representation.

    ``value`` and ``norm_input`` are the same preregistered pre-norm residual
    coordinate: the final state token for ``VLM_CONTEXT`` and the mean-pooled
    action-token residual for expert locations.  Both fields are detached clones,
    never references to the live model activation.
    """

    location: str
    phase: CallPhase
    run_index: int
    denoising_step: int | None
    flow_time: float | None
    value: Tensor
    norm_input: Tensor
    source_shape: tuple[int, ...]
    token_count: int
    patched: bool


@dataclass(slots=True)
class _CallFrame:
    phase: CallPhase
    run_index: int
    denoising_step: int | None = None
    flow_time: float | None = None
    patched_locations: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _PatchSpec:
    location: str
    shift: Tensor
    denoising_step: int | None


def probe_subspace_shift(
    probe_rows: Tensor,
    donor_activation: Tensor,
    recipient_activation: Tensor,
    *,
    alpha: float = 1.0,
) -> Tensor:
    """Return ``alpha * P(h_d - h_r)`` for the probe row-space projector.

    ``probe_rows`` has shape ``(probe_dim, hidden_dim)`` and activations may have
    shape ``(hidden_dim,)`` or ``(batch, hidden_dim)``.  ``pinv`` makes the
    operation well-defined even if probe rows are not orthogonal or are rank
    deficient.
    """

    if probe_rows.ndim != 2:
        raise ValueError(
            f"probe_rows must be rank 2, got shape {tuple(probe_rows.shape)}"
        )
    if donor_activation.shape != recipient_activation.shape:
        raise ValueError(
            "donor_activation and recipient_activation must have identical shapes; "
            f"got {tuple(donor_activation.shape)} and {tuple(recipient_activation.shape)}"
        )
    if donor_activation.ndim not in (1, 2):
        raise ValueError(
            "activations must have shape (hidden_dim,) or (batch, hidden_dim); "
            f"got {tuple(donor_activation.shape)}"
        )
    if donor_activation.shape[-1] != probe_rows.shape[-1]:
        raise ValueError(
            "probe and activation hidden dimensions differ: "
            f"{probe_rows.shape[-1]} != {donor_activation.shape[-1]}"
        )
    if not math.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha!r}")

    if donor_activation.device != recipient_activation.device:
        raise ValueError("donor and recipient activations must be on the same device")

    compute_dtype = (
        torch.float64
        if torch.float64
        in (probe_rows.dtype, donor_activation.dtype, recipient_activation.dtype)
        else torch.float32
    )
    delta = donor_activation.to(dtype=compute_dtype) - recipient_activation.to(
        dtype=compute_dtype
    )
    rows = probe_rows.to(device=delta.device, dtype=compute_dtype)
    gram_pinv = torch.linalg.pinv(rows @ rows.mT)
    projected = (delta @ rows.mT) @ gram_pinv @ rows
    return projected * alpha


class SmolVLAInstrumentation:
    """Capture and causally patch preregistered SmolVLA representations.

    Args:
        model: Either the flow model exposing ``vlm_with_expert`` or a policy
            whose ``model`` attribute exposes it.
        capture_steps: Denoising indices to retain.  The preregistered default is
            ``(0, 5)`` for ten flow-matching steps (times 1.0 and 0.5).
        expected_action_tokens: Required expert sequence length.  The protocol
            fixes this at SmolVLA's 50-token action chunk.
        copy_to_cpu: Move detached activation copies to CPU for storage.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        capture_steps: Sequence[int] = (0, 5),
        expected_action_tokens: int = 50,
        copy_to_cpu: bool = True,
    ) -> None:
        self.flow_model = self._resolve_flow_model(model)
        self.vlm_with_expert = self.flow_model.vlm_with_expert

        self.capture_steps = frozenset(int(step) for step in capture_steps)
        self.expected_action_tokens = int(expected_action_tokens)
        self.copy_to_cpu = bool(copy_to_cpu)
        config = getattr(self.flow_model, "config", None)
        self.num_steps = int(getattr(config, "num_steps", 10))

        if not self.capture_steps:
            raise ValueError("capture_steps may not be empty")
        if min(self.capture_steps) < 0 or max(self.capture_steps) >= self.num_steps:
            raise ValueError(
                f"capture_steps {sorted(self.capture_steps)} fall outside num_steps={self.num_steps}"
            )
        if self.capture_steps != frozenset((0, 5)):
            raise InstrumentationError(
                "the preregistered expert capture steps must be exactly (0, 5)"
            )
        if self.expected_action_tokens <= 0:
            raise ValueError("expected_action_tokens must be positive")
        if self.num_steps != 10:
            raise InstrumentationError(
                f"pinned SmolVLA requires num_steps=10, got {self.num_steps}"
            )
        if self.expected_action_tokens != 50:
            raise InstrumentationError(
                "the preregistered expert activation requires exactly 50 action tokens"
            )
        if (
            config is not None
            and hasattr(config, "chunk_size")
            and int(config.chunk_size) != 50
        ):
            raise InstrumentationError(
                f"pinned SmolVLA requires chunk_size=50, got {config.chunk_size}"
            )
        if (
            config is not None
            and hasattr(config, "use_cache")
            and config.use_cache is not True
        ):
            raise InstrumentationError(
                f"pinned SmolVLA requires use_cache=True, got {config.use_cache!r}"
            )

        self._norms = self._resolve_norm_modules()
        self._records: list[ActivationRecord] = []
        self._calls: list[CallRecord] = []
        self._handles: list[Any] = []
        self._previous_forward: Any | None = None
        self._wrapped_forward: Any | None = None
        self._installed = False
        self._run_index = -1
        self._next_denoising_step = 0
        self._lock = threading.RLock()
        identity = hex(id(self))
        self._frame_var: contextvars.ContextVar[tuple[_CallFrame, ...]] = (
            contextvars.ContextVar(
                f"smolvla_instrumentation_frames_{identity}", default=()
            )
        )
        self._patch_var: contextvars.ContextVar[tuple[_PatchSpec, ...]] = (
            contextvars.ContextVar(
                f"smolvla_instrumentation_patches_{identity}", default=()
            )
        )

    @staticmethod
    def _resolve_flow_model(model: nn.Module) -> nn.Module:
        if hasattr(model, "vlm_with_expert"):
            return model
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "vlm_with_expert"):
            return inner
        raise InstrumentationError(
            "expected a SmolVLA flow model with `vlm_with_expert`, or a policy with "
            "`model.vlm_with_expert`"
        )

    def _resolve_norm_modules(self) -> dict[str, nn.Module]:
        try:
            vlm_layers = self.vlm_with_expert.get_vlm_model().text_model.layers
            layers = self.vlm_with_expert.lm_expert.layers
            vlm_norm = vlm_layers[12].input_layernorm
            early_norm = layers[4].input_layernorm
            late_norm = layers[12].input_layernorm
        except (AttributeError, IndexError, TypeError) as exc:
            raise InstrumentationError(
                "model does not expose the preregistered SmolVLA norm paths "
                "(VLM layer 12 and expert layers 4/12 input_layernorm)"
            ) from exc

        modules = {
            VLM_CONTEXT: vlm_norm,
            EXPERT_EARLY: early_norm,
            EXPERT_LATE: late_norm,
        }
        for location, module in modules.items():
            if not isinstance(module, nn.Module):
                raise InstrumentationError(
                    f"{location} target is not a torch.nn.Module"
                )
        return modules

    @property
    def is_installed(self) -> bool:
        return self._installed

    @property
    def records(self) -> tuple[ActivationRecord, ...]:
        with self._lock:
            return tuple(self._records)

    @property
    def calls(self) -> tuple[CallRecord, ...]:
        with self._lock:
            return tuple(self._calls)

    def clear(self) -> None:
        """Clear captured data without disturbing installed hooks."""

        with self._lock:
            self._records.clear()
            self._calls.clear()

    def activations(
        self,
        location: str,
        *,
        denoising_step: int | None = None,
    ) -> tuple[ActivationRecord, ...]:
        """Return records filtered by location and optional denoising step."""

        self._validate_location(location)
        with self._lock:
            return tuple(
                record
                for record in self._records
                if record.location == location
                and (denoising_step is None or record.denoising_step == denoising_step)
            )

    def install(self) -> SmolVLAInstrumentation:
        """Install all instrumentation.  Repeated calls are idempotent."""

        with self._lock:
            if self._installed:
                return self
            self._installed = True

        try:
            self._handles.append(
                self._norms[VLM_CONTEXT].register_forward_pre_hook(self._vlm_pre_hook)
            )
            self._handles.append(
                self._norms[EXPERT_EARLY].register_forward_pre_hook(
                    functools.partial(self._expert_pre_hook, EXPERT_EARLY)
                )
            )
            self._handles.append(
                self._norms[EXPERT_LATE].register_forward_pre_hook(
                    functools.partial(self._expert_pre_hook, EXPERT_LATE)
                )
            )
            self._install_forward_wrapper()
        except BaseException:
            self.remove()
            raise
        return self

    def remove(self) -> None:
        """Remove hooks and restore the original forward method safely."""

        with self._lock:
            handles, self._handles = self._handles, []
            was_installed = self._installed
            self._installed = False

        for handle in reversed(handles):
            handle.remove()

        if was_installed and self._wrapped_forward is not None:
            current = self.vlm_with_expert.forward
            # Do not clobber another tool that wrapped forward after this one.
            if current is self._wrapped_forward:
                self.vlm_with_expert.forward = self._previous_forward
            self._wrapped_forward = None
            self._previous_forward = None

    def __enter__(self) -> Self:
        return self.install()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.remove()

    def _install_forward_wrapper(self) -> None:
        previous_forward = self.vlm_with_expert.forward

        @functools.wraps(previous_forward)
        def instrumented_forward(*args: Any, **kwargs: Any) -> Any:
            if not self._installed:
                return previous_forward(*args, **kwargs)
            frame = self._new_call_frame(args, kwargs)
            stack = self._frame_var.get()
            token = self._frame_var.set((*stack, frame))
            try:
                return previous_forward(*args, **kwargs)
            finally:
                self._frame_var.reset(token)

        self._previous_forward = previous_forward
        self._wrapped_forward = instrumented_forward
        self.vlm_with_expert.forward = instrumented_forward

    def _new_call_frame(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> _CallFrame:
        past_key_values = kwargs.get(
            "past_key_values", args[2] if len(args) > 2 else None
        )
        fill_kv_cache = kwargs.get("fill_kv_cache", args[5] if len(args) > 5 else None)

        with self._lock:
            if fill_kv_cache is True:
                self._run_index += 1
                self._next_denoising_step = 0
                frame = _CallFrame(CallPhase.PREFIX_CACHE, self._run_index)
            elif fill_kv_cache is False and past_key_values is not None:
                step = self._next_denoising_step
                self._next_denoising_step += 1
                frame = _CallFrame(
                    CallPhase.DENOISING,
                    max(self._run_index, 0),
                    denoising_step=step,
                    flow_time=1.0 - step / self.num_steps,
                )
            elif fill_kv_cache is False:
                frame = _CallFrame(CallPhase.FULL_FORWARD, max(self._run_index, 0))
            else:
                frame = _CallFrame(CallPhase.UNKNOWN, max(self._run_index, 0))

            self._calls.append(
                CallRecord(
                    phase=frame.phase,
                    run_index=frame.run_index,
                    denoising_step=frame.denoising_step,
                    flow_time=frame.flow_time,
                )
            )
        return frame

    def _current_frame(self) -> _CallFrame | None:
        stack = self._frame_var.get()
        return stack[-1] if stack else None

    def _vlm_pre_hook(self, module: nn.Module, inputs: tuple[Any, ...]) -> None:
        del module
        frame = self._current_frame()
        if frame is None or frame.phase is not CallPhase.PREFIX_CACHE:
            return
        tensor = self._first_tensor(inputs, VLM_CONTEXT)
        self._validate_sequence_tensor(tensor, VLM_CONTEXT, expected_tokens=None)
        self._apply_active_patches(VLM_CONTEXT, tensor, frame)
        state_token = tensor[:, -1, :]
        self._append_record(
            location=VLM_CONTEXT,
            frame=frame,
            value=state_token,
            norm_input=state_token,
            source_shape=tuple(tensor.shape),
            token_count=1,
        )

    def _expert_pre_hook(
        self,
        location: str,
        module: nn.Module,
        inputs: tuple[Any, ...],
    ) -> None:
        del module
        frame = self._current_frame()
        if frame is None or frame.phase is not CallPhase.DENOISING:
            return
        tensor = self._first_tensor(inputs, location)
        self._validate_sequence_tensor(
            tensor, location, expected_tokens=self.expected_action_tokens
        )
        self._apply_active_patches(location, tensor, frame)
        if frame.denoising_step in self.capture_steps:
            pooled = tensor.mean(dim=1)
            self._append_record(
                location=location,
                frame=frame,
                value=pooled,
                norm_input=pooled,
                source_shape=tuple(tensor.shape),
                token_count=tensor.shape[1],
            )

    @staticmethod
    def _first_tensor(inputs: tuple[Any, ...], location: str) -> Tensor:
        if not inputs or not isinstance(inputs[0], Tensor):
            raise ActivationShapeError(
                f"{location} expected a Tensor as its first norm input"
            )
        return inputs[0]

    @staticmethod
    def _validate_sequence_tensor(
        tensor: Tensor,
        location: str,
        *,
        expected_tokens: int | None,
    ) -> None:
        if tensor.ndim != 3:
            raise ActivationShapeError(
                f"{location} expected shape (batch, tokens, hidden), got {tuple(tensor.shape)}"
            )
        if tensor.shape[0] <= 0 or tensor.shape[1] <= 0 or tensor.shape[2] <= 0:
            raise ActivationShapeError(
                f"{location} received an empty dimension: {tuple(tensor.shape)}"
            )
        if expected_tokens is not None and tensor.shape[1] != expected_tokens:
            raise ActivationShapeError(
                f"{location} expected {expected_tokens} action tokens, got {tensor.shape[1]}"
            )

    def _append_record(
        self,
        *,
        location: str,
        frame: _CallFrame,
        value: Tensor,
        norm_input: Tensor,
        source_shape: tuple[int, ...],
        token_count: int,
    ) -> None:
        record = ActivationRecord(
            location=location,
            phase=frame.phase,
            run_index=frame.run_index,
            denoising_step=frame.denoising_step,
            flow_time=frame.flow_time,
            value=self._copy_activation(value),
            norm_input=self._copy_activation(norm_input),
            source_shape=source_shape,
            token_count=token_count,
            patched=location in frame.patched_locations,
        )
        with self._lock:
            self._records.append(record)

    def _copy_activation(self, value: Tensor) -> Tensor:
        copied = value.detach().clone()
        return copied.cpu() if self.copy_to_cpu else copied

    @contextlib.contextmanager
    def patch(
        self,
        location: str,
        shift: Tensor,
        *,
        denoising_step: int | None = None,
    ) -> Iterator[None]:
        """Temporarily add ``shift`` at a preregistered activation location.

        Expert shifts are broadcast over all 50 action tokens.  VLM shifts are
        applied only to the final state token entering VLM layer 12.  Every expert
        patch must name exactly one denoising step explicitly; applying an
        intervention at both candidate times by default is prohibited.
        """

        if not self._installed:
            raise InstrumentationError(
                "install instrumentation before entering a patch context"
            )
        self._validate_location(location)
        if not isinstance(shift, Tensor):
            raise TypeError("shift must be a torch.Tensor")
        if shift.ndim not in (1, 2):
            raise ValueError(
                f"shift must have shape (hidden,) or (batch, hidden), got {tuple(shift.shape)}"
            )
        if not torch.isfinite(shift).all().item():
            raise ValueError("shift must contain only finite values")

        if location == VLM_CONTEXT:
            if denoising_step is not None:
                raise ValueError(
                    "denoising_step does not apply to the VLM prefix context"
                )
            step = None
        else:
            if denoising_step is None:
                raise ValueError(
                    "expert patches require exactly one explicit denoising_step"
                )
            if not isinstance(denoising_step, int) or isinstance(denoising_step, bool):
                raise TypeError("denoising_step must be one integer step index")
            step = denoising_step
            if step < 0 or step >= self.num_steps:
                raise ValueError(
                    f"denoising_step {step} falls outside num_steps={self.num_steps}"
                )

        spec = _PatchSpec(
            location=location,
            shift=shift.detach().clone(),
            denoising_step=step,
        )
        current = self._patch_var.get()
        token = self._patch_var.set((*current, spec))
        try:
            yield
        finally:
            self._patch_var.reset(token)

    @contextlib.contextmanager
    def patch_probe_subspace(
        self,
        location: str,
        probe_rows: Tensor,
        donor_activation: Tensor,
        recipient_activation: Tensor,
        *,
        alpha: float = 1.0,
        denoising_step: int | None = None,
    ) -> Iterator[Tensor]:
        """Patch with the preregistered ``alpha P(h_d - h_r)`` shift."""

        shift = probe_subspace_shift(
            probe_rows,
            donor_activation,
            recipient_activation,
            alpha=alpha,
        )
        with self.patch(location, shift, denoising_step=denoising_step):
            yield shift

    def _apply_active_patches(
        self, location: str, tensor: Tensor, frame: _CallFrame
    ) -> None:
        for spec in self._patch_var.get():
            if spec.location != location:
                continue
            if (
                location in _EXPERT_LOCATIONS
                and frame.denoising_step != spec.denoising_step
            ):
                continue
            if self.flow_model.training:
                raise InstrumentationError(
                    "causal activation patches are inference-only; call model.eval() first"
                )
            if torch.is_grad_enabled():
                raise InstrumentationError(
                    "causal activation patches require torch.no_grad() or torch.inference_mode()"
                )
            shift = self._broadcast_shift(spec.shift, tensor, location)
            with torch.no_grad():
                if location == VLM_CONTEXT:
                    tensor[:, -1, :].add_(shift)
                else:
                    tensor.add_(shift[:, None, :])
            frame.patched_locations.add(location)

    @staticmethod
    def _broadcast_shift(shift: Tensor, target: Tensor, location: str) -> Tensor:
        converted = shift.to(device=target.device, dtype=target.dtype)
        if converted.shape[-1] != target.shape[-1]:
            raise ActivationShapeError(
                f"{location} patch hidden size {converted.shape[-1]} does not match "
                f"activation hidden size {target.shape[-1]}"
            )
        if converted.ndim == 1:
            return converted.unsqueeze(0).expand(target.shape[0], -1)
        if converted.shape[0] == 1:
            return converted.expand(target.shape[0], -1)
        if converted.shape[0] != target.shape[0]:
            raise ActivationShapeError(
                f"{location} patch batch size {converted.shape[0]} does not match "
                f"activation batch size {target.shape[0]}"
            )
        return converted

    @staticmethod
    def _validate_location(location: str) -> None:
        if location not in _VALID_LOCATIONS:
            raise ValueError(
                f"unknown activation location {location!r}; expected one of {sorted(_VALID_LOCATIONS)}"
            )


__all__ = [
    "EXPERT_EARLY",
    "EXPERT_LATE",
    "VLM_CONTEXT",
    "ActivationRecord",
    "ActivationShapeError",
    "CallPhase",
    "CallRecord",
    "InstrumentationError",
    "SmolVLAInstrumentation",
    "probe_subspace_shift",
]
