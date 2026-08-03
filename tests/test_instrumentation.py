from __future__ import annotations

import math
import os
import pathlib
import subprocess
import sys
import unittest
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch import nn

from mech_int_vla.instrumentation import (
    EXPERT_EARLY,
    EXPERT_LATE,
    VLM_CONTEXT,
    ActivationShapeError,
    CallPhase,
    InstrumentationError,
    SmolVLAInstrumentation,
    minimum_norm_circular_probe_shift,
    probe_subspace_shift,
)

BRANCH_SCALE = 0.1
CONTEXT_SCALE = 0.01


class FakeResidualLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.Identity()


class FakeTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(FakeResidualLayer() for _ in range(16))
        self.norm = nn.Identity()


class FakeVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_model = FakeTextModel()


class FakeVLMWithExpert(nn.Module):
    """Small model preserving SmolVLA's explicit residual identity paths."""

    def __init__(self) -> None:
        super().__init__()
        self.vlm = FakeVLM()
        self.lm_expert = nn.Module()
        self.lm_expert.layers = nn.ModuleList(FakeResidualLayer() for _ in range(16))
        self.last_cache = None

    def get_vlm_model(self) -> FakeVLM:
        return self.vlm

    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=True,
        fill_kv_cache=True,
    ):
        del attention_mask, position_ids, use_cache
        if fill_kv_cache:
            prefix = inputs_embeds[0]
            layer_cache = []
            for layer in self.vlm.text_model.layers:
                residual = prefix
                normalized = layer.input_layernorm(residual)
                layer_cache.append(normalized[:, -1, :].detach().clone())
                prefix = residual + BRANCH_SCALE * normalized

            cache = {
                "layers": layer_cache,
                "context": torch.stack(layer_cache[12:], dim=0).mean(dim=0),
            }
            self.last_cache = cache
            return [self.vlm.text_model.norm(prefix), None], cache

        suffix = inputs_embeds[1]
        context = past_key_values["context"][:, None, :]
        for layer in self.lm_expert.layers:
            residual = suffix
            normalized = layer.input_layernorm(residual)
            branch = BRANCH_SCALE * normalized + CONTEXT_SCALE * context
            suffix = residual + branch
        return [None, suffix], past_key_values


class FakeFlowModel(nn.Module):
    def __init__(
        self,
        *,
        action_tokens: int = 50,
        hidden_size: int = 4,
        configured_chunk_size: int = 50,
        configured_num_steps: int = 10,
        configured_use_cache: bool = True,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_steps=configured_num_steps,
            chunk_size=configured_chunk_size,
            use_cache=configured_use_cache,
        )
        self.vlm_with_expert = FakeVLMWithExpert()
        self.action_tokens = action_tokens
        self.hidden_size = hidden_size

    def sample_actions(self, batch_size: int = 2):
        prefix = torch.arange(
            batch_size * 4 * self.hidden_size, dtype=torch.float32
        ).reshape(batch_size, 4, self.hidden_size)
        prefix_output, cache = self.vlm_with_expert.forward(
            past_key_values=None,
            inputs_embeds=[prefix, None],
            fill_kv_cache=True,
        )
        expert_outputs = []
        for step in range(self.config.num_steps):
            suffix = torch.full(
                (batch_size, self.action_tokens, self.hidden_size),
                float(step),
            )
            output, _ = self.vlm_with_expert.forward(
                past_key_values=cache,
                inputs_embeds=[None, suffix],
                fill_kv_cache=False,
            )
            expert_outputs.append(output[1])
        return prefix, prefix_output[0], expert_outputs


def residual_before_layer(
    value: torch.Tensor,
    layer_index: int,
    *,
    context: torch.Tensor | None = None,
) -> torch.Tensor:
    result = value
    for _ in range(layer_index):
        result = result + BRANCH_SCALE * result
        if context is not None:
            result = result + CONTEXT_SCALE * context
    return result


class InstrumentationTests(unittest.TestCase):
    def test_imports_do_not_eagerly_load_torch_lerobot_or_transformers(self) -> None:
        package_only = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import mech_int_vla; assert 'torch' not in sys.modules",
            ],
            check=True,
            env={**dict(os.environ), "PYTHONPATH": str(ROOT / "src")},
        )
        self.assertEqual(package_only.returncode, 0)

        instrumentation_import = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import mech_int_vla.instrumentation; "
                    "assert 'lerobot' not in sys.modules; "
                    "assert 'transformers' not in sys.modules"
                ),
            ],
            check=True,
            env={**dict(os.environ), "PYTHONPATH": str(ROOT / "src")},
        )
        self.assertEqual(instrumentation_import.returncode, 0)

    def test_captures_five_preregistered_pre_norm_candidates(self) -> None:
        model = FakeFlowModel()
        instrumentation = SmolVLAInstrumentation(model)

        with instrumentation:
            prefix, _, _ = model.sample_actions()

        self.assertEqual(len(instrumentation.records), 5)
        self.assertEqual(len(instrumentation.activations(VLM_CONTEXT)), 1)
        self.assertEqual(
            [r.denoising_step for r in instrumentation.activations(EXPERT_EARLY)],
            [0, 5],
        )
        self.assertEqual(
            [r.denoising_step for r in instrumentation.activations(EXPERT_LATE)],
            [0, 5],
        )

        vlm_record = instrumentation.activations(VLM_CONTEXT)[0]
        expected_vlm = residual_before_layer(prefix[:, -1, :], 12)
        torch.testing.assert_close(vlm_record.value, expected_vlm)
        torch.testing.assert_close(vlm_record.norm_input, expected_vlm)
        self.assertEqual(vlm_record.phase, CallPhase.PREFIX_CACHE)
        self.assertEqual(vlm_record.token_count, 1)

        context = model.vlm_with_expert.last_cache["context"][:, None, :]
        early_zero = instrumentation.activations(EXPERT_EARLY, denoising_step=0)[0]
        late_five = instrumentation.activations(EXPERT_LATE, denoising_step=5)[0]
        expected_early = residual_before_layer(
            torch.zeros((2, 1, 4)), 4, context=context
        ).squeeze(1)
        expected_late = residual_before_layer(
            torch.full((2, 1, 4), 5.0), 12, context=context
        ).squeeze(1)
        torch.testing.assert_close(early_zero.value, expected_early)
        torch.testing.assert_close(late_five.value, expected_late)
        self.assertEqual(early_zero.source_shape, (2, 50, 4))
        self.assertEqual(early_zero.token_count, 50)
        self.assertAlmostEqual(early_zero.flow_time, 1.0)
        self.assertAlmostEqual(late_five.flow_time, 0.5)
        self.assertIsNone(vlm_record.value.grad_fn)
        self.assertFalse(vlm_record.value.requires_grad)

        self.assertEqual(instrumentation.calls[0].phase, CallPhase.PREFIX_CACHE)
        self.assertTrue(
            all(call.phase is CallPhase.DENOISING for call in instrumentation.calls[1:])
        )
        self.assertEqual(
            [call.denoising_step for call in instrumentation.calls[1:]], list(range(10))
        )

    def test_hooks_are_removed_after_normal_and_exceptional_exit(self) -> None:
        model = FakeFlowModel()
        original_forward = model.vlm_with_expert.forward
        instrumentation = SmolVLAInstrumentation(model)

        with instrumentation:
            model.sample_actions(batch_size=1)
        self.assertFalse(instrumentation.is_installed)
        self.assertEqual(model.vlm_with_expert.forward, original_forward)

        instrumentation.clear()
        with self.assertRaisesRegex(RuntimeError, "sentinel"), instrumentation:
            raise RuntimeError("sentinel")
        model.sample_actions(batch_size=1)
        self.assertEqual(instrumentation.records, ())
        self.assertEqual(instrumentation.calls, ())

    def test_rejects_non_50_token_expert_activation(self) -> None:
        model = FakeFlowModel(action_tokens=49, configured_chunk_size=50)
        instrumentation = SmolVLAInstrumentation(model)

        with (
            instrumentation,
            self.assertRaisesRegex(ActivationShapeError, "expected 50 action tokens"),
        ):
            model.sample_actions(batch_size=1)

    def test_asserts_pinned_inference_configuration(self) -> None:
        cases = (
            ({"configured_num_steps": 9}, "num_steps=10"),
            ({"configured_chunk_size": 49}, "chunk_size=50"),
            ({"configured_use_cache": False}, "use_cache=True"),
        )
        for kwargs, message in cases:
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(InstrumentationError, message),
            ):
                SmolVLAInstrumentation(FakeFlowModel(**kwargs))

        with self.assertRaisesRegex(InstrumentationError, r"exactly \(0, 5\)"):
            SmolVLAInstrumentation(FakeFlowModel(), capture_steps=(0, 4))

    def test_expert_patch_requires_one_explicit_step(self) -> None:
        model = FakeFlowModel().eval()
        instrumentation = SmolVLAInstrumentation(model)

        with instrumentation:
            with (
                self.assertRaisesRegex(ValueError, "one explicit denoising_step"),
                instrumentation.patch(EXPERT_EARLY, torch.ones(4)),
            ):
                pass
            with (
                self.assertRaisesRegex(TypeError, "one integer step"),
                instrumentation.patch(
                    EXPERT_EARLY, torch.ones(4), denoising_step=(0, 5)
                ),
            ):
                pass

    def test_expert_patch_mutates_residual_identity_path_and_actions(self) -> None:
        model = FakeFlowModel().eval()
        instrumentation = SmolVLAInstrumentation(model)
        rows = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        donor = torch.tensor([2.0, 3.0, 99.0, -99.0])
        recipient = torch.zeros(4)

        with torch.no_grad(), instrumentation:
            _, _, baseline_outputs = model.sample_actions(batch_size=1)
            baseline_early = instrumentation.activations(
                EXPERT_EARLY, denoising_step=0
            )[0].value.clone()
            instrumentation.clear()

            with instrumentation.patch_probe_subspace(
                EXPERT_EARLY,
                rows,
                donor,
                recipient,
                alpha=0.5,
                denoising_step=0,
            ) as shift:
                _, _, patched_outputs = model.sample_actions(batch_size=1)

            expected_shift = torch.tensor([1.0, 1.5, 0.0, 0.0])
            torch.testing.assert_close(shift, expected_shift)
            action_delta = patched_outputs[0] - baseline_outputs[0]
            expected_delta = expected_shift * (1.0 + BRANCH_SCALE) ** 12
            torch.testing.assert_close(
                action_delta,
                expected_delta[None, None, :].expand(1, 50, -1),
            )
            torch.testing.assert_close(patched_outputs[5], baseline_outputs[5])

            patched_record = instrumentation.activations(
                EXPERT_EARLY, denoising_step=0
            )[0]
            self.assertTrue(patched_record.patched)
            torch.testing.assert_close(
                patched_record.value, baseline_early + expected_shift
            )

            instrumentation.clear()
            _, _, unpatched_outputs = model.sample_actions(batch_size=1)
            torch.testing.assert_close(unpatched_outputs[0], baseline_outputs[0])

    def test_vlm_patch_changes_layer12_cache_and_downstream_actions(self) -> None:
        model = FakeFlowModel().eval()
        instrumentation = SmolVLAInstrumentation(model)
        shift = torch.tensor([1.0, 2.0, 3.0, 4.0])

        with torch.no_grad(), instrumentation:
            _, baseline_prefix_output, baseline_outputs = model.sample_actions(
                batch_size=1
            )
            baseline_record = instrumentation.activations(VLM_CONTEXT)[0]
            baseline_cache = [
                value.clone() for value in model.vlm_with_expert.last_cache["layers"]
            ]
            instrumentation.clear()

            with instrumentation.patch(VLM_CONTEXT, shift):
                _, patched_prefix_output, patched_outputs = model.sample_actions(
                    batch_size=1
                )
            patched_cache = model.vlm_with_expert.last_cache["layers"]

        for layer_index in range(12):
            torch.testing.assert_close(
                patched_cache[layer_index], baseline_cache[layer_index]
            )
        torch.testing.assert_close(patched_cache[12], baseline_cache[12] + shift)
        self.assertFalse(torch.equal(patched_cache[15], baseline_cache[15]))
        torch.testing.assert_close(
            patched_prefix_output[:, :-1], baseline_prefix_output[:, :-1]
        )
        self.assertFalse(
            torch.equal(patched_prefix_output[:, -1], baseline_prefix_output[:, -1])
        )
        self.assertFalse(torch.equal(patched_outputs[0], baseline_outputs[0]))

        patched_record = instrumentation.activations(VLM_CONTEXT)[0]
        self.assertTrue(patched_record.patched)
        torch.testing.assert_close(patched_record.value, baseline_record.value + shift)

    def test_causal_patch_requires_eval_and_no_grad(self) -> None:
        shift = torch.ones(4)

        training_model = FakeFlowModel()
        training_instrumentation = SmolVLAInstrumentation(training_model)
        with (
            torch.no_grad(),
            training_instrumentation,
            training_instrumentation.patch(VLM_CONTEXT, shift),
            self.assertRaisesRegex(InstrumentationError, "inference-only"),
        ):
            training_model.sample_actions(batch_size=1)

        grad_model = FakeFlowModel().eval()
        grad_instrumentation = SmolVLAInstrumentation(grad_model)
        with (
            grad_instrumentation,
            grad_instrumentation.patch(VLM_CONTEXT, shift),
            self.assertRaisesRegex(InstrumentationError, "torch.no_grad"),
        ):
            grad_model.sample_actions(batch_size=1)

    def test_probe_projection_computes_in_stable_precision(self) -> None:
        rows = torch.tensor([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0]], dtype=torch.float16)
        donor = torch.tensor([3.0, 1.0, 10.0], dtype=torch.float16)
        recipient = torch.zeros(3, dtype=torch.float16)

        shift = probe_subspace_shift(rows, donor, recipient, alpha=0.25)

        self.assertEqual(shift.dtype, torch.float32)
        torch.testing.assert_close(
            shift, torch.tensor([0.75, 0.25, 0.0]), rtol=1e-5, atol=1e-5
        )

        shift64 = probe_subspace_shift(
            rows.double(), donor.double(), recipient.double(), alpha=0.25
        )
        self.assertEqual(shift64.dtype, torch.float64)

    def test_probe_projection_rejects_rank_deficient_rows(self) -> None:
        rows = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "rank-two"):
            probe_subspace_shift(rows, torch.ones(3), torch.zeros(3))

    def test_minimum_norm_circular_probe_shift_hits_rotated_raw_target(self) -> None:
        rows = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        activation = torch.tensor([1.5, 2.5, 7.0])
        feature_center = torch.tensor([0.5, 0.5, 0.0])
        target_center = torch.tensor([0.0, 0.0])

        shift = minimum_norm_circular_probe_shift(
            rows,
            activation,
            feature_center,
            target_center,
            scaled_angle_radians=math.pi / 2,
        )
        raw = (activation - feature_center) @ rows.mT + target_center
        moved = (activation + shift - feature_center) @ rows.mT + target_center
        expected = torch.tensor([-raw[1], raw[0]])
        torch.testing.assert_close(moved, expected)
        self.assertEqual(float(shift[2]), 0.0)

    def test_minimum_norm_circular_probe_shift_rejects_zero_resultant(self) -> None:
        rows = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "zero probe resultant"):
            minimum_norm_circular_probe_shift(
                rows,
                torch.zeros(2),
                torch.zeros(2),
                torch.zeros(2),
                scaled_angle_radians=0.1,
            )


if __name__ == "__main__":
    unittest.main()
