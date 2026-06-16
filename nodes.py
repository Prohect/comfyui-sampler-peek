"""
Core nodes for comfyui-sampler-peek.

Provides:
  - SamplerPeekAdvanced: Wraps SamplerCustomAdvanced with step-by-step latent decoding.
  - StepExpression: Evaluates math expressions in the context of a sampling step.
  - PeekStepIndex: Outputs step index for use in downstream expression nodes.
"""

from typing import Any

import torch

import comfy.latent_formats
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict

from .expressions import evaluate_condition, evaluate_number

# ──────────────────────────────────────────────────────────────
#  SamplerPeekAdvanced
# ──────────────────────────────────────────────────────────────


class SamplerPeekAdvanced(ComfyNodeABC):
    """
    A drop-in replacement for SamplerCustomAdvanced that can decode
    intermediate latents during sampling using a VAE.

    At steps matching the decode_expression, the current latent (x0
    prediction) is decoded through the VAE and collected into an image
    batch.  Step indices are also recorded.

    Inputs:
        noise:        NOISE source
        guider:       GUIDER (e.g., CFGGuider)
        sampler:      SAMPLER
        sigmas:       SIGMAS schedule
        latent_image: LATENT to sample from
        vae:          VAE model for decoding
        decode_expr:  STRING expression; when True, decode at this step.
                      Variables: step (1..n), n (total steps).
                      Default: all steps (empty string).
        start_step:   INT (1-based). Only decode at or after this step.
        end_step:     INT (1-based). Only decode at or before this step.
                      Set to 0 for no upper bound.  Default: 0.
        max_previews: INT. Maximum number of previews to store in RAM.
                      Default: 0 (unlimited).

    Outputs:
        output:          LATENT   — final sampled latent
        denoised_output: LATENT   — final x0 prediction
        preview_images:  IMAGE    — batch of decoded preview images
        step_indices:    INT      — step indices corresponding to each preview
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "noise": (IO.NOISE, {"tooltip": "Noise source for sampling"}),
                "guider": (IO.GUIDER, {"tooltip": "Guider (e.g., CFGGuider)"}),
                "sampler": (IO.SAMPLER, {"tooltip": "Sampler to use"}),
                "sigmas": (IO.SIGMAS, {"tooltip": "Sigmas schedule"}),
                "latent_image": (IO.LATENT, {"tooltip": "Latent image to sample from"}),
                "vae": (
                    IO.VAE,
                    {"tooltip": "VAE model for decoding intermediate latents"},
                ),
                "decode_expr": (
                    IO.STRING,
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": (
                            "Expression that determines when to decode. "
                            "Variables: step (1..n), n (total steps). "
                            "e.g., 'step % 5 == 0'. Empty = every step."
                        ),
                    },
                ),
                "start_step": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10000,
                        "step": 1,
                        "tooltip": "Only decode at or after this step (1-based)",
                    },
                ),
                "end_step": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "step": 1,
                        "tooltip": "Only decode at or before this step. 0 = no limit",
                    },
                ),
                "max_previews": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10000,
                        "step": 1,
                        "tooltip": "Max previews to store in RAM. 0 = unlimited",
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.LATENT, IO.LATENT, IO.IMAGE, IO.INT)
    RETURN_NAMES = ("output", "denoised_output", "preview_images", "step_indices")
    OUTPUT_TOOLTIPS = (
        "Final sampled latent",
        "Final x0 prediction",
        "Batch of decoded preview images from intermediate steps",
        "Step indices (1-based) for each preview image",
    )
    FUNCTION = "sample"
    CATEGORY = "sampling/peek"
    DESCRIPTION = (
        "Extended SamplerCustomAdvanced that decodes intermediate latents "
        "using a VAE at specified steps.  Useful for visualizing the "
        "sampling process or creating step-by-step animations."
    )

    def sample(
        self,
        noise: Any,
        guider: Any,
        sampler: Any,
        sigmas: torch.Tensor,
        latent_image: dict,
        vae: Any,
        decode_expr: str = "",
        start_step: int = 1,
        end_step: int = 0,
        max_previews: int = 0,
    ):
        # ── prepare latent ──────────────────────────────────
        latent = latent_image
        latent_samples = latent["samples"]
        latent = latent.copy()
        latent_samples = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            latent_samples,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        latent["samples"] = latent_samples

        noise_mask = latent.get("noise_mask", None)

        # ── setup ────────────────────────────────────────────
        total_steps = sigmas.shape[-1] - 1
        if end_step <= 0:
            effective_end = total_steps
        else:
            effective_end = min(end_step, total_steps)
        effective_start = max(start_step, 1)

        # Accumulators for previews
        preview_images: list[torch.Tensor] = []
        step_indices: list[int] = []

        # Shared dict for x0 output from the original callback
        x0_output: dict = {}

        # ── build chained callback ───────────────────────────
        original_callback = latent_preview.prepare_callback(
            guider.model_patcher, total_steps, x0_output
        )

        def peek_callback(step: int, x0: torch.Tensor, x: torch.Tensor, total: int):
            """Called at each sampling step.  Decodes x0 when conditions met."""
            # Always forward to the original preview callback
            if original_callback is not None:
                original_callback(step, x0, x, total)

            current_step = step + 1  # 1-based

            # Check range bounds
            if current_step < effective_start or current_step > effective_end:
                return

            # Check max limit
            if max_previews > 0 and len(preview_images) >= max_previews:
                return

            # Check expression
            try:
                should_decode = evaluate_condition(decode_expr, current_step, total)
            except Exception:
                # If expression is invalid, skip this step
                return

            if not should_decode:
                return

            # Decode x0 through VAE
            try:
                # x0 is the predicted clean image at this step
                # Move to appropriate device and decode
                x0_decoded = vae.decode(x0[:1])
                preview_images.append(x0_decoded)
                step_indices.append(current_step)
            except Exception:
                # If decoding fails, skip this step silently
                pass

        # ── run sampling ─────────────────────────────────────
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(
            noise.generate_noise(latent),
            latent_samples,
            sampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=peek_callback,
            disable_pbar=disable_pbar,
            seed=noise.seed,
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        # ── build outputs ────────────────────────────────────
        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples

        if "x0" in x0_output:
            x0_out = guider.model_patcher.model.process_latent_out(
                x0_output["x0"].cpu()
            )
            if samples.is_nested:
                from comfy import nested_tensor

                latent_shapes = [s.shape for s in samples.unbind()]
                x0_out = nested_tensor.NestedTensor(
                    comfy.utils.unpack_latents(x0_out, latent_shapes)
                )
            out_denoised = latent.copy()
            out_denoised["samples"] = x0_out
        else:
            out_denoised = out

        # ── assemble preview image batch ─────────────────────
        if len(preview_images) > 0:
            preview_batch = torch.cat(preview_images, dim=0)
        else:
            # Return a 1x1 black image as placeholder
            preview_batch = torch.zeros(1, 1, 1, 3, device="cpu", dtype=torch.float32)

        # Step indices as a 1D int tensor
        step_tensor = torch.tensor(
            step_indices if step_indices else [0],
            dtype=torch.int32,
            device="cpu",
        )

        return (out, out_denoised, preview_batch, step_tensor)


# ──────────────────────────────────────────────────────────────
#  StepExpression
# ──────────────────────────────────────────────────────────────


class StepExpression(ComfyNodeABC):
    """
    Evaluate a math expression in the context of a sampling step.

    Takes step index and total steps, evaluates an expression,
    and returns a float value and boolean condition.

    Inputs:
        step_index:  INT — current step index (1-based)
        total_steps: INT — total number of steps
        expression:  STRING — math expression. Variables: step, n.

    Outputs:
        value:     FLOAT   — evaluated expression as float
        condition: BOOLEAN — boolean truth of the expression
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "step_index": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "Current step index (1-based)",
                    },
                ),
                "total_steps": (
                    IO.INT,
                    {
                        "default": 20,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "Total number of sampling steps",
                    },
                ),
                "expression": (
                    IO.STRING,
                    {
                        "multiline": False,
                        "default": "step % 5 == 0",
                        "tooltip": (
                            "Math expression to evaluate. "
                            "Variables: step (current), n (total steps). "
                            "Returns both float value and bool condition."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.FLOAT, IO.BOOLEAN)
    RETURN_NAMES = ("value", "condition")
    OUTPUT_TOOLTIPS = (
        "Evaluated expression as a float",
        "Boolean truth of the expression",
    )
    FUNCTION = "evaluate"
    CATEGORY = "sampling/peek"
    DESCRIPTION = (
        "Evaluate a math expression in step context. "
        "Useful for creating step-dependent schedules or conditions."
    )

    def evaluate(self, step_index: int, total_steps: int, expression: str):
        from .expressions import evaluate_expression

        result = evaluate_expression(expression, step_index, total_steps)
        return (float(result), bool(result))


# ──────────────────────────────────────────────────────────────
#  PeekStepIndex
# ──────────────────────────────────────────────────────────────


class PeekStepIndex(ComfyNodeABC):
    """
    Simple pass-through node for step index from SamplerPeekAdvanced.

    Use this to route step indices to downstream nodes that need
    step-dependent evaluation during sampling.

    Inputs:
        step_indices: INT tensor from SamplerPeekAdvanced

    Outputs:
        step_index: INT — the first step index value (for scalar use)
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "step_indices": (
                    IO.INT,
                    {
                        "tooltip": "Step indices tensor from SamplerPeekAdvanced",
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.INT,)
    RETURN_NAMES = ("step_index",)
    FUNCTION = "passthrough"
    CATEGORY = "sampling/peek"
    DESCRIPTION = "Extracts a scalar step index from a step indices tensor."

    def passthrough(self, step_indices):
        if isinstance(step_indices, torch.Tensor):
            if step_indices.numel() > 0:
                val = int(step_indices.flatten()[0].item())
            else:
                val = 0
        elif isinstance(step_indices, (list, tuple)):
            val = int(step_indices[0]) if len(step_indices) > 0 else 0
        else:
            val = int(step_indices)
        return (val,)


# ──────────────────────────────────────────────────────────────
#  PeekConditionGate
# ──────────────────────────────────────────────────────────────


class PeekConditionGate(ComfyNodeABC):
    """
    A conditional gate: passes through inputs only when the condition
    is true.  Useful for step-dependent control flow in sampling.

    When condition is false, outputs a default/empty value.

    Inputs:
        condition:  BOOLEAN — gate condition
        true_value: FLOAT  — value to pass through when condition is True
        false_value: FLOAT — value to pass through when condition is False

    Outputs:
        value: FLOAT — the selected value
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "condition": (
                    IO.BOOLEAN,
                    {"tooltip": "Gate condition"},
                ),
                "true_value": (
                    IO.FLOAT,
                    {
                        "default": 1.0,
                        "min": -10000.0,
                        "max": 10000.0,
                        "tooltip": "Value when condition is True",
                    },
                ),
                "false_value": (
                    IO.FLOAT,
                    {
                        "default": 0.0,
                        "min": -10000.0,
                        "max": 10000.0,
                        "tooltip": "Value when condition is False",
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.FLOAT,)
    RETURN_NAMES = ("value",)
    FUNCTION = "gate"
    CATEGORY = "sampling/peek"
    DESCRIPTION = (
        "Passes through true_value when condition is True, "
        "false_value otherwise.  Useful for step-dependent routing."
    )

    def gate(self, condition: bool, true_value: float, false_value: float):
        return (true_value if condition else false_value,)


# ──────────────────────────────────────────────────────────────
#  Registry
# ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "SamplerPeekAdvanced": SamplerPeekAdvanced,
    "StepExpression": StepExpression,
    "PeekStepIndex": PeekStepIndex,
    "PeekConditionGate": PeekConditionGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerPeekAdvanced": "Sampler Peek (Advanced)",
    "StepExpression": "Step Expression",
    "PeekStepIndex": "Peek Step Index",
    "PeekConditionGate": "Peek Condition Gate",
}
