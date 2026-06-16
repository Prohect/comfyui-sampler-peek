# comfyui-sampler-peek

**Peek into ComfyUI sampling loops.** Decode intermediate latents using a VAE
during `SamplerCustomAdvanced` execution, controlled by math expressions.

![category: sampling](https://img.shields.io/badge/category-sampling-blue)

## Overview

Ever wanted to see what your image looks like *during* the sampling process?
This extension lets you do exactly that — and more.

It replaces `SamplerCustomAdvanced` with `SamplerPeekAdvanced`, which wraps
the same sampling loop but adds a step-by-step callback. At steps matching a
configurable math expression, the current latent (x₀ prediction) is decoded
through a VAE and collected into an image batch. Step indices are recorded
alongside the images.

## Nodes

### SamplerPeekAdvanced (sampling/peek)
A drop-in replacement for `SamplerCustomAdvanced`. It accepts all the same
inputs, plus a VAE and decode configuration.

| Input          | Type     | Description                                   |
|----------------|----------|-----------------------------------------------|
| noise          | NOISE    | Noise source                                  |
| guider         | GUIDER   | Guider (e.g., CFGGuider)                      |
| sampler        | SAMPLER  | Sampler to use                                |
| sigmas         | SIGMAS   | Sigmas schedule                               |
| latent_image   | LATENT   | Latent image to sample from                   |
| vae            | VAE      | VAE model for decoding intermediate latents   |
| decode_expr    | STRING   | Expression for when to decode (see below)     |
| start_step     | INT      | Only decode at ≥ this step (1-based)          |
| end_step       | INT      | Only decode at ≤ this step (0 = no limit)     |
| max_previews   | INT      | Max previews to store in RAM (0 = unlimited) |

| Output           | Type   | Description                                    |
|------------------|--------|------------------------------------------------|
| output           | LATENT | Final sampled latent                           |
| denoised_output  | LATENT | Final x₀ prediction                           |
| preview_images   | IMAGE  | Batch of decoded preview images                |
| step_indices     | INT    | Step indices (1-based) for each preview image  |

### StepExpression (sampling/peek)
Evaluate a math expression with step context. Returns both the float value
and boolean truth.

| Input       | Type     | Description                          |
|-------------|----------|--------------------------------------|
| step_index  | INT      | Current step index (1-based)         |
| total_steps | INT      | Total number of steps                |
| expression  | STRING   | Math expression to evaluate          |

### PeekStepIndex (sampling/peek)
Extracts a scalar step index from the tensor output of `SamplerPeekAdvanced`.

### PeekConditionGate (sampling/peek)
A conditional multiplexer: passes `true_value` when condition is True,
`false_value` otherwise.

## Expression Syntax

The `decode_expr` and `StepExpression` nodes use a safe math expression
evaluator. Available variables and functions:

| Variable    | Description           |
|-------------|-----------------------|
| `step`      | Current step (1..n)   |
| `n`         | Total steps           |
| `total_steps` | Total steps (alias)  |

| Function    | Description            |
|-------------|------------------------|
| `sin`, `cos`, `tan` | Trig functions  |
| `sqrt`, `log`, `exp` | Math functions  |
| `ceil`, `floor`      | Rounding        |
| `clamp(x, lo, hi)`   | Clamp value     |
| `lerp(a, b, t)`      | Linear interpolate |

**Examples:**

- `step % 5 == 0` — decode every 5th step
- `step >= 10 and step <= 20` — decode steps 10–20
- `step == 1 or step == n` — decode first and last step
- `sin(step / n * pi)` — sine ramp for dynamic schedules
- `step % 2 == 0` — decode even-numbered steps

## Installation

1. Clone into ComfyUI's `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/Prohect/comfyui-sampler-peek.git
   ```
2. Restart ComfyUI.

No additional dependencies required — it uses only ComfyUI's built-in
libraries (PyTorch, etc.).

## Usage

1. Replace `SamplerCustomAdvanced` with `SamplerPeekAdvanced` in your workflow.
2. Connect a VAE to the `vae` input.
3. Set `decode_expr` to control which steps produce previews.
4. Connect `preview_images` to a `SaveImage` or `PreviewImage` node.
5. Queue and watch the sampling process unfold!

## License

MIT
