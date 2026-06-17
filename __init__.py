"""
comfyui-sampler-peek: Peek into the sampling process.

Decode latents at intermediate steps during SamplerCustomAdvanced execution
using a given VAE, controlled by math expressions.

Optionally supports step-dependent CFG modulation for dynamic guidance.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

VERSION = "0.2.0"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "VERSION"]
