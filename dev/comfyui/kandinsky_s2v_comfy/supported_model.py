"""ComfyUI supported-model registration for Kandinsky S2V 1.0."""

from __future__ import annotations

import torch

import comfy.latent_formats
import comfy.supported_models_base

from .contract import PIFLOW_SHIFT
from .model_base import KandinskyS2VModel

IMAGE_MODEL = "kandinsky_s2v"


class KandinskyS2V(comfy.supported_models_base.BASE):
    # Matched against the unet_config produced by our detector (register.py).
    unet_config = {"image_model": IMAGE_MODEL}
    sampling_settings = {"shift": PIFLOW_SHIFT}
    latent_format = comfy.latent_formats.HunyuanVideo
    memory_usage_factor = 1.0
    # Weights are bf16 apart from fp32 norms and time embedding; the forward
    # runs under bf16 autocast either way.
    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    def get_model(self, state_dict, prefix="", device=None):
        return KandinskyS2VModel(self, device=device)

    def clip_target(self, state_dict=None):
        # The speech and text encoder is Qwen2.5-Omni, which ComfyUI does not
        # carry; it is loaded by the Kandinsky S2V Encoder Loader node.
        return None
