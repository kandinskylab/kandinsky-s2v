"""Register Kandinsky S2V 1.0 as a ComfyUI model type and checkpoint detector.

The stock ``UNETLoader`` then loads both published layouts unchanged: the
native ``kandinsky_s2v_{lite,pro}_distill.safetensors`` and the Diffusers bundle's
``transformer/diffusion_pytorch_model.safetensors``. The Diffusers port keeps
every training parameter name, so the two files hold the same keys and need no
renaming here.
"""

from __future__ import annotations

import math
from functools import wraps

import comfy.model_detection
import comfy.supported_models

from .contract import DIT_DEFAULTS, RELEASE_DIT_CONFIGS
from .supported_model import IMAGE_MODEL, KandinskyS2V

_DETECTOR_MARKER = "_comfyui_kandinsky_s2v_detector"
_REQUIRED_KEYS = (
    "mask_embeddings.weight",
    "audio_embeddings.in_layer.weight",
    "visual_embeddings.in_layer.weight",
    "visual_embeddings_proj4x.in_layer.weight",
    "time_embeddings.out_layer.weight",
    "pooled_text_embeddings.in_layer.weight",
    "text_embeddings.in_layer.weight",
    "audio_transformer_blocks.0.self_attention.to_query.weight",
    "visual_transformer_blocks.0.visual_modulation.out_layer.weight",
    "out_layer.out_layer.weight",
)
# RoPE axes per attention head width; head_dim = sum(axes_dims).
_AXES_DIMS = {64: (16, 24, 24), 128: (32, 48, 48)}


def _count(state_dict, prefix: str, stem: str, leaf: str) -> int:
    index = 0
    while f"{prefix}{stem}.{index}.{leaf}" in state_dict:
        index += 1
    return index


def detect_s2v_config(state_dict, prefix: str) -> dict:
    """The DiT configuration, read off the tensor shapes."""

    def shape(name):
        return state_dict[f"{prefix}{name}"].shape

    patch_size = tuple(DIT_DEFAULTS["patch_size"])
    patch_volume = math.prod(patch_size)
    visual_input = shape("visual_embeddings.in_layer.weight")[1]
    if visual_input % patch_volume:
        raise ValueError(
            f"Kandinsky S2V visual input projection {visual_input} is not divisible by the patch volume {patch_volume}"
        )
    in_visual_dim = visual_input // patch_volume
    out_dim = shape("out_layer.out_layer.weight")[0] // patch_volume
    if out_dim % in_visual_dim:
        raise ValueError(f"Kandinsky S2V output head width {out_dim} is not a multiple of {in_visual_dim} channels")
    head_dim = shape("visual_transformer_blocks.0.self_attention.query_norm.weight")[0]
    if head_dim not in _AXES_DIMS:
        raise ValueError(f"Kandinsky S2V checkpoint has an unreleased head width {head_dim}")
    num_visual_blocks = _count(state_dict, prefix, "visual_transformer_blocks", "self_attention.to_query.weight")

    return {
        "image_model": IMAGE_MODEL,
        **DIT_DEFAULTS,
        "in_visual_dim": in_visual_dim,
        "n_grid": out_dim // in_visual_dim,
        "in_text_dim": shape("text_embeddings.in_layer.weight")[1],
        "in_text_dim2": shape("pooled_text_embeddings.in_layer.weight")[1],
        "in_audio_dim": shape("audio_embeddings.in_layer.weight")[1],
        "time_dim": shape("time_embeddings.out_layer.weight")[0],
        "model_dim": shape("visual_embeddings.in_layer.weight")[0],
        "ff_dim": shape("visual_transformer_blocks.0.feed_forward.in_layer.weight")[0],
        "num_text_blocks": _count(state_dict, prefix, "text_transformer_blocks", "self_attention.to_query.weight"),
        "num_audio_blocks": _count(state_dict, prefix, "audio_transformer_blocks", "self_attention.to_query.weight"),
        "num_visual_blocks": num_visual_blocks,
        "axes_dims": _AXES_DIMS[head_dim],
        "audio_inject_layers": tuple(
            i
            for i in range(num_visual_blocks)
            if f"{prefix}visual_transformer_blocks.{i}.cross_attention_v2a.to_query.weight" in state_dict
        ),
    }


def check_release(config: dict) -> str:
    """The released model this configuration is; refuse any other.

    Lite and Pro (not distilled) load as the same model type but need another
    sampler, so they fail here, with every mismatch against the closest release
    named, rather than halfway through sampling.
    """
    mismatches = {
        release: [
            f"{name}: checkpoint={config[name]!r}, release={expected!r}"
            for name, expected in expected_config.items()
            if config[name] != expected
        ]
        for release, expected_config in RELEASE_DIT_CONFIGS.items()
    }
    closest = min(mismatches, key=lambda release: len(mismatches[release]))
    if mismatches[closest]:
        raise ValueError(
            "This checkpoint is not Kandinsky S2V 1.0 Lite distilled or Pro distilled, the models these nodes "
            f"run (closest: {closest}; {'; '.join(mismatches[closest])})."
        )
    return closest


def _register_model(model_class) -> None:
    # Dynamic custom-node reloads create a new class object. Replace our old
    # registration in place instead of accumulating duplicate model entries.
    for index, model in enumerate(comfy.supported_models.models):
        if model.__module__ == model_class.__module__ and model.__name__ == model_class.__name__:
            comfy.supported_models.models[index] = model_class
            return
    comfy.supported_models.models.append(model_class)


def register() -> None:
    _register_model(KandinskyS2V)

    current_detector = comfy.model_detection.detect_unet_config
    if getattr(current_detector, _DETECTOR_MARKER, False):
        return

    @wraps(current_detector)
    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        if all(f"{key_prefix}{name}" in state_dict for name in _REQUIRED_KEYS):
            config = detect_s2v_config(state_dict, key_prefix)
            check_release(config)
            return config
        return current_detector(state_dict, key_prefix, *args, **kwargs)

    setattr(detect_unet_config, _DETECTOR_MARKER, True)
    comfy.model_detection.detect_unet_config = detect_unet_config
