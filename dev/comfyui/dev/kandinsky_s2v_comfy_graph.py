"""The example graph in ComfyUI's API format."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kandinsky_s2v_comfy.contract import PROMPT  # noqa: E402

# Installed into ComfyUI's input/ on first load (example_inputs.py).
# Each workflow opens with its own reference frame and speech track.
EXAMPLE_INPUTS = {
    "lite": ("kandinsky_s2v_reference.png", "kandinsky_s2v_speech.wav"),
    "pro": ("kandinsky_s2v_pro_reference.png", "kandinsky_s2v_pro_speech.wav"),
}
# The DiT file of each released model, in models/diffusion_models.
DIT_FILES = {"lite": "kandinsky_s2v_lite_distill.safetensors", "pro": "kandinsky_s2v_pro_distill.safetensors"}


def api_prompt(  # noqa: PLR0913
    image: str | None = None,
    audio: str | None = None,
    chunks: int = 0,
    prompt: str | None = None,
    model: str = "lite",
) -> dict:
    default_image, default_audio = EXAMPLE_INPUTS[model]
    image, audio = image or default_image, audio or default_audio
    return {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": DIT_FILES[model], "weight_dtype": "default"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": "hunyuan_video_vae_bf16.safetensors"}},
        "3": {"class_type": "KandinskyS2VEncoderLoader", "inputs": {"omni": "Qwen2.5-Omni-7B"}},
        "14": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "clip_l.safetensors", "type": "stable_diffusion", "device": "default"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image}},
        "5": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "6": {"class_type": "KandinskyS2VReferenceImage", "inputs": {"image": ["4", 0]}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["2", 0]}},
        "8": {"class_type": "KandinskyS2VEncode", "inputs": {
            "encoder": ["3", 0], "clip": ["14", 0], "audio": ["5", 0], "prompt": prompt or PROMPT, "chunks": chunks}},
        "9": {"class_type": "KandinskyS2VSampler", "inputs": {
            "model": ["1", 0], "conditioning": ["8", 0], "reference_latent": ["7", 0], "seed": 5, "steps": 16}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["2", 0]}},
        "11": {"class_type": "KandinskyS2VTrimToAudio", "inputs": {"images": ["10", 0], "conditioning": ["8", 0]}},
        "12": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "fps": ["11", 1], "audio": ["5", 0]}},
        "13": {"class_type": "SaveVideo", "inputs": {
            "video": ["12", 0], "filename_prefix": "kandinsky_s2v/S2V", "format": "mp4", "format.codec": "h264"}},
    }
