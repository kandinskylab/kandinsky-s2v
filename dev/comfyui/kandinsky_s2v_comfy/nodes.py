"""Kandinsky S2V 1.0 nodes for ComfyUI.

The graph keeps stock nodes wherever the model allows:

    UNETLoader ─────────────────────────────────────────────────────┐
    LoadImage → S2V Reference Image → VAEEncode ────────────────────┤
    LoadAudio ─┬→ S2V Encode (← S2V Encoder Loader, ← Load CLIP) ───┼→ S2V Sampler → VAEDecode
               │                                                                       │
               └──────────────────→ CreateVideo ← S2V Trim To Audio ←──────────────────┘

Only what ComfyUI cannot express itself is custom: the Qwen2.5-Omni speech and
text encoder, the aspect-ratio bucket of the reference frame, the chunked
π-Flow sampler with its FramePack memory, and trimming the padded tail. CLIP-L
is ComfyUI's own, from Load CLIP.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
from PIL import Image

import comfy.model_management as mm
import comfy.model_patcher
import folder_paths
import latent_preview

from . import contract
from .encoder import S2VConditioningEncoder, is_omni_directory, resample_speech
from .model_base import KandinskyS2VModel
from .sampling import PiFlowSettings, sample_clip

CATEGORY = "Kandinsky S2V"
ENCODER_TYPE = "KANDINSKY_S2V_ENCODER"
CONDITIONING_TYPE = "KANDINSKY_S2V_CONDITIONING"


def _text_encoder_entries(accept_dir) -> list[str]:
    entries = set()
    for root in folder_paths.get_folder_paths("text_encoders"):
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path) and accept_dir(path):
                entries.add(name)
    return sorted(entries)


def _text_encoder_path(name: str) -> str:
    for root in folder_paths.get_folder_paths("text_encoders"):
        path = os.path.join(root, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{name} is not in any text_encoders folder")


def clip_l_pooled(clip, text: str) -> torch.Tensor:
    """The pooled CLIP-L embedding of ``text``, as the reference computes it.

    The reference truncates the prompt at 77 tokens and takes the unprojected
    pooled output. ComfyUI's tokenizer would instead split a long prompt into
    windows at word boundaries and parse ``(emphasis:1.2)``, so the token ids
    are built here with its CLIP tokenizer directly.
    """
    tokenizer = getattr(clip.tokenizer, "clip_l", None)
    if tokenizer is None or type(clip.cond_stage_model).__name__ != "SD1ClipModel":
        raise ValueError(
            "Kandinsky S2V needs CLIP-L alone: load clip_l.safetensors with Load CLIP, type stable_diffusion"
        )
    ids = tokenizer.tokenizer(text, truncation=True, max_length=contract.CLIP_MAX_LENGTH)["input_ids"]
    ids += [tokenizer.pad_token] * (contract.CLIP_MAX_LENGTH - len(ids))
    _, pooled = clip.encode_from_tokens({"l": [[(token, 1.0) for token in ids]]}, return_pooled="unprojected")
    return pooled


class KandinskyS2VEncoderLoader:
    @classmethod
    def INPUT_TYPES(cls):
        omni = _text_encoder_entries(is_omni_directory) or ["Qwen2.5-Omni-7B"]
        return {
            "required": {
                "omni": (omni, {"tooltip": "Qwen2.5-Omni-7B directory under models/text_encoders."}),
            }
        }

    RETURN_TYPES = (ENCODER_TYPE,)
    RETURN_NAMES = ("encoder",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Loads the speech and text encoder of Kandinsky S2V, the Qwen2.5-Omni Thinker (its Talker stays on disk). "
        "CLIP-L comes from the stock Load CLIP."
    )

    def load(self, omni):
        encoder = S2VConditioningEncoder(_text_encoder_path(omni))
        patcher = comfy.model_patcher.ModelPatcher(
            encoder,
            load_device=mm.text_encoder_device(),
            offload_device=mm.text_encoder_offload_device(),
        )
        return (patcher,)


class KandinskyS2VEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encoder": (ENCODER_TYPE,),
                "clip": ("CLIP", {"tooltip": "CLIP-L from Load CLIP (clip_l.safetensors, type stable_diffusion)."}),
                "audio": ("AUDIO", {"tooltip": "The speech that drives the avatar."}),
                "prompt": ("STRING", {"multiline": True, "default": contract.PROMPT}),
                "chunks": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 60,
                        "tooltip": (
                            f"Number of {contract.CHUNK_SECONDS:g}-second chunks. 0 covers the whole speech track; "
                            "a smaller number generates only the beginning of it."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = (CONDITIONING_TYPE,)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Encodes the prompt and the speech track. The track is padded with silence to whole "
        f"{contract.CHUNK_SECONDS:g}-second chunks; the padded tail is trimmed after decoding."
    )

    def encode(self, encoder, clip, audio, prompt, chunks):
        waveform, sample_rate = audio["waveform"], int(audio["sample_rate"])
        if waveform.shape[0] != 1:
            raise ValueError(f"Kandinsky S2V takes one speech track, got a batch of {waveform.shape[0]}")
        track_seconds = waveform.shape[-1] / sample_rate
        if track_seconds <= 0:
            raise ValueError("The speech track is empty")
        needed = max(1, math.ceil(track_seconds / contract.CHUNK_SECONDS))
        chunks = needed if chunks == 0 else chunks
        target_seconds = chunks * contract.CHUNK_SECONDS

        model: S2VConditioningEncoder = encoder.model
        if target_seconds > model.max_audio_seconds + 0.01:
            raise ValueError(
                f"{chunks} chunks need {target_seconds:g} s of speech, but the Qwen2.5-Omni feature extractor "
                f"truncates at {model.max_audio_seconds:g} s; trim the track or set fewer chunks"
            )

        pooled = clip_l_pooled(clip, prompt)
        speech = resample_speech(waveform[0], sample_rate, target_seconds)
        mm.load_models_gpu([encoder], force_full_load=True)
        text_embeds, audio_embeds = model.encode(prompt, speech, encoder.load_device)
        # The DiT's text RoPE table ends here; past it the sampler would hit a CUDA
        # device-side assert, which takes the whole ComfyUI process down.
        max_tokens = contract.DIT_DEFAULTS["text_rope_max_pos"]
        if text_embeds.shape[0] > max_tokens:
            raise ValueError(
                f"The prompt is {text_embeds.shape[0]} Qwen2.5-Omni tokens long; Kandinsky S2V takes at most "
                f"{max_tokens}. Shorten the prompt."
            )

        out_device = mm.intermediate_device()
        conditioning = {
            "text_embeds": text_embeds.to(out_device, torch.bfloat16),
            "pooled_embed": pooled.to(out_device, torch.bfloat16),
            "audio_embeds": audio_embeds.to(out_device, torch.bfloat16),
            "chunks": chunks,
            "track_seconds": min(track_seconds, target_seconds),
        }
        return (conditioning,)


def _crop_and_resize(image: Image.Image, height: int, width: int) -> Image.Image:
    """Resize to cover, then center-crop, as the reference preprocesses the frame."""
    from torchvision.transforms import InterpolationMode  # noqa: PLC0415
    from torchvision.transforms import functional as TF  # noqa: N812, PLC0415

    scale = max(width / image.width, height / image.height)
    image = TF.resize(
        image,
        (round(image.height * scale), round(image.width * scale)),
        interpolation=InterpolationMode.BILINEAR,
    )
    return TF.center_crop(image, (height, width))


class KandinskyS2VReferenceImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "prepare"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Fits the reference frame to the closest of the trained resolutions (512x512, 512x768 or 768x512, "
        "by aspect ratio): resize to cover, then center-crop. Encode the result with the stock VAE Encode."
    )

    def prepare(self, image):
        frame = image[0]
        source_height, source_width = frame.shape[0], frame.shape[1]
        aspect = source_width / source_height
        height, width = min(contract.RESOLUTIONS, key=lambda res: abs(aspect - res[1] / res[0]))

        pixels = (frame[..., :3].cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        fitted = _crop_and_resize(Image.fromarray(pixels), height, width)
        out = torch.from_numpy(np.asarray(fitted).astype(np.float32) / 255.0).unsqueeze(0)
        return (out,)


def _reference_latent(model, reference_latent) -> torch.Tensor:
    """Stock VAE Encode output → the scaled ``(1, H, W, C)`` latent the DiT takes."""
    samples = reference_latent["samples"]
    if getattr(samples, "is_nested", False):
        raise ValueError("Kandinsky S2V expects the LATENT of one encoded reference frame")
    if samples.ndim == 4:  # noqa: PLR2004
        samples = samples.unsqueeze(2)
    if samples.ndim != 5 or samples.shape[0] != 1 or samples.shape[2] != 1:  # noqa: PLR2004
        raise ValueError(
            "Kandinsky S2V expects one encoded reference frame shaped [1, C, 1, H, W], "
            f"got {tuple(samples.shape)}. Encode a single image with the HunyuanVideo VAE."
        )
    size = (samples.shape[3] * contract.SPATIAL_COMPRESSION, samples.shape[4] * contract.SPATIAL_COMPRESSION)
    if size not in contract.RESOLUTIONS:
        # Outside the trained buckets the RoPE tables can run out (2048 px), and
        # on CUDA that is a device-side assert that takes ComfyUI down with it.
        trained = ", ".join(f"{w}x{h}" for h, w in contract.RESOLUTIONS)
        raise ValueError(
            f"Kandinsky S2V expects a reference frame of {trained}, got {size[1]}x{size[0]}. "
            "Fit the image with Kandinsky S2V Reference Image before VAE Encode."
        )
    dit = model.model.diffusion_model
    if samples.shape[1] != dit.in_visual_dim:
        raise ValueError(
            f"Kandinsky S2V expects {dit.in_visual_dim} latent channels, got {samples.shape[1]}. "
            "Encode the reference frame with the HunyuanVideo VAE."
        )
    scaled = model.model.latent_format.process_in(samples.float())
    return scaled[:, :, 0].permute(0, 2, 3, 1)


def _memory_estimate(dit, latent_height: int, latent_width: int) -> int:
    tokens_per_frame = (latent_height // dit.patch_size[1]) * (latent_width // dit.patch_size[2])
    tokens = tokens_per_frame * (contract.LATENT_FRAMES + 1 + sum(dit.frame_buckets))
    return tokens * dit.ff_dim * 2 * 4


class KandinskyS2VSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": (CONDITIONING_TYPE,),
                "reference_latent": (
                    "LATENT",
                    {"tooltip": "The reference frame from Kandinsky S2V Reference Image, encoded by VAE Encode."},
                ),
                "seed": (
                    "INT",
                    {"default": contract.SEED, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True},
                ),
                "steps": (
                    "INT",
                    {
                        "default": contract.PIFLOW_STEPS,
                        "min": 1,
                        "max": 100,
                        "tooltip": "π-Flow segments per chunk, one model call each. The model was distilled at 16.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Generates the clip chunk by chunk with the π-Flow sampler of the distilled model; each chunk "
        "continues from the last frames of the previous one. Decode the result with the stock VAE Decode."
    )

    def sample(self, model, conditioning, reference_latent, seed, steps):
        if not isinstance(model.model, KandinskyS2VModel):
            raise ValueError("Kandinsky S2V Sampler needs a Kandinsky S2V model from Load Diffusion Model")
        dit = model.model.diffusion_model
        if dit.n_grid < 2:  # noqa: PLR2004
            raise ValueError(
                "This checkpoint is not a π-Flow distilled one; "
                "only Kandinsky S2V 1.0 Lite distilled and Pro distilled are supported"
            )

        ref = _reference_latent(model, reference_latent)
        mm.load_models_gpu([model], memory_required=_memory_estimate(dit, ref.shape[1], ref.shape[2]))
        device = model.load_device
        if device.type != "cuda":
            raise RuntimeError("Kandinsky S2V runs on a CUDA device")

        chunks = conditioning["chunks"]
        total_steps = chunks * steps
        preview = latent_preview.prepare_callback(model, total_steps)

        def step_callback(chunk, step, x0, x):
            mm.throw_exception_if_processing_interrupted()
            global_step = chunk * steps + step
            preview(global_step, x0.permute(3, 0, 1, 2).unsqueeze(0), x, total_steps)

        settings = PiFlowSettings(
            num_steps=steps,
            shift=contract.PIFLOW_SHIFT,
            num_policy_substeps=contract.PIFLOW_POLICY_SUBSTEPS,
            final_step_size_scale=contract.PIFLOW_FINAL_STEP_SIZE_SCALE,
            eps=contract.PIFLOW_EPS,
        )
        transformer_options = model.model_options.get("transformer_options", {})
        model.pre_run()
        try:
            latents = sample_clip(
                dit,
                ref.to(device),
                conditioning["text_embeds"].to(device),
                conditioning["pooled_embed"].to(device),
                conditioning["audio_embeds"].to(device),
                chunks=chunks,
                latent_frames=contract.LATENT_FRAMES,
                seed=seed,
                settings=settings,
                scale_factor=contract.ROPE_SCALE_FACTOR,
                transformer_options=transformer_options,
                step_callback=step_callback,
            )
        finally:
            model.cleanup()

        # One clip per batch entry: VAE Decode decodes each chunk on its own, as
        # the reference does, and concatenates the frames in order.
        samples = torch.stack(latents).permute(0, 4, 1, 2, 3)
        samples = model.model.latent_format.process_out(samples).to(mm.intermediate_device(), torch.float32)
        return ({"samples": samples},)


class KandinskyS2VTrimToAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "conditioning": (CONDITIONING_TYPE,)}}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "fps")
    FUNCTION = "trim"
    CATEGORY = CATEGORY
    DESCRIPTION = "Drops the frames generated over the silence padding of the last chunk; returns the 25 fps rate."

    def trim(self, images, conditioning):
        keep = min(images.shape[0], int(conditioning["track_seconds"] * contract.FPS))
        return (images[:keep], float(contract.FPS))


NODE_CLASS_MAPPINGS = {
    "KandinskyS2VEncoderLoader": KandinskyS2VEncoderLoader,
    "KandinskyS2VEncode": KandinskyS2VEncode,
    "KandinskyS2VReferenceImage": KandinskyS2VReferenceImage,
    "KandinskyS2VSampler": KandinskyS2VSampler,
    "KandinskyS2VTrimToAudio": KandinskyS2VTrimToAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KandinskyS2VEncoderLoader": "Kandinsky S2V Encoder Loader",
    "KandinskyS2VEncode": "Kandinsky S2V Encode (speech + prompt)",
    "KandinskyS2VReferenceImage": "Kandinsky S2V Reference Image",
    "KandinskyS2VSampler": "Kandinsky S2V Sampler",
    "KandinskyS2VTrimToAudio": "Kandinsky S2V Trim To Audio",
}
