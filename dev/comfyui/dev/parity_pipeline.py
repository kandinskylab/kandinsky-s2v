"""Stage-by-stage parity: the ComfyUI nodes against the native kandinsky_s2v pipeline.

1. conditioning: the Encode node path (ComfyUI's audio loader, resampling,
   Qwen2.5-Omni, CLIP-L from the stock CLIP loader) against the native
   embedder on the padded file;
2. reference frame: the Reference Image node against the native crop/resize;
3. sampling: the Sampler node against the native chunk loop, both fed the
   same embeddings and the same reference latent (the native VAE samples the
   posterior, ComfyUI takes its mode, so that stage is pinned).

    COMFYUI=... python dev/parity_pipeline.py [--model pro] --audio speech.wav --image ref.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
# The kandinsky-s2v checkout this extension lives in, at dev/comfyui.
ROOT = REPO.parents[1]
COMFYUI = Path(os.environ["COMFYUI"])
sys.path.insert(0, str(COMFYUI))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO))

import comfy.model_management as mm  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
from comfy_extras.nodes_audio import load as load_audio_file  # noqa: E402
from kandinsky_s2v.core.algo import prepare_latents as native_latents  # noqa: E402
from kandinsky_s2v.core.algo import prepare_ropes as native_ropes  # noqa: E402
from kandinsky_s2v.core.algo.audio_io import pad_with_silence  # noqa: E402
from kandinsky_s2v.core.algo.chunks import split_audio_embeds as native_split  # noqa: E402
from kandinsky_s2v.core.algo.denoise_loop import ChunkContext, TextContext  # noqa: E402
from kandinsky_s2v.core.algo.piflow_sampler import piflow_denoise_chunk  # noqa: E402
from kandinsky_s2v.core.components.omni_embedder import S2VTextAudioEmbedder  # noqa: E402
from kandinsky_s2v.pipeline.config import load_config  # noqa: E402
from kandinsky_s2v.pipeline.factory import create_bare_dit  # noqa: E402
from kandinsky_s2v_comfy import contract  # noqa: E402
from kandinsky_s2v_comfy_graph import DIT_FILES  # noqa: E402
from kandinsky_s2v_comfy.encoder import S2VConditioningEncoder, resample_speech  # noqa: E402
from kandinsky_s2v_comfy.register import register  # noqa: E402
from kandinsky_s2v_comfy.nodes import KandinskyS2VReferenceImage, KandinskyS2VSampler, clip_l_pooled  # noqa: E402

MODELS = COMFYUI / "models"
VAE = MODELS / "vae/hunyuan_video_vae_bf16.safetensors"
OMNI = MODELS / "text_encoders/Qwen2.5-Omni-7B"
CLIP_DIR = MODELS / "text_encoders/clip-vit-large-patch14"  # the native embedder's CLIP
CLIP_FILE = MODELS / "text_encoders/clip_l.safetensors"


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def report(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    same_shape = actual.shape == expected.shape
    line = f"{name:28s} shape {tuple(actual.shape)} vs {tuple(expected.shape)}"
    if same_shape:
        max_abs = (actual.float() - expected.float()).abs().max().item()
        line += f"  rel L2 {rel_l2(actual, expected):.3e}  max abs {max_abs:.3e}"
        line += f"  identical {torch.equal(actual, expected)}"
    print(line, flush=True)


def conditioning(cfg, audio: str, device):
    track = load_audio_file(audio)
    waveform, sample_rate = track
    track_seconds = waveform.shape[-1] / sample_rate
    chunks = max(1, int(np.ceil(track_seconds / contract.CHUNK_SECONDS)))
    target = chunks * contract.CHUNK_SECONDS
    prompt = cfg.generation.prompt
    print("contract.PROMPT == config prompt:", contract.PROMPT == prompt)

    native = S2VTextAudioEmbedder(str(OMNI), str(CLIP_DIR), device=device)
    padded = pad_with_silence(audio, target)
    try:
        expected = native.encode(prompt, str(padded or audio), 0.0, target)
    finally:
        if padded is not None:
            padded.unlink(missing_ok=True)
    del native
    torch.cuda.empty_cache()

    ours = S2VConditioningEncoder(str(OMNI)).to(device)
    speech = resample_speech(waveform, sample_rate, target)
    text_embeds, audio_embeds = ours.encode(prompt, speech, device)
    report("text_embeds", text_embeds, expected[0])
    report("audio_embeds", audio_embeds, expected[2])
    del ours
    torch.cuda.empty_cache()

    # What Load CLIP (clip_l.safetensors, type stable_diffusion) hands the Encode node.
    clip = comfy.sd.load_clip(ckpt_paths=[str(CLIP_FILE)], clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
    report("pooled (Load CLIP)", clip_l_pooled(clip, prompt).to(expected[1].device), expected[1])
    return expected, chunks


def reference_frame(image_path: str):
    resolution = native_latents.select_best_resolution(image_path, contract.RESOLUTIONS)
    with Image.open(image_path) as raw:
        expected = np.asarray(native_latents.crop_and_resize(raw.convert("RGB"), *resolution))
    with Image.open(image_path) as raw:
        loaded = ImageOps.exif_transpose(raw).convert("RGB")
    comfy_image = torch.from_numpy(np.asarray(loaded).astype(np.float32) / 255.0)[None]
    (out,) = KandinskyS2VReferenceImage().prepare(comfy_image)
    actual = (out[0].numpy() * 255.0).round().astype(np.uint8)
    print(f"reference frame {tuple(actual.shape)} (native {tuple(expected.shape)}), "
          f"identical pixels: {np.array_equal(actual, expected)}")
    return out


def native_sampling(cfg, dit, ref, embeds, chunks, device) -> list[torch.Tensor]:
    text, pooled, audio = (e.to(device) for e in embeds)
    dcfg = cfg.dit
    coefs = dit_coefs(dcfg)
    h, w = ref.shape[1], ref.shape[2]
    memory = native_latents.init_memory(dcfg.frame_buckets, coefs, dcfg.patch_size, h, w, dcfg.in_visual_dim, device)
    generator = torch.Generator(device=device).manual_seed(contract.SEED)
    frames = cfg.generation.latent_frames
    vis_pos = native_ropes.visual_rope_pos(frames, h, w, dcfg.patch_size, device)
    ref_pos = native_ropes.ref_rope_pos(dcfg.index_rope_ref_image, h, w, dcfg.patch_size, device)
    text_ctx = TextContext(text.bfloat16(), pooled.bfloat16(), native_ropes.text_rope_pos(text.shape[0], device))
    per_chunk = native_split(audio, chunks)
    outputs = []
    for index in range(chunks):
        latent = native_latents.init_noise(frames, h, w, dcfg.in_visual_dim, dcfg.patch_size, generator, device)
        ctx = ChunkContext(
            ref=ref,
            memory=memory,
            audio_embeds=per_chunk[index].bfloat16(),
            visual_rope_pos=vis_pos,
            ref_rope_pos=ref_pos,
            memory_rope_pos=native_ropes.memory_rope_pos(memory, dcfg.frame_buckets, coefs, dcfg.patch_size, device),
            audio_rope_pos=native_ropes.audio_rope_pos(per_chunk[index].shape[0], frames, device),
            duration=frames,
            scale_factor=cfg.generation.scale_factor,
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            latent = piflow_denoise_chunk(
                latent, dit, ctx, text_ctx, text_ctx,
                num_steps=cfg.generation.num_steps,
                scheduler_scale=cfg.generation.scheduler_scale,
                guidance_weight=1.0,
                num_policy_substeps=cfg.piflow.num_policy_substeps,
                final_step_size_scale=cfg.piflow.final_step_size_scale,
                shift=cfg.piflow.shift,
                eps=cfg.piflow.eps,
            )
        memory = native_latents.roll_memory(latent, dcfg.frame_buckets, coefs, dcfg.patch_size)
        outputs.append(latent)
    return outputs


def dit_coefs(dcfg):
    return (dcfg.patch_proj1x_coef, dcfg.patch_proj2x_coef, dcfg.patch_proj4x_coef)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", choices=sorted(DIT_FILES), default="lite")
    args = parser.parse_args()
    dit = MODELS / "diffusion_models" / DIT_FILES[args.model]

    register()
    device = torch.device("cuda")
    cfg = load_config(ROOT / f"src/kandinsky_s2v/configs/s2v_{args.model}_distill.yaml")
    cfg = cfg.model_copy(update={"paths": cfg.paths.model_copy(update={"dit": str(dit)})})

    embeds, chunks = conditioning(cfg, args.audio, device)
    pixels = reference_frame(args.image)

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(VAE)))
    vae_latent = vae.encode(pixels[:, :, :, :3])
    print("VAE Encode latent", tuple(vae_latent.shape))

    patcher = comfy.sd.load_diffusion_model(str(dit))
    conditioning_dict = {
        "text_embeds": embeds[0].bfloat16().cpu(),
        "pooled_embed": embeds[1].bfloat16().cpu(),
        "audio_embeds": embeds[2].bfloat16().cpu(),
        "chunks": chunks,
        "track_seconds": 0.0,
    }
    (out,) = KandinskyS2VSampler().sample(patcher, conditioning_dict, {"samples": vae_latent}, contract.SEED, 16)
    actual = patcher.model.latent_format.process_in(out["samples"]).permute(0, 2, 3, 4, 1)

    ref = patcher.model.latent_format.process_in(vae_latent.float())[:, :, 0].permute(0, 2, 3, 1).to(device)
    mm.unload_all_models()
    native = create_bare_dit(cfg, device, "sdpa")
    expected = native_sampling(cfg, native, ref, embeds, chunks, device)
    for index in range(chunks):
        report(f"chunk {index} latent", actual[index].to(device), expected[index])


if __name__ == "__main__":
    main()
