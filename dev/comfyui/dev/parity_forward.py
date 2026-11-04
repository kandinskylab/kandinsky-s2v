"""One DiT forward: the ComfyUI port against the native kandinsky_s2v DiT.

Loads the checkpoint the way the stock UNETLoader does and the native DiT the
way kandinsky_s2v's factory does, feeds both the same inputs and compares.

    COMFYUI=/path/to/ComfyUI python dev/parity_forward.py [--model pro] [--dynamic]

The checkpoint is $COMFYUI/models/diffusion_models/<the model's file>, or DIT=/path/to/model.safetensors.
The native forward runs first and is freed before the port loads, so Pro fits
one 80 GB GPU.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
# The kandinsky-s2v checkout this extension lives in, at dev/comfyui.
ROOT = REPO.parents[1]
sys.path.insert(0, os.environ["COMFYUI"])
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO))

DYNAMIC_VRAM = "--dynamic" in sys.argv
if DYNAMIC_VRAM:
    import comfy_aimdo.control

    comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=True)

import comfy.memory_management  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.sd  # noqa: E402
from kandinsky_s2v.core.algo import prepare_latents as native_latents  # noqa: E402
from kandinsky_s2v.core.algo import prepare_ropes as native_ropes  # noqa: E402
from kandinsky_s2v.core.algo.denoise_loop import ChunkContext, TextContext, dit_forward  # noqa: E402
from kandinsky_s2v.pipeline.config import load_config  # noqa: E402
from kandinsky_s2v.pipeline.factory import create_bare_dit  # noqa: E402
from kandinsky_s2v_comfy_graph import DIT_FILES  # noqa: E402
from kandinsky_s2v_comfy import sampling  # noqa: E402
from kandinsky_s2v_comfy.register import register  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(DIT_FILES), default="lite")
    parser.add_argument("--dynamic", action="store_true", help="ComfyUI's dynamic-VRAM model patcher")
    args = parser.parse_args()
    default_dit = Path(os.environ["COMFYUI"]) / "models/diffusion_models" / DIT_FILES[args.model]
    dit_path = os.environ.get("DIT") or str(default_dit)
    if DYNAMIC_VRAM:
        devices = ((d.index, 0) for d in mm.get_all_torch_devices())
        if not comfy_aimdo.control.init_devices(devices):
            raise RuntimeError("comfy-aimdo did not initialize")
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True
    register()
    device = torch.device("cuda")
    cfg = load_config(ROOT / f"src/kandinsky_s2v/configs/s2v_{args.model}_distill.yaml")
    cfg = cfg.model_copy(update={"paths": cfg.paths.model_copy(update={"dit": dit_path})})
    print("checkpoint:", dit_path)

    native = create_bare_dit(cfg, device, "sdpa")
    fp32_native = {k for k, v in native.state_dict().items() if v.dtype == torch.float32}

    gen = torch.Generator(device=device).manual_seed(0)
    h, w, frames = 64, 96, 32
    ref = torch.randn(1, h, w, 16, device=device, generator=gen)
    latent = torch.randn(frames, h, w, 16, device=device, generator=gen)
    coefs = (cfg.dit.patch_proj1x_coef, cfg.dit.patch_proj2x_coef, cfg.dit.patch_proj4x_coef)
    memory = [torch.randn(*m.shape, device=device, generator=gen) for m in native_latents.init_memory(
        cfg.dit.frame_buckets, coefs, cfg.dit.patch_size, h, w, 16, device)]
    text = torch.randn(120, 3584, device=device, generator=gen).to(torch.bfloat16)
    pooled = torch.randn(1, 768, device=device, generator=gen).to(torch.bfloat16)
    audio = torch.randn(125, 3584, device=device, generator=gen).to(torch.bfloat16)
    sigma = torch.tensor([0.7], device=device)

    ctx = ChunkContext(
        ref=ref,
        memory=memory,
        audio_embeds=audio,
        visual_rope_pos=native_ropes.visual_rope_pos(frames, h, w, cfg.dit.patch_size, device),
        ref_rope_pos=native_ropes.ref_rope_pos(40, h, w, cfg.dit.patch_size, device),
        memory_rope_pos=native_ropes.memory_rope_pos(memory, cfg.dit.frame_buckets, coefs, cfg.dit.patch_size, device),
        audio_rope_pos=native_ropes.audio_rope_pos(125, frames, device),
        duration=frames,
        scale_factor=cfg.generation.scale_factor,
    )
    text_ctx = TextContext(text, pooled, native_ropes.text_rope_pos(text.shape[0], device))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = dit_forward(native, latent, sigma, ctx, text_ctx)
    del native
    torch.cuda.empty_cache()

    patcher = comfy.sd.load_diffusion_model(dit_path)
    mm.load_models_gpu([patcher])
    print("patcher:", type(patcher).__name__)
    port = patcher.model.diffusion_model
    assert port.memory_coefs == coefs, (port.memory_coefs, coefs)
    fp32_port = {k for k, v in port.state_dict().items() if v.dtype == torch.float32}
    print("fp32 tensors native/port:", len(fp32_native), len(fp32_port), "same set:", fp32_native == fp32_port)

    inputs = sampling.ChunkInputs(
        ref=ref,
        memory=memory,
        text_embeds=text,
        pooled_embed=pooled,
        audio_embeds=audio,
        visual_rope_pos=ctx.visual_rope_pos,
        ref_rope_pos=ctx.ref_rope_pos,
        memory_rope_pos=ctx.memory_rope_pos,
        text_rope_pos=text_ctx.rope_pos,
        audio_rope_pos=ctx.audio_rope_pos,
        scale_factor=cfg.generation.scale_factor,
    )
    patcher.pre_run()
    try:
        with torch.no_grad():
            actual = sampling._dit_forward(port, latent, sigma, inputs, {})  # noqa: SLF001
    finally:
        patcher.cleanup()

    diff = (actual.float() - expected.float()).abs()
    print("shape", tuple(actual.shape), tuple(expected.shape), actual.dtype, expected.dtype)
    scale = expected.float().abs().mean().item()
    print(f"max abs diff {diff.max().item():.3e}  mean {diff.mean().item():.3e}  ref scale {scale:.3e}")
    print("bit-identical:", torch.equal(actual, expected))


if __name__ == "__main__":
    main()
