"""Independent CPU checks of the port: parity with the native code on tiny random
models, release detection on meta-built releases, node logic and the download
routes. No weights, no network, no GPU.

The native implementation is imported from this repository's ``src``; the tests
that need it skip when it is not importable.
"""

from __future__ import annotations

import asyncio
import errno
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import comfy.latent_formats
import comfy.model_detection
import comfy.ops
import comfy.supported_models
import model_downloads
from kandinsky_s2v_comfy import contract, nodes, sampling
from kandinsky_s2v_comfy.ldm import model as port_model
from kandinsky_s2v_comfy.ldm.model import KandinskyS2VTransformer
from kandinsky_s2v_comfy.register import check_release, detect_s2v_config, register

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
CONFIGS = REPO / "src" / "kandinsky_s2v" / "configs"
WORKFLOWS = PACKAGE / "example_workflows"


def _native(module: str):
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return pytest.importorskip(module)


# --------------------------------------------------------------------------- tiny models


def _tiny_config(axes_dims: tuple[int, int, int]) -> dict:
    head_dim = sum(axes_dims)
    return {
        "in_visual_dim": 4,
        "n_grid": 3,
        "in_text_dim": 24,
        "in_text_dim2": 12,
        "in_audio_dim": 20,
        "time_dim": 32,
        "model_dim": 2 * head_dim,
        "ff_dim": 3 * head_dim,
        "num_text_blocks": 2,
        "num_audio_blocks": 1,
        "num_visual_blocks": 3,
        "axes_dims": axes_dims,
        "audio_inject_layers": (0, 2),
    }


AXES = {"head64": (16, 24, 24), "head128": (32, 48, 48)}


def _build_pair(axes_dims, seed=0):
    """A native π-Flow DiT with random weights and the port loaded from its state dict."""
    piflow_dit = _native("kandinsky_s2v.core.components.piflow_dit")
    cfg = _tiny_config(axes_dims)
    native_kwargs = {k: v for k, v in cfg.items() if k != "n_grid"}
    native = piflow_dit.PiFlowDiffusionTransformer3DS2V(
        n_grid=cfg["n_grid"], out_visual_dim=cfg["in_visual_dim"], attention_engine="sdpa", **native_kwargs
    ).eval()
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in native.parameters():
            # Modulations are zero-initialised natively; random everywhere exposes wiring mistakes.
            param.copy_(torch.randn(param.shape, generator=gen) * 0.2)
    port = KandinskyS2VTransformer(**cfg, operations=comfy.ops.disable_weight_init, device="cpu").eval()
    native_sd = native.state_dict()
    port_keys = set(port.state_dict())
    assert port_keys == set(native_sd), (sorted(port_keys ^ set(native_sd)))[:10]
    for key, tensor in port.state_dict().items():
        assert tensor.shape == native_sd[key].shape, key
    port.load_state_dict(native_sd, strict=True)
    return native, port, cfg


def _chunk_inputs(cfg, height=8, width=8, frames=32, audio_tokens=125, text_tokens=7, seed=1):
    gen = torch.Generator().manual_seed(seed)
    c = cfg["in_visual_dim"]
    patch = contract.DIT_DEFAULTS["patch_size"]
    buckets = contract.DIT_DEFAULTS["frame_buckets"]
    coefs = (
        contract.DIT_DEFAULTS["patch_proj1x_coef"],
        contract.DIT_DEFAULTS["patch_proj2x_coef"],
        contract.DIT_DEFAULTS["patch_proj4x_coef"],
    )
    memory = [
        torch.randn(m.shape, generator=gen)
        for m in sampling.init_memory(buckets, coefs, patch, height, width, c, "cpu")
    ]
    return {
        "latent": torch.randn(frames, height, width, c, generator=gen),
        "ref": torch.randn(1, height, width, c, generator=gen),
        "memory": memory,
        "text": torch.randn(text_tokens, cfg["in_text_dim"], generator=gen),
        "pooled": torch.randn(1, cfg["in_text_dim2"], generator=gen),
        "audio": torch.randn(audio_tokens, cfg["in_audio_dim"], generator=gen),
        "patch": patch,
        "buckets": buckets,
        "coefs": coefs,
    }


def _native_ctx(inp, scale_factor):
    loop = _native("kandinsky_s2v.core.algo.denoise_loop")
    ropes = _native("kandinsky_s2v.core.algo.prepare_ropes")
    frames, height, width = inp["latent"].shape[:3]
    ctx = loop.ChunkContext(
        ref=inp["ref"],
        memory=inp["memory"],
        audio_embeds=inp["audio"],
        visual_rope_pos=ropes.visual_rope_pos(frames, height, width, inp["patch"], "cpu"),
        ref_rope_pos=ropes.ref_rope_pos(40, height, width, inp["patch"], "cpu"),
        memory_rope_pos=ropes.memory_rope_pos(inp["memory"], inp["buckets"], inp["coefs"], inp["patch"], "cpu"),
        audio_rope_pos=ropes.audio_rope_pos(inp["audio"].shape[0], frames, "cpu"),
        duration=frames,
        scale_factor=scale_factor,
    )
    text = loop.TextContext(inp["text"], inp["pooled"], ropes.text_rope_pos(inp["text"].shape[0], "cpu"))
    return ctx, text


def _port_inputs(inp, scale_factor):
    """Built with the port's own helpers only."""
    frames, height, width = inp["latent"].shape[:3]
    return sampling.ChunkInputs(
        ref=inp["ref"],
        memory=inp["memory"],
        text_embeds=inp["text"],
        pooled_embed=inp["pooled"],
        audio_embeds=inp["audio"],
        visual_rope_pos=sampling.visual_rope_pos(frames, height, width, inp["patch"], "cpu"),
        ref_rope_pos=sampling.ref_rope_pos(40, height, width, inp["patch"], "cpu"),
        memory_rope_pos=sampling.memory_rope_pos(inp["memory"], inp["buckets"], inp["coefs"], inp["patch"], "cpu"),
        text_rope_pos=torch.arange(inp["text"].shape[0]),
        audio_rope_pos=sampling.audio_rope_pos(inp["audio"].shape[0], frames, "cpu"),
        scale_factor=scale_factor,
    )


# --------------------------------------------------------------------------- 1. DiT parity


@pytest.mark.parametrize("axes", list(AXES.values()), ids=list(AXES))
@pytest.mark.parametrize("sigma", [1.0, 0.37, 0.0])
def test_dit_forward_matches_native(axes, sigma):
    loop = _native("kandinsky_s2v.core.algo.denoise_loop")
    native, port, cfg = _build_pair(axes)
    inp = _chunk_inputs(cfg)
    time = torch.tensor([sigma])
    ctx, text = _native_ctx(inp, contract.ROPE_SCALE_FACTOR)
    with torch.no_grad():
        expected = loop.dit_forward(native, inp["latent"], time, ctx, text)
        actual = sampling._dit_forward(port, inp["latent"], time, _port_inputs(inp, contract.ROPE_SCALE_FACTOR), {})  # noqa: SLF001
    assert actual.shape == (32, cfg["n_grid"], 8, 8, cfg["in_visual_dim"])
    assert actual.shape == expected.shape
    assert expected.abs().mean() > 1e-2  # a dead network would compare equal trivially
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_dit_forward_runs_with_fp8_weights():
    # UNETLoader weight_dtype=fp8_*: weights stored in fp8, cast per op (manual_cast).
    _, port, cfg = _build_pair(next(iter(AXES.values())))
    fp8 = KandinskyS2VTransformer(
        **cfg, operations=comfy.ops.manual_cast, device="cpu", dtype=torch.float8_e4m3fn
    ).eval()
    fp8.load_state_dict(
        {k: v.to(fp8.state_dict()[k].dtype) for k, v in port.state_dict().items()}, strict=True
    )
    assert fp8.mask_embeddings.weight.dtype == torch.float8_e4m3fn
    inp = _chunk_inputs(cfg)
    with torch.no_grad():
        out = sampling._dit_forward(  # noqa: SLF001
            fp8, inp["latent"], torch.tensor([0.5]), _port_inputs(inp, contract.ROPE_SCALE_FACTOR), {}
        )
    assert out.shape == (32, cfg["n_grid"], 8, 8, cfg["in_visual_dim"])
    assert torch.isfinite(out).all()


def test_dit_forward_depends_on_every_input():
    """The parity above is meaningful only if each input reaches the output."""
    _, port, cfg = _build_pair(AXES["head64"])
    inp = _chunk_inputs(cfg)
    base = sampling._dit_forward(port, inp["latent"], torch.tensor([0.5]), _port_inputs(inp, (1.0, 2.0, 2.0)), {})  # noqa: SLF001
    for name in ("ref", "text", "pooled", "audio"):
        changed = dict(inp, **{name: inp[name] + 0.5})
        out = sampling._dit_forward(
            port, inp["latent"], torch.tensor([0.5]), _port_inputs(changed, (1.0, 2.0, 2.0)), {}
        )  # noqa: SLF001
        assert not torch.allclose(out, base), name
    for index in range(3):
        memory = [m.clone() for m in inp["memory"]]
        memory[index] += 0.5
        out = sampling._dit_forward(  # noqa: SLF001
            port, inp["latent"], torch.tensor([0.5]), _port_inputs(dict(inp, memory=memory), (1.0, 2.0, 2.0)), {}
        )
        assert not torch.allclose(out, base), f"memory[{index}]"


def test_grouped_attention_is_block_diagonal():
    """The frame-aligned audio adapter: query group i attends to key group i only."""
    gen = torch.Generator().manual_seed(0)
    q_len = torch.tensor([3, 3, 3])
    kv_len = torch.tensor([1, 4, 2])
    q = torch.randn(9, 2, 8, generator=gen)
    k = torch.randn(7, 2, 8, generator=gen)
    v = torch.randn(7, 2, 8, generator=gen)
    out = port_model.grouped_attention(q, k, v, q_len, kv_len)
    expected = []
    q_off = kv_off = 0
    for ql, kl in zip(q_len.tolist(), kv_len.tolist(), strict=False):
        attn = torch.softmax(
            torch.einsum("qhd,khd->hqk", q[q_off : q_off + ql], k[kv_off : kv_off + kl]) / 8**0.5, dim=-1
        )
        expected.append(torch.einsum("hqk,khd->qhd", attn, v[kv_off : kv_off + kl]))
        q_off += ql
        kv_off += kl
    torch.testing.assert_close(out, torch.cat(expected), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("axes", list(AXES.values()), ids=list(AXES))
@pytest.mark.parametrize("num_steps", [1, 3])
def test_piflow_chunk_matches_native(axes, num_steps):
    sampler = _native("kandinsky_s2v.core.algo.piflow_sampler")
    native, port, cfg = _build_pair(axes)
    inp = _chunk_inputs(cfg)
    ctx, text = _native_ctx(inp, contract.ROPE_SCALE_FACTOR)
    settings = sampling.PiFlowSettings(num_steps=num_steps, num_policy_substeps=16)
    expected = sampler.piflow_denoise_chunk(
        inp["latent"],
        native,
        ctx,
        text,
        text,
        num_steps=num_steps,
        scheduler_scale=10.0,
        num_policy_substeps=16,
        final_step_size_scale=0.5,
        shift=10.0,
        eps=1e-6,
    )
    seen = []
    actual = sampling.piflow_denoise_chunk(
        inp["latent"],
        port,
        _port_inputs(inp, contract.ROPE_SCALE_FACTOR),
        settings,
        {},
        step_callback=lambda step, x0, x: seen.append((step, tuple(x0.shape), tuple(x.shape))),
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert [s[0] for s in seen] == list(range(num_steps))
    assert all(s[1] == s[2] == tuple(inp["latent"].shape) for s in seen)


def test_sample_clip_matches_native_chunk_loop():
    """Two chunks: noise stream, audio split, memory roll and its RoPE, as the native pipeline loop."""
    sampler = _native("kandinsky_s2v.core.algo.piflow_sampler")
    lat = _native("kandinsky_s2v.core.algo.prepare_latents")
    ropes = _native("kandinsky_s2v.core.algo.prepare_ropes")
    chunks_mod = _native("kandinsky_s2v.core.algo.chunks")
    loop = _native("kandinsky_s2v.core.algo.denoise_loop")
    native, port, cfg = _build_pair(AXES["head64"])
    gen = torch.Generator().manual_seed(3)
    c, height, width, frames, chunks, seed, steps = cfg["in_visual_dim"], 9, 11, 32, 2, 1234, 2
    ref = torch.randn(1, height, width, c, generator=gen)  # odd sizes: cropped to whole patches
    text = torch.randn(5, cfg["in_text_dim"], generator=gen)
    pooled = torch.randn(1, cfg["in_text_dim2"], generator=gen)
    audio = torch.randn(2 * 125 + 3, cfg["in_audio_dim"], generator=gen)  # ragged tail dropped
    patch = contract.DIT_DEFAULTS["patch_size"]
    buckets = contract.DIT_DEFAULTS["frame_buckets"]
    coefs = port.memory_coefs
    scale = contract.ROPE_SCALE_FACTOR

    # Native, written out as KandinskyS2VPipeline.__call__ does it.
    nref = lat.crop_reference_latent(ref, patch)
    h, w = nref.shape[1], nref.shape[2]
    per_chunk = chunks_mod.split_audio_embeds(audio, chunks)
    text_ctx = loop.TextContext(text, pooled, ropes.text_rope_pos(text.shape[0], "cpu"))
    memory = lat.init_memory(buckets, coefs, patch, h, w, c, "cpu")
    g = torch.Generator().manual_seed(seed)
    expected = []
    for index in range(chunks):
        latent = lat.init_noise(frames, h, w, c, patch, g, "cpu")
        ctx = loop.ChunkContext(
            ref=nref,
            memory=memory,
            audio_embeds=per_chunk[index],
            visual_rope_pos=ropes.visual_rope_pos(frames, h, w, patch, "cpu"),
            ref_rope_pos=ropes.ref_rope_pos(40, h, w, patch, "cpu"),
            memory_rope_pos=ropes.memory_rope_pos(memory, buckets, coefs, patch, "cpu"),
            audio_rope_pos=ropes.audio_rope_pos(per_chunk[index].shape[0], frames, "cpu"),
            duration=frames,
            scale_factor=scale,
        )
        latent = sampler.piflow_denoise_chunk(
            latent,
            native,
            ctx,
            text_ctx,
            text_ctx,
            num_steps=steps,
            scheduler_scale=10.0,
            num_policy_substeps=8,
            final_step_size_scale=0.5,
            shift=10.0,
            eps=1e-6,
        )
        memory = lat.roll_memory(latent, buckets, coefs, patch)
        expected.append(latent)

    calls = []
    actual = sampling.sample_clip(
        port,
        ref,
        text,
        pooled,
        audio,
        chunks=chunks,
        latent_frames=frames,
        seed=seed,
        settings=sampling.PiFlowSettings(num_steps=steps, num_policy_substeps=8),
        scale_factor=scale,
        transformer_options={},
        step_callback=lambda chunk, step, x0, x: calls.append((chunk, step)),
    )
    assert len(actual) == chunks
    for a, e in zip(actual, expected, strict=False):
        assert a.shape == (frames, 8, 10, c)
        torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-5)
    assert calls == [(i, s) for i in range(chunks) for s in range(steps)]


# --------------------------------------------------------------------------- 2. sampler pieces


def test_shift_timesteps_matches_native():
    loop = _native("kandinsky_s2v.core.algo.denoise_loop")
    t = torch.rand(1000, generator=torch.Generator().manual_seed(0))
    for shift in (1.0, 3.0, 10.0):
        assert torch.equal(sampling.shift_timesteps(t, shift), loop.shift_timesteps(t, shift))


@pytest.mark.parametrize("n_grid", [1, 2, 5])
@pytest.mark.parametrize("raw_src", [1.0, 0.4, 0.05])
def test_dx_policy_rollout_matches_native(n_grid, raw_src):
    pm = _native("kandinsky_s2v.core.algo.piflow_math")
    gen = torch.Generator().manual_seed(n_grid)
    tokens = 6
    x = torch.randn(tokens, 3, 4, 2, generator=gen)
    grid = torch.randn(tokens, n_grid, 3, 4, 2, generator=gen)
    shift = 10.0
    raw = torch.full((1,), raw_src)
    sigma = sampling.shift_timesteps(raw, shift)
    sigma_tok = sigma.repeat_interleave(tokens).reshape(tokens, 1, 1, 1)
    seg = torch.full_like(sigma_tok, 0.3)  # larger than raw_src for the last case: clamp(min=0) path
    raw_dst = (raw - 0.3).clamp(min=1e-6)

    def run(module):
        policy = module.DXPolicy(grid, x, sigma_tok, seg, shift=shift, eps=1e-6)
        probe = policy.pi(x + 0.1, sigma_tok * 0.9)
        out = module.policy_rollout_fm(
            x, sigma_tok, raw.repeat_interleave(tokens), raw_dst.repeat_interleave(tokens), 64, policy
        )
        return policy.denoising_output_x_0, probe, out

    for a, e in zip(run(sampling), run(pm), strict=False):
        assert torch.equal(a, e)


def test_dx_interpolate_matches_native_at_and_between_grid_points():
    pm = _native("kandinsky_s2v.core.algo.piflow_math")
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(4, 5, 3, generator=gen)
    t = torch.tensor([-0.5, 0.0, 0.5, 1.5]).view(4, 1)
    out = sampling.DXPolicy._interpolate(x, t)  # noqa: SLF001
    assert torch.equal(out, pm.DXPolicy._interpolate(x, t))  # noqa: SLF001
    torch.testing.assert_close(out[0], x[0, 0])  # clamped below
    torch.testing.assert_close(out[2], x[2, 2])  # midpoint of 5 points
    torch.testing.assert_close(out[3], x[3, 4])  # clamped above


@pytest.mark.parametrize(("height", "width"), [(64, 96), (13, 22), (9, 9), (96, 64)])
def test_latent_and_rope_helpers_match_native(height, width):
    lat = _native("kandinsky_s2v.core.algo.prepare_latents")
    ropes = _native("kandinsky_s2v.core.algo.prepare_ropes")
    patch = contract.DIT_DEFAULTS["patch_size"]
    buckets = contract.DIT_DEFAULTS["frame_buckets"]
    coefs = (
        contract.DIT_DEFAULTS["patch_proj1x_coef"],
        contract.DIT_DEFAULTS["patch_proj2x_coef"],
        contract.DIT_DEFAULTS["patch_proj4x_coef"],
    )
    ref = torch.randn(1, height, width, 16)
    assert torch.equal(sampling.crop_reference_latent(ref, patch), lat.crop_reference_latent(ref, patch))
    a = sampling.init_noise(32, height, width, 16, patch, torch.Generator().manual_seed(7), "cpu")
    b = lat.init_noise(32, height, width, 16, patch, torch.Generator().manual_seed(7), "cpu")
    assert torch.equal(a, b)
    port_mem = sampling.init_memory(buckets, coefs, patch, height, width, 16, "cpu")
    native_mem = lat.init_memory(buckets, coefs, patch, height, width, 16, "cpu")
    assert [m.shape for m in port_mem] == [m.shape for m in native_mem]
    latent = torch.randn(32, height // 2 * 2, width // 2 * 2, 16)
    rolled, native_rolled = (
        sampling.roll_memory(latent, buckets, coefs, patch),
        lat.roll_memory(latent, buckets, coefs, patch),
    )
    for p, n in zip(rolled, native_rolled, strict=False):
        assert torch.equal(p, n)
    for p, n in zip(
        sampling.memory_rope_pos(rolled, buckets, coefs, patch, "cpu"),
        ropes.memory_rope_pos(native_rolled, buckets, coefs, patch, "cpu"),
        strict=False,
    ):
        assert all(torch.equal(x, y) for x, y in zip(p, n, strict=False))
    for p, n in zip(
        sampling.visual_rope_pos(32, height, width, patch, "cpu"),
        ropes.visual_rope_pos(32, height, width, patch, "cpu"),
        strict=False,
    ):
        assert torch.equal(p, n)
    for p, n in zip(
        sampling.ref_rope_pos(40, height, width, patch, "cpu"),
        ropes.ref_rope_pos(40, height, width, patch, "cpu"),
        strict=False,
    ):
        assert torch.equal(p, n)


def test_memory_rope_pos_lies_before_the_chunk_and_covers_each_bucket():
    patch = contract.DIT_DEFAULTS["patch_size"]
    buckets = contract.DIT_DEFAULTS["frame_buckets"]
    coefs = ((1, 1, 1), (2, 2, 2), (4, 4, 4))
    memory = sampling.init_memory(buckets, coefs, patch, 64, 96, 16, "cpu")
    pos = sampling.memory_rope_pos(memory, buckets, coefs, patch, "cpu")
    assert pos[0][0].tolist() == [-2, -1]
    assert pos[1][0].tolist() == [-6, -4]
    assert pos[2][0].tolist() == list(range(-30, -6, 4))
    # Each memory buffer embeds to as many tokens per axis as it has positions.
    for buffer, p, coef in zip(memory, pos, coefs, strict=False):
        assert buffer.shape[0] // coef[0] == len(p[0])
        assert buffer.shape[1] // (patch[1] * coef[1]) == len(p[1])
        assert buffer.shape[2] // (patch[2] * coef[2]) == len(p[2])


@pytest.mark.parametrize(("tokens", "duration"), [(125, 32), (126, 32), (300, 32), (2, 2), (40, 7), (31, 32)])
def test_audio_groups_match_native(tokens, duration):
    dit = _native("kandinsky_s2v.core.components.dit")
    ropes = _native("kandinsky_s2v.core.algo.prepare_ropes")
    port_lengths = port_model.audio_group_lengths(tokens, duration, "cpu")
    assert torch.equal(port_lengths, dit.audio_group_lengths(tokens, duration, "cpu"))
    assert int(port_lengths.sum()) == tokens
    assert port_lengths[0] == 1
    assert torch.equal(sampling.audio_rope_pos(tokens, duration, "cpu"), ropes.audio_rope_pos(tokens, duration, "cpu"))
    for total, groups in ((tokens, duration), (500, 32), (7, 3)):
        assert torch.equal(port_model.split_lengths(total, groups, "cpu"), dit.split_lengths(total, groups, "cpu"))
    with pytest.raises(ValueError, match="duration"):
        port_model.audio_group_lengths(tokens, 1, "cpu")


@pytest.mark.parametrize(("tokens", "chunks"), [(125, 1), (377, 3), (1000, 7), (5, 5)])
def test_split_audio_embeds_matches_native(tokens, chunks):
    ch = _native("kandinsky_s2v.core.algo.chunks")
    embeds = torch.randn(tokens, 3)
    port_parts = sampling.split_audio_embeds(embeds, chunks)
    native_parts = ch.split_audio_embeds(embeds, chunks)
    assert len(port_parts) == len(native_parts) == chunks
    assert all(torch.equal(a, b) for a, b in zip(port_parts, native_parts, strict=False))
    with pytest.raises(ValueError, match="cannot be split"):
        sampling.split_audio_embeds(embeds[: chunks - 1], chunks)


@pytest.mark.parametrize("axes", list(AXES.values()), ids=list(AXES))
def test_rope_tables_match_native(axes):
    rope = _native("kandinsky_s2v.core.components.rope")
    head = sum(axes)
    pos = [torch.arange(-30, -6, 4), torch.arange(0, 8, 2), torch.arange(0, 12, 2)]
    for scale in ((1.0, 1.0, 1.0), (1.0, 2.0, 2.0)):
        a = port_model.RoPE3D(axes)((6, 4, 6), pos, scale)
        b = rope.RoPE3D(axes)((6, 4, 6), pos, scale)
        assert torch.equal(a, b)
    text_pos = torch.tensor([0, 5, 4095])
    assert torch.equal(port_model.RoPE1D(head, max_pos=4096)(text_pos), rope.RoPE1D(head, max_pos=4096)(text_pos))


# --------------------------------------------------------------------------- 3. release detection


def _native_meta_state_dict(config_name: str) -> tuple[dict, object]:
    config = _native("kandinsky_s2v.pipeline.config")
    piflow_dit = _native("kandinsky_s2v.core.components.piflow_dit")
    dit = _native("kandinsky_s2v.core.components.dit")
    cfg = config.load_config(CONFIGS / config_name)
    kwargs = cfg.dit.model_dump()
    with torch.device("meta"):
        model = (
            piflow_dit.PiFlowDiffusionTransformer3DS2V(n_grid=cfg.piflow.n_grid, **kwargs)
            if cfg.piflow.enabled
            else dit.DiffusionTransformer3DS2V(**kwargs)
        )
    return model.state_dict(), cfg


@pytest.mark.parametrize(
    ("config_name", "release"),
    [("s2v_lite_distill.yaml", "Lite distilled"), ("s2v_pro_distill.yaml", "Pro distilled")],
)
def test_native_release_state_dicts_are_detected(config_name, release):
    state_dict, cfg = _native_meta_state_dict(config_name)
    config = detect_s2v_config(state_dict, "")
    assert check_release(config) == release
    for key in ("model_dim", "ff_dim", "time_dim", "num_visual_blocks", "num_text_blocks", "num_audio_blocks"):
        assert config[key] == getattr(cfg.dit, key), key
    assert config["axes_dims"] == tuple(cfg.dit.axes_dims)
    assert config["audio_inject_layers"] == tuple(cfg.dit.audio_inject_layers)
    assert config["n_grid"] == cfg.piflow.n_grid
    assert tuple(config["patch_size"]) == tuple(cfg.dit.patch_size)
    assert tuple(config["frame_buckets"]) == tuple(cfg.dit.frame_buckets)
    assert config["index_rope_ref_image"] == cfg.dit.index_rope_ref_image
    # The port built from the detected config has exactly the checkpoint's keys and shapes.
    port = KandinskyS2VTransformer(**config, operations=comfy.ops.disable_weight_init, device="meta")
    port_sd = port.state_dict()
    assert set(port_sd) == set(state_dict)
    assert all(port_sd[k].shape == v.shape for k, v in state_dict.items())


@pytest.mark.parametrize("config_name", ["s2v_lite.yaml", "s2v_pro.yaml"])
def test_native_base_models_are_refused(config_name):
    state_dict, _ = _native_meta_state_dict(config_name)
    config = detect_s2v_config(state_dict, "")
    assert config["n_grid"] == 1
    with pytest.raises(ValueError, match="n_grid: checkpoint=1"):
        check_release(config)


def test_contract_matches_the_distilled_yaml_configs():
    config = _native("kandinsky_s2v.pipeline.config")
    for name, release in (("s2v_lite_distill.yaml", "Lite distilled"), ("s2v_pro_distill.yaml", "Pro distilled")):
        cfg = config.load_config(CONFIGS / name)
        expected = contract.RELEASE_DIT_CONFIGS[release]
        for key, value in expected.items():
            native_value = cfg.piflow.n_grid if key == "n_grid" else getattr(cfg.dit, key)
            assert (tuple(native_value) if isinstance(value, tuple) else native_value) == value, (release, key)
        for key, value in contract.DIT_DEFAULTS.items():
            native_value = getattr(cfg.dit, key)
            assert (tuple(native_value) if isinstance(value, tuple) else native_value) == value, key
        gen, pf = cfg.generation, cfg.piflow
        assert gen.latent_frames == contract.LATENT_FRAMES
        assert gen.fps == contract.FPS
        assert tuple(map(tuple, gen.allowed_resolutions)) == contract.RESOLUTIONS
        assert tuple(gen.scale_factor) == contract.ROPE_SCALE_FACTOR
        assert gen.num_steps == contract.PIFLOW_STEPS
        assert pf.shift == contract.PIFLOW_SHIFT
        assert pf.num_policy_substeps == contract.PIFLOW_POLICY_SUBSTEPS
        assert pf.final_step_size_scale == contract.PIFLOW_FINAL_STEP_SIZE_SCALE
        assert pf.eps == contract.PIFLOW_EPS
        assert cfg.text.clip_max_length == contract.CLIP_MAX_LENGTH


def _port_meta(config: dict) -> dict:
    return KandinskyS2VTransformer(**config, operations=comfy.ops.disable_weight_init, device="meta").state_dict()


LITE = contract.RELEASE_DIT_CONFIGS["Lite distilled"]
MUTANTS = {
    "in_visual_dim": 32,
    "n_grid": 4,
    "in_text_dim": 3584 + 8,
    "in_text_dim2": 1024,
    "in_audio_dim": 1280,
    "time_dim": 768,
    "model_dim": 1792 + 64,
    "ff_dim": 7168 + 64,
    "num_text_blocks": 3,
    "num_audio_blocks": 1,
    "num_visual_blocks": 33,
    "axes_dims": (32, 48, 48),
    "audio_inject_layers": (0, 3, 6, 9, 12, 15, 18, 21, 23, 25, 27, 29),
}


@pytest.mark.parametrize("field", sorted(MUTANTS))
def test_lite_mutants_are_refused(field):
    config = {**LITE, field: MUTANTS[field]}
    detected = detect_s2v_config(_port_meta(config), "")
    assert detected[field] == MUTANTS[field]
    with pytest.raises(ValueError, match=f"{field}: checkpoint="):
        check_release(detected)


def test_detection_rejects_unreleased_shapes():
    sd = _port_meta({**LITE, "axes_dims": (32, 32, 32), "model_dim": 1728})
    with pytest.raises(ValueError, match="head width 96"):
        detect_s2v_config(sd, "")
    sd = _port_meta(LITE)
    sd["out_layer.out_layer.weight"] = torch.empty(4 * 17, 1792, device="meta")
    with pytest.raises(ValueError, match="not a multiple"):
        detect_s2v_config(sd, "")
    sd = _port_meta(LITE)
    sd["visual_embeddings.in_layer.weight"] = torch.empty(1792, 66, device="meta")
    with pytest.raises(ValueError, match="patch volume"):
        detect_s2v_config(sd, "")


def test_registered_detector_hooks_comfy_model_detection():
    register()
    register()  # idempotent: the detector is wrapped once
    assert sum(getattr(m, "__name__", "") == "KandinskyS2V" for m in comfy.supported_models.models) == 1
    prefix = "model.diffusion_model."
    for release in ("Lite distilled", "Pro distilled"):
        sd = {prefix + k: v for k, v in _port_meta(contract.RELEASE_DIT_CONFIGS[release]).items()}
        model_config = comfy.model_detection.model_config_from_unet(sd, prefix)
        assert type(model_config).__name__ == "KandinskyS2V"
        assert model_config.unet_config["model_dim"] == contract.RELEASE_DIT_CONFIGS[release]["model_dim"]
    sd = {prefix + k: v for k, v in _port_meta({**LITE, "n_grid": 1}).items()}
    with pytest.raises(ValueError, match="n_grid"):
        comfy.model_detection.detect_unet_config(sd, prefix)
    # A state dict without every S2V key goes on to ComfyUI's own detector, unrefused.
    sd = {prefix + k: v for k, v in _port_meta({**LITE, "n_grid": 1}).items() if k != "mask_embeddings.weight"}
    assert comfy.model_detection.detect_unet_config(sd, prefix)["image_model"] != "kandinsky_s2v"
    assert comfy.model_detection.detect_unet_config({prefix + "foo.weight": torch.empty(1)}, prefix) is None


def test_port_fp32_set_is_the_converters():
    convert = _native("kandinsky_s2v.ckpt.convert")
    port = KandinskyS2VTransformer(
        **LITE, operations=comfy.ops.disable_weight_init, device="meta", dtype=torch.bfloat16
    )
    for key, tensor in port.state_dict().items():
        assert tensor.dtype == convert.tensor_dtype(key, torch.bfloat16), key


# --------------------------------------------------------------------------- 4. nodes


@pytest.mark.parametrize(
    ("width", "height"),
    [(300, 200), (200, 300), (333, 333), (1000, 10), (10, 1000), (625, 500), (575, 500), (1024, 683)],
)
def test_reference_image_matches_native_preprocessing(tmp_path, width, height):
    lat = _native("kandinsky_s2v.core.algo.prepare_latents")
    from torchvision import transforms  # noqa: PLC0415

    rng = np.random.default_rng(width * 7 + height)
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    path = tmp_path / "ref.png"
    Image.fromarray(pixels).save(path)
    resolution = lat.select_best_resolution(path, contract.RESOLUTIONS)
    with Image.open(path) as raw:
        expected = transforms.ToTensor()(lat.crop_and_resize(raw.convert("RGB"), *resolution)).permute(1, 2, 0)
    # IMAGE as ComfyUI's Load Image gives it, with an alpha channel the node must drop.
    rgba = np.concatenate([pixels, np.full((height, width, 1), 255, np.uint8)], axis=-1)
    image = torch.from_numpy(rgba.astype(np.float32) / 255.0).unsqueeze(0)
    (out,) = nodes.KandinskyS2VReferenceImage().prepare(image)
    assert out.shape == (1, *resolution, 3)
    assert out.dtype == torch.float32
    assert torch.equal(out[0], expected)


def test_trim_to_audio():
    images = torch.zeros(250, 4, 4, 3)
    trimmed, fps = nodes.KandinskyS2VTrimToAudio().trim(images, {"track_seconds": 7.3})
    assert fps == 25.0
    assert trimmed.shape[0] == 182
    trimmed, _ = nodes.KandinskyS2VTrimToAudio().trim(images, {"track_seconds": 20.0})
    assert trimmed.shape[0] == 250


class _FakeOmni:
    max_audio_seconds = 300.0

    def __init__(self):
        self.calls = []

    def encode(self, prompt, speech, device):
        self.calls.append((prompt, speech, device))
        return torch.ones(9, 3584), torch.ones(125 * 2, 3584)


class _FakeEncoderPatcher:
    def __init__(self):
        self.model = _FakeOmni()
        self.load_device = torch.device("cpu")


@pytest.fixture
def encode_node(monkeypatch):
    monkeypatch.setattr(nodes.mm, "load_models_gpu", lambda *a, **k: None)
    monkeypatch.setattr(nodes, "clip_l_pooled", lambda clip, text: torch.full((1, 768), 0.5))
    return nodes.KandinskyS2VEncode()


@pytest.mark.parametrize(
    ("seconds", "chunks_in", "chunks_out", "track_out"),
    [(7.3, 0, 2, 7.3), (7.3, 1, 1, 5.0), (7.3, 3, 3, 7.3), (5.0, 0, 1, 5.0), (0.2, 0, 1, 0.2), (10.0, 0, 2, 10.0)],
)
def test_encode_plans_chunks_and_pads_speech(encode_node, seconds, chunks_in, chunks_out, track_out):
    rate = 16000
    audio = {"waveform": torch.rand(1, 2, int(round(seconds * rate))) - 0.5, "sample_rate": rate}
    encoder = _FakeEncoderPatcher()
    (cond,) = encode_node.encode(encoder, object(), audio, "a prompt", chunks_in)
    assert cond["chunks"] == chunks_out
    assert cond["track_seconds"] == pytest.approx(track_out)
    ((prompt, speech, device),) = encoder.model.calls
    assert prompt == "a prompt"
    assert speech.ndim == 1
    assert abs(len(speech) - chunks_out * contract.CHUNK_SECONDS * 16000) <= 1
    for key in ("text_embeds", "pooled_embed", "audio_embeds"):
        assert cond[key].dtype == torch.bfloat16, key
    # The native planning, for comparison.
    audio_io = _native("kandinsky_s2v.core.algo.audio_io")
    needed = audio_io.num_chunks_for(audio["waveform"].shape[-1] / rate, audio_io.chunk_duration_sec(32, 25))
    assert cond["chunks"] == (needed if chunks_in == 0 else chunks_in)


def test_encode_refuses_bad_tracks(encode_node):
    encoder = _FakeEncoderPatcher()
    with pytest.raises(ValueError, match="one speech track"):
        encode_node.encode(encoder, None, {"waveform": torch.zeros(2, 1, 1600), "sample_rate": 16000}, "p", 0)
    with pytest.raises(ValueError, match="empty"):
        encode_node.encode(encoder, None, {"waveform": torch.zeros(1, 1, 0), "sample_rate": 16000}, "p", 0)
    with pytest.raises(ValueError, match="truncates at 300"):
        encode_node.encode(encoder, None, {"waveform": torch.zeros(1, 1, 1600), "sample_rate": 16000}, "p", 61)
    assert encoder.model.calls == []
    # 60 chunks = 300 s exactly is allowed.
    (cond,) = encode_node.encode(encoder, None, {"waveform": torch.zeros(1, 1, 1600), "sample_rate": 16000}, "p", 60)
    assert cond["chunks"] == 60


def test_encode_refuses_a_prompt_past_the_text_rope(encode_node):
    audio = {"waveform": torch.zeros(1, 1, 1600), "sample_rate": 16000}
    limit = contract.DIT_DEFAULTS["text_rope_max_pos"]
    encoder = _FakeEncoderPatcher()
    encoder.model.encode = lambda prompt, speech, device: (torch.ones(limit, 3584), torch.ones(125, 3584))
    (cond,) = encode_node.encode(encoder, None, audio, "p", 0)
    assert cond["text_embeds"].shape[0] == limit
    encoder.model.encode = lambda prompt, speech, device: (torch.ones(limit + 1, 3584), torch.ones(125, 3584))
    with pytest.raises(ValueError, match=f"{limit + 1} Qwen2.5-Omni tokens"):
        encode_node.encode(encoder, None, audio, "p", 0)


class _FakeDiT:
    in_visual_dim = 16


class _FakeBaseModel:
    def __init__(self):
        self.diffusion_model = _FakeDiT()
        self.latent_format = comfy.latent_formats.HunyuanVideo()


class _FakeModelPatcher:
    def __init__(self):
        self.model = _FakeBaseModel()


def test_reference_latent_is_scaled_and_channels_last():
    samples = torch.randn(1, 16, 1, 64, 96)  # 512x768
    out = nodes._reference_latent(_FakeModelPatcher(), {"samples": samples})  # noqa: SLF001
    assert out.shape == (1, 64, 96, 16)
    scale = comfy.latent_formats.HunyuanVideo().scale_factor
    torch.testing.assert_close(out, samples[:, :, 0].permute(0, 2, 3, 1) * scale)
    # A 4-D latent is read as one frame.
    out4 = nodes._reference_latent(_FakeModelPatcher(), {"samples": samples[:, :, 0]})  # noqa: SLF001
    assert torch.equal(out4, out)


@pytest.mark.parametrize("height, width", [(512, 512), (768, 512), (512, 768)])
def test_reference_latent_accepts_every_trained_resolution(height, width):
    samples = torch.zeros(1, 16, 1, height // 8, width // 8)
    assert nodes._reference_latent(_FakeModelPatcher(), {"samples": samples}).shape[1:3] == (height // 8, width // 8)  # noqa: SLF001


# 1024x1024 is off the buckets; 2160x3840 would also overrun the 128-position RoPE tables.
@pytest.mark.parametrize("height, width", [(48, 80), (1024, 1024), (2160, 3840), (512, 776)])
def test_reference_latent_refuses_untrained_resolutions(height, width):
    samples = torch.zeros(1, 16, 1, height // 8, width // 8)
    with pytest.raises(ValueError, match="Kandinsky S2V Reference Image"):
        nodes._reference_latent(_FakeModelPatcher(), {"samples": samples})  # noqa: SLF001


@pytest.mark.parametrize("shape", [(2, 16, 1, 64, 96), (1, 16, 3, 64, 96), (1, 4, 1, 64, 96), (16, 64, 96)])
def test_reference_latent_refuses_other_shapes(shape):
    with pytest.raises(ValueError, match="Kandinsky S2V expects"):
        nodes._reference_latent(_FakeModelPatcher(), {"samples": torch.zeros(shape)})  # noqa: SLF001


def test_node_input_defaults_follow_the_contract():
    sampler = nodes.KandinskyS2VSampler.INPUT_TYPES()["required"]
    assert sampler["seed"][1]["default"] == contract.SEED
    assert sampler["steps"][1]["default"] == contract.PIFLOW_STEPS
    encode = nodes.KandinskyS2VEncode.INPUT_TYPES()["required"]
    assert encode["prompt"][1]["default"] == contract.PROMPT
    assert encode["chunks"][1]["max"] * contract.CHUNK_SECONDS <= 300
    assert set(nodes.NODE_CLASS_MAPPINGS) == set(nodes.NODE_DISPLAY_NAME_MAPPINGS)


# --------------------------------------------------------------------------- workflows


def _build_workflow_module():
    dev = str(PACKAGE / "dev")
    if dev not in sys.path:
        sys.path.insert(0, dev)
    import build_workflow  # noqa: PLC0415

    return build_workflow


@pytest.mark.parametrize("model", ["lite", "pro"])
def test_example_workflows_are_the_generated_ones(model):
    bw = _build_workflow_module()
    name = bw.RELEASES[model][0]
    shipped = json.loads((WORKFLOWS / f"{name}.json").read_text(encoding="utf-8"))
    assert shipped == json.loads(json.dumps(bw.build(model)))


@pytest.mark.parametrize("stem", ["Kandinsky S2V 1.0 Lite distilled", "Kandinsky S2V 1.0 Pro distilled"])
def test_workflow_download_note_names_its_own_workflow(stem):
    graph = json.loads((WORKFLOWS / f"{stem}.json").read_text(encoding="utf-8"))
    by_type = {node["type"]: node for node in graph["nodes"]}
    assert by_type["MarkdownNote"]["properties"]["kandinsky_s2v_workflow"] == stem
    assert stem in model_downloads.model_plan(WORKFLOWS)
    sampler = by_type["KandinskyS2VSampler"]["widgets_values"]
    assert sampler[0] == contract.SEED and sampler[2] == contract.PIFLOW_STEPS
    unet = by_type["UNETLoader"]["widgets_values"][0]
    assert ("lite" if " Lite " in stem else "pro") in unet
    assert by_type["UNETLoader"]["properties"]["models"][0]["name"] == unet


# --------------------------------------------------------------------------- 5. model downloads


def _hf(repo, path):
    return f"https://huggingface.co/{repo}/resolve/main/{path}"


def _write_workflow(directory: Path, stem: str, items, field="models"):
    directory.mkdir(parents=True, exist_ok=True)
    graph = {"nodes": [{"id": 1, "properties": {field: items}}, {"id": 2}]}
    (directory / f"{stem}.json").write_text(json.dumps(graph), encoding="utf-8")


GOOD = {"name": "a.safetensors", "url": _hf("org/repo", "sub/a.safetensors"), "directory": "vae"}


def test_model_plan_parses_entries(tmp_path):
    _write_workflow(tmp_path, "one", [GOOD])
    _write_workflow(
        tmp_path,
        "two",
        [{**GOOD, "name": "dir/b.json", "directory": "text_encoders"}],
        field="kandinsky_s2v_required_files",
    )
    plan = model_downloads.model_plan(tmp_path)
    assert set(plan) == {"one", "two"}
    (entry,) = plan["one"]
    assert entry["repo_id"] == "org/repo"
    assert entry["filename"] == "sub/a.safetensors"
    assert plan["two"][0]["name"] == "dir/b.json"


@pytest.mark.parametrize(
    "item",
    [
        {**GOOD, "name": "../escape.safetensors"},
        {**GOOD, "name": "a/../../escape.safetensors"},
        {**GOOD, "name": "/abs.safetensors"},
        {**GOOD, "name": "a\\b.safetensors"},
        {**GOOD, "directory": "checkpoints"},
        {**GOOD, "directory": "../vae"},
        {**GOOD, "url": "http://huggingface.co/org/repo/resolve/main/a.safetensors"},
        {**GOOD, "url": "https://evil.example/org/repo/resolve/main/a.safetensors"},
        {**GOOD, "url": "https://huggingface.co.evil.example/org/repo/resolve/main/a.safetensors"},
        {**GOOD, "url": "https://user@huggingface.co/org/repo/resolve/main/a.safetensors"},
        {**GOOD, "url": "https://huggingface.co/org/repo/resolve/dev/a.safetensors"},
        {**GOOD, "url": "https://huggingface.co/org/repo/blob/main/a.safetensors"},
        {**GOOD, "url": "https://huggingface.co/org/repo/resolve/main/"},
        {**GOOD, "url": "https://huggingface.co/org/repo/resolve/main/../x"},
        {**GOOD, "url": "https://huggingface.co/org/repo/resolve/main//x"},
        {**GOOD, "url": "https://huggingface.co/org/repo/resolve/main/a?x=1"},
        {**GOOD, "url": "https://huggingface.co/org/repo/resolve/main/a#frag"},
    ],
)
def test_model_plan_rejects_bad_entries(tmp_path, item):
    _write_workflow(tmp_path, "w", [item])
    with pytest.raises(ValueError, match="Invalid packaged model manifest"):
        model_downloads.model_plan(tmp_path)


def test_model_plan_rejects_conflicts_and_empty(tmp_path):
    _write_workflow(tmp_path / "c", "one", [GOOD])
    _write_workflow(tmp_path / "c", "two", [{**GOOD, "url": _hf("org/other", "a.safetensors")}])
    with pytest.raises(ValueError, match="Conflicting"):
        model_downloads.model_plan(tmp_path / "c")
    # The same destination twice with the same source is shared, not a conflict.
    _write_workflow(tmp_path / "s", "one", [GOOD])
    _write_workflow(tmp_path / "s", "two", [GOOD, GOOD])
    assert {k: len(v) for k, v in model_downloads.model_plan(tmp_path / "s").items()} == {"one": 1, "two": 1}
    _write_workflow(tmp_path / "e", "one", [GOOD])
    _write_workflow(tmp_path / "e", "empty", [])
    with pytest.raises(ValueError, match="Empty"):
        model_downloads.model_plan(tmp_path / "e")
    (tmp_path / "none").mkdir()
    with pytest.raises(ValueError, match="Empty"):
        model_downloads.model_plan(tmp_path / "none")


def test_existing_model_searches_extra_roots(tmp_path):
    entry = {"directory": "text_encoders", "name": "Qwen/model.safetensors"}
    models, extra = tmp_path / "models", tmp_path / "extra"
    assert model_downloads.existing_model(entry, models, {}) is None
    target = extra / "Qwen" / "model.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    assert model_downloads.existing_model(entry, models, {"text_encoders": [str(extra)]}) is None  # empty file
    target.write_bytes(b"x")
    assert model_downloads.existing_model(entry, models, {"text_encoders": [str(extra)]}) == target
    assert model_downloads.existing_model(entry, models, {"vae": [str(extra)]}) is None  # other kind
    local = models / "text_encoders" / "Qwen" / "model.safetensors"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"y")
    assert model_downloads.existing_model(entry, models, {"text_encoders": [str(extra)]}) == local  # own first


def test_model_directory_counts_in_the_first_root_that_has_it(tmp_path):
    """A partial Qwen directory first in ComfyUI's order is what the loader opens; files elsewhere don't help."""
    config = {"directory": "text_encoders", "name": "Qwen/config.json"}
    shard = {"directory": "text_encoders", "name": "Qwen/model.safetensors"}
    models, extra = tmp_path / "models", tmp_path / "extra"
    (extra / "Qwen").mkdir(parents=True)
    (extra / "Qwen" / "config.json").write_text("{}")
    (models / "text_encoders" / "Qwen").mkdir(parents=True)
    (models / "text_encoders" / "Qwen" / "model.safetensors").write_bytes(b"w")
    # extra first, as an is_default extra path puts it.
    order = {"text_encoders": [str(extra), str(models / "text_encoders")]}
    assert model_downloads.existing_model(config, models, order) == extra / "Qwen" / "config.json"
    assert model_downloads.existing_model(shard, models, order) is None
    # The packaged folder first: its directory is the one, and lacks the config.
    order = {"text_encoders": [str(models / "text_encoders"), str(extra)]}
    assert model_downloads.existing_model(config, models, order) is None
    assert (
        model_downloads.existing_model(shard, models, order) == models / "text_encoders" / "Qwen" / "model.safetensors"
    )


def test_download_completes_a_partial_directory_where_it_is(tmp_path, monkeypatch):
    def fake_hf(repo_id, filename, revision, local_dir):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
        return str(path)

    monkeypatch.setattr(model_downloads, "hf_hub_download", fake_hf)
    models, extra = tmp_path / "models", tmp_path / "extra"
    (extra / "Qwen").mkdir(parents=True)
    order = {"text_encoders": [str(extra), str(models / "text_encoders")]}
    entry = {
        "directory": "text_encoders",
        "name": "Qwen/model.safetensors",
        "repo_id": "o/r",
        "filename": "model.safetensors",
    }
    assert model_downloads.download_model(entry, models, order) == "downloaded"
    assert (extra / "Qwen" / "model.safetensors").read_bytes() == b"weights"
    assert not (models / "text_encoders" / "Qwen").exists()
    assert model_downloads.download_model(entry, models, order) == "reused"


def test_download_model_moves_into_place_and_reuses(tmp_path, monkeypatch):
    calls = []

    def fake_hf(repo_id, filename, revision, local_dir):
        calls.append((repo_id, filename, revision))
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
        return str(path)

    monkeypatch.setattr(model_downloads, "hf_hub_download", fake_hf)
    (entry,) = [
        e
        for e in model_downloads.model_plan(WORKFLOWS)["Kandinsky S2V 1.0 Lite distilled"]
        if e["name"].endswith("model.safetensors.index.json")
    ]
    models = tmp_path / "models"
    assert model_downloads.download_model(entry, models, {}) == "downloaded"
    target = models / "text_encoders" / "Qwen2.5-Omni-7B" / "model.safetensors.index.json"
    assert target.read_bytes() == b"weights"
    assert calls == [("Qwen/Qwen2.5-Omni-7B", "model.safetensors.index.json", "main")]
    assert model_downloads.download_model(entry, models, {}) == "reused"
    assert len(calls) == 1
    # Present in an extra root: reused, nothing written under models/.
    other = tmp_path / "other_models"
    assert model_downloads.download_model(entry, other, {"text_encoders": [str(models / "text_encoders")]}) == "reused"
    assert not other.exists()


def test_download_errors_are_explained():
    import httpx  # noqa: PLC0415
    from huggingface_hub.errors import HfHubHTTPError  # noqa: PLC0415

    request = httpx.Request("GET", "https://huggingface.co/x")
    for status in (401, 403):
        error = HfHubHTTPError("denied", response=httpx.Response(status, request=request))
        assert "hf auth login" in model_downloads.download_error(error)
    error = HfHubHTTPError("missing", response=httpx.Response(404, request=request))
    assert "Download failed" in model_downloads.download_error(error)
    assert "disk space" in model_downloads.download_error(OSError(errno.ENOSPC, "full"))
    assert "permissions" in model_downloads.download_error(PermissionError("nope"))
    assert "Download failed" in model_downloads.download_error(RuntimeError("x"))


def _run_app(models_root, extra_roots, scenario):
    from aiohttp import web  # noqa: PLC0415
    from aiohttp.test_utils import TestClient, TestServer  # noqa: PLC0415

    async def main():
        routes = web.RouteTableDef()
        model_downloads.register_routes(routes, "kandinsky_s2v", WORKFLOWS, models_root, extra_roots)
        app = web.Application()
        app.add_routes(routes)
        async with TestClient(TestServer(app)) as client:
            return await scenario(client)

    return asyncio.run(main())


def test_route_lists_models_per_workflow(tmp_path):
    ready = tmp_path / "models" / "vae" / "hunyuan_video_vae_bf16.safetensors"
    ready.parent.mkdir(parents=True)
    ready.write_bytes(b"x")

    async def scenario(client):
        response = await client.get("/kandinsky_s2v/models")
        assert response.status == 200
        return await response.json()

    body = _run_app(tmp_path / "models", {}, scenario)
    plans = model_downloads.model_plan(WORKFLOWS)
    assert set(body["workflows"]) == set(plans)
    for stem, items in body["workflows"].items():
        assert [i["path"] for i in items] == [e["directory"] + "/" + e["name"] for e in plans[stem]]
        assert {i["path"] for i in items if i["ready"]} == {"vae/hunyuan_video_vae_bf16.safetensors"}


@pytest.mark.parametrize(
    ("body", "content_type", "message"),
    [
        ('{"confirm": true, "workflow": "Kandinsky S2V 1.0 Lite distilled"}', "text/plain", "JSON confirmation is required"),
        ("{not json", "application/json", "Invalid JSON"),
        ('{"workflow": "Kandinsky S2V 1.0 Lite distilled"}', "application/json", "confirmation is required"),
        ('{"confirm": 1, "workflow": "Kandinsky S2V 1.0 Lite distilled"}', "application/json", "confirmation is required"),
        ('{"confirm": "true", "workflow": "Kandinsky S2V 1.0 Lite distilled"}', "application/json", "confirmation is required"),
        ('{"confirm": true, "workflow": "Kandinsky S2V 1.0 Lite distilled", "x": 1}', "application/json", "confirmation"),
        ("[true]", "application/json", "confirmation is required"),
        ('{"confirm": true, "workflow": "Nope"}', "application/json", "Unknown workflow"),
        ('{"confirm": true, "workflow": ["Kandinsky S2V 1.0 Lite distilled"]}', "application/json", "Unknown workflow"),
        ('{"confirm": true, "workflow": null}', "application/json", "Unknown workflow"),
    ],
)
def test_route_download_refuses_without_confirmation(tmp_path, monkeypatch, body, content_type, message):
    monkeypatch.setattr(model_downloads, "download_model", lambda *a: pytest.fail("must not download"))
    monkeypatch.setattr(model_downloads, "hf_hub_download", lambda *a, **k: pytest.fail("must not download"))

    async def scenario(client):
        response = await client.post(
            "/kandinsky_s2v/download-models", data=body, headers={"Content-Type": content_type}
        )
        return response.status, await response.text()

    status, text = _run_app(tmp_path / "models", {}, scenario)
    assert status == 400
    assert message in text


def _post_download(client, stem):
    return client.post("/kandinsky_s2v/download-models", json={"confirm": True, "workflow": stem})


def test_route_download_streams_only_the_chosen_workflow(tmp_path, monkeypatch):
    seen = []

    def fake_download(entry, models_root, extra_roots):
        seen.append(entry["directory"] + "/" + entry["name"])
        return "reused" if entry["directory"] == "vae" else "downloaded"

    monkeypatch.setattr(model_downloads, "download_model", fake_download)
    monkeypatch.setattr(model_downloads, "hf_hub_download", lambda *a, **k: pytest.fail("network"))

    async def scenario(client):
        response = await _post_download(client, "Kandinsky S2V 1.0 Pro distilled")
        assert response.status == 200
        assert response.headers["Content-Type"].startswith("application/x-ndjson")
        return [json.loads(line) for line in (await response.text()).splitlines()]

    events = _run_app(tmp_path / "models", {}, scenario)
    plan = model_downloads.model_plan(WORKFLOWS)["Kandinsky S2V 1.0 Pro distilled"]
    paths = [e["directory"] + "/" + e["name"] for e in plan]
    assert seen == paths
    assert "diffusion_models/kandinsky_s2v_lite_distill.safetensors" not in seen
    assert events[-1] == {"status": "complete", "total": len(plan)}
    per_file = events[:-1]
    assert len(per_file) == 2 * len(plan)
    for i, path in enumerate(paths):
        checking, done = per_file[2 * i], per_file[2 * i + 1]
        assert checking == {"path": path, "index": i + 1, "total": len(plan), "status": "checking"}
        assert done["path"] == path
        assert done["status"] == ("reused" if path.startswith("vae/") else "downloaded")


def test_route_download_reports_errors_and_stops(tmp_path, monkeypatch):
    seen = []

    def fake_download(entry, models_root, extra_roots):
        seen.append(entry["name"])
        if len(seen) == 2:
            raise PermissionError("read-only")
        return "downloaded"

    monkeypatch.setattr(model_downloads, "download_model", fake_download)

    async def scenario(client):
        response = await _post_download(client, "Kandinsky S2V 1.0 Lite distilled")
        return [json.loads(line) for line in (await response.text()).splitlines()]

    events = _run_app(tmp_path / "models", {}, scenario)
    assert len(seen) == 2
    assert events[-1]["status"] == "error"
    assert "permissions" in events[-1]["message"]
    assert not any(e["status"] == "complete" for e in events)


def test_route_download_refuses_a_second_concurrent_download(tmp_path, monkeypatch):
    import threading  # noqa: PLC0415

    release = threading.Event()

    def slow_download(entry, models_root, extra_roots):
        release.wait(10)
        return "downloaded"

    monkeypatch.setattr(model_downloads, "download_model", slow_download)

    async def scenario(client):
        first = asyncio.ensure_future(_post_download(client, "Kandinsky S2V 1.0 Lite distilled"))
        for _ in range(100):
            await asyncio.sleep(0.02)
            second = await _post_download(client, "Kandinsky S2V 1.0 Pro distilled")
            if second.status == 409:
                break
        status = second.status
        release.set()
        response = await first
        lines = [json.loads(line) for line in (await response.text()).splitlines()]
        return status, lines

    status, lines = _run_app(tmp_path / "models", {}, scenario)
    assert status == 409
    assert lines[-1]["status"] == "complete"


def test_shipped_manifest_is_valid_and_hf_only():
    plans = model_downloads.model_plan(WORKFLOWS)
    for stem, plan in plans.items():
        keys = [e["directory"] + "/" + e["name"] for e in plan]
        assert len(keys) == len(set(keys)), stem
        for entry in plan:
            assert entry["url"] == _hf(entry["repo_id"], entry["filename"])
    dits = {stem: [e for e in plan if e["directory"] == "diffusion_models"] for stem, plan in plans.items()}
    assert dits["Kandinsky S2V 1.0 Lite distilled"][0]["repo_id"] == "kandinskylab/Kandinsky-S2V-1.0-Lite-distill"
    assert dits["Kandinsky S2V 1.0 Pro distilled"][0]["repo_id"] == "kandinskylab/Kandinsky-S2V-1.0-Pro-distill"
