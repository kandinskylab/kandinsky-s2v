"""π-Flow sampling of a speech-driven clip, chunk by chunk.

A port of ``kandinsky_s2v.core.algo`` (``piflow_math.py``, ``piflow_sampler.py``,
``prepare_latents.py``, ``prepare_ropes.py``, ``chunks.py``) and of the chunk loop
in ``KandinskyS2VPipeline.__call__``. Latents are ``(T, H, W, C)`` and already
scaled by the VAE scaling factor, as in the reference.

The π-Flow code is a literal port of the code the distilled checkpoints were
trained with, including two spots that look like defects and are not: see the
notes on ``clamp`` in ``DXPolicy`` and ``piflow_denoise_chunk``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from .ldm.model import audio_group_lengths

_MIN_GRID_POINTS = 2


def shift_timesteps(t: Tensor, shift: float) -> Tensor:
    return shift * t / (1 + (shift - 1) * t)


class DXPolicy:
    """Dynamic-x0 policy: a grid of x0 predictions, linearly interpolated in raw time."""

    def __init__(  # noqa: PLR0913
        self,
        denoising_output: Tensor,
        x_t_src: Tensor,
        sigma_t_src: Tensor,
        segment_size: Tensor,
        shift: float = 1.0,
        eps: float = 1e-4,
    ):
        self.x_t_src = x_t_src
        self.ndim = x_t_src.dim()
        self.shift = shift
        self.eps = eps

        self.sigma_t_src = sigma_t_src.reshape(*sigma_t_src.size(), *((self.ndim - sigma_t_src.dim()) * [1]))
        self.raw_t_src = self._unwarp_t(self.sigma_t_src)
        seg = segment_size
        if seg.dim() < self.raw_t_src.dim():
            seg = seg.reshape(*seg.size(), *((self.ndim - seg.dim()) * [1]))
        # clamp(min=0) here while the sampler clamps to eps: the distillation code
        # does exactly this, and the student was trained against it. Keep both.
        self.raw_t_dst = (self.raw_t_src - seg).clamp(min=0)
        self.segment_size = (self.raw_t_src - self.raw_t_dst).clamp(min=eps)

        self.denoising_output_x_0 = x_t_src.unsqueeze(1) - self.sigma_t_src.unsqueeze(1) * denoising_output

    @staticmethod
    def _interpolate(x: Tensor, t: Tensor) -> Tensor:
        n = x.size(1)
        if n < _MIN_GRID_POINTS:
            return x.squeeze(1)
        t = t.clamp(min=0, max=1) * (n - 1)
        t0 = t.floor().to(torch.long).clamp(min=0, max=n - 2)
        t1 = t0 + 1
        t0t1 = torch.stack([t0, t1], dim=1)
        x0x1 = torch.gather(x, dim=1, index=t0t1.expand(-1, -1, *x.shape[2:]))
        return (t1 - t) * x0x1[:, 0] + (t - t0) * x0x1[:, 1]

    def _unwarp_t(self, sigma_t: Tensor) -> Tensor:
        return sigma_t / (self.shift + (1 - self.shift) * sigma_t)

    def pi(self, x_t: Tensor, sigma_t: Tensor) -> Tensor:
        sigma_t = sigma_t.reshape(*sigma_t.size(), *((self.ndim - sigma_t.dim()) * [1]))
        raw_t = self._unwarp_t(sigma_t)
        x_0 = self._interpolate(self.denoising_output_x_0, (raw_t - self.raw_t_dst) / self.segment_size)
        return (x_t - x_0) / sigma_t.clamp(min=self.eps)


def policy_rollout_fm(  # noqa: PLR0913
    x_t_start: Tensor,
    sigma_t_start: Tensor,
    raw_t_start: Tensor,
    raw_t_end: Tensor,
    total_substeps: int,
    policy: DXPolicy,
) -> Tensor:
    """Integrate the policy ODE across a segment with a fixed substep size."""
    num_tokens = x_t_start.size(0)
    ndim = x_t_start.dim()
    raw_t_start = raw_t_start.reshape(num_tokens, *((ndim - 1) * [1]))
    raw_t_end = raw_t_end.reshape(num_tokens, *((ndim - 1) * [1]))

    delta_raw_t = raw_t_start - raw_t_end
    num_substeps = (delta_raw_t * total_substeps).round().to(torch.long).clamp(min=1)
    substep_size = delta_raw_t / num_substeps

    raw_t, sigma_t, x_t = raw_t_start, sigma_t_start, x_t_start
    for substep_id in range(int(num_substeps.max().item())):
        u = policy.pi(x_t, sigma_t)
        raw_t_minus = (raw_t - substep_size).clamp(min=0)
        sigma_t_minus = shift_timesteps(raw_t_minus, policy.shift)
        x_t_minus = x_t + u * (sigma_t_minus - sigma_t)
        active = num_substeps > substep_id
        x_t = torch.where(active, x_t_minus, x_t)
        sigma_t = torch.where(active, sigma_t_minus, sigma_t)
        raw_t = torch.where(active, raw_t_minus, raw_t)
    return x_t


@dataclass
class ChunkInputs:
    """Everything that stays fixed while one chunk is denoised."""

    ref: Tensor
    memory: list[Tensor]  # [1x, 2x, 4x]
    text_embeds: Tensor
    pooled_embed: Tensor
    audio_embeds: Tensor
    visual_rope_pos: list[Tensor]
    ref_rope_pos: list[Tensor]
    memory_rope_pos: list[list[Tensor]]
    text_rope_pos: Tensor
    audio_rope_pos: Tensor
    scale_factor: tuple[float, float, float]


@dataclass(frozen=True)
class PiFlowSettings:
    num_steps: int = 16
    shift: float = 10.0
    num_policy_substeps: int = 128
    final_step_size_scale: float = 0.5
    eps: float = 1e-6


def _dit_forward(dit, latent: Tensor, sigma: Tensor, inputs: ChunkInputs, transformer_options: dict) -> Tensor:
    return dit(
        latent,
        inputs.ref,
        inputs.memory,
        inputs.text_embeds,
        inputs.audio_embeds,
        inputs.pooled_embed,
        sigma * 1000,
        inputs.visual_rope_pos,
        inputs.ref_rope_pos,
        inputs.memory_rope_pos,
        inputs.text_rope_pos,
        inputs.audio_rope_pos,
        inputs.scale_factor,
        transformer_options,
    )


@torch.no_grad()
def piflow_denoise_chunk(  # noqa: PLR0913
    latent: Tensor,
    dit,
    inputs: ChunkInputs,
    settings: PiFlowSettings,
    transformer_options: dict,
    step_callback: Callable[[int, Tensor, Tensor], None] | None = None,
) -> Tensor:
    """Denoise one chunk in ``num_steps`` DX segments, one DiT forward each."""
    device = latent.device
    num_tokens = latent.shape[0]
    extra_dims = (latent.dim() - 1) * [1]
    shift = settings.shift

    raw_t_src = torch.ones(1, device=device, dtype=torch.float32)
    base_segment_size = 1.0 / (float(settings.num_steps) - (1.0 - settings.final_step_size_scale))

    for step_id in range(settings.num_steps):
        is_final_step = step_id == settings.num_steps - 1
        segment_size = base_segment_size * (settings.final_step_size_scale if is_final_step else 1.0)
        # eps here, 0 inside DXPolicy: see the note there.
        raw_t_dst = (raw_t_src - segment_size).clamp(min=settings.eps)

        sigma_t_src = shift_timesteps(raw_t_src, shift)
        sigma_t_src_tok = sigma_t_src.repeat_interleave(num_tokens).reshape(num_tokens, *extra_dims)
        segment_size_tok = torch.full_like(sigma_t_src_tok, segment_size)

        grid = _dit_forward(dit, latent, sigma_t_src, inputs, transformer_options)
        policy = DXPolicy(
            denoising_output=grid,
            x_t_src=latent,
            sigma_t_src=sigma_t_src_tok,
            segment_size=segment_size_tok,
            shift=shift,
            eps=settings.eps,
        )
        latent = policy_rollout_fm(
            x_t_start=latent,
            sigma_t_start=sigma_t_src_tok,
            raw_t_start=raw_t_src.repeat_interleave(num_tokens),
            raw_t_end=raw_t_dst.repeat_interleave(num_tokens),
            total_substeps=settings.num_policy_substeps,
            policy=policy,
        )
        raw_t_src = raw_t_dst
        if step_callback is not None:
            step_callback(step_id, policy.denoising_output_x_0[:, 0], latent)
    return latent


def _align(size: int, multiple: int) -> int:
    return (size // multiple) * multiple


def crop_reference_latent(ref: Tensor, patch_size: tuple[int, int, int]) -> Tensor:
    """Crop spatial dims of ``(1, H, W, C)`` to a whole number of patches."""
    return ref[:, : _align(ref.shape[1], patch_size[1]), : _align(ref.shape[2], patch_size[2]), :]


def init_noise(  # noqa: PLR0913
    duration: int, height: int, width: int, dim: int, patch_size, generator: torch.Generator, device
) -> Tensor:
    return torch.randn(
        duration,
        _align(height, patch_size[1]),
        _align(width, patch_size[2]),
        dim,
        device=device,
        generator=generator,
    )


def init_memory(frame_buckets, coefs, patch_size, height: int, width: int, dim: int, device) -> list[Tensor]:
    """Zero FramePack memory for the first chunk, in ``[1x, 2x, 4x]`` order."""
    buffers = []
    for bucket, coef in zip(frame_buckets, coefs, strict=True):
        h_mult = patch_size[1] * coef[1]
        w_mult = patch_size[2] * coef[2]
        buffers.append(torch.zeros(bucket, _align(height, h_mult), _align(width, w_mult), dim, device=device))
    return buffers


def roll_memory(latent: Tensor, frame_buckets, coefs, patch_size) -> list[Tensor]:
    """The tail of this chunk becomes the next chunk's ``[1x, 2x, 4x]`` memory."""
    height, width = latent.shape[1], latent.shape[2]
    offsets = [0]
    for bucket in frame_buckets:
        offsets.append(offsets[-1] + bucket)
    buffers = []
    for i, coef in enumerate(coefs):
        start, stop = offsets[i + 1], offsets[i]
        window = latent[latent.shape[0] - start : latent.shape[0] - stop]
        h_mult = patch_size[1] * coef[1]
        w_mult = patch_size[2] * coef[2]
        buffers.append(window[:, : _align(height, h_mult), : _align(width, w_mult), :].clone())
    return buffers


def visual_rope_pos(duration: int, height: int, width: int, patch_size, device) -> list[Tensor]:
    return [
        torch.arange(duration, device=device),
        torch.arange(height // patch_size[1], device=device),
        torch.arange(width // patch_size[2], device=device),
    ]


def ref_rope_pos(index_rope_ref_image: int, height: int, width: int, patch_size, device) -> list[Tensor]:
    return [
        torch.tensor([index_rope_ref_image], device=device),
        torch.arange(height // patch_size[1], device=device),
        torch.arange(width // patch_size[2], device=device),
    ]


def memory_rope_pos(memory: list[Tensor], frame_buckets, coefs, patch_size, device) -> list[list[Tensor]]:
    """Memory sits at negative temporal positions, just before the chunk."""
    bounds = [0]
    for bucket in frame_buckets:
        bounds.append(bounds[-1] + bucket)
    out = []
    for i, (buffer, coef) in enumerate(zip(memory, coefs, strict=True)):
        start, stop = -bounds[i + 1], -bounds[i]
        height = buffer.shape[1] // patch_size[1]
        width = buffer.shape[2] // patch_size[2]
        out.append(
            [
                torch.arange(start, stop, coef[0], device=device),
                torch.arange(0, height, coef[1], device=device),
                torch.arange(0, width, coef[2], device=device),
            ]
        )
    return out


def audio_rope_pos(num_audio_tokens: int, duration: int, device) -> Tensor:
    """Audio token → index of the latent frame it belongs to."""
    lengths = audio_group_lengths(num_audio_tokens, duration, device)
    return torch.arange(duration, device=device).repeat_interleave(lengths).to(torch.int32)


def split_audio_embeds(audio_embeds: Tensor, chunks: int) -> list[Tensor]:
    """Split ``(S_audio, D)`` evenly over ``chunks``, dropping the ragged tail."""
    usable = (audio_embeds.shape[0] // chunks) * chunks
    if usable == 0:
        raise ValueError(f"{audio_embeds.shape[0]} audio tokens cannot be split into {chunks} chunks")
    return list(audio_embeds[:usable].view(chunks, -1, audio_embeds.shape[1]).unbind(0))


@torch.no_grad()
def sample_clip(  # noqa: PLR0913
    dit,
    ref: Tensor,
    text_embeds: Tensor,
    pooled_embed: Tensor,
    audio_embeds: Tensor,
    chunks: int,
    latent_frames: int,
    seed: int,
    settings: PiFlowSettings,
    scale_factor: tuple[float, float, float],
    transformer_options: dict,
    step_callback: Callable[[int, int, Tensor, Tensor], None] | None = None,
) -> list[Tensor]:
    """Generate ``chunks`` chunks; returns one ``(T, H, W, C)`` latent per chunk.

    ``ref`` is the scaled reference latent ``(1, H, W, C)`` on the target device.
    """
    device = ref.device
    patch_size = dit.patch_size
    frame_buckets = dit.frame_buckets
    coefs = dit.memory_coefs
    ref = crop_reference_latent(ref, patch_size)
    latent_height, latent_width = ref.shape[1], ref.shape[2]

    memory = init_memory(frame_buckets, coefs, patch_size, latent_height, latent_width, dit.in_visual_dim, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    vis_pos = visual_rope_pos(latent_frames, latent_height, latent_width, patch_size, device)
    ref_pos = ref_rope_pos(dit.index_rope_ref_image, latent_height, latent_width, patch_size, device)
    text_pos = torch.arange(text_embeds.shape[0], device=device)
    audio_per_chunk = split_audio_embeds(audio_embeds, chunks)

    outputs = []
    for index in range(chunks):
        latent = init_noise(
            latent_frames, latent_height, latent_width, dit.in_visual_dim, patch_size, generator, device
        )
        inputs = ChunkInputs(
            ref=ref,
            memory=memory,
            text_embeds=text_embeds,
            pooled_embed=pooled_embed,
            audio_embeds=audio_per_chunk[index],
            visual_rope_pos=vis_pos,
            ref_rope_pos=ref_pos,
            memory_rope_pos=memory_rope_pos(memory, frame_buckets, coefs, patch_size, device),
            text_rope_pos=text_pos,
            audio_rope_pos=audio_rope_pos(audio_per_chunk[index].shape[0], latent_frames, device),
            scale_factor=scale_factor,
        )
        chunk_callback = None
        if step_callback is not None:

            def chunk_callback(step, x0, x, chunk=index):
                step_callback(chunk, step, x0, x)

        latent = piflow_denoise_chunk(latent, dit, inputs, settings, transformer_options, chunk_callback)
        memory = roll_memory(latent, frame_buckets, coefs, patch_size)
        outputs.append(latent)
    return outputs
