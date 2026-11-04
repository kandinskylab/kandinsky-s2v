"""Kandinsky S2V 1.0 diffusion transformer, built from ComfyUI operations.

A port of ``kandinsky_s2v.core.components`` (``dit.py``, ``nn.py``, ``rope.py``,
``piflow_dit.py``). Module and parameter names are those of the released
checkpoints, so the published safetensors load unchanged. The arithmetic is the
reference's: the forward runs under bf16 autocast, with the AdaLN path, the time
embedding and every normalisation in fp32.

Two departures, neither of which changes the result:

* layers come from ComfyUI's ``operations``, so model management, offloading
  and weight patches apply to them;
* dense attention goes through ``optimized_attention``, which honours the
  attention backend ComfyUI was started with.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from comfy.ldm.modules.attention import optimized_attention

# The first latent frame takes a single audio token, so the rest need at least one frame.
_MIN_DURATION = 2

# Stored in fp32 by the checkpoint converter; built in fp32 here so loading keeps them so.
_FP32_KEY_PARTS = ("time_embeddings", "norm")


def _settings_for(operation_settings: dict, name: str) -> dict:
    if any(part in name for part in _FP32_KEY_PARTS):
        return {**operation_settings, "dtype": torch.float32}
    return operation_settings


def _linear(settings: dict, in_features: int, out_features: int, bias: bool = True):
    return settings["operations"].Linear(
        in_features, out_features, bias=bias, device=settings["device"], dtype=settings["dtype"]
    )


def _layer_norm(settings: dict, dim: int, elementwise_affine: bool):
    return settings["operations"].LayerNorm(
        dim, elementwise_affine=elementwise_affine, device=settings["device"], dtype=settings["dtype"]
    )


def _rms_norm(settings: dict, dim: int):
    return settings["operations"].RMSNorm(dim, device=settings["device"], dtype=settings["dtype"])


def get_freqs(dim: int, max_period: float = 10000.0) -> Tensor:
    return torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim)


def split_lengths(total: int, groups: int, device: torch.device) -> Tensor:
    """Split ``total`` items into ``groups``; boundaries are ``total*i // groups``."""
    bounds = torch.arange(groups + 1, device=device) * total // groups
    return torch.diff(bounds)


def audio_group_lengths(num_audio_tokens: int, duration: int, device: torch.device) -> Tensor:
    """First latent frame gets one audio token, the rest split the remainder evenly."""
    if duration < _MIN_DURATION:
        raise ValueError(f"duration must be >= {_MIN_DURATION} to split audio tokens, got {duration}")
    rest = split_lengths(num_audio_tokens - 1, duration - 1, device)
    first = torch.ones(1, dtype=rest.dtype, device=device)
    return torch.cat([first, rest])


@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_scale_shift_norm(norm, x: Tensor, scale: Tensor, shift: Tensor, idx: Tensor) -> Tensor:
    return norm(x.float()) * (scale.index_select(0, idx) + 1.0) + shift.index_select(0, idx)


@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_gate_sum(x: Tensor, out: Tensor, gate: Tensor, idx: Tensor) -> Tensor:
    return x + gate.index_select(0, idx) * out


@torch.autocast(device_type="cuda", enabled=False)
def apply_rotary(x: Tensor, rope: Tensor) -> Tensor:
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
    x_out = rope[..., 0] * x_[..., 0] + rope[..., 1] * x_[..., 1]
    return x_out.reshape(*x.shape)


def _to_rope(args: Tensor) -> Tensor:
    rope = torch.stack([torch.cos(args), -torch.sin(args), torch.sin(args), torch.cos(args)], dim=-1)
    return rope.view(*rope.shape[:-1], 2, 2).unsqueeze(-4)


class RoPE1D(nn.Module):
    def __init__(self, dim: int, max_pos: int = 1024, max_period: float = 10000.0):
        super().__init__()
        freq = get_freqs(dim // 2, max_period)
        pos = torch.arange(max_pos, dtype=freq.dtype)
        self.register_buffer("args", torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, pos: Tensor) -> Tensor:
        return _to_rope(self.args.to(pos.device)[pos])


class RoPE3D(nn.Module):
    """3-D RoPE; the temporal axis spans negative positions for memory frames."""

    def __init__(
        self,
        axes_dims: tuple[int, int, int],
        max_pos: tuple[int, int, int] = (128, 128, 128),
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.offsets_t = max_pos[0]
        for i, (axes_dim, ax_max_pos) in enumerate(zip(axes_dims, max_pos, strict=True)):
            freq = get_freqs(axes_dim // 2, max_period)
            pos = (
                torch.arange(-ax_max_pos, ax_max_pos, dtype=freq.dtype)
                if i == 0
                else torch.arange(ax_max_pos, dtype=freq.dtype)
            )
            self.register_buffer(f"args_{i}", torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(
        self,
        shape: tuple[int, int, int],
        pos: list[Tensor],
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tensor:
        duration, height, width = shape
        device = pos[0].device
        args_t = self.args_0.to(device)[pos[0] + self.offsets_t] / scale_factor[0]
        args_h = self.args_1.to(device)[pos[1]] / scale_factor[1]
        args_w = self.args_2.to(device)[pos[2]] / scale_factor[2]
        args = torch.cat(
            [
                args_t.view(duration, 1, 1, -1).repeat(1, height, width, 1),
                args_h.view(1, height, 1, -1).repeat(duration, 1, width, 1),
                args_w.view(1, 1, width, -1).repeat(duration, height, 1, 1),
            ],
            dim=-1,
        )
        return _to_rope(args)


class TimeEmbeddings(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, settings: dict, max_period: float = 10000.0):
        super().__init__()
        self.register_buffer("freqs", get_freqs(model_dim // 2, max_period), persistent=False)
        self.in_layer = _linear(settings, model_dim, time_dim)
        self.activation = nn.SiLU()
        self.out_layer = _linear(settings, time_dim, time_dim)

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, time: Tensor) -> Tensor:
        args = torch.outer(time.float(), self.freqs.to(device=time.device))
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.out_layer(self.activation(self.in_layer(time_embed)))


class GeneralEmbeddings(nn.Module):
    """Linear projection + LayerNorm; used for text, pooled text and audio."""

    def __init__(self, dim: int, model_dim: int, settings: dict, norm_settings: dict):
        super().__init__()
        self.in_layer = _linear(settings, dim, model_dim)
        self.norm = _layer_norm(norm_settings, model_dim, elementwise_affine=True)

    def forward(self, embed: Tensor) -> Tensor:
        embed = self.in_layer(embed)
        return self.norm(embed.float()).type_as(embed)


class VisualEmbeddings(nn.Module):
    def __init__(self, visual_dim: int, model_dim: int, patch_size: tuple[int, int, int], settings: dict):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = _linear(settings, math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x: Tensor) -> Tensor:
        """``(T, H, W, C)`` → ``(T/p0, H/p1, W/p2, model_dim)``."""
        duration, height, width, dim = x.shape
        x = (
            x.view(
                duration // self.patch_size[0],
                self.patch_size[0],
                height // self.patch_size[1],
                self.patch_size[1],
                width // self.patch_size[2],
                self.patch_size[2],
                dim,
            )
            .permute(0, 2, 4, 1, 3, 5, 6)
            .flatten(3, 6)
        )
        return self.in_layer(x)


class Modulation(nn.Module):
    def __init__(self, time_dim: int, model_dim: int, num_params: int, settings: dict):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = _linear(settings, time_dim, num_params * model_dim)

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.activation(x.float()))


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int, settings: dict):
        super().__init__()
        self.in_layer = _linear(settings, dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = _linear(settings, ff_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.activation(self.in_layer(x)))


def _dense_attention(query: Tensor, key: Tensor, value: Tensor, transformer_options: dict) -> Tensor:
    """Packed ``(S, heads, head_dim)`` in, ``(S, heads * head_dim)`` out."""
    heads = query.shape[1]
    out = optimized_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        heads,
        skip_reshape=True,
        transformer_options=transformer_options,
    )
    return out.squeeze(0)


def _pad_groups(x: Tensor, lengths: Tensor, max_len: int) -> tuple[Tensor, Tensor]:
    positions = torch.arange(max_len, device=x.device)
    valid = positions[None, :] < lengths[:, None]
    out = x.new_zeros(lengths.shape[0], max_len, *x.shape[1:])
    out[valid] = x
    return out, valid


def grouped_attention(q: Tensor, k: Tensor, v: Tensor, q_lengths: Tensor, kv_lengths: Tensor) -> Tensor:
    """Block-diagonal attention: query group ``i`` attends exactly to key group ``i``."""
    q_pad, q_valid = _pad_groups(q, q_lengths, int(q_lengths.max()))
    k_pad, kv_valid = _pad_groups(k, kv_lengths, int(kv_lengths.max()))
    v_pad, _ = _pad_groups(v, kv_lengths, int(kv_lengths.max()))
    out = F.scaled_dot_product_attention(
        q_pad.transpose(1, 2),
        k_pad.transpose(1, 2),
        v_pad.transpose(1, 2),
        attn_mask=kv_valid[:, None, None, :],
    ).transpose(1, 2)
    return out[q_valid]


class _QKVAttention(nn.Module):
    def __init__(self, num_channels: int, head_dim: int, settings: dict, norm_settings: dict):
        super().__init__()
        self.num_heads = num_channels // head_dim
        self.to_query = _linear(settings, num_channels, num_channels)
        self.to_key = _linear(settings, num_channels, num_channels)
        self.to_value = _linear(settings, num_channels, num_channels)
        self.query_norm = _rms_norm(norm_settings, head_dim)
        self.key_norm = _rms_norm(norm_settings, head_dim)
        self.out_layer = _linear(settings, num_channels, num_channels)

    def _norm_qk(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        return self.query_norm(q.float()).type_as(q), self.key_norm(k.float()).type_as(k)


class MultiheadSelfAttention(_QKVAttention):
    def forward(self, x: Tensor, rope: Tensor, transformer_options: dict) -> Tensor:
        shape = x.shape[:-1]
        query = self.to_query(x).reshape(*shape, self.num_heads, -1)
        key = self.to_key(x).reshape(*shape, self.num_heads, -1)
        value = self.to_value(x).reshape(*shape, self.num_heads, -1)
        query, key = self._norm_qk(query, key)
        query = apply_rotary(query, rope).type_as(query)
        key = apply_rotary(key, rope).type_as(key)
        return self.out_layer(_dense_attention(query, key, value, transformer_options))


class MultiheadCrossAttention(_QKVAttention):
    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        transformer_options: dict,
        q_lengths: Tensor | None = None,
        kv_lengths: Tensor | None = None,
    ) -> Tensor:
        """Dense cross-attention, or block-diagonal when group lengths are given."""
        shape, cond_shape = x.shape[:-1], cond.shape[:-1]
        query = self.to_query(x).reshape(*shape, self.num_heads, -1)
        key = self.to_key(cond).reshape(*cond_shape, self.num_heads, -1)
        value = self.to_value(cond).reshape(*cond_shape, self.num_heads, -1)
        query, key = self._norm_qk(query, key)
        if q_lengths is None:
            out = _dense_attention(query, key, value, transformer_options)
        else:
            out = grouped_attention(query, key, value, q_lengths, kv_lengths).flatten(-2, -1)
        return self.out_layer(out)


class OutLayer(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, visual_dim: int, patch_size: tuple[int, int, int], settings):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Modulation(time_dim, model_dim, 2, settings)
        self.norm = _layer_norm(settings, model_dim, elementwise_affine=False)
        self.out_layer = _linear(settings, model_dim, math.prod(patch_size) * visual_dim)

    def forward(self, visual_embed: Tensor, time_embed: Tensor, time_embed_idx: Tensor) -> Tensor:
        """``(T, H, W, model_dim)`` → ``(T*p0, H*p1, W*p2, visual_dim)``."""
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        visual_embed = apply_scale_shift_norm(
            self.norm, visual_embed, scale[:, None, None], shift[:, None, None], time_embed_idx
        ).type_as(visual_embed)
        x = self.out_layer(visual_embed)
        duration, height, width, _ = x.shape
        return (
            x.view(duration, height, width, -1, *self.patch_size)
            .permute(0, 4, 1, 5, 2, 6, 3)
            .flatten(0, 1)
            .flatten(1, 2)
            .flatten(2, 3)
        )


class TransformerTextEncoderBlock(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, ff_dim: int, head_dim: int, settings, norm_settings):
        super().__init__()
        self.text_modulation = Modulation(time_dim, model_dim, 6, settings)
        self.self_attention_norm = _layer_norm(settings, model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim, settings, norm_settings)
        self.feed_forward_norm = _layer_norm(settings, model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim, settings)

    def forward(self, x, time_embed, rope, time_embed_idx, transformer_options):
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.self_attention_norm, x, scale, shift, time_embed_idx).type_as(x)
        out = self.self_attention(out, rope, transformer_options)
        x = apply_gate_sum(x, out, gate, time_embed_idx).type_as(x)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift, time_embed_idx).type_as(x)
        out = self.feed_forward(out)
        return apply_gate_sum(x, out, gate, time_embed_idx).type_as(x)


class TransformerAudioEncoderBlock(nn.Module):
    """Plain pre-norm transformer block — no time modulation."""

    def __init__(self, model_dim: int, ff_dim: int, head_dim: int, settings, norm_settings):
        super().__init__()
        self.self_attention_norm = _layer_norm(norm_settings, model_dim, elementwise_affine=True)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim, settings, norm_settings)
        self.feed_forward_norm = _layer_norm(norm_settings, model_dim, elementwise_affine=True)
        self.feed_forward = FeedForward(model_dim, ff_dim, settings)

    def forward(self, x, rope, transformer_options):
        x = x + self.self_attention(self.self_attention_norm(x.float()).type_as(x), rope, transformer_options)
        return x + self.feed_forward(self.feed_forward_norm(x.float()).type_as(x))


class TransformerDecoderBlock(nn.Module):
    """Visual block; ``enable_audio`` adds the frame-aligned audio adapter."""

    def __init__(  # noqa: PLR0913
        self, model_dim, time_dim, ff_dim, head_dim, settings, norm_settings, enable_audio=False
    ):
        super().__init__()
        self.visual_modulation = Modulation(time_dim, model_dim, 9, settings)
        self.self_attention_norm = _layer_norm(settings, model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttention(model_dim, head_dim, settings, norm_settings)
        self.cross_attention_norm = _layer_norm(settings, model_dim, elementwise_affine=False)
        self.cross_attention = MultiheadCrossAttention(model_dim, head_dim, settings, norm_settings)
        self.enable_audio = enable_audio
        if enable_audio:
            self.cross_attention_norm_v2a = _layer_norm(norm_settings, model_dim, elementwise_affine=True)
            self.cross_attention_v2a = MultiheadCrossAttention(model_dim, head_dim, settings, norm_settings)
        self.feed_forward_norm = _layer_norm(settings, model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim, settings)

    def forward(  # noqa: PLR0913
        self,
        visual_embed_full,
        text_embed,
        audio_embed,
        time_embed_full,
        visual_rope_full,
        time_embed_idx_full,
        num_visual_tokens,
        visual_group_lengths,
        audio_group_lengths_,
        transformer_options,
    ):
        self_attn_params, cross_attn_params, ff_params = torch.chunk(self.visual_modulation(time_embed_full), 3, dim=-1)

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = apply_scale_shift_norm(
            self.self_attention_norm, visual_embed_full, scale, shift, time_embed_idx_full
        ).type_as(visual_embed_full)
        out = self.self_attention(out, visual_rope_full, transformer_options)
        visual_embed_full = apply_gate_sum(visual_embed_full, out, gate, time_embed_idx_full).type_as(visual_embed_full)

        shift, scale, gate = torch.chunk(cross_attn_params, 3, dim=-1)
        out = apply_scale_shift_norm(
            self.cross_attention_norm, visual_embed_full, scale, shift, time_embed_idx_full
        ).type_as(visual_embed_full)
        out = self.cross_attention(out, text_embed, transformer_options)
        visual_embed_full = apply_gate_sum(visual_embed_full, out, gate, time_embed_idx_full).type_as(visual_embed_full)

        if self.enable_audio:
            # Audio conditions only the noisy part of the sequence, frame by frame,
            # through an ungated residual.
            head, visual_embed = visual_embed_full[:-num_visual_tokens], visual_embed_full[-num_visual_tokens:]
            out = self.cross_attention_v2a(
                self.cross_attention_norm_v2a(visual_embed.float()).type_as(visual_embed),
                audio_embed,
                transformer_options,
                visual_group_lengths,
                audio_group_lengths_,
            )
            visual_embed_full = torch.cat([head, visual_embed + out], dim=0)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = apply_scale_shift_norm(
            self.feed_forward_norm, visual_embed_full, scale, shift, time_embed_idx_full
        ).type_as(visual_embed_full)
        out = self.feed_forward(out)
        return apply_gate_sum(visual_embed_full, out, gate, time_embed_idx_full).type_as(visual_embed_full)


class KandinskyS2VTransformer(nn.Module):
    """Speech-to-video DiT over one packed sequence ``[memory | ref | visual]``.

    ``memory`` (FramePack context at 4x / 2x / 1x temporal downsampling) and
    ``ref`` (the reference frame) are clean, so they are modulated with ``t = 0``
    while ``visual`` uses the real timestep. With ``n_grid > 1`` (the π-Flow
    distilled checkpoints) the head is ``n_grid`` times wider and the output
    carries a grid axis: ``(T, n_grid, H, W, C)``.
    """

    def __init__(  # noqa: PLR0913
        self,
        in_visual_dim: int = 16,
        n_grid: int = 1,
        in_text_dim: int = 3584,
        in_text_dim2: int = 768,
        in_audio_dim: int = 3584,
        time_dim: int = 512,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        frame_buckets: tuple[int, int, int] = (2, 4, 24),
        index_rope_ref_image: int = 40,
        patch_proj1x_coef: tuple[int, int, int] = (1, 1, 1),
        patch_proj2x_coef: tuple[int, int, int] = (2, 2, 2),
        patch_proj4x_coef: tuple[int, int, int] = (4, 4, 4),
        model_dim: int = 1792,
        ff_dim: int = 7168,
        num_text_blocks: int = 2,
        num_audio_blocks: int = 2,
        num_visual_blocks: int = 32,
        axes_dims: tuple[int, int, int] = (16, 24, 24),
        audio_inject_layers: tuple[int, ...] = (),
        text_rope_max_pos: int = 4096,
        image_model=None,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        settings = {"operations": operations, "device": device, "dtype": dtype}
        norm_settings = _settings_for(settings, "norm")
        head_dim = sum(axes_dims)
        self.dtype = dtype
        self.in_visual_dim = in_visual_dim
        self.n_grid = n_grid
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.patch_size = tuple(patch_size)
        self.frame_buckets = tuple(frame_buckets)
        self.index_rope_ref_image = index_rope_ref_image
        self.memory_coefs = (tuple(patch_proj1x_coef), tuple(patch_proj2x_coef), tuple(patch_proj4x_coef))
        self.audio_inject_layers = tuple(audio_inject_layers)

        self.time_embeddings = TimeEmbeddings(model_dim, time_dim, _settings_for(settings, "time_embeddings"))
        self.text_embeddings = GeneralEmbeddings(in_text_dim, model_dim, settings, norm_settings)
        self.pooled_text_embeddings = GeneralEmbeddings(in_text_dim2, time_dim, settings, norm_settings)
        self.audio_embeddings = GeneralEmbeddings(in_audio_dim, model_dim, settings, norm_settings)

        self.visual_embeddings = VisualEmbeddings(in_visual_dim, model_dim, self.patch_size, settings)
        for name, coef in zip(("proj1x", "proj2x", "proj4x"), self.memory_coefs, strict=True):
            scaled = tuple(p * c for p, c in zip(self.patch_size, coef, strict=True))
            setattr(self, f"visual_embeddings_{name}", VisualEmbeddings(in_visual_dim, model_dim, scaled, settings))
        # 0 = visual, 1 = reference frame, 2 = memory
        self.mask_embeddings = operations.Embedding(3, model_dim, device=device, dtype=dtype)

        self.text_rope_embeddings = RoPE1D(head_dim, max_pos=text_rope_max_pos)
        self.text_transformer_blocks = nn.ModuleList(
            [
                TransformerTextEncoderBlock(model_dim, time_dim, ff_dim, head_dim, settings, norm_settings)
                for _ in range(num_text_blocks)
            ]
        )

        self.audio_rope_embeddings = RoPE1D(head_dim)
        self.audio_transformer_blocks = nn.ModuleList(
            [
                TransformerAudioEncoderBlock(model_dim, ff_dim, head_dim, settings, norm_settings)
                for _ in range(num_audio_blocks)
            ]
        )

        self.visual_rope_embeddings = RoPE3D(tuple(axes_dims))
        self.visual_transformer_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    model_dim,
                    time_dim,
                    ff_dim,
                    head_dim,
                    settings,
                    norm_settings,
                    enable_audio=(i in self.audio_inject_layers),
                )
                for i in range(num_visual_blocks)
            ]
        )

        self.out_layer = OutLayer(model_dim, time_dim, in_visual_dim * n_grid, self.patch_size, settings)

    def _embed_memory(self, memory: list[Tensor], memory_rope_pos: list[list[Tensor]], scale_factor):
        """``memory`` in ``[1x, 2x, 4x]`` order; tokens are packed ``4x | 2x | 1x``."""
        embedders = (self.visual_embeddings_proj1x, self.visual_embeddings_proj2x, self.visual_embeddings_proj4x)
        embeds, ropes = [], []
        for index in (2, 1, 0):
            embed = embedders[index](memory[index])
            rope = self.visual_rope_embeddings(embed.shape[:-1], memory_rope_pos[index], scale_factor)
            embeds.append(embed.flatten(0, 2))
            ropes.append(rope.flatten(0, 2))
        return torch.cat(embeds, dim=0), torch.cat(ropes, dim=0)

    def forward(  # noqa: PLR0913
        self,
        x: Tensor,
        ref: Tensor,
        memory: list[Tensor],
        text_embed: Tensor,
        audio_embed: Tensor,
        pooled_text_embed: Tensor,
        time: Tensor,
        visual_rope_pos: list[Tensor],
        ref_rope_pos: list[Tensor],
        memory_rope_pos: list[list[Tensor]],
        text_rope_pos: Tensor,
        audio_rope_pos: Tensor,
        scale_factor: tuple[float, float, float],
        transformer_options: dict | None = None,
    ) -> Tensor:
        """Latents are ``(T, H, W, C)``; ``time`` is the flow time in ``[0, 1000]``."""
        transformer_options = {} if transformer_options is None else transformer_options
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self._forward(
                x,
                ref,
                memory,
                text_embed,
                audio_embed,
                pooled_text_embed,
                time,
                visual_rope_pos,
                ref_rope_pos,
                memory_rope_pos,
                text_rope_pos,
                audio_rope_pos,
                scale_factor,
                transformer_options,
            )
        if self.n_grid == 1:
            return out
        # (T, H, W, n_grid * C) → (T, n_grid, H, W, C). The split is grid-major,
        # as the distilled head was trained; channel-major would shuffle the grid.
        grid = out.view(*out.shape[:-1], self.n_grid, self.in_visual_dim)
        return grid.permute(0, 3, 1, 2, 4)

    def _forward(  # noqa: PLR0913
        self,
        x,
        ref,
        memory,
        text_embed,
        audio_embed,
        pooled_text_embed,
        time,
        visual_rope_pos,
        ref_rope_pos,
        memory_rope_pos,
        text_rope_pos,
        audio_rope_pos,
        scale_factor,
        transformer_options,
    ):
        device = x.device
        duration = x.shape[0]

        text_embed = self.text_embeddings(text_embed)
        audio_embed = self.audio_embeddings(audio_embed)

        time_embed = self.time_embeddings(time)
        time_embed_clean = self.time_embeddings(torch.zeros_like(time))
        pooled = self.pooled_text_embeddings(pooled_text_embed)
        time_embed = time_embed + pooled
        time_embed_clean = time_embed_clean + pooled
        # row 0 → memory / ref (t=0), row 1 → visual (t)
        time_embed_full = torch.stack((time_embed_clean, time_embed), dim=1).flatten(0, 1)

        text_rope = self.text_rope_embeddings(text_rope_pos)
        text_time_embed_idx = torch.zeros(text_embed.shape[0], dtype=torch.int32, device=device)
        for block in self.text_transformer_blocks:
            text_embed = block(text_embed, time_embed, text_rope, text_time_embed_idx, transformer_options)

        audio_rope = self.audio_rope_embeddings(audio_rope_pos)
        for block in self.audio_transformer_blocks:
            audio_embed = block(audio_embed, audio_rope, transformer_options)

        visual_embed = self.visual_embeddings(x)
        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor).flatten(0, 2)
        visual_embed = visual_embed.flatten(0, 2)

        ref_embed = self.visual_embeddings(ref)
        ref_rope = self.visual_rope_embeddings(ref_embed.shape[:-1], ref_rope_pos, scale_factor).flatten(0, 2)
        ref_embed = ref_embed.flatten(0, 2)

        memory_embed, memory_rope = self._embed_memory(memory, memory_rope_pos, scale_factor)

        mask_ids = torch.arange(3, device=device)
        # out_dtype: with fp8 weights (UNETLoader weight_dtype) the embedding would
        # come back in fp8, which torch cannot add to the bf16/fp32 stream.
        visual_embed = visual_embed + self.mask_embeddings(mask_ids[0], out_dtype=visual_embed.dtype)
        ref_embed = ref_embed + self.mask_embeddings(mask_ids[1], out_dtype=ref_embed.dtype)
        memory_embed = memory_embed + self.mask_embeddings(mask_ids[2], out_dtype=memory_embed.dtype)

        num_visual_tokens = visual_embed.shape[0]
        num_clean_tokens = memory_embed.shape[0] + ref_embed.shape[0]
        visual_embed_full = torch.cat([memory_embed, ref_embed, visual_embed], dim=0)
        visual_rope_full = torch.cat([memory_rope, ref_rope, visual_rope], dim=0)
        time_embed_idx_full = torch.cat(
            [
                torch.zeros(num_clean_tokens, dtype=torch.int32, device=device),
                torch.ones(num_visual_tokens, dtype=torch.int32, device=device),
            ]
        )

        visual_groups = split_lengths(num_visual_tokens, duration, device)
        audio_groups = audio_group_lengths(audio_embed.shape[0], duration, device)

        transformer_options = transformer_options.copy()
        transformer_options["total_blocks"] = len(self.visual_transformer_blocks)
        for index, block in enumerate(self.visual_transformer_blocks):
            transformer_options["block_index"] = index
            visual_embed_full = block(
                visual_embed_full,
                text_embed,
                audio_embed,
                time_embed_full,
                visual_rope_full,
                time_embed_idx_full,
                num_visual_tokens,
                visual_groups,
                audio_groups,
                transformer_options,
            )

        visual_embed = visual_embed_full[-num_visual_tokens:].reshape(*visual_shape, self.model_dim)
        visual_time_embed_idx = torch.zeros(visual_shape[0], dtype=torch.int32, device=device)
        return self.out_layer(visual_embed, time_embed, visual_time_embed_idx)
