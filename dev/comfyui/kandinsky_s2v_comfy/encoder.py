"""Text and speech conditioning: the Qwen2.5-Omni Thinker.

A port of the Omni half of ``kandinsky_s2v.core.components.omni_embedder`` that
takes a waveform instead of a file path. The Thinker supplies the dense text
tokens (last hidden layer, sliced to the user text span) and, through a forward
hook on its audio tower, the speech features that condition the DiT: 25 tokens
per second at 16 kHz.

ComfyUI has no Qwen2.5-Omni text encoder, so it runs through transformers. The
pooled CLIP-L embedding of the other half comes from ComfyUI's own CLIP: see
``nodes.clip_l_pooled``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import transformers
from torch import Tensor, nn

AUDIO_SAMPLE_RATE = 16000

_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)
# The chat template only needs to know an audio item is there; the waveform
# itself goes to the processor.
_AUDIO_PLACEHOLDER = "speech.wav"

_DTYPE_KWARG = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"  # noqa: PLR2004

@contextmanager
def _quiet_transformers() -> Iterator[None]:
    from transformers.utils import logging as hf_logging  # noqa: PLC0415

    verbosity, bars = hf_logging.get_verbosity(), hf_logging.is_progress_bar_enabled()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        yield
    finally:
        hf_logging.set_verbosity(verbosity)
        if bars:
            hf_logging.enable_progress_bar()


def _thinker_attention() -> str:
    from transformers.utils import import_utils  # noqa: PLC0415

    for probe, name in (
        ("is_flash_attn_3_available", "flash_attention_3"),
        ("is_flash_attn_2_available", "flash_attention_2"),
    ):
        checker = getattr(import_utils, probe, None)
        if checker is not None and checker():
            return name
    return "sdpa"


def load_thinker(omni_path: str):
    """The Thinker alone, out of the full Omni checkpoint (Talker weights are skipped)."""
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration  # noqa: PLC0415

    with _quiet_transformers():
        loaded: Any = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            omni_path,
            attn_implementation=_thinker_attention(),
            output_loading_info=True,
            **{_DTYPE_KWARG: torch.bfloat16},
        )
    thinker, info = loaded
    if info["missing_keys"]:
        raise RuntimeError(f"Thinker weights missing from {omni_path}: {sorted(info['missing_keys'])[:5]}")
    return thinker.eval()


class S2VConditioningEncoder(nn.Module):
    """The Qwen2.5-Omni Thinker and its processor, as one module for ComfyUI's model management."""

    def __init__(self, omni_path: str):
        super().__init__()
        from transformers import Qwen2_5OmniProcessor  # noqa: PLC0415

        self.thinker = load_thinker(omni_path)
        # The processor parses the whole Omni config, whose Talker carries token ids
        # transformers checks against its own, smaller vocabulary.
        with _quiet_transformers():
            self.processor = Qwen2_5OmniProcessor.from_pretrained(omni_path)

    @property
    def max_audio_seconds(self) -> float:
        extractor = self.processor.feature_extractor
        return extractor.n_samples / extractor.sampling_rate

    def _conversation(self, text: str) -> list[dict]:
        return [
            {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": text}, {"type": "audio", "audio": _AUDIO_PLACEHOLDER}],
            },
        ]

    @torch.no_grad()
    def encode(self, text: str, speech: np.ndarray, device: torch.device) -> tuple[Tensor, Tensor]:
        """``speech`` is mono float32 at 16 kHz.

        Returns ``(text_embeds (S_text, 3584), audio_embeds (S_audio, 3584))``.
        """
        chat_text = self.processor.apply_chat_template(
            [self._conversation(text)], add_generation_prompt=False, tokenize=False
        )
        inputs = self.processor(
            text=chat_text,
            audio=[speech],
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(device).to(self.thinker.dtype)

        captured: dict[str, Tensor] = {}

        def hook(_module, _inputs, output):
            captured["features"] = getattr(output, "last_hidden_state", output)

        handle = self.thinker.audio_tower.register_forward_hook(hook)
        try:
            output = self.thinker(**inputs, output_hidden_states=True, return_dict=True)
        finally:
            handle.remove()

        text_embeds = self._slice_user_text(output.hidden_states[-1], inputs)
        return text_embeds, captured["features"]

    def _slice_user_text(self, hidden_states: Tensor, inputs) -> Tensor:
        """Hidden states of the user's text span only: after ``user\\n``, before the audio."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        seq_len = input_ids.shape[1]
        device = input_ids.device

        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        audio_start_token_id = self.thinker.config.audio_start_token_id
        user_role_len = len(self.processor.tokenizer("user\n", add_special_tokens=False).input_ids)

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        last_im_start = torch.where(input_ids == im_start_id, positions, -1).max(dim=1).values
        start = (last_im_start + 1 + user_role_len).unsqueeze(1)
        end = torch.where(input_ids == audio_start_token_id, positions, seq_len - 2).min(dim=1).values.unsqueeze(1)
        mask = (positions >= start) & (positions < end) & (attention_mask == 1)
        return hidden_states[mask]

def resample_speech(waveform: Tensor, sample_rate: int, seconds: float) -> np.ndarray:
    """``(C, N)`` waveform → mono 16 kHz float32 covering exactly ``seconds``.

    Mirrors the reference, which pads the track with silence to whole chunks,
    reads that many seconds and lets librosa downmix and resample.
    """
    import librosa  # noqa: PLC0415

    data = waveform.detach().cpu().to(torch.float32).numpy()
    original = data.shape[-1] / sample_rate
    if seconds > original + 0.01:
        pad = int((seconds - original) * sample_rate)
        data = np.concatenate([data, np.zeros((data.shape[0], pad), dtype=data.dtype)], axis=-1)
    data = data[:, : int(seconds * sample_rate)]
    mono = librosa.to_mono(data) if data.shape[0] > 1 else data[0]
    return librosa.resample(mono, orig_sr=sample_rate, target_sr=AUDIO_SAMPLE_RATE, res_type="soxr_hq")


def is_omni_directory(path: str) -> bool:
    config = os.path.join(path, "config.json")
    if not os.path.isfile(config):
        return False
    try:
        with open(config, encoding="utf-8") as handle:
            model_type = json.load(handle).get("model_type", "")
    except (OSError, ValueError):
        return False
    return model_type.startswith("qwen2_5_omni")

