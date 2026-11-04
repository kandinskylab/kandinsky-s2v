"""CPU checks that need no weights. Parity with the reference lives in dev/."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import comfy.ops
import comfy.utils
import example_inputs
from kandinsky_s2v_comfy import contract, sampling
from kandinsky_s2v_comfy.encoder import AUDIO_SAMPLE_RATE, resample_speech
from kandinsky_s2v_comfy.ldm.model import KandinskyS2VTransformer
from kandinsky_s2v_comfy.nodes import clip_l_pooled
from kandinsky_s2v_comfy.register import check_release, detect_s2v_config
from model_downloads import model_plan

PACKAGE = Path(__file__).resolve().parents[1]

TINY = {
    "in_visual_dim": 16,
    "n_grid": 5,
    "in_text_dim": 48,
    "in_text_dim2": 24,
    "in_audio_dim": 40,
    "time_dim": 32,
    "model_dim": 128,
    "ff_dim": 256,
    "num_text_blocks": 1,
    "num_audio_blocks": 2,
    "num_visual_blocks": 3,
    "axes_dims": (16, 24, 24),
    "audio_inject_layers": (0, 2),
}


def test_detection_recovers_the_architecture():
    model = KandinskyS2VTransformer(**TINY, operations=comfy.ops.disable_weight_init, device="meta")
    config = detect_s2v_config(model.state_dict(), "")
    for key, value in TINY.items():
        assert config[key] == value, key


@pytest.mark.parametrize("release", sorted(contract.RELEASE_DIT_CONFIGS))
def test_release_check_accepts_both_distilled_releases(release):
    assert check_release(dict(contract.RELEASE_DIT_CONFIGS[release])) == release


def test_release_check_refuses_the_rest_against_the_closest_release():
    with pytest.raises(ValueError, match="model_dim: checkpoint=128") as error:
        check_release(dict(TINY))
    assert "n_grid" not in str(error.value)  # the tiny model keeps the distilled head
    # Lite and Pro base: the same architecture with a one-cell head.
    for release in ("Lite distilled", "Pro distilled"):
        with pytest.raises(ValueError, match=f"closest: {release}; n_grid: checkpoint=1, release=5\\)"):
            check_release({**contract.RELEASE_DIT_CONFIGS[release], "n_grid": 1})


def test_workflow_manifests_are_per_model():
    plans = {
        stem: {f"{entry['directory']}/{entry['name']}" for entry in plan}
        for stem, plan in model_plan(PACKAGE / "example_workflows").items()
    }
    lite, pro = plans["Kandinsky S2V 1.0 Lite distilled"], plans["Kandinsky S2V 1.0 Pro distilled"]
    assert set(plans) == {"Kandinsky S2V 1.0 Lite distilled", "Kandinsky S2V 1.0 Pro distilled"}
    assert lite - pro == {"diffusion_models/kandinsky_s2v_lite_distill.safetensors"}
    assert pro - lite == {"diffusion_models/kandinsky_s2v_pro_distill.safetensors"}
    for plan in (lite, pro):
        assert "vae/hunyuan_video_vae_bf16.safetensors" in plan
        assert "text_encoders/clip_l.safetensors" in plan
        omni = [path for path in plan if path.startswith("text_encoders/Qwen2.5-Omni-7B/")]
        assert "text_encoders/Qwen2.5-Omni-7B/model.safetensors.index.json" in omni
        assert sum(path.endswith(".safetensors") for path in omni) == 5  # noqa: PLR2004


class _FakeCLIP:
    """Just enough of comfy.sd.CLIP with an SD1 CLIP-L to see the tokens it is given."""

    def __init__(self):
        from comfy.sd1_clip import SD1Tokenizer  # noqa: PLC0415

        self.tokenizer = SD1Tokenizer()
        self.cond_stage_model = type("SD1ClipModel", (), {})()
        self.calls = []

    def encode_from_tokens(self, tokens, return_pooled=False):
        self.calls.append((tokens, return_pooled))
        return None, torch.zeros(1, 768)


def test_clip_tokens_are_the_reference_truncation():
    from transformers import CLIPTokenizer  # noqa: PLC0415

    clip = _FakeCLIP()
    clip_l_pooled(clip, contract.PROMPT)
    (tokens, return_pooled), = clip.calls
    assert return_pooled == "unprojected"
    ids = [token for token, weight in tokens["l"][0]]
    reference = CLIPTokenizer.from_pretrained(os.path.join(os.path.dirname(comfy.utils.__file__), "sd1_tokenizer"))
    expected = reference(
        [contract.PROMPT], max_length=contract.CLIP_MAX_LENGTH, truncation=True, padding="max_length"
    )["input_ids"][0]
    assert ids == expected


def test_clip_other_than_clip_l_is_refused():
    clip = _FakeCLIP()
    clip.cond_stage_model = object()
    with pytest.raises(ValueError, match="stable_diffusion"):
        clip_l_pooled(clip, "a prompt")


def test_example_inputs_never_overwrite(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "reference.png").write_bytes(b"bundled")
    monkeypatch.setattr(example_inputs, "EXAMPLE_INPUTS", bundled)
    inputs = tmp_path / "input"
    target = inputs / "reference.png"
    assert example_inputs.install_example_inputs(inputs) == (target,)
    target.write_bytes(b"edited")
    assert example_inputs.install_example_inputs(inputs) == ()
    assert target.read_bytes() == b"edited"


def test_workflow_media_inputs_are_bundled_top_level_files():
    # Load Image and Load Audio list only the top of input/; a subfolder path is
    # shown by the frontend as "no file selected".
    bundled = {path.name for path in (PACKAGE / "example_inputs").iterdir()}
    examples = set()
    for workflow in (PACKAGE / "example_workflows").glob("*.json"):
        graph = json.loads(workflow.read_text(encoding="utf-8"))
        media = {node["type"]: node["widgets_values"][0] for node in graph["nodes"]
                 if node["type"] in ("LoadImage", "LoadAudio")}
        assert set(media) == {"LoadImage", "LoadAudio"}, workflow.name
        assert set(media.values()) <= bundled, (workflow.name, media)
        examples.add(tuple(sorted(media.items())))
    assert len(examples) == 2  # noqa: PLR2004 - Lite and Pro open with different examples


def test_norms_and_time_embeddings_stay_fp32():
    model = KandinskyS2VTransformer(
        **TINY, operations=comfy.ops.disable_weight_init, device="meta", dtype=torch.bfloat16
    )
    for name, tensor in model.state_dict().items():
        expected = torch.float32 if "norm" in name or "time_embeddings" in name else torch.bfloat16
        assert tensor.dtype == expected, name


@pytest.mark.parametrize("channels", [1, 2])
def test_speech_is_padded_and_cut_to_whole_chunks(channels):
    rate = 44100
    waveform = torch.rand(channels, int(7.3 * rate)) - 0.5
    for seconds in (5.0, 10.0):
        speech = resample_speech(waveform, rate, seconds)
        assert speech.ndim == 1
        assert speech.dtype == np.float32
        assert abs(len(speech) - seconds * AUDIO_SAMPLE_RATE) <= 1
    padded = resample_speech(waveform, rate, 10.0)
    assert np.abs(padded[int(7.5 * AUDIO_SAMPLE_RATE) :]).max() < 1e-3


def test_audio_tokens_split_evenly_over_chunks():
    embeds = torch.arange(377 * 2, dtype=torch.float32).view(377, 2)
    parts = sampling.split_audio_embeds(embeds, 3)
    assert [p.shape[0] for p in parts] == [125, 125, 125]
    assert torch.equal(torch.cat(parts), embeds[:375])


def test_audio_rope_covers_every_latent_frame():
    pos = sampling.audio_rope_pos(125, contract.LATENT_FRAMES, "cpu")
    assert pos.shape == (125,)
    assert pos[0] == 0
    assert (pos == 0).sum() == 1
    assert torch.equal(pos.unique(), torch.arange(contract.LATENT_FRAMES, dtype=torch.int32))


def test_memory_rolls_the_tail_of_the_chunk():
    coefs = (contract.DIT_DEFAULTS["patch_proj1x_coef"], contract.DIT_DEFAULTS["patch_proj2x_coef"],
             contract.DIT_DEFAULTS["patch_proj4x_coef"])
    buckets = contract.DIT_DEFAULTS["frame_buckets"]
    patch = contract.DIT_DEFAULTS["patch_size"]
    latent = torch.arange(contract.LATENT_FRAMES, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 64, 96, 16)
    memory = sampling.roll_memory(latent, buckets, coefs, patch)
    assert [m.shape[0] for m in memory] == list(buckets)
    assert memory[0][:, 0, 0, 0].tolist() == [30.0, 31.0]
    assert memory[1][:, 0, 0, 0].tolist() == [26.0, 27.0, 28.0, 29.0]
    assert memory[2][0, 0, 0, 0].item() == 2.0
    initial = sampling.init_memory(buckets, coefs, patch, 64, 96, 16, "cpu")
    assert [m.shape for m in initial] == [m.shape for m in memory]
