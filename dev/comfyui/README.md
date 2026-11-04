# Kandinsky S2V

Kandinsky S2V 1.0 for ComfyUI: a talking avatar from one reference image and a
speech track.

This extension runs the two π-Flow students of Kandinsky S2V 1.0:
**Lite distilled** (2B) and **Pro distilled** (21B). Both take 16 model calls
per 5-second chunk at 25 fps, 512x512, 512x768 or 768x512; the same nodes run
either, and **Load Diffusion Model** picks which. Clips longer than 5 seconds are generated chunk by chunk, each chunk
continuing from the last frames of the previous one, up to the 300 seconds the
speech encoder accepts.

Requires ComfyUI **0.38.0+**, Python 3.10+ and a CUDA GPU. Model loading and
offloading are managed by ComfyUI; the Lite workflow runs with ComfyUI
limited to 24 GB of VRAM, because it unloads the speech encoder (17.5 GB)
before sampling. Pro distilled runs in the same 24 GB: ComfyUI's dynamic VRAM
keeps only part of its 40 GB DiT on the GPU. Measured on an H100 with ComfyUI left 24 GB,
the example workflow cut to two chunks (10 s) took 115 s on Lite and 470 s on
Pro, model loading included.

## Install

Once the Registry version is available:

1. Open **ComfyUI Manager** and its custom-node list.
2. Search for `kandinsky-s2v` (publisher `kandinskylab`) and click **Install**.
3. Restart ComfyUI, then follow **Models** and **Run** below.

Alternatively, with [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)
and ComfyUI Manager installed:

```bash
comfy --workspace /path/to/ComfyUI node install kandinsky-s2v
```

For manual installation, clone `kandinskylab/kandinsky-s2v` outside
`custom_nodes/`, copy the contents of `dev/comfyui/` into
`ComfyUI/custom_nodes/kandinsky-s2v/`, then install the dependencies with
**ComfyUI's Python** and restart:

```bash
python -m pip install -r ComfyUI/custom_nodes/kandinsky-s2v/requirements.txt
```

## Models

Open a bundled workflow and click **Download models** in its **Models**
note, or choose **Kandinsky S2V → Kandinsky S2V — Download models** from the
top menu (it fetches the models of the open workflow, or asks which).

The button downloads that workflow's DiT, the HunyuanVideo VAE, CLIP-L and
Qwen2.5-Omni-7B: about 27 GB for Lite, 22 GB of it Qwen, and about 66 GB for
Pro, 43 GB of it the Pro DiT. The two workflows share everything but the DiT. Files are placed in ComfyUI's
model folders; existing files, including configured extra model paths, are
reused. Nothing is downloaded during install or startup. For gated or private
weights, run `hf auth login` on the ComfyUI server first.

### Manual downloads

Destinations are relative to your **ComfyUI folder**; create missing
directories.

| Download | Destination |
| --- | --- |
| [Kandinsky S2V 1.0 Lite distilled — DiT](https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Lite-distill/resolve/main/model/kandinsky_s2v_lite_distill.safetensors) | `models/diffusion_models/kandinsky_s2v_lite_distill.safetensors` |
| [Kandinsky S2V 1.0 Pro distilled — DiT](https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Pro-distill/resolve/main/model/kandinsky_s2v_pro_distill.safetensors) | `models/diffusion_models/kandinsky_s2v_pro_distill.safetensors` |
| [HunyuanVideo VAE](https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/vae/hunyuan_video_vae_bf16.safetensors) | `models/vae/hunyuan_video_vae_bf16.safetensors` |
| [CLIP-L — pooled text embedding](https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/clip_l.safetensors) | `models/text_encoders/clip_l.safetensors` |
| [Qwen2.5-Omni-7B — speech and text encoder](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) (the whole repository) | `models/text_encoders/Qwen2.5-Omni-7B/` |

```bash
cd ComfyUI/models
hf download kandinskylab/Kandinsky-S2V-1.0-Lite-distill model/kandinsky_s2v_lite_distill.safetensors --local-dir /tmp/s2v
mv /tmp/s2v/model/kandinsky_s2v_lite_distill.safetensors diffusion_models/
# Pro distilled, if wanted:
hf download kandinskylab/Kandinsky-S2V-1.0-Pro-distill model/kandinsky_s2v_pro_distill.safetensors --local-dir /tmp/s2v
mv /tmp/s2v/model/kandinsky_s2v_pro_distill.safetensors diffusion_models/
hf download Comfy-Org/HunyuanVideo_repackaged split_files/vae/hunyuan_video_vae_bf16.safetensors split_files/text_encoders/clip_l.safetensors --local-dir /tmp/s2v
mv /tmp/s2v/split_files/vae/hunyuan_video_vae_bf16.safetensors vae/
mv /tmp/s2v/split_files/text_encoders/clip_l.safetensors text_encoders/
hf download Qwen/Qwen2.5-Omni-7B --local-dir text_encoders/Qwen2.5-Omni-7B
```

Only the Thinker half of Qwen2.5-Omni-7B is loaded; its Talker weights stay on
disk. The DiT of a Diffusers bundle,
`Kandinsky-S2V-1.0-{Lite,Pro}-distill-Diffusers/transformer/diffusion_pytorch_model.safetensors`,
is the same file under another name and loads as well.

## Run

Open **Workflow → Browse Templates**, find this extension and choose
`Kandinsky S2V 1.0 Lite distilled` or `Kandinsky S2V 1.0 Pro distilled`, or drag
[`example_workflows/Kandinsky S2V 1.0 Lite distilled.json`](example_workflows/Kandinsky%20S2V%201.0%20Lite%20distilled.json)
or [`… Pro distilled.json`](example_workflows/Kandinsky%20S2V%201.0%20Pro%20distilled.json) onto the
canvas. The two differ in the DiT selected in **Load Diffusion Model** and in
their example: each opens with its own bundled reference image and speech track
(`kandinsky_s2v_reference.png` and `kandinsky_s2v_speech.wav` for Lite,
`kandinsky_s2v_pro_reference.png` and `kandinsky_s2v_pro_speech.wav` for Pro),
installed into `ComfyUI/input/` on first start, never overwriting a file already
there; pick your own in
**Load Image** and **Load Audio**, then run it to get an `.mp4` with the speech
as its sound.

The graph uses stock nodes wherever the model allows:

```text
Load Diffusion Model ─────────────────────────────────────────────────────────┐
Load Image → Kandinsky S2V Reference Image → VAE Encode ──────────────────────┤
Load Audio ─┬→ Kandinsky S2V Encode (← Kandinsky S2V Encoder Loader, ← Load CLIP) ┼→ Kandinsky S2V Sampler → VAE Decode
            │                                                                                                    │
            └──→ Create Video ← Kandinsky S2V Trim To Audio ←────────────────────────────────────────────────────┘
                     │
                     └→ Save Video
```

| Node | What it does |
|---|---|
| **Kandinsky S2V Encoder Loader** | Loads the Qwen2.5-Omni Thinker from `models/text_encoders`. |
| **Kandinsky S2V Encode** | Encodes the prompt and the speech with Qwen2.5-Omni, and the prompt with CLIP-L from the stock **Load CLIP** (`clip_l.safetensors`, type `stable_diffusion`). The track is padded with silence to whole 5-second chunks. `chunks = 0` covers the whole track; a smaller number generates only its beginning. |
| **Kandinsky S2V Reference Image** | Fits the image to the closest trained resolution by aspect ratio: resize to cover, then center-crop. |
| **Kandinsky S2V Sampler** | The chunked π-Flow sampler. `steps` is the number of π-Flow segments per chunk; the model was distilled at 16. Guidance is distilled into the model, so there is no negative prompt and no CFG scale. |
| **Kandinsky S2V Trim To Audio** | Drops the frames generated over the silence padding and returns the 25 fps rate for **Create Video**. |

The prompt describes the scene and how the speaker should behave; the default
is the one from the reference configuration. The lip-sync follows the audio,
not the prompt. **Load Diffusion Model** refuses any checkpoint other than Lite
distilled and Pro distilled with the list of what differs: Lite and Pro (not
distilled) need another sampler. The HD grid of Pro distilled
(`s2v_pro_distill_hd.yaml` in the reference) is not supported.

## Fidelity

The nodes reproduce the reference implementation
([kandinskylab/kandinsky-s2v](https://github.com/kandinskylab/kandinsky-s2v))
on the same inputs, Lite distilled and Pro distilled alike:

- one DiT forward is bit-identical, with both the legacy and the dynamic-VRAM
  model patcher, under ComfyUI's default PyTorch attention (sage, xformers or
  flash attention compute it differently);
- text and speech embeddings and the preprocessed reference frame are
  bit-identical;
- the pooled CLIP-L embedding gets the same token ids as the reference and
  matches it to 6e-4 relative L2: **Load CLIP** runs `clip_l.safetensors`
  in fp16, the reference CLIP in fp32;
- with the same embeddings and reference latent, three chunks of sampling
  match to 6e-8 relative L2 (the rounding of the latent scale factor).

The one other intended difference: the reference samples the VAE posterior of
the reference frame, while ComfyUI's **VAE Encode** takes its mean.

## License

MIT, see [LICENSE](LICENSE). The model weights carry their own licenses.
