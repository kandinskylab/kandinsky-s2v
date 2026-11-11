# What does this PR do?

Adds Kandinsky S2V 1.0, a speech-to-video model. Given a reference frame and a speech track, the
model animates the person in the frame to match the speech: the whole figure, not only the
head. Audio is a condition, not an output.

It was reviewed and approved in the staging repository:
https://github.com/huggingface/diffusers-new-model-addition-kandinskyX-avatar/pull/3

Since that approval, one commit addresses the self-review below (compile-friendly speech
attention, the full set of model testers, workflow and guider contracts, dead code):
https://github.com/huggingface/diffusers/pull/15164/commits/9ba233beabbbbb2f0589c27c685c2044ae3bc056
Everything else is the approved code, rebased on current `main`.

It is versioned apart from the other Kandinsky models, so it gets files of its own, and it is
integrated as a modular pipeline only — no `DiffusionPipeline` class, nothing in `pipelines/`:

- `KandinskyS2VTransformer3DModel` — `models/transformers/transformer_kandinsky_s2v.py`
- the distilled checkpoints use the existing `PiflowScheduler`, fixed to match the reference
  sampler in kandinskylab/kandinsky-6 bit for bit (it differed on the final step and on the
  raw-time walk)
- `KandinskyS2VAutoBlocks` / `KandinskyS2VModularPipeline` and
  `KandinskyS2VDistilledAutoBlocks` / `KandinskyS2VDistilledModularPipeline` —
  `modular_pipelines/kandinsky_s2v/`

The pipeline takes a PIL image and a mono 16 kHz waveform and reads no files. Its top-level
blocks — `text_encoder`, `vae_encoder`, `denoise` and `decode` — each run on their own; the text
encoder plans the chunks from the speech, and declares no guider.

A clip is generated chunk by chunk (5 s each) with FramePack memory carried between chunks.
The chunk loop is a `LoopSequentialPipelineBlocks`; the timestep loop is a plain loop inside
one block, as in Helios, since loop blocks cannot nest. Base and distilled are separate block
sets rather than a switch: the distilled scheduler and 16-step default need component specs of
their own, and the distilled blocks have no guider and no negative prompt, since guidance is
distilled into the student. Each checkpoint names its blocks in `modular_model_index.json`, so
`ModularPipeline.from_pretrained` picks the variant from the checkpoint alone.

Docs: a pipeline page and a transformer page; the `PiflowScheduler` page mentions S2V.

## Usage

```python
import librosa
import torch
from diffusers import ModularPipeline
from diffusers.utils import encode_video, load_image

pipe = ModularPipeline.from_pretrained("kandinskylab/Kandinsky-S2V-1.0-Lite-distill-Diffusers")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

image = load_image("reference.png")
audio, _ = librosa.load("speech.wav", sr=16000)  # a mono waveform at 16 kHz, for the model
speech, sample_rate = librosa.load("speech.wav", sr=None)  # the original track, added to the output video

video = pipe(image=image, audio=audio, output="videos")[0]
encode_video(
    video,
    fps=25,
    output_path="avatar.mp4",
    audio=torch.from_numpy(speech)[None],
    audio_sample_rate=sample_rate,
)
```

Pro does not fit an 80 GB card fully resident; the pipeline page shows offloading through a
`ComponentsManager` with `memory_reserve_margin="40GiB"` (50.4 GiB peak on Pro distilled).

## Checkpoints

Four bundles, one per configuration, each with a `modular_model_index.json`:
`kandinskylab/Kandinsky-S2V-1.0-{Lite,Pro,Lite-distill,Pro-distill}-Diffusers`.

They are private until release; the reviewers of the staging PR have access. To run the usage
example above, set `HF_TOKEN` to a token with access to `kandinskylab`. The tests do not need
it: the two tiny repositories they load are public.

## Tests

- the transformer, with the testers `utils/generate_model_tests.py` emits: model, memory,
  compile (including dynamic shapes and repeated blocks), training and attention;
- for base and distilled, a tester config with the fast, loading, memory and workflow mixins
  (`expected_workflow_defaults` pins each one's components, inputs and defaults), plus the
  guider mixin for base; each loads a tiny public repository —
  `kandinskylab/tiny-kandinsky-s2v-modular-pipe` and
  `kandinskylab/tiny-kandinsky-s2v-distilled-modular-pipe` (happy to move them under
  `hf-internal-testing`);
- that memory carries over two chunks, that a track longer than the chunks is cut to them, and
  that speech past the encoder limit is rejected;
- that the standalone text encoder encodes a negative prompt only when one is passed;
- that a multichannel, empty or integer waveform is rejected;
- fp16 inference with the pipeline loaded in fp16, which keeps `_keep_in_fp32_modules` in fp32;
- that the RoPE tables stay fp32 after `.to(torch.bfloat16)`;
- `PiflowScheduler`'s final step against a hand-rolled policy and rollout.

Skipped, each for a property of the model:

- batching and `num_images_per_prompt`, for base and distilled: one prompt, one video per call;
- `test_fuse_unfuse_qkv_projections`: the projections are named `to_query`/`to_key`/`to_value`
  after the training checkpoints, as in Kandinsky 6;
- `test_layerwise_casting_memory`: the affine normalisations take fp32 input, as the
  checkpoints keep them in fp32, and the test first casts the whole model with `.to()`;
- `test_compile_works_with_aot`: AOTInductor lowers the lookup into the RoPE tables, plain fp32
  attributes so that casting the model keeps them fp32, to a CPU kernel on a CUDA tensor.

On one H100, on this branch (rebased on current `main`): 112 passed, 16 skipped (the above, one
test that needs several GPUs, and the workflow-map tests for a pipeline with a single
workflow). The Kandinsky 6 tests give the same result as on `main`: 166 passed, and the same 3
failures (`test_pipeline_kandinsky6_sr::test_inference_batch_single_identical` and
`test_pipeline_kandinsky6_ti2va::…::test_group_offloading_inference_block_level`, both
variants). `make quality` (with the pinned ruff 0.9.10), `check_copies`, `check_dummies`,
`check_doc_toc`, `check_support_list`, `check_forward_call_docstrings`,
`check_return_annotations` and `modular_auto_docstring` pass.

## Self-review notes

I ran the `self-review` skill, with the rubric from current `main`, in two rounds and fixed
what it reported as blocking:

- `FromOriginalModelMixin` without single-file loading wired up: removed;
- model tests now compose every tester `generate_model_tests.py` emits;
- the speech cross-attention broke `torch.compile(fullgraph=True)` on data-dependent shapes: it
  now computes the per-frame audio groups as Python ints and gathers them into a padded, masked
  batch. One forward stays bit-identical to the original model on all four checkpoints;
- `expected_workflow_defaults` for both variants, and the guider tester for base.

Left as they are, for your call:

- `prompt` is optional and defaults to the prompt the checkpoints were trained with (an empty
  prompt gives no text tokens). The default is long, so it shows in the auto docstrings.
- There is no `latents` input: noise is drawn per chunk from `generator`, as in `wan_animate_2`.
- In the base pipeline, `num_inference_steps=N` gives N−1 Euler updates: the raw grid is the
  one the original sampler uses, and the block description says so.
- The VAE decode runs under bf16 autocast on CUDA, as in the original pipeline.
- The text encoder reads `requires_unconditional_embeds` without declaring a guider, as
  `wan_animate_2` does, so that it runs on its own.
- The `PiflowScheduler` fix changes Kandinsky 6 distilled outputs slightly. No Kandinsky 6 test
  pins output slices, and they pass as on `main`.
- The speech adapter always passes a padding mask, so attention backends that reject masks
  (`flash`, `_flash_3`, `sage`, `aiter`) fail in those blocks; native and the varlen backends
  work.
- `test_layerwise_casting_memory` and `test_compile_works_with_aot` are skipped with a reason;
  strict `xfail` would also fit. `_skip_layerwise_casting_patterns` is not set.
- The tiny test repositories live under `kandinskylab`.

## Before submitting
- [x] Did you use an AI agent (Claude Code, Codex, Cursor, etc.) to help with this PR? If so:
  - [x] Did you read the [Coding with AI agents](https://huggingface.co/docs/diffusers/main/en/conceptual/contribution#coding-with-ai-agents) guide?
  - [x] Did you run the [`self-review`](https://github.com/huggingface/diffusers/blob/main/.ai/skills/self-review/SKILL.md) skill on the diff?
  - [x] Did you share the final self-review notes in the PR description or a comment?
- [x] Did you read the [contributor guideline](https://huggingface.co/docs/diffusers/main/en/conceptual/contribution)?
- [x] Did you read our [philosophy doc](https://huggingface.co/docs/diffusers/main/en/conceptual/philosophy)? (important for complex PRs)
- [x] Was this discussed/approved via a GitHub issue or the [forum](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers/63)? Please add a link to it if that's the case.
- [x] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/diffusers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/diffusers/tree/main/docs#writing-source-documentation).
- [x] Did you write any new necessary tests?
- [x] Are you the author (or part of the team) of the model/pipeline (only applicable for model/pipeline related PRs)?

## Who can review?

@yiyixuxu @sayakpaul
