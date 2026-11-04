"""Constants of Kandinsky S2V 1.0 Lite and Pro distilled, from ``configs/s2v_{lite,pro}_distill.yaml``.

The two share every generation setting; only the DiT architecture differs.

Kept free of ComfyUI and model imports so the node definitions can be
inspected without loading anything.
"""

from typing import Final

# DiT settings the checkpoint does not reveal by its tensor shapes.
DIT_DEFAULTS: Final[dict] = {
    "patch_size": (1, 2, 2),
    "frame_buckets": (2, 4, 24),
    "index_rope_ref_image": 40,
    "patch_proj1x_coef": (1, 1, 1),
    "patch_proj2x_coef": (2, 2, 2),
    "patch_proj4x_coef": (4, 4, 4),
    "text_rope_max_pos": 4096,
}

# The released architectures, as the detector reads them off the tensor shapes.
# A checkpoint that matches none is refused at load time: the sampler and its
# constants below are those of these two models.
_SHARED_DIT_CONFIG: Final[dict] = {
    "in_visual_dim": 16,
    "n_grid": 5,
    "in_text_dim": 3584,
    "in_text_dim2": 768,
    "in_audio_dim": 3584,
}
RELEASE_DIT_CONFIGS: Final[dict[str, dict]] = {
    "Lite distilled": {
        **_SHARED_DIT_CONFIG,
        "time_dim": 512,
        "model_dim": 1792,
        "ff_dim": 7168,
        "num_text_blocks": 2,
        "num_audio_blocks": 2,
        "num_visual_blocks": 32,
        "axes_dims": (16, 24, 24),
        "audio_inject_layers": (0, 3, 6, 9, 12, 15, 18, 21, 23, 25, 27, 29, 31),
    },
    "Pro distilled": {
        **_SHARED_DIT_CONFIG,
        "time_dim": 1024,
        "model_dim": 4096,
        "ff_dim": 16384,
        "num_text_blocks": 4,
        "num_audio_blocks": 4,
        "num_visual_blocks": 60,
        "axes_dims": (32, 48, 48),
        "audio_inject_layers": (0, 4, 8, 12, 16, 20, 24, 28, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59),
    },
}

FPS: Final[int] = 25
LATENT_FRAMES: Final[int] = 32  # per chunk: (32 - 1) * 4 + 1 = 125 pixel frames, 5 s
TEMPORAL_COMPRESSION: Final[int] = 4
SPATIAL_COMPRESSION: Final[int] = 8  # HunyuanVideo VAE, pixels per latent cell
CHUNK_FRAMES: Final[int] = (LATENT_FRAMES - 1) * TEMPORAL_COMPRESSION + 1
CHUNK_SECONDS: Final[float] = CHUNK_FRAMES / FPS

# (height, width) buckets; the reference frame picks the closest aspect ratio.
RESOLUTIONS: Final[tuple[tuple[int, int], ...]] = ((512, 512), (768, 512), (512, 768))
# RoPE scale of the 512 training buckets.
ROPE_SCALE_FACTOR: Final[tuple[float, float, float]] = (1.0, 2.0, 2.0)

PIFLOW_STEPS: Final[int] = 16
PIFLOW_SHIFT: Final[float] = 10.0
PIFLOW_POLICY_SUBSTEPS: Final[int] = 128
PIFLOW_FINAL_STEP_SIZE_SCALE: Final[float] = 0.5
PIFLOW_EPS: Final[float] = 1e-6
SEED: Final[int] = 5

CLIP_MAX_LENGTH: Final[int] = 77

PROMPT: Final[str] = (
    "The video presents a stable medium close-up of a subject facing the camera directly. The subject exhibits "
    "natural, lifelike body dynamics: the head and shoulders move fluidly with the rhythm of speech. The lighting "
    "is even, maintaining a balanced natural exposure to clearly define facial features without washout. "
    "Throughout the sequence, the subject engages in continuous speech marked by precise, high-fidelity "
    "articulation. The lips move in a rhythmic, natural pattern: alternating between brief firm closures and "
    "relaxed openings. During these openings, the tips of the upper teeth are momentarily and realistically "
    "visible, exhibiting natural enamel texture. The upper lip lifts subtly over the teeth during enunciation, "
    "while the jaw performs minimal vertical oscillations synchronized with the speech cadence. Facial "
    "micro-expressions are active yet restrained; eyebrows rise and lower slightly with minute asymmetry. The "
    "eyes maintain a steady, engaged gaze forward, punctuated by smooth, deliberate blinks."
)
