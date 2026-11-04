"""ComfyUI ``BaseModel`` for Kandinsky S2V 1.0.

Wrapping the DiT in a ``BaseModel`` gives it ComfyUI's model management,
offloading and weight patches like any built-in model. Sampling is not the
stock KSampler's: see ``nodes.KandinskyS2VSampler``.
"""

from __future__ import annotations

import comfy.model_base

from .ldm.model import KandinskyS2VTransformer


class KandinskyS2VModel(comfy.model_base.BaseModel):
    def __init__(self, model_config, model_type=comfy.model_base.ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=KandinskyS2VTransformer)

    def apply_model(self, *args, **kwargs):
        raise RuntimeError(
            "Kandinsky S2V generates a clip chunk by chunk from a reference frame and a speech track, "
            "which the stock samplers cannot do. Sample it with the 'Kandinsky S2V Sampler' node."
        )
