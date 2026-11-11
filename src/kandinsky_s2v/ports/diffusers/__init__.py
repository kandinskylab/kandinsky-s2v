"""Diffusers port of the Kandinsky S2V 1.0 model, scheduler and modular pipeline.

The files here follow upstream Diffusers conventions — self-contained modules,
``KandinskyS2V``-prefixed classes, absolute ``diffusers.*`` imports — so they
can be dropped into a Diffusers checkout with the imports rewritten to relative
ones. The pipeline is the ``modular`` package: Diffusers carries S2V as a
modular pipeline only.

The distilled checkpoints use Diffusers' own ``PiflowScheduler``, which no
release ships yet; ``scheduling_piflow.py`` is a copy of it, including the
final-step fix the S2V pull request makes to it.
"""

from .scheduling_piflow import PiflowScheduler
from .transformer_kandinsky_s2v import KandinskyS2VTransformer3DModel

__all__ = [
    "PiflowScheduler",
    "KandinskyS2VTransformer3DModel",
]
