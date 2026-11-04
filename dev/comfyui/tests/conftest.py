"""The nodes import ComfyUI; point COMFYUI at a checkout to run the tests."""

import os
import sys
from pathlib import Path

import pytest

COMFYUI = os.environ.get("COMFYUI")
if not COMFYUI:
    pytest.exit("Set COMFYUI to a ComfyUI checkout", returncode=4)

sys.path.insert(0, COMFYUI)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import comfy.cli_args  # noqa: E402

if not torch.cuda.is_available():
    comfy.cli_args.args.cpu = True
