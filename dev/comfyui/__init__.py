from pathlib import Path

import folder_paths
from server import PromptServer

from .example_inputs import install_example_inputs
from .kandinsky_s2v_comfy.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .kandinsky_s2v_comfy.register import register
from .model_downloads import register_routes

register()
install_example_inputs()

if getattr(PromptServer, "instance", None) is not None:
    register_routes(
        PromptServer.instance.routes,
        "kandinsky_s2v",
        Path(__file__).parent / "example_workflows",
        Path(folder_paths.models_dir),
        {
            kind: list(folder_paths.folder_names_and_paths.get(kind, ([], set()))[0])
            for kind in ("diffusion_models", "text_encoders", "vae")
        },
    )

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
