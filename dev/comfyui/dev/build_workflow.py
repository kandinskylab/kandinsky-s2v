"""Write example_workflows/Kandinsky S2V 1.0 {Lite,Pro}.json, the UI form of the graph in
kandinsky_s2v_comfy_graph.py, one per released model.

    python dev/build_workflow.py

The loaders carry the model manifest: ``models`` is what the ComfyUI frontend
checks for missing files, ``kandinsky_s2v_required_files`` the rest of the
Qwen2.5-Omni directory, and model_downloads.py fetches both on request.
"""

from __future__ import annotations

import json
from pathlib import Path

from kandinsky_s2v_comfy_graph import DIT_FILES, EXAMPLE_INPUTS, PROMPT

REPO = Path(__file__).resolve().parents[1]
PACKAGE = "kandinsky-s2v"  # the Registry id, [project].name in pyproject.toml


def hf(repo: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{path}"


# model -> workflow name, the label in its texts, its Hub repo, the download
# size of the workflow's whole manifest, and a fixed workflow id.
RELEASES = {
    "lite": ("Kandinsky S2V 1.0 Lite distilled", "Lite distilled", "kandinskylab/Kandinsky-S2V-1.0-Lite-distill",
             "about 27 GB, 22 GB of it Qwen", "4f1d7a52-8c1e-4d3b-9d7e-5b2a6c0e9a11"),
    "pro": ("Kandinsky S2V 1.0 Pro distilled", "Pro distilled", "kandinskylab/Kandinsky-S2V-1.0-Pro-distill",
            "about 66 GB, 43 GB of it the Pro DiT and 22 GB Qwen", "9c2e6b18-3f4a-4e7d-8a51-0d6f2b7c4e93"),
}


def dit_entry(model: str) -> dict:
    repo = RELEASES[model][2]
    return {"name": DIT_FILES[model], "url": hf(repo, f"model/{DIT_FILES[model]}"), "directory": "diffusion_models"}


VAE = {
    "name": "hunyuan_video_vae_bf16.safetensors",
    "url": hf("Comfy-Org/HunyuanVideo_repackaged", "split_files/vae/hunyuan_video_vae_bf16.safetensors"),
    "directory": "vae",
}
CLIP_L = {
    "name": "clip_l.safetensors",
    "url": hf("Comfy-Org/HunyuanVideo_repackaged", "split_files/text_encoders/clip_l.safetensors"),
    "directory": "text_encoders",
}
OMNI_DIR = "Qwen2.5-Omni-7B"
# Everything the Thinker and its processor read; the Talker's spk_dict.pt is left out.
OMNI_FILES = [
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    *(f"model-{i:05d}-of-00005.safetensors" for i in range(1, 6)),
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]
OMNI = [
    {"name": f"{OMNI_DIR}/{name}", "url": hf("Qwen/Qwen2.5-Omni-7B", name), "directory": "text_encoders"}
    for name in OMNI_FILES
]


def note(model: str) -> str:
    _, label, _, size, _ = RELEASES[model]
    return "\n\n".join([
        "Install **Kandinsky S2V** (`kandinsky-s2v`) in Manager; restart.",
        f"Click **Download models** below (or the **Kandinsky S2V** menu). Downloads the {label} DiT, "
        f"the HunyuanVideo VAE, CLIP-L and Qwen2.5-Omni-7B: {size}.",
        "Pick a reference image in **Load Image** and a speech track in **Load Audio**. The prompt describes the "
        "scene; the lip-sync follows the audio. `chunks = 0` covers the whole track, up to 300 s.",
        f"Default: **{label}, 16 π-Flow steps per 5-second chunk**. Guidance is distilled into the model: "
        "no negative prompt, no CFG.",
        "ComfyUI **0.38+** required. For gated/private HF weights, run `hf auth login` on the server.",
    ])


def graph_nodes(model: str) -> list:
    # id, type, pos, size, inputs [(name, type, source id, source slot, widget?)], outputs [(name, type)], widgets
    return [
        (1, "UNETLoader", [40, 40], [340, 90], [], [("MODEL", "MODEL")], [DIT_FILES[model], "default"]),
        (2, "VAELoader", [40, 180], [340, 60], [], [("VAE", "VAE")], [VAE["name"]]),
        (3, "KandinskyS2VEncoderLoader", [40, 290], [340, 60], [], [("encoder", "KANDINSKY_S2V_ENCODER")], [OMNI_DIR]),
        (14, "CLIPLoader", [40, 400], [340, 110], [], [("CLIP", "CLIP")],
         [CLIP_L["name"], "stable_diffusion", "default"]),
        (4, "LoadImage", [40, 560], [340, 330], [], [("IMAGE", "IMAGE"), ("MASK", "MASK")], [EXAMPLE_INPUTS[model][0], "image"]),
        (5, "LoadAudio", [40, 940], [340, 140], [], [("AUDIO", "AUDIO")], [EXAMPLE_INPUTS[model][1]]),
        (6, "KandinskyS2VReferenceImage", [440, 560], [280, 30], [("image", "IMAGE", 4, 0, False)],
         [("image", "IMAGE")], []),
        (7, "VAEEncode", [440, 660], [220, 50], [("pixels", "IMAGE", 6, 0, False), ("vae", "VAE", 2, 0, False)],
         [("LATENT", "LATENT")], []),
        (8, "KandinskyS2VEncode", [440, 780], [400, 280],
         [("encoder", "KANDINSKY_S2V_ENCODER", 3, 0, False), ("clip", "CLIP", 14, 0, False),
          ("audio", "AUDIO", 5, 0, False)],
         [("conditioning", "KANDINSKY_S2V_CONDITIONING")], [PROMPT, 0]),
        (9, "KandinskyS2VSampler", [900, 40], [320, 150],
         [("model", "MODEL", 1, 0, False), ("conditioning", "KANDINSKY_S2V_CONDITIONING", 8, 0, False),
          ("reference_latent", "LATENT", 7, 0, False)],
         [("LATENT", "LATENT")], [5, "fixed", 16]),
        (10, "VAEDecode", [1280, 40], [220, 50], [("samples", "LATENT", 9, 0, False), ("vae", "VAE", 2, 0, False)],
         [("IMAGE", "IMAGE")], []),
        (11, "KandinskyS2VTrimToAudio", [1280, 160], [260, 50],
         [("images", "IMAGE", 10, 0, False), ("conditioning", "KANDINSKY_S2V_CONDITIONING", 8, 0, False)],
         [("images", "IMAGE"), ("fps", "FLOAT")], []),
        (12, "CreateVideo", [1600, 40], [280, 100],
         [("images", "IMAGE", 11, 0, False), ("audio", "AUDIO", 5, 0, False), ("fps", "FLOAT", 11, 1, True)],
         [("VIDEO", "VIDEO")], [25.0, "auto", "sRGB", "none"]),
        (13, "SaveVideo", [1940, 40], [420, 480], [("video", "VIDEO", 12, 0, False)], [("video", "VIDEO")],
         ["kandinsky_s2v/S2V", "mp4", "h264"]),
        (15, "MarkdownNote", [900, 260], [620, 420], [], [], [note(model)]),
    ]


def properties(model: str) -> dict:
    return {
        1: {"models": [dit_entry(model)]},
        2: {"models": [VAE]},
        3: {"kandinsky_s2v_required_files": OMNI},
        14: {"models": [CLIP_L]},
        # The Download models button fetches this workflow's manifest only.
        15: {"kandinsky_s2v_download_package": "kandinsky_s2v", "kandinsky_s2v_workflow": RELEASES[model][0]},
    }


EXTRA = {15: {"title": "Models", "color": "#432", "bgcolor": "#653"}}


def build(model: str) -> dict:
    workflow_name, label, _, _, workflow_id = RELEASES[model]
    description = (
        f"Talking avatar from one reference image and a speech track: Kandinsky S2V 1.0 {label}, "
        "16 pi-Flow steps per 5-second chunk, 25 fps."
    )
    node_properties = properties(model)
    links = []
    outgoing: dict[tuple[int, int], list[int]] = {}
    nodes = []
    for order, (node_id, node_type, pos, size, inputs, outputs, widgets) in enumerate(graph_nodes(model)):
        node_inputs = []
        for name, kind, src, slot, is_widget in inputs:
            link_id = len(links) + 1
            target_slot = len(node_inputs)
            links.append([link_id, src, slot, node_id, target_slot, kind])
            outgoing.setdefault((src, slot), []).append(link_id)
            entry = {"name": name, "type": kind, "link": link_id}
            if is_widget:
                entry["widget"] = {"name": name}
            node_inputs.append(entry)
        nodes.append({
            "id": node_id,
            "type": node_type,
            "pos": pos,
            "size": size,
            "flags": {},
            "order": order,
            "mode": 0,
            "inputs": node_inputs,
            "outputs": [{"name": name, "type": kind, "links": [], "slot_index": i}
                        for i, (name, kind) in enumerate(outputs)],
            "properties": {
                "Node name for S&R": node_type,
                "cnr_id": PACKAGE if node_type.startswith("KandinskyS2V") else "comfy-core",
                **node_properties.get(node_id, {}),
            },
            "widgets_values": widgets,
            **EXTRA.get(node_id, {}),
        })
    for node in nodes:
        for slot, output in enumerate(node["outputs"]):
            output["links"] = outgoing.get((node["id"], slot), [])
    return {
        "id": workflow_id,
        "revision": 0,
        "last_node_id": max(n["id"] for n in nodes),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"name": workflow_name, "description": description},
        "version": 0.4,
        "name": workflow_name,
        "description": description,
    }


def main() -> None:
    for model, (name, *_) in RELEASES.items():
        out = REPO / "example_workflows" / f"{name}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(build(model), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
