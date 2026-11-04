"""Queue the example graph on a running ComfyUI server and wait for the video.

    python dev/run_api.py --server 127.0.0.1:8188 [--model pro] [--image reference.png --audio speech.wav]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

from kandinsky_s2v_comfy_graph import api_prompt


def post(server: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"http://{server}{path}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request) as response:  # noqa: S310
        return json.load(response)


def get(server: str, path: str) -> dict:
    with urllib.request.urlopen(f"http://{server}{path}") as response:  # noqa: S310
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="127.0.0.1:8188")
    parser.add_argument("--image", help="default: the example of --model")
    parser.add_argument("--audio", help="default: the example of --model")
    parser.add_argument("--chunks", type=int, default=0)
    parser.add_argument("--model", choices=("lite", "pro"), default="lite")
    args = parser.parse_args()

    queued = post(args.server, "/prompt", {"prompt": api_prompt(args.image, args.audio, args.chunks, model=args.model)})
    prompt_id = queued["prompt_id"]
    start = time.time()
    while True:
        history = get(args.server, f"/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry["status"]
            print(json.dumps(status.get("messages", [])[-1:], indent=1))
            print(json.dumps(entry["outputs"], indent=1))
            print(f"done in {time.time() - start:.0f} s, status {status['status_str']}")
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
