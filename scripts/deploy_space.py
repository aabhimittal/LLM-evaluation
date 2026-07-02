"""Deploy the Gradio app to a Hugging Face Space.

Stages the Space files (space/* plus a vendored copy of src/caliper) and
uploads them with huggingface_hub. The token needs write scope.

Usage:
    HF_TOKEN=hf_... python scripts/deploy_space.py [--repo-id user/caliper] [--private]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def stage(tmp: Path) -> None:
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy2(ROOT / "space" / name, tmp / name)
    target = tmp / "src" / "caliper"
    shutil.copytree(
        ROOT / "src" / "caliper", target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=None, help="e.g. abhimittal/caliper")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--wait", action="store_true", help="poll until RUNNING")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN to a write-scope token.", file=sys.stderr)
        return 2
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = args.repo_id or f"{user}/caliper"

    api.create_repo(
        repo_id=repo_id, repo_type="space", space_sdk="gradio",
        private=args.private, exist_ok=True,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        stage(tmp)
        api.upload_folder(
            folder_path=str(tmp), repo_id=repo_id, repo_type="space",
            commit_message="Deploy Caliper Space",
        )
    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"uploaded -> {url}")

    if args.wait:
        for _ in range(60):
            runtime = api.get_space_runtime(repo_id)
            print(f"stage: {runtime.stage}")
            if runtime.stage == "RUNNING":
                print(f"Space is live: {url}")
                return 0
            if runtime.stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"):
                print("Build failed — check the Space logs.", file=sys.stderr)
                return 1
            time.sleep(10)
        print("timed out waiting for RUNNING", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
