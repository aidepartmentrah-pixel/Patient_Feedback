"""
Downloads approved model artifacts from Hugging Face, per models.lock.yaml,
pinned to the exact revision recorded there -- never "main"/latest.

Usage:
    python scripts/models/download_models.py                # download all configured, pinned models
    python scripts/models/download_models.py --only pfms-classification-models

Intended for internet-connected engineering machines (e.g. the Lenovo Legion)
restoring models_directory/ after a fresh git clone. Never run this on an
offline/production machine -- offline installs get models from the approved
Application Release package, not from Hugging Face directly.
"""
import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lock_utils import REPO_ROOT, load_lock, load_token


def download_one(token: str, entry: dict) -> None:
    model_id = entry["id"]
    repository = entry.get("repository")
    revision = entry.get("revision")

    if not repository:
        print(f"[SKIP] {model_id}: no repository configured yet in models.lock.yaml")
        return
    if not revision:
        print(f"[SKIP] {model_id}: no revision recorded -- run upload_models.py first, or this "
              f"model hasn't been published yet")
        return

    required_files = entry.get("required_files") or []
    if not required_files:
        print(f"[SKIP] {model_id}: models.lock.yaml has no required_files recorded")
        return

    target_dir = REPO_ROOT / entry["target_directory"]
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{model_id}] downloading {len(required_files)} files from {repository} @ {revision[:12]} ...")
    for rel_path in required_files:
        cached_path = hf_hub_download(
            repo_id=repository,
            filename=rel_path,
            revision=revision,
            repo_type="model",
            token=token,
        )
        dest = target_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_path, dest)
        print(f"    {rel_path}")

    print(f"[{model_id}] done -> {target_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Only download this model id from models.lock.yaml")
    args = parser.parse_args()

    token = load_token()
    lock = load_lock()

    for entry in lock["models"]:
        if args.only and entry["id"] != args.only:
            continue
        download_one(token, entry)


if __name__ == "__main__":
    main()
