"""
Uploads approved model artifacts to their Hugging Face repos, per models.lock.yaml.

Usage:
    python scripts/models/upload_models.py                 # upload all configured models
    python scripts/models/upload_models.py --only pfms-classification-models

Never uploads training data, patient/complaint text, or anything outside the
include_patterns configured per model in models.lock.yaml. Skips any model
entry whose "repository" is still null (not yet created).
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lock_utils import REPO_ROOT, load_lock, save_lock, load_token, collect_files, sha256_of, write_manifest


def upload_one(api: HfApi, entry: dict) -> None:
    model_id = entry["id"]
    repository = entry.get("repository")
    if not repository:
        print(f"[SKIP] {model_id}: no repository configured yet in models.lock.yaml")
        return

    target_dir = REPO_ROOT / entry["target_directory"]
    if not target_dir.exists():
        print(f"[SKIP] {model_id}: target_directory {target_dir} does not exist")
        return

    files = collect_files(target_dir, entry["include_patterns"], entry.get("exclude_patterns", []))
    if not files:
        print(f"[SKIP] {model_id}: no files matched include_patterns under {target_dir}")
        return

    print(f"[{model_id}] hashing {len(files)} files ...")
    hashes = [(rel_path, sha256_of(abs_path)) for abs_path, rel_path in files]

    manifest_path = REPO_ROOT / "scripts" / "models" / entry["sha256_manifest"]
    write_manifest(manifest_path, hashes)

    # Single commit for the whole batch -- HF rate-limits to 128 commits/hour per repo,
    # so per-file upload_file() calls (one commit each) blow through that on any real
    # model set. upload_folder() batches everything into one commit instead.
    print(f"[{model_id}] uploading {len(files)} files + manifest to {repository} in a single commit ...")
    api.upload_folder(
        folder_path=str(target_dir),
        repo_id=repository,
        repo_type="model",
        allow_patterns=entry["include_patterns"],
        ignore_patterns=entry.get("exclude_patterns", []),
        commit_message=f"Upload {model_id} {entry['release_version']} ({len(files)} files)",
    )
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo=entry["sha256_manifest"],
        repo_id=repository,
        repo_type="model",
        commit_message=f"Update checksum manifest ({model_id} {entry['release_version']})",
    )

    repo_info = api.repo_info(repo_id=repository, repo_type="model")
    revision = repo_info.sha

    entry["revision"] = revision
    entry["required_files"] = [rel for _, rel in files]
    print(f"[{model_id}] done. revision={revision}, {len(files)} files, manifest written to {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Only upload this model id from models.lock.yaml")
    args = parser.parse_args()

    token = load_token()
    api = HfApi(token=token)
    lock = load_lock()

    for entry in lock["models"]:
        if args.only and entry["id"] != args.only:
            continue
        upload_one(api, entry)

    save_lock(lock)
    print("\nmodels.lock.yaml updated with new revision(s) and file list(s).")


if __name__ == "__main__":
    main()
