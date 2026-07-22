"""Shared helpers for upload_models.py / download_models.py / verify_models.py."""
import hashlib
import fnmatch
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOCK_FILE = REPO_ROOT / "models.lock.yaml"
TOKEN_FILE = REPO_ROOT / ".hf_token"


def load_lock():
    with open(LOCK_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_lock(data):
    with open(LOCK_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def load_token():
    if not TOKEN_FILE.exists():
        raise SystemExit(
            f"Missing {TOKEN_FILE}. Create it with a Hugging Face access token "
            f"(see database/docs or ask for the setup steps) before running this script."
        )
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit(f"{TOKEN_FILE} exists but is empty.")
    return token


def matches_any(rel_path: str, patterns) -> bool:
    posix = rel_path.replace("\\", "/")
    return any(fnmatch.fnmatch(posix, pat) for pat in patterns)


def collect_files(target_dir: Path, include_patterns, exclude_patterns):
    """Returns list of (absolute_path, relative_posix_path) matching include but not exclude."""
    result = []
    for p in target_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(target_dir).as_posix()
        if matches_any(rel, include_patterns) and not matches_any(rel, exclude_patterns):
            result.append((p, rel))
    return sorted(result, key=lambda x: x[1])


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(manifest_path: Path, files_with_hashes):
    with open(manifest_path, "w", encoding="utf-8") as f:
        for rel, digest in files_with_hashes:
            f.write(f"{digest}  {rel}\n")


def read_manifest(manifest_path: Path):
    entries = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            digest, rel = line.split("  ", 1)
            entries[rel] = digest
    return entries
