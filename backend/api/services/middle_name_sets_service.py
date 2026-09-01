"""
Middle-Name Candidate Sets Service

Manages named, switchable lists of candidate father's/middle names used by
the patient-search "middle name assist" chip UI (see
HCAT-Middle-Name-Search-Assist-Plan.md). File-based, no database — one JSON
file per set plus a small pointer file, all inside the existing config
volume (same persistent, editable-without-rebuild volume db_settings.json
already lives in, see core/config_loader.py's _CONFIG_DIR).

    backend/config/middle_name_sets/
        thirty_names.json   {"display_name": "...", "names": [...]}
        _active.json         {"active_set": "thirty_names"}

Set identifiers are slugified display names (lowercase, [a-z0-9_] only)
used directly as filenames — see _slugify. This is a security boundary:
never build a path from a set identifier without passing it through
_slugify first, or a crafted identifier could escape the directory
(path traversal).

The active set is read fresh from disk on every call (no caching), so a
change made through the CRUD endpoints takes effect on the very next
patient search with no backend restart — same convention as the Hospital
Directory API settings.
"""

import json
import re
import threading
from pathlib import Path
from typing import List, Optional

from core.config_loader import _CONFIG_DIR

SETS_DIR = _CONFIG_DIR / "middle_name_sets"
ACTIVE_FILE = SETS_DIR / "_active.json"

# One process-wide lock: these files are small and rarely written, and this
# avoids any read-modify-write race between two admins editing sets at once.
_lock = threading.Lock()

_DEFAULT_SET_ID = "thirty_names"
_DEFAULT_SET_DISPLAY_NAME = "Starting set (32 names)"
# The real, curated Lebanese Shia male given-name list from
# HCAT-Middle-Name-Search-Assist-Plan.md's "Starting set" block, transcribed
# directly from that file's UTF-8 bytes (not retyped) -- see
# C:\Users\it\Desktop\API Middle Name Coverage Test\name_sets\thirty_names.json
# for the extraction this was copied from.
_DEFAULT_NAMES = [
    "محمد", "علي", "حسين", "حسن", "عباس", "أحمد", "محمود", "مهدي",
    "مصطفى", "حيدر", "هادي", "رضا", "جعفر", "قاسم", "جواد", "مرتضى",
    "كاظم", "إبراهيم", "موسى", "إسماعيل", "عمار", "ياسر", "خليل", "بلال",
    "نبيه", "فاضل", "حسنَين", "صادق", "باقر", "وسام", "يوسف", "فؤاد",
]


class MiddleNameSetsError(Exception):
    """Raised for invalid set operations (bad id, duplicate, delete-active, etc.)."""


def _slugify(raw: str) -> str:
    """
    Lowercase, [a-z0-9_] only. Used both to derive a set id from a display
    name and to validate any set id received from a client before it
    touches the filesystem -- reject anything that doesn't already match
    this shape rather than silently mangling it, so a caller always knows
    exactly what id a name became.
    """
    slug = raw.strip().lower()
    slug = re.sub(r"[\s-]+", "_", slug)
    slug = re.sub(r"[^a-z0-9_]", "", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


def _validate_set_id(set_id: str) -> str:
    if not set_id or set_id != _slugify(set_id):
        raise MiddleNameSetsError(
            f"Invalid set id {set_id!r} -- must be lowercase [a-z0-9_] only."
        )
    return set_id


def _set_path(set_id: str) -> Path:
    return SETS_DIR / f"{_validate_set_id(set_id)}.json"


def _ensure_seeded():
    """Create the default set + activate it, but only on a genuinely empty
    directory -- never overwrites existing sets, including one named
    thirty_names that an admin has already edited."""
    SETS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [p for p in SETS_DIR.glob("*.json") if p.name != "_active.json"]
    if existing:
        return
    _write_set_file(_DEFAULT_SET_ID, _DEFAULT_SET_DISPLAY_NAME, _DEFAULT_NAMES)
    ACTIVE_FILE.write_text(
        json.dumps({"active_set": _DEFAULT_SET_ID}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_set_file(set_id: str, display_name: str, names: List[str]):
    path = _set_path(set_id)
    path.write_text(
        json.dumps({"display_name": display_name, "names": names}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_set_file(set_id: str) -> dict:
    path = _set_path(set_id)
    if not path.exists():
        raise MiddleNameSetsError(f"Set '{set_id}' does not exist.")
    return json.loads(path.read_text(encoding="utf-8"))


def _get_active_set_id() -> str:
    if not ACTIVE_FILE.exists():
        raise MiddleNameSetsError("No active set configured.")
    return json.loads(ACTIVE_FILE.read_text(encoding="utf-8"))["active_set"]


def list_sets() -> List[dict]:
    _ensure_seeded()
    active_id = _get_active_set_id()
    result = []
    for path in sorted(SETS_DIR.glob("*.json")):
        if path.name == "_active.json":
            continue
        set_id = path.stem
        data = json.loads(path.read_text(encoding="utf-8"))
        result.append({
            "id": set_id,
            "display_name": data.get("display_name", set_id),
            "name_count": len(data.get("names", [])),
            "is_active": set_id == active_id,
        })
    return result


def get_active_set() -> dict:
    """Read fresh from disk -- called by the search-time chip UI on every
    request, no caching, so a Settings-tab change takes effect immediately."""
    _ensure_seeded()
    active_id = _get_active_set_id()
    data = _read_set_file(active_id)
    return {"id": active_id, "display_name": data.get("display_name", active_id), "names": data.get("names", [])}


def get_set(set_id: str) -> dict:
    _ensure_seeded()
    data = _read_set_file(set_id)
    return {"id": _validate_set_id(set_id), "display_name": data.get("display_name", set_id), "names": data.get("names", [])}


def create_set(display_name: str, names: Optional[List[str]] = None) -> dict:
    with _lock:
        _ensure_seeded()
        set_id = _slugify(display_name)
        if not set_id:
            raise MiddleNameSetsError("Display name must contain at least one letter/number.")
        if _set_path(set_id).exists():
            raise MiddleNameSetsError(f"A set with id '{set_id}' already exists.")
        _write_set_file(set_id, display_name.strip(), names or [])
        return get_set(set_id)


def update_set(set_id: str, display_name: Optional[str] = None, names: Optional[List[str]] = None) -> dict:
    """Rename and/or bulk-replace the name list. Renaming does NOT change
    the set's id/filename -- only its display_name -- so the active-set
    pointer and any external reference to this id stay valid."""
    with _lock:
        current = _read_set_file(set_id)
        new_display_name = display_name.strip() if display_name is not None else current.get("display_name", set_id)
        new_names = names if names is not None else current.get("names", [])
        _write_set_file(set_id, new_display_name, new_names)
        return get_set(set_id)


def delete_set(set_id: str):
    with _lock:
        _validate_set_id(set_id)
        if _get_active_set_id() == set_id:
            raise MiddleNameSetsError("Cannot delete the active set -- switch to another set first.")
        path = _set_path(set_id)
        if not path.exists():
            raise MiddleNameSetsError(f"Set '{set_id}' does not exist.")
        remaining = [p for p in SETS_DIR.glob("*.json") if p.name not in ("_active.json", path.name)]
        if not remaining:
            raise MiddleNameSetsError("Cannot delete the only remaining set.")
        path.unlink()


def set_active(set_id: str):
    with _lock:
        _validate_set_id(set_id)
        if not _set_path(set_id).exists():
            raise MiddleNameSetsError(f"Set '{set_id}' does not exist.")
        ACTIVE_FILE.write_text(
            json.dumps({"active_set": set_id}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def add_name(set_id: str, name: str) -> dict:
    with _lock:
        data = _read_set_file(set_id)
        names = data.get("names", [])
        name = name.strip()
        if not name:
            raise MiddleNameSetsError("Name cannot be empty.")
        if name in names:
            raise MiddleNameSetsError(f"'{name}' is already in this set.")
        names.append(name)
        _write_set_file(set_id, data.get("display_name", set_id), names)
        return get_set(set_id)


def update_name(set_id: str, old_name: str, new_name: str) -> dict:
    with _lock:
        data = _read_set_file(set_id)
        names = data.get("names", [])
        if old_name not in names:
            raise MiddleNameSetsError(f"'{old_name}' is not in this set.")
        new_name = new_name.strip()
        if not new_name:
            raise MiddleNameSetsError("Name cannot be empty.")
        names[names.index(old_name)] = new_name
        _write_set_file(set_id, data.get("display_name", set_id), names)
        return get_set(set_id)


def delete_name(set_id: str, name: str) -> dict:
    with _lock:
        data = _read_set_file(set_id)
        names = data.get("names", [])
        if name not in names:
            raise MiddleNameSetsError(f"'{name}' is not in this set.")
        names.remove(name)
        _write_set_file(set_id, data.get("display_name", set_id), names)
        return get_set(set_id)
