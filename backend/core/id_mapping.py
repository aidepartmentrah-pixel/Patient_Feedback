import json
from pathlib import Path
from typing import Any, Dict

# Module-level cache for mapping
_MAPPING_CACHE: Dict[str, Any] | None = None


def _config_path() -> Path:
    """Return absolute path to classification_id_mapping.json."""
    # backend/core/id_mapping.py -> backend/config/classification_id_mapping.json
    return Path(__file__).resolve().parent.parent / "config" / "classification_id_mapping.json"


def get_id_mapping() -> Dict[str, Any]:
    """Load and cache the mapping configuration.

    Returns a dict with keys like 'idMap' and 'labels'. If the file is
    missing or invalid, returns an empty mapping structure.
    """
    global _MAPPING_CACHE
    if _MAPPING_CACHE is not None:
        return _MAPPING_CACHE

    path = _config_path()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # Basic shape check
            if not isinstance(data, dict):
                raise ValueError("Mapping JSON is not a dict")
            _MAPPING_CACHE = data
            return _MAPPING_CACHE
    except Exception:
        # Fallback to empty structure
        _MAPPING_CACHE = {"idMap": {}, "labels": {}}
        return _MAPPING_CACHE


def remap_classification_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """Remap ID fields in classification output using configured idMap.

    Expects keys in result:
      - domain_id, category_id, sub_category_id,
      - severity_id, stage_id, harm_level_id

    If a mapping exists and is non-null, replaces the id with the mapped value.
    Missing or null mapping leaves the id unchanged.
    """
    mapping = get_id_mapping()
    id_map = mapping.get("idMap", {}) or {}

    # Map from config section -> output id key
    fields = {
        "domain": "domain_id",
        "category": "category_id",
        "subcategory": "sub_category_id",
        "severity_level": "severity_id",
        "stage": "stage_id",
        "harm_level": "harm_level_id",
    }

    updated = dict(result)  # shallow copy

    for section, id_key in fields.items():
        if id_key in result and result[id_key] is not None:
            old_id = result[id_key]
            try:
                section_map = id_map.get(section, {}) or {}
                new_id = section_map.get(str(old_id), None)
                if isinstance(new_id, int):
                    updated[id_key] = new_id
            except Exception:
                # Be resilient; keep original id on any error
                pass

    return updated
