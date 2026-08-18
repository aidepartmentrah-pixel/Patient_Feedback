"""
Stage A supplement -- fetches the 6 org units that are individually retrievable
but excluded from the type-classified listing endpoints (see the Stage A gap
report: IDs 22, 138, 167, 168, 172, 177 -- structurally orphaned/incomplete
rows in the source, not frozen/archived by any observable flag). Captured
separately here, reproducibly, rather than relying on ad hoc manual checks.

Read-only: GET /api/org-units/unit/{id} only.
"""
import json
import urllib3
from datetime import datetime, timezone
from pathlib import Path

import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SOURCE_BASE = "https://170.70.32.34"
OUT_DIR = Path(__file__).resolve().parent / "raw_extract"
ORPHANED_IDS = [22, 138, 167, 168, 172, 177]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for uid in ORPHANED_IDS:
        url = f"{SOURCE_BASE}/api/org-units/unit/{uid}"
        print(f"GET {url} ...")
        resp = requests.get(url, verify=False, timeout=10)
        print(f"  HTTP {resp.status_code}")
        results[str(uid)] = {"status": resp.status_code, "body": resp.json() if resp.status_code == 200 else None}

    out_path = OUT_DIR / "orphaned_units.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "note": "Units referenced by APP_OrgUnitPolicy but absent from /api/org-units/{administrations,departments,sections} "
                        "due to unresolvable ParentID chains (empty ancestors) and, for most, a NULL Type column. "
                        "Not excluded due to any Frozen/active flag -- get_admin_unit_tree() applies no such filter.",
                "units": results,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
