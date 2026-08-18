"""
Stage A (unauthenticated part) -- pulls the organizational-unit hierarchy from
the old HCAT system's own read-only, no-auth REST API. No credentials involved,
no writes performed against the source system: plain HTTP GET only.

Source: http://170.70.32.34/ (self-signed cert -- verification is disabled
below deliberately, matching how the app's own browser access already works
per DEPLOYMENT_CHECKLIST.txt; this is read-only regardless of cert trust).

Usage:
    python database/sqlserver/seed/extract_org_units.py
"""
import json
import sys
import urllib3
from datetime import datetime, timezone
from pathlib import Path

import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SOURCE_BASE = "https://170.70.32.34"
OUT_DIR = Path(__file__).resolve().parent / "raw_extract"

ENDPOINTS = {
    "administrations": "/api/org-units/administrations",
    "departments": "/api/org-units/departments",
    "sections": "/api/org-units/sections",
    "summary": "/api/org-units/summary",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    statuses = {}

    for name, path in ENDPOINTS.items():
        url = SOURCE_BASE + path
        print(f"GET {url} ...")
        resp = requests.get(url, verify=False, timeout=15)
        statuses[name] = resp.status_code
        if resp.status_code != 200:
            print(f"  ERROR: HTTP {resp.status_code}")
            sys.exit(1)
        results[name] = resp.json()
        print(f"  OK ({resp.status_code})")

    out_path = OUT_DIR / "org_units.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    manifest_path = OUT_DIR / "org_units_manifest.json"
    manifest = {
        "source_url": SOURCE_BASE,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "endpoints": {name: {"path": path, "status": statuses[name]} for name, path in ENDPOINTS.items()},
        "counts": {
            "administrations": len(results["administrations"].get("administrations", [])),
            "departments": len(results["departments"].get("departments", [])),
            "sections": len(results["sections"].get("sections", [])),
            "summary": results["summary"],
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {out_path}")
    print(f"Wrote {manifest_path}")
    print(f"\nCounts: {json.dumps(manifest['counts'], indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
