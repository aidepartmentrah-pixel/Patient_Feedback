"""
Stage A analysis -- produces the extraction + gap report for human review.
Read-only analysis of the raw extract files; writes nothing back to any
database, does not touch the source system.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw_extract"

org_units = json.load(open(RAW_DIR / "org_units.json", encoding="utf-8"))
org_manifest = json.load(open(RAW_DIR / "org_units_manifest.json", encoding="utf-8"))
users = json.load(open(RAW_DIR / "user_credentials.json", encoding="utf-8"))
user_manifest = json.load(open(RAW_DIR / "user_credentials_manifest.json", encoding="utf-8"))

print("=" * 70)
print("1. EXTRACTION MANIFEST / ENDPOINT RESULTS")
print("=" * 70)
print(json.dumps({"org_units": org_manifest, "users": user_manifest}, indent=2, ensure_ascii=False))

print()
print("=" * 70)
print("2. ORG-UNIT COUNTS BY TYPE")
print("=" * 70)
admins = org_units["administrations"]["administrations"]
depts = org_units["departments"]["departments"]
secs = org_units["sections"]["sections"]
print(f"Administrations: {len(admins)}")
print(f"Departments:     {len(depts)}")
print(f"Sections:        {len(secs)}")
print(f"Total:           {len(admins) + len(depts) + len(secs)}")

all_units = {}
for a in admins:
    all_units[a["id"]] = {"name": a["name"], "name_ar": a.get("name_ar"), "parent_id": None, "type": 323}
for d in depts:
    all_units[d["id"]] = {"name": d["name"], "name_ar": d.get("name_ar"), "parent_id": d.get("administration_id"), "type": 325}
for s in secs:
    all_units[s["id"]] = {"name": s["name"], "name_ar": s.get("name_ar"), "parent_id": s.get("department_id"), "type": 324}

print()
print("=" * 70)
print("3. ROOT UNITS AND ORPHANED PARENT REFERENCES")
print("=" * 70)
root_units = [uid for uid, u in all_units.items() if u["parent_id"] is None]
print(f"Root units (parent_id is null -- expected: administrations): {len(root_units)}")
orphaned_parents = []
for uid, u in all_units.items():
    if u["parent_id"] is not None and u["parent_id"] not in all_units:
        orphaned_parents.append((uid, u["name"], u["parent_id"]))
print(f"Units whose parent_id does not resolve to a known unit: {len(orphaned_parents)}")
for uid, name, pid in orphaned_parents:
    print(f"  unit {uid} ({name!r}) -> missing parent {pid}")

print()
print("=" * 70)
print("4. RECONCILIATION AGAINST APP_OrgUnitPolicy'S 179 SEEDED ENTRIES")
print("=" * 70)
import re
sql = open(Path(__file__).resolve().parents[1] / "install" / "009_seed_configuration.sql", encoding="utf-8").read()
matches = re.findall(r'VALUES \((\d+), (\d+), (\d+),', sql)
policy_ids = set(int(m[1]) for m in matches)
api_ids = set(all_units.keys())
print(f"OrgUnitIDs in seeded APP_OrgUnitPolicy: {len(policy_ids)}")
print(f"OrgUnitIDs returned live by the API:    {len(api_ids)}")
missing_from_api = sorted(policy_ids - api_ids)
extra_in_api = sorted(api_ids - policy_ids)
print(f"In policy but NOT in live API (orphaned policy refs): {missing_from_api}")
print(f"In live API but NOT in policy (unpoliced units):      {extra_in_api}")

print()
print("=" * 70)
print("5. USER COUNTS: TOTAL / ACTIVE / INACTIVE")
print("=" * 70)
active = [u for u in users if u.get("active")]
inactive = [u for u in users if not u.get("active")]
print(f"Total: {len(users)}   Active: {len(active)}   Inactive: {len(inactive)}")
print("Inactive accounts:")
for u in inactive:
    print(f"  {u['username']} (role={u.get('role')}, org_unit={u.get('org_unit')!r})")

print()
print("=" * 70)
print("6. USERS BY ROLE")
print("=" * 70)
role_counts = Counter(u.get("role") or "(none)" for u in users)
for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
    print(f"  {role}: {count}")

print()
print("=" * 70)
print("7. DUPLICATE USERNAMES")
print("=" * 70)
username_counts = Counter(u["username"] for u in users)
dupes = {name: c for name, c in username_counts.items() if c > 1}
print(f"Duplicate usernames: {len(dupes)}")
for name, c in dupes.items():
    print(f"  {name}: {c} occurrences")

print()
print("=" * 70)
print("8. USERS WITHOUT AN ORGANIZATIONAL UNIT OR SCOPE")
print("=" * 70)
no_org = [u for u in users if not u.get("org_unit_id")]
print(f"Users with no org_unit_id: {len(no_org)}")
for u in no_org:
    print(f"  {u['username']} (role={u.get('role')}, org_unit={u.get('org_unit')!r}, active={u.get('active')})")

print()
print("=" * 70)
print("9. UNRESOLVED ROLE CODES / ORGANIZATIONAL REFERENCES")
print("=" * 70)
known_roles = {"WORKER", "COMPLAINT_SUPERVISOR", "SECTION_ADMIN", "DEPARTMENT_ADMIN", "ADMINISTRATION_ADMIN", "SOFTWARE_ADMIN"}
unknown_roles = [u for u in users if u.get("role") and u["role"] not in known_roles]
print(f"Users with an unrecognized role code: {len(unknown_roles)}")
for u in unknown_roles:
    print(f"  {u['username']}: role={u['role']!r}")

unresolved_org = [u for u in users if u.get("org_unit_id") and u["org_unit_id"] not in all_units]
print(f"Users whose org_unit_id doesn't resolve to a known unit: {len(unresolved_org)}")
for u in unresolved_org:
    print(f"  {u['username']}: org_unit_id={u['org_unit_id']} (claimed name: {u.get('org_unit')!r})")

print()
print("=" * 70)
print("10. FLAGGED / SUSPICIOUS / NULL-NAMED / TEST-LOOKING RECORDS")
print("=" * 70)
print("-- Org units with null/empty/'NULL' names or unknown type --")
for uid, u in all_units.items():
    if not u["name"] or u["name"].strip() in ("", "NULL") or u["type"] is None:
        print(f"  unit {uid}: name={u['name']!r} type={u['type']}")
print()
print("-- Users with test-looking usernames or org units --")
suspicious_keywords = ["test", "what so ever", "universal_section_user", "whatsoever"]
for u in users:
    haystack = f"{u['username']} {u.get('org_unit') or ''}".lower()
    if any(k in haystack for k in suspicious_keywords):
        print(f"  {u['username']} (role={u.get('role')}, org_unit={u.get('org_unit')!r}, active={u.get('active')})")

print()
print("=" * 70)
print("11. SOURCE-SYSTEM WRITE CONFIRMATION")
print("=" * 70)
print(f"source_write_operations_performed: {user_manifest['source_write_operations_performed']}")
print("Only GET/POST-login/POST-logout calls were made. No PUT/PATCH/DELETE, no")
print("data-modifying endpoint was ever called against 170.70.32.34.")
