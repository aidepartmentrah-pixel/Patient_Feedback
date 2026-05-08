"""
99_cleanup.py — Remove all T100_ test cases and their related data.

Deletes in safe order:
  1. APP_SubcaseActionItem   (for T100_ subcases)
  2. APP_AdministrativeSubcase (for T100_ cases)
  3. APP_IncidentCaseDoctor
  4. APP_IncidentCaseEmployee
  5. APP_IncidentCaseTargetDepartment
  6. APP_IncidentCase          (T100_ patient names)
"""

import sys, os, json

_HERE    = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.join(_HERE, '..', '..')
_REPO    = os.path.join(_BACKEND, '..')
sys.path.insert(0, os.path.abspath(_BACKEND))
sys.path.insert(0, os.path.abspath(_REPO))

from backend.core.database import get_connection

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def main():
    print("=" * 60)
    print("99_CLEANUP — Removing T100_ Test Data")
    print("=" * 60)

    conn = get_connection()
    cur  = conn.cursor()

    try:
        # Find T100_ case IDs
        cur.execute("""
            SELECT IncidentRequestCaseID
            FROM dbo.APP_IncidentCase
            WHERE PatientName LIKE 'T100_%'
        """)
        ids = [r[0] for r in cur.fetchall()]
        print(f"\nFound {len(ids)} T100_ cases to delete")

        if not ids:
            print("Nothing to delete.")
            return

        ids_in = "(" + ",".join(str(i) for i in ids) + ")"

        steps = [
            ("APP_SubcaseActionItem via subcases", f"""
                DELETE FROM dbo.APP_SubcaseActionItem
                WHERE SubcaseID IN (
                    SELECT SubcaseID FROM dbo.APP_AdministrativeSubcase
                    WHERE IncidentRequestCaseID IN {ids_in}
                )
            """),
            ("APP_AdministrativeSubcase", f"""
                DELETE FROM dbo.APP_AdministrativeSubcase
                WHERE IncidentRequestCaseID IN {ids_in}
            """),
            ("APP_IncidentCaseDoctor", f"""
                DELETE FROM dbo.APP_IncidentCaseDoctor
                WHERE IncidentRequestCaseID IN {ids_in}
            """),
            ("APP_IncidentCaseEmployee", f"""
                DELETE FROM dbo.APP_IncidentCaseEmployee
                WHERE IncidentRequestCaseID IN {ids_in}
            """),
            ("APP_IncidentCaseTargetDepartment", f"""
                DELETE FROM dbo.APP_IncidentCaseTargetDepartment
                WHERE IncidentRequestCaseID IN {ids_in}
            """),
            ("APP_IncidentCase", f"""
                DELETE FROM dbo.APP_IncidentCase
                WHERE IncidentRequestCaseID IN {ids_in}
            """),
        ]

        for label, sql in steps:
            cur.execute(sql)
            print(f"  Deleted {cur.rowcount:>4} rows from {label}")

        conn.commit()
        print(f"\nCommitted. All {len(ids)} T100_ cases removed.")

        # Also clean the data folder JSON files
        for fname in ['inserted_ids.json', 'verification_report.json']:
            fpath = os.path.join(DATA_DIR, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
                print(f"Removed {fname}")

    except Exception as e:
        conn.rollback()
        print(f"\n[ERROR] {e}")
        print("Rolled back — no data deleted.")
        raise
    finally:
        cur.close()
        conn.close()

    print("=" * 60)


if __name__ == "__main__":
    main()
