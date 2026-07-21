"""
ML Architecture Consolidation — Stage 12: SQL Server Restore Drill

Proves the Stage 1 baseline .bak (C:\\SQLBackup\\IncidentManager_ml_stage1_manual_20260716.bak)
restores cleanly — the one clause of the original Stage 12 plan
(jazzy-imagining-allen.md line 103, "Confirm the SQL Server .bak from Stage 1
restores cleanly as a final fallback") that had zero existing tooling behind
it anywhere in this repo (confirmed by research: every existing backup
script/doc only ever backs up or restores IN PLACE onto the live
IncidentManager name — never to an isolated name).

Restores to a brand-new, isolated database name
(IncidentManager_Stage12_RollbackTest) so the live IncidentManager DB is
never touched. The app's own SQL-authenticated login lacks BACKUP/RESTORE
DATABASE permission (confirmed during Stage 1/8) — this script uses `sqlcmd
-E` (Windows trusted auth, sysadmin via the HCAT\\Administrator account
already used for the original backups), exactly like the manual fallback
Stage 1/8 already relied on, rather than backend/core/database.py's
connection/config.

Spot-checks a handful of known counts against the Stage 1 baseline, then
drops the isolated test database, leaving no trace and never touching the
live IncidentManager DB.

Run from the backend/ directory:
    python -m scripts.ml_stage12_restore_drill
"""

import subprocess
import sys

BAK_PATH = r"C:\SQLBackup\IncidentManager_ml_stage1_manual_20260716.bak"
TEST_DB_NAME = "IncidentManager_Stage12_RollbackTest"
SQL_SERVER = "127.0.0.1,1433"


def run_sqlcmd(query: str, database: str = "master") -> str:
    """Runs a T-SQL query via sqlcmd -E (Windows trusted auth, sysadmin),
    never through the app's own SQL-authenticated connection. Raises on a
    non-zero exit code so a real restore failure isn't silently swallowed."""
    result = subprocess.run(
        ["sqlcmd", "-S", SQL_SERVER, "-E", "-d", database, "-Q", query, "-W", "-s", "|"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sqlcmd failed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}")
    return result.stdout


def parse_filelist(filelist_output: str):
    lines = [l for l in filelist_output.splitlines() if l.strip() and "|" in l]
    header = lines[0].split("|")
    logical_idx = header.index("LogicalName")
    type_idx = header.index("Type")
    data_rows = [l.split("|") for l in lines[2:] if l.split("|")[0].strip()]
    return [(row[logical_idx].strip(), row[type_idx].strip()) for row in data_rows]


def find_stage1_backup_set_position(header_output: str) -> int:
    """
    This .bak file has TWO appended backup sets (confirmed via RESTORE
    HEADERONLY during Stage 12 investigation): an older, unrelated
    'PreForceCloseTest' backup from 2026-06-18, and the actual Stage 1
    baseline from 2026-07-16 (matching the Stage 1 manifest's recorded
    backed_up_at). RESTORE DATABASE without WITH FILE = <n> silently
    restores Position 1 (the OLDER set) by default — this is exactly what
    happened on the first drill attempt (got 156 rows instead of the
    manifest's recorded 173). Always pick the set with the LATEST
    BackupFinishDate, never assume Position 1.
    """
    lines = [l for l in header_output.splitlines() if l.strip() and "|" in l]
    header = lines[0].split("|")
    position_idx = header.index("Position")
    finish_date_idx = header.index("BackupFinishDate")
    data_rows = [l.split("|") for l in lines[2:] if l.split("|")[0].strip()]
    sets = [(int(row[position_idx]), row[finish_date_idx].strip()) for row in data_rows]
    sets.sort(key=lambda x: x[1])  # ISO-like datetime strings sort correctly lexicographically
    print(f"    Backup sets found in file: {sets}")
    return sets[-1][0]


def main():
    print("=" * 70)
    print("STAGE 12 — SQL Server Restore Drill")
    print(f"Source: {BAK_PATH}")
    print(f"Target (isolated, never the live DB): {TEST_DB_NAME}")
    print("=" * 70)

    try:
        print("\n[1] RESTORE HEADERONLY — identifying which backup set is the Stage 1 baseline...")
        header_output = run_sqlcmd(f"RESTORE HEADERONLY FROM DISK = '{BAK_PATH}'")
        backup_position = find_stage1_backup_set_position(header_output)
        print(f"    Using backup set Position={backup_position} (most recent = Stage 1 baseline)")

        print(f"\n[2] RESTORE FILELISTONLY (FILE={backup_position}) — reading real logical file names...")
        filelist_output = run_sqlcmd(f"RESTORE FILELISTONLY FROM DISK = '{BAK_PATH}' WITH FILE = {backup_position}")
        files = parse_filelist(filelist_output)
        print(f"    Files in backup: {files}")
        assert len(files) == 2, f"Expected exactly 2 files (data + log), got {files}"
        data_logical = next(name for name, ftype in files if ftype == "D")
        log_logical = next(name for name, ftype in files if ftype == "L")

        print("\n[3] Reading SQL Server's default data path...")
        path_output = run_sqlcmd("SELECT SERVERPROPERTY('InstanceDefaultDataPath')")
        data_path = [l.strip() for l in path_output.splitlines() if l.strip() and "-" not in l and "row" not in l.lower()][0]
        print(f"    Default data path: {data_path}")

        mdf_path = f"{data_path}{TEST_DB_NAME}.mdf"
        ldf_path = f"{data_path}{TEST_DB_NAME}_log.ldf"

        print(f"\n[4] RESTORE DATABASE {TEST_DB_NAME} (isolated name — live IncidentManager untouched)...")
        restore_query = (
            f"RESTORE DATABASE [{TEST_DB_NAME}] FROM DISK = '{BAK_PATH}' "
            f"WITH FILE = {backup_position}, "
            f"MOVE '{data_logical}' TO '{mdf_path}', "
            f"MOVE '{log_logical}' TO '{ldf_path}', REPLACE"
        )
        run_sqlcmd(restore_query)
        print(f"    Restored successfully to {mdf_path}")

        print("\n[5] Spot-checking known counts against the Stage 1 baseline manifest "
              "(C:\\SQLBackup\\ml_stage1_archive_20260716_124546\\stage1_manifest.json: "
              "app_incident_case_row_count=173, table_count=72)...")
        case_count_output = run_sqlcmd("SELECT COUNT(*) FROM dbo.APP_IncidentCase", database=TEST_DB_NAME)
        case_count = int([l.strip() for l in case_count_output.splitlines() if l.strip().isdigit()][0])
        print(f"    dbo.APP_IncidentCase row count: {case_count}")
        assert case_count == 173, f"Expected 173 (Stage 1 baseline manifest), got {case_count}"

        table_count_output = run_sqlcmd("SELECT COUNT(*) FROM sys.tables", database=TEST_DB_NAME)
        table_count = int([l.strip() for l in table_count_output.splitlines() if l.strip().isdigit()][0])
        print(f"    sys.tables count: {table_count}")
        assert table_count == 72, f"Expected 72 (Stage 1 baseline manifest), got {table_count}"

        text_output = run_sqlcmd(
            "SELECT TOP 1 LEN(ComplaintText) FROM dbo.APP_IncidentCase WHERE ComplaintText IS NOT NULL",
            database=TEST_DB_NAME,
        )
        text_len = int([l.strip() for l in text_output.splitlines() if l.strip().isdigit()][0])
        print(f"    Sample ComplaintText readable, length={text_len} chars")
        assert text_len > 0, "Expected restored row data to be genuinely readable, not just present"

        print("\n" + "=" * 70)
        print("RESTORE DRILL PASSED — .bak restores cleanly to an isolated database")
        print("=" * 70)

    finally:
        print(f"\n[Cleanup] Dropping {TEST_DB_NAME}...")
        try:
            run_sqlcmd(
                f"IF DB_ID('{TEST_DB_NAME}') IS NOT NULL "
                f"BEGIN ALTER DATABASE [{TEST_DB_NAME}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE; "
                f"DROP DATABASE [{TEST_DB_NAME}]; END"
            )
            print("    Dropped. Live IncidentManager DB was never touched.")
        except Exception as e:
            print(f"    [WARNING] Cleanup failed — {TEST_DB_NAME} may still exist, drop it manually: {e}")


if __name__ == "__main__":
    main()
