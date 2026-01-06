import sqlite3
import json
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
DB_PATH = Path(r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\models_directory\patient_feedback_ml.db")
MAPPING_PATH = Path(r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\backend\config\ML_To_Database_Encoding.json")

SRC_TABLE = "patient_feedback_encoded"
NEW_TABLE = "patient_feedback_encoded_new"

# ----------------------------
# Load mapping (idMap only)
# ----------------------------
with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    id_map = json.load(f)["idMap"]

# Columns we will re-encode (mapping keys)
ENCODE_COLS = [
    "domain",
    "category",
    "subcategory",  # mapping key name -> DB column is sub_category
    "severity_level",
    "stage",
    "harm_level",
    "improvement_opportunity_type",
    "feedback_type",
    "classification_en",  # ✅ NEW
]

# Mapping key name -> DB column name
DB_COL_RENAME = {
    "subcategory": "sub_category"
}

def normalize_new_value(new_val):
    """
    Your JSON sometimes maps like "73": [130] (list),
    and sometimes like "1": 3 (int).
    This returns a single int.
    """
    if isinstance(new_val, list):
        if len(new_val) == 0:
            return None
        return int(new_val[0])
    if new_val is None:
        return None
    return int(new_val)

def build_case_expr(db_col: str, mapping_dict: dict) -> str:
    """
    SQL:
    (CASE db_col WHEN old THEN new ... ELSE db_col END)
    """
    parts = []
    for old, new in mapping_dict.items():
        new_int = normalize_new_value(new)
        if new_int is None:
            # If mapping has empty list / null, skip it (do not change those)
            continue
        parts.append(f"WHEN {int(old)} THEN {new_int}")

    if not parts:
        # nothing to map; keep as-is
        return db_col

    return f"(CASE {db_col} " + " ".join(parts) + f" ELSE {db_col} END)"

def table_exists(cur, table_name: str) -> bool:
    r = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    ).fetchone()
    return r is not None

def get_columns(cur, table_name: str):
    rows = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [r[1] for r in rows]

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

try:
    cur.execute("BEGIN")

    # 1) Ensure source exists
    if not table_exists(cur, SRC_TABLE):
        raise RuntimeError(f"Source table not found: {SRC_TABLE}")

    # 2) Drop new table if exists
    if table_exists(cur, NEW_TABLE):
        cur.execute(f"DROP TABLE {NEW_TABLE}")

    # 3) Create new table with same columns
    cur.execute(f"CREATE TABLE {NEW_TABLE} AS SELECT * FROM {SRC_TABLE} WHERE 0")

    src_cols = get_columns(cur, SRC_TABLE)
    new_cols = get_columns(cur, NEW_TABLE)
    if src_cols != new_cols:
        raise RuntimeError("New table schema mismatch after creation.")

    # 4) Build SELECT list (transform where needed)
    select_exprs = []
    for col in src_cols:
        mapping_key = None

        for key in ENCODE_COLS:
            db_col = DB_COL_RENAME.get(key, key)
            if col == db_col and key in id_map:
                mapping_key = key
                break

        if mapping_key is None:
            select_exprs.append(col)
        else:
            select_exprs.append(build_case_expr(col, id_map[mapping_key]) + f" AS {col}")

    select_sql = ",\n    ".join(select_exprs)

    # 5) Filter out 2026 records while copying
    filter_sql = """
    WHERE feedback_received_date IS NULL
       OR (
            feedback_received_date NOT LIKE '2026%'
        AND feedback_received_date NOT LIKE '%/2026'
        AND feedback_received_date NOT LIKE '%/2026 %'
       )
    """

    # 6) Copy with transformation
    insert_sql = f"""
    INSERT INTO {NEW_TABLE}
    SELECT
    {select_sql}
    FROM {SRC_TABLE}
    {filter_sql}
    """
    cur.execute(insert_sql)

    conn.commit()

    old_count = cur.execute(f"SELECT COUNT(*) FROM {SRC_TABLE}").fetchone()[0]
    new_count = cur.execute(f"SELECT COUNT(*) FROM {NEW_TABLE}").fetchone()[0]
    print(f"✅ Done. {SRC_TABLE} rows: {old_count} | {NEW_TABLE} rows: {new_count}")

except Exception:
    conn.rollback()
    raise
finally:
    conn.close()
