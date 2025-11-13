"""
create_patient_feedback_db.py  (FIXED)

- Reads:  data_exploration/original_data.xls (or .xlsx)
- Creates: data_exploration/patient_feedback.db (SQLite)
- Table:   patient_feedback (all original columns + embedding placeholder columns)
- Embedding columns are created but left NULL for now.

Fixes:
- Properly escapes double-quotes in column names for SQL (" -> "").
- Inserts rows in chunks for robustness.
"""

import pandas as pd
import os
import sqlite3
import datetime

# === CONFIG ===
# Excel file expected next to this script

SCRIPT_DIR = os.path.dirname(__file__)
EXCEL_PATH = os.path.join( "original_data.xls")   # change to .xlsx if needed
OUTPUT_DB = os.path.join( "patient_feedback.db")
SHEET_NAME = None  # set to sheet name (str) to force a sheet, or None to auto-pick the first sheet

# tune chunk size for inserts
INSERT_CHUNK = 500

# === Utility functions ===

def load_excel_auto(path, sheet_name=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found: {os.path.abspath(path)}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xls":
        loaded = pd.read_excel(path, sheet_name=sheet_name, engine="xlrd")
    elif ext == ".xlsx":
        loaded = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    else:
        raise ValueError("Unsupported Excel file extension: " + ext)

    # If sheet_name=None, pandas returns a dict of DataFrames (one per sheet).
    if isinstance(loaded, dict):
        if sheet_name:
            if sheet_name in loaded:
                df = loaded[sheet_name]
                chosen = sheet_name
            else:
                raise ValueError(f"Requested sheet '{sheet_name}' not found. Available: {list(loaded.keys())}")
        else:
            first_key = list(loaded.keys())[0]
            df = loaded[first_key]
            chosen = first_key
        print(f"Note: Excel had multiple sheets. Using sheet: '{chosen}'")
        return df
    else:
        return loaded

def infer_sql_type(series: pd.Series) -> str:
    """Infer a simple SQLite type for the column."""
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "DATE"
    # try to detect date-like strings
    sample = series.dropna().astype(str).head(10)
    if not sample.empty and sample.str.match(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}").any():
        return "DATE"
    return "TEXT"

def normalize_value(v):
    """Convert pandas/numpy values to Python native types suitable for sqlite3."""
    if pd.isna(v):
        return None
    # datetimes -> ISO string
    if isinstance(v, (pd.Timestamp, datetime.datetime, datetime.date)):
        return v.isoformat()
    # numpy scalar -> native python
    try:
        return v.item()
    except Exception:
        return v

def escape_identifier(name: str) -> str:
    """Escape SQL identifier for SQLite by doubling internal double-quotes."""
    return name.replace('"', '""')

# === Main flow ===

def main():
    print("Loading Excel:", EXCEL_PATH)
    df = load_excel_auto(EXCEL_PATH, sheet_name=SHEET_NAME)
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError("Loaded object is not a DataFrame.")
    print(f"Loaded DataFrame with {len(df)} rows and {len(df.columns)} columns.\n")

    # === Clean and normalize column names ===
    def clean_sql_name(name: str) -> str:
        # Convert to string and strip whitespace/newlines
        n = str(name).replace("\n", " ").replace("\r", " ").strip()
        n = n.replace('"', "'")  # avoid double quotes
        # Replace forbidden/special symbols with underscores
        for ch in ["(", ")", ",", ".", "-", "/", "+", "&", ":", ";"]:
            n = n.replace(ch, "_")
        # Replace multiple spaces/underscores
        while "__" in n:
            n = n.replace("__", "_")
        n = n.replace(" ", "_")
        # Truncate long names
        if len(n) > 80:
            n = n[:80]
        return n

    cleaned_cols = [clean_sql_name(c) for c in df.columns]
    rename_map = dict(zip(df.columns, cleaned_cols))
    df.columns = cleaned_cols

    print("Normalized column names:")
    for orig, clean in rename_map.items():
        print(f" - {orig}  -->  {clean}")
    print()

    # Save the mapping for traceability
    with open(os.path.join(SCRIPT_DIR, "column_name_mapping.txt"), "w", encoding="utf-8") as f:
        for orig, clean in rename_map.items():
            f.write(f"{orig} --> {clean}\n")
    print("🗂️ Column name mapping saved to column_name_mapping.txt\n")

    # === Build the schema ===
    original_cols = list(df.columns)
    col_types = {col: infer_sql_type(df[col]) for col in original_cols}

    embedding_columns = [
        ("embedding_text1", "TEXT"),
        ("embedding_text2", "TEXT"),
        ("embedding_text3", "TEXT"),
        ("embedding_text123", "TEXT"),
        ("embedding_text23", "TEXT"),
        ("embedding_model", "TEXT"),
    ]

    create_cols_sql = ['    id INTEGER PRIMARY KEY AUTOINCREMENT']
    for col in original_cols:
        sql_type = col_types.get(col, "TEXT")
        esc = escape_identifier(col)
        create_cols_sql.append(f'    "{esc}" {sql_type}')
    for name, dtype in embedding_columns:
        esc = escape_identifier(name)
        create_cols_sql.append(f'    "{esc}" {dtype}')

    create_table_sql = "DROP TABLE IF EXISTS patient_feedback; CREATE TABLE patient_feedback (\n" + ",\n".join(
        create_cols_sql) + "\n);"

    print("Creating SQLite database at:", OUTPUT_DB)
    conn = sqlite3.connect(OUTPUT_DB)
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS patient_feedback")
        cur.execute(
            "CREATE TABLE patient_feedback (\n" + ",\n".join(create_cols_sql) + "\n);")
        conn.commit()
        print("Table patient_feedback created (or already exists).")

        all_columns = original_cols + [name for name, _ in embedding_columns]
        quoted_cols = ", ".join([f'"{escape_identifier(c)}"' for c in all_columns])
        placeholders = ", ".join(["?"] * len(all_columns))
        insert_sql = f'INSERT INTO patient_feedback ({quoted_cols}) VALUES ({placeholders})'

        print("Inserting rows into database (embedding columns will be NULL)...")
        rows = []
        total = 0
        for _, row in df.iterrows():
            values = [normalize_value(row[c]) for c in original_cols]
            for _ in embedding_columns:
                values.append(None)
            rows.append(tuple(values))

            if len(rows) >= INSERT_CHUNK:
                cur.executemany(insert_sql, rows)
                conn.commit()
                total += len(rows)
                print(f"  inserted {total} rows...", end="\r")
                rows = []

        if rows:
            cur.executemany(insert_sql, rows)
            conn.commit()
            total += len(rows)
            print(f"  inserted {total} rows.", end="\n")

        print(f"Inserted total {total} rows into patient_feedback.")
    finally:
        conn.close()

    print("\n✅ Done. Database ready at:", os.path.abspath(OUTPUT_DB))
    print("Note: embedding columns are present and currently NULL — ready for later population.")


if __name__ == "__main__":
    main()
