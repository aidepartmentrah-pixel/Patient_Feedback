import sqlite3
from pathlib import Path

# --------------------------------------------------
# ML DATABASE PATH (SQLite)
# --------------------------------------------------
DB_PATH = Path(
    r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\models_directory\patient_feedback_ml.db"
)


conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# --------------------------------------------------
# Get all tables
# --------------------------------------------------
tables = cursor.execute("""
SELECT name
FROM sqlite_master
WHERE type='table'
AND name NOT LIKE 'sqlite_%'
ORDER BY name
""").fetchall()

output = []

for (table_name,) in tables:
    output.append("=" * 70)
    output.append(f"TABLE: {table_name}")
    output.append("=" * 70)
    output.append("COLUMNS:")

    # --------------------------------------------------
    # Columns
    # --------------------------------------------------
    columns = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()

    for cid, name, dtype, notnull, default, pk in columns:
        line = f"  - {name}: {dtype}"
        if notnull:
            line += " (NOT NULL)"
        if default is not None:
            line += f" (Default={default})"
        if pk:
            line += " (PRIMARY KEY)"
        output.append(line)

    # --------------------------------------------------
    # Foreign Keys
    # --------------------------------------------------
    fks = cursor.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()

    output.append("\nFOREIGN KEYS:")
    if fks:
        for fk in fks:
            _, _, ref_table, from_col, to_col, *_ = fk
            output.append(f"  - {from_col} -> {ref_table}.{to_col}")
    else:
        output.append("  None")

    output.append("\n")

# --------------------------------------------------
# Write schema to file
# --------------------------------------------------
OUTPUT_FILE = "ml_sqlite_schema.txt"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(output))

conn.close()

print(f"✅ ML SQLite schema exported to {OUTPUT_FILE}")
