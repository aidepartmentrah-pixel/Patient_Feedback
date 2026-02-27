import sqlite3

DB_PATH = "patient_feedback_ml.db"

TABLES = [
    "patient_feedback_encoded",
    "table_feedback_train",
    "table_feedback_test",
]

# ---------------------------
# Your FIXING REPORT mappings
# ---------------------------
FIX_MAP = {
    "sub_category": {
        7: 6,
        17: 16,
        20: 19,
        25: 1
    },
    "severity_level": {
        4: 3,
        5: 3
    },
    "stage": {
        3: 2,
        5: 4,
        7: 1
    },
    "status": {
        2: 1
    },
    "harm_level": {
        7: 1
    }
}


def apply_fixes(conn, table_name):
    cursor = conn.cursor()
    print(f"\nProcessing table: {table_name}")

    for col, mapping in FIX_MAP.items():
        for old_val, new_val in mapping.items():
            cursor.execute(
                f"""
                UPDATE {table_name}
                SET {col} = ?
                WHERE {col} = ?;
                """,
                (new_val, old_val)
            )
            print(f"   ✔ Updated {col}: {old_val} → {new_val}")

    conn.commit()


def main():
    conn = sqlite3.connect(DB_PATH)

    for table in TABLES:
        apply_fixes(conn, table)

    conn.close()
    print("\n✅ All fixes applied successfully!")


if __name__ == "__main__":
    main()
