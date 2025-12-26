import csv
from backend.core.database import get_connection

CSV_PATH = r"C:\Users\IT\OneDrive\Obsidain Directory\AI Department Documentation\Project 1 ; Patient Feedback\Documents from the Hospital\Data From Hussein Borji\administration.csv"


def clean_key(key: str) -> str:
    """Remove BOM and whitespace from CSV headers."""
    return key.strip().replace("\ufeff", "")


def normalize_int(value):
    if value in (None, "", "NULL"):
        return None
    return int(value)


# ================================
# START INSERT (CORRECT WAY)
# ================================

conn = get_connection()
cursor = conn.cursor()

# IMPORTANT: enable identity insert
cursor.execute("SET IDENTITY_INSERT dbo.AdminsrationUnit ON")

with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for raw_row in reader:
        row = {clean_key(k): v for k, v in raw_row.items()}

        cursor.execute(
            """
            INSERT INTO dbo.AdminsrationUnit
            (
                UniqueID,
                Name,
                ParentID,
                Frozen,
                Type,
                CreateDate,
                CreateID,
                UpdateDate,
                UpdateUser
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            normalize_int(row.get("UniqueID")),
            row.get("Name"),
            normalize_int(row.get("ParentID")),
            normalize_int(row.get("Frozen")) or 0,
            normalize_int(row.get("Type")),
            None,  # legacy CreateDate discarded
            normalize_int(row.get("CreateID")),
            None,  # legacy UpdateDate discarded
            normalize_int(row.get("UpdateUser")),
        )

# IMPORTANT: disable identity insert
cursor.execute("SET IDENTITY_INSERT dbo.AdminsrationUnit OFF")

conn.commit()
conn.close()

print("✅ Admin units inserted successfully with hierarchy preserved")
