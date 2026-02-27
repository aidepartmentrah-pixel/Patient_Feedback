import sqlite3
import re

# ---------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------
DB_PATH = r"patient_feedback_ml.db"

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"

TEXT1 = "complaint_text"
TEXT2 = "immediate_action"
TEXT3 = "taken_action"


# ---------------------------------------------------
# 2. Arabic text cleaning function
# ---------------------------------------------------
def clean_arabic(text: str) -> str:
    if text is None:
        return ""

    # Convert to string
    text = str(text)

    # Remove Arabic diacritics (Tashkeel)
    diacritics_pattern = re.compile(r"[\u064B-\u065F]")
    text = re.sub(diacritics_pattern, "", text)

    # Normalize characters
    replacements = {
        "أ": "ا", "إ": "ا", "آ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Remove non-Arabic letters and common punctuation
    text = re.sub(r"[^؀-ۿ0-9\s]+", " ", text)

    # Collapse extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------
# 3. Function to update a table
# ---------------------------------------------------
def clean_table(table_name: str):

    print(f"\nCleaning table: {table_name}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Fetch needed columns
    cursor.execute(
        f"SELECT rowid, {TEXT1}, {TEXT2}, {TEXT3} FROM {table_name}"
    )
    rows = cursor.fetchall()

    for rowid, t1, t2, t3 in rows:

        new_t1 = clean_arabic(t1)
        new_t2 = clean_arabic(t2)
        new_t3 = clean_arabic(t3)

        cursor.execute(f"""
            UPDATE {table_name}
            SET {TEXT1}=?,
                {TEXT2}=?,
                {TEXT3}=?
            WHERE rowid=?
        """, (new_t1, new_t2, new_t3, rowid))

    conn.commit()
    conn.close()
    print(f"Completed cleaning: {table_name}")


# ---------------------------------------------------
# 4. RUN
# ---------------------------------------------------
clean_table(TRAIN_TABLE)
clean_table(TEST_TABLE)

print("\nALL text fields cleaned and overwritten successfully!")
