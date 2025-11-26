import sqlite3
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
from models_directory.Classification_Models.Stage.modular_functions import get_embedding

# ---------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------
DB_PATH = r"patient_feedback_ml.db"
MODEL_PATH = r"Classification_Models/model_storage/mpnet_embeddings"

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"

# Original 3 text columns
TEXT1 = "complaint_text"
TEXT2 = "immediate_action"
TEXT3 = "taken_action"

# Embedding columns in the database
EMB1 = "embedding_text1"
EMB2 = "embedding_text2"
EMB3 = "embedding_text3"
EMB23 = "embedding_text23"
EMB123 = "embedding_text123"

# ---------------------------------------------------
# 2. LOAD OFFLINE MODEL
# ---------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH)
model.eval()


# ---------------------------------------------------
# 4. Process a table
# ---------------------------------------------------
def process_table(table_name: str):
    print(f"\n🔄 Processing table: {table_name}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Fetch original 3 text fields
    cur.execute(f"SELECT rowid, {TEXT1}, {TEXT2}, {TEXT3} FROM {table_name}")
    rows = cur.fetchall()

    for rowid, t1, t2, t3 in tqdm(rows, desc=f"Updating {table_name}"):

        # Convert None → empty string
        t1 = t1 or ""
        t2 = t2 or ""
        t3 = t3 or ""

        # Create all required embeddings
        emb1 = get_embedding(t1)
        emb2 = get_embedding(t2)
        emb3 = get_embedding(t3)

        emb23 = get_embedding(t2 + " " + t3)
        emb123 = get_embedding(t1 + " " + t2 + " " + t3)

        # Update DB row
        cur.execute(f"""
            UPDATE {table_name}
            SET {EMB1}=?,
                {EMB2}=?,
                {EMB3}=?,
                {EMB23}=?,
                {EMB123}=?
            WHERE rowid=?
        """, (emb1, emb2, emb3, emb23, emb123, rowid))

    conn.commit()
    conn.close()
    print(f"✅ Completed: {table_name}")


# ---------------------------------------------------
# 5. Run on train & test tables
# ---------------------------------------------------
process_table(TRAIN_TABLE)
process_table(TEST_TABLE)

print("\n🎉 ALL embeddings overwritten successfully!")
