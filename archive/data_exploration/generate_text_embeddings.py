"""
generate_text_embeddings.py

Purpose:
- Generate embeddings for text1, text2, text3 fields in patient_feedback_encoded
- Model: aubmindlab/bert-base-arabertv2
- Saves embeddings as JSON arrays into the same DB

CPU-optimized version (batch processing + progress logging)
"""

import os
import time
import json
import sqlite3
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# === CONFIG ===
DATA_DIR = "data_exploration"
DB_PATH = os.path.join("patient_feedback.db")
TABLE_NAME = "patient_feedback_encoded"
BATCH_SIZE = 8  # small batch for CPU
DEVICE = torch.device("cpu")  # use CPU on purpose

TEXT_COLUMNS = {
    "embedding_text1": "complaint_text",
    "embedding_text2": "immediate_action",
    "embedding_text3": "taken_action",
}

MODEL_NAME = "aubmindlab/bert-base-arabertv2"

# === FUNCTIONS ===
def mean_pooling(model_output, attention_mask):
    """Compute mean pooling on token embeddings (CLS excluded)."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def generate_embedding(texts, tokenizer, model):
    """Generate mean pooled embeddings for a list of texts."""
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**encoded_input.to(DEVICE))
    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    return embeddings.cpu().numpy()


def update_embeddings_in_db(conn, df_chunk, column_name):
    """Update embeddings into SQLite for a given column."""
    cursor = conn.cursor()
    for _, row in df_chunk.iterrows():
        cursor.execute(
            f"UPDATE {TABLE_NAME} SET {column_name} = ? WHERE id = ?",
            (json.dumps(row[column_name]), int(row["id"]))
        )
    conn.commit()


def main():
    start_time = time.time()

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    print("🔗 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully.\n")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT id, complaint_text, immediate_action, taken_action FROM {TABLE_NAME}", conn)
    print(f"✅ Loaded {len(df)} records from {TABLE_NAME}\n")

    for embed_col, text_col in TEXT_COLUMNS.items():
        print(f"🧠 Generating embeddings for {text_col} → {embed_col}")
        df[embed_col] = None

        for i in tqdm(range(0, len(df), BATCH_SIZE)):
            batch = df.iloc[i:i + BATCH_SIZE]
            texts = [t if isinstance(t, str) else "" for t in batch[text_col]]
            if not any(texts):
                continue

            embeddings = generate_embedding(texts, tokenizer, model)

            # safely assign embeddings row by row
            for j, idx in enumerate(batch.index):
                df.at[idx, embed_col] = embeddings[j].tolist()

            # periodic DB commit
            update_embeddings_in_db(conn, df.iloc[i:i + BATCH_SIZE][["id", embed_col]], embed_col)

        print(f"✅ Done embedding for {text_col}.\n")

    conn.close()
    print(f"\n🏁 All embeddings generated and stored successfully!")
    print(f"🕒 Total time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
