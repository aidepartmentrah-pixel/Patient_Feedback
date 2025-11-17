import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report

from package_models import predict_embedding

# ============================================================
# CONFIG
# ============================================================

HERE = Path(__file__).resolve().parent
DB_PATH = HERE.parent / "patient_feedback_ml.db"
TEST_TABLE = "table_feedback_test"

EMBED_COL = "embedding_text1"
DOMAIN_COL = "domain"
CATEGORY_COL = "category"
SUBCAT_COL = "sub_category"


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_table(name):
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(f"SELECT * FROM {name}", conn)
    conn.close()
    return df


df = load_table(TEST_TABLE)
print("Loaded:", len(df))


# ============================================================
# PARSE EMBEDDINGS
# ============================================================

def parse_embedding(v):
    if isinstance(v, str):
        return np.array(eval(v), dtype=np.float32)
    if isinstance(v, list):
        return np.array(v, dtype=np.float32)
    if isinstance(v, (bytes, bytearray)):
        return np.frombuffer(v, dtype=np.float32)
    raise ValueError(type(v))


emb_list = [parse_embedding(v) for v in df[EMBED_COL]]
embeddings = np.vstack(emb_list)


# ============================================================
# PREDICTION
# ============================================================

pred_domain = []
pred_category = []
pred_subcat = []

for emb in embeddings:
    out = predict_embedding(emb)
    pred_domain.append(int(out["domain"]))
    pred_category.append(int(out["category"]))
    pred_subcat.append(int(out["sub_category"]))


# ============================================================
# LABELS → MUST BE INTEGERS
# ============================================================

true_domain = df[DOMAIN_COL].astype(int).tolist()
true_category = df[CATEGORY_COL].astype(int).tolist()
true_subcat = df[SUBCAT_COL].astype(int).tolist()


# ============================================================
# SAVE REPORTS
# ============================================================

with open("eval_domain.txt", "w", encoding="utf-8") as f:
    f.write(classification_report(true_domain, pred_domain, zero_division=0))

with open("eval_category.txt", "w", encoding="utf-8") as f:
    f.write(classification_report(true_category, pred_category, zero_division=0))

with open("eval_sub_category.txt", "w", encoding="utf-8") as f:
    f.write(classification_report(true_subcat, pred_subcat, zero_division=0))

print("\nDONE ✓ Reports generated:")
print(" → eval_domain.txt")
print(" → eval_category.txt")
print(" → eval_sub_category.txt")
