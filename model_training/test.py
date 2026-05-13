import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

DB_PATH = str(Path(__file__).resolve().parent.parent / "models_directory" / "patient_feedback_ml.db")
TABLE = "table_feedback_test"  # your test data

# ===========================================================
# LOAD DATA
# ===========================================================
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(f"SELECT * FROM {TABLE}", conn)
conn.close()

print("Columns in DB:", df.columns.tolist())

# ===========================================================
# SELECT RELEVANT TARGET FEATURES
# ===========================================================
TARGET_COLS = [
    "domain",
    "category",
    "sub_category",
    "severity_level",
    "stage",
    "harm_level"
]

# Filter missing columns (just in case)
available = [c for c in TARGET_COLS if c in df.columns]

df_c = df[available].copy()

# Drop rows where any target is NaN
df_c = df_c.dropna()

# Convert everything to integers
df_c = df_c.astype(int)

print(f"\nUsing {len(df_c)} clean rows for correlation analysis")

# ===========================================================
# CORRELATION MATRIX
# ===========================================================
corr = df_c.corr()

print("\n=== CORRELATION MATRIX ===")
print(corr)

# ===========================================================
# SORTED PAIRWISE CORRELATIONS
# ===========================================================
pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .rename("correlation")
        .reset_index()
)
pairs = pairs.sort_values(by="correlation", ascending=False)

print("\n=== SORTED PAIRWISE CORRELATIONS ===")
print(pairs)

# ===========================================================
# SAVE HEATMAP
# ===========================================================
plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation Matrix — Domain/Category/Sub/Severity/Stage/Harm")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=300)

print("\nSaved: correlation_matrix.png")
