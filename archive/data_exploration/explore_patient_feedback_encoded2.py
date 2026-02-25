#!/usr/bin/env python3
"""
explore_patient_feedback_encoded2.py

Extended numeric/categorical exploration:
 - Histograms for each numeric column
 - Summary statistics Excel
 - Spearman correlation matrix + heatmap
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# === CONFIG ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_DB_PATHS = [
    os.path.join(SCRIPT_DIR, "patient_feedback.db"),
    os.path.join(SCRIPT_DIR, "data_exploration", "patient_feedback.db"),
    os.path.join(SCRIPT_DIR, "..", "data_exploration", "patient_feedback.db"),
]
TABLE_PREFERENCES = ["patient_feedback_encoded", "patient_feedback_clean"]

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_DIR = os.path.join(SCRIPT_DIR, "reports", f"exploration2_{TIMESTAMP}")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SUMMARY_EXCEL = os.path.join(REPORT_DIR, "feature_summary.xlsx")

TEXT_COLS = {
    "patient_full_name", "complaint_text", "immediate_action", "taken_action",
    "embedding_text1", "embedding_text2", "embedding_text3",
    "embedding_text123", "embedding_text23", "embedding_model"
}

# === UTILITIES ===
def find_db():
    for p in POSSIBLE_DB_PATHS:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None

def load_table(db_path, prefs):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    chosen = next((t for t in prefs if t in tables), None)
    if chosen is None:
        raise RuntimeError(f"No valid table found in {tables}")
    df = pd.read_sql_query(f"SELECT * FROM {chosen}", conn)
    conn.close()
    return df, chosen

def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

# === MAIN ===
def main():
    db = find_db()
    if not db:
        sys.exit("❌ Database not found.")
    df, table = load_table(db, TABLE_PREFERENCES)
    print(f"✅ Loaded {table} from {db} ({len(df)} rows)")

    # Select numeric and encoded categorical columns
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in TEXT_COLS
    ]
    print(f"📊 Numeric columns detected: {numeric_cols}\n")

    # === HISTOGRAMS ===
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[col].dropna(), bins=20, edgecolor="black", alpha=0.7)
        ax.set_title(f"Histogram: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        save_fig(fig, f"hist_{col}.png")

    # === SUMMARY STATISTICS ===
    summary = []
    for col in numeric_cols:
        s = df[col].dropna()
        summary.append({
            "Feature": col,
            "Count": len(s),
            "Missing": df[col].isna().sum(),
            "Missing_%": 100 * df[col].isna().mean(),
            "Mean": s.mean() if len(s) else np.nan,
            "Std": s.std() if len(s) else np.nan,
            "Min": s.min() if len(s) else np.nan,
            "Max": s.max() if len(s) else np.nan,
            "Unique": df[col].nunique(dropna=True)
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_excel(SUMMARY_EXCEL, index=False)
    print(f"📘 Feature summary saved at: {SUMMARY_EXCEL}")

    # === CORRELATION MATRIX ===
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr(method="spearman")
        corr_path = os.path.join(REPORT_DIR, "correlation_matrix.csv")
        corr.to_csv(corr_path)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
        ax.set_title("Spearman Correlation Heatmap")
        save_fig(fig, "correlation_heatmap.png")
        print(f"📈 Correlation heatmap and matrix saved at: {corr_path}")
    else:
        print("⚠️ Not enough numeric columns for correlation analysis.")

    print(f"\n✅ Report ready at: {os.path.abspath(REPORT_DIR)}")

if __name__ == "__main__":
    main()
