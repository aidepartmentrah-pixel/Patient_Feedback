#!/usr/bin/env python3
"""
explore_patient_feedback_encoded.py

Produces a full exploratory data analysis report for the HCAT dataset.

Outputs (in reports/exploration_YYYYMMDD_HHMMSS/):
 - exploration_report.md  (human-readable markdown report)
 - figures/*.png          (plots)
 - tables/*.csv           (summary CSVs)
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
from itertools import combinations
from datetime import datetime

# === CONFIG ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_DB_PATHS = [
    os.path.join(SCRIPT_DIR, "patient_feedback.db"),
    os.path.join(SCRIPT_DIR, "data_exploration", "patient_feedback.db"),
    os.path.join(SCRIPT_DIR, "..", "data_exploration", "patient_feedback.db"),
]
TABLE_PREFERENCES = ["patient_feedback_encoded", "patient_feedback_clean", "patient_feedback"]

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_DIR = os.path.join(SCRIPT_DIR, "reports", f"exploration_{TIMESTAMP}")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
TABLE_DIR = os.path.join(REPORT_DIR, "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

DEFAULT_TEXT_COLS = [
    "patient_full_name", "complaint_text", "immediate_action", "taken_action",
    "Raw_complaint_text", "محتوى_الشكوى__Raw_Content_",
    "embedding_text1", "embedding_text2", "embedding_text3",
    "embedding_text123", "embedding_text23", "embedding_model"
]

# === Utility ===
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
        raise RuntimeError(f"No matching table found. Available: {tables}")
    df = pd.read_sql_query(f"SELECT * FROM {chosen}", conn)
    conn.close()
    return df, chosen, tables

def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
def tokenize_text(s):
    if pd.isna(s): return []
    return TOKEN_RE.findall(str(s).lower())

def cramers_v(x, y):
    confusion = pd.crosstab(x, y)
    chi2 = (((confusion - confusion.mean())**2) / (confusion.mean() + 1e-9)).to_numpy().sum()
    n = confusion.sum().sum()
    phi2 = chi2 / n if n > 0 else 0
    r, k = confusion.shape
    denom = max(min((k-1), (r-1)), 1)
    return np.sqrt(phi2 / denom)

# === Analysis ===
def general_summary(df):
    return {
        "rows": len(df),
        "cols": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "missing": df.isna().sum().to_dict(),
        "uniques": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
        "types": df.dtypes.apply(str).to_dict()
    }

def categorical_distributions(df, cat_cols):
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False)
        vc.to_csv(os.path.join(TABLE_DIR, f"dist_{col}.csv"))
        fig, ax = plt.subplots(figsize=(8, max(3, min(12, len(vc)*0.25))))
        ax.bar(vc.index.astype(str), vc.values)
        ax.set_title(f"Distribution: {col}")
        ax.set_xticklabels([str(x) for x in vc.index], rotation=45, ha="right")
        save_fig(fig, f"dist_{col}.png")

def numeric_summary(df, num_cols):
    for col in num_cols:
        stats = df[col].describe().to_dict()
        pd.DataFrame([stats]).to_csv(os.path.join(TABLE_DIR, f"numeric_{col}.csv"))

def correlation_analysis(df, cat_cols, num_cols):
    if len(cat_cols) > 1:
        cramer = pd.DataFrame(index=cat_cols, columns=cat_cols)
        for a, b in combinations(cat_cols, 2):
            try: cramer.loc[a,b] = cramers_v(df[a].astype(str), df[b].astype(str))
            except: cramer.loc[a,b] = np.nan
        cramer.to_csv(os.path.join(TABLE_DIR, "cramers_v.csv"))
    if len(num_cols) > 1:
        corr = df[num_cols].corr(method="spearman")
        corr.to_csv(os.path.join(TABLE_DIR, "numeric_corr.csv"))

def time_series_analysis(df):
    if "feedback_received_date" not in df.columns:
        return
    d = pd.to_datetime(df["feedback_received_date"], errors="coerce")
    daily = d.dt.date.value_counts().sort_index()
    pd.DataFrame({"date": daily.index, "count": daily.values}).to_csv(
        os.path.join(TABLE_DIR, "feedbacks_per_day.csv"), index=False)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(daily.index, daily.values)
    ax.set_title("Feedbacks per Day")
    save_fig(fig, "timeseries_feedbacks.png")

def text_field_analysis(df, text_cols):
    for col in text_cols:
        if col not in df: continue
        texts = df[col].dropna().astype(str)
        lens = texts.str.len()
        token_counter = Counter()
        for t in texts:
            token_counter.update(tokenize_text(t))
        pd.DataFrame(token_counter.most_common(30),
                     columns=["token","count"]).to_csv(
            os.path.join(TABLE_DIR, f"top_tokens_{col}.csv"), index=False)
        fig, ax = plt.subplots()
        ax.hist(lens, bins=30)
        ax.set_title(f"Text length: {col}")
        save_fig(fig, f"text_len_{col}.png")

# === Main ===
def run():
    db = find_db()
    if not db:
        sys.exit("❌ Could not find database.")
    df, table, _ = load_table(db, TABLE_PREFERENCES)
    print(f"✅ Loaded {table} from {db}")

    summary = general_summary(df)
    pd.DataFrame([summary]).to_csv(os.path.join(TABLE_DIR, "basic_summary.csv"), index=False)

    text_cols = [c for c in df.columns if c in DEFAULT_TEXT_COLS or "text" in c.lower()]
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in text_cols + num_cols + date_cols + ["id"]]

    categorical_distributions(df, cat_cols)
    numeric_summary(df, num_cols)
    correlation_analysis(df, cat_cols, num_cols)
    time_series_analysis(df)
    text_field_analysis(df, text_cols)

    with open(os.path.join(REPORT_DIR, "exploration_report.md"), "w", encoding="utf-8") as f:
        f.write(f"# HCAT Data Exploration Report\n\n")
        f.write(f"**Database:** {db}\n\n**Table:** {table}\n\n")
        f.write(f"Rows: {summary['rows']} | Columns: {summary['cols']} | Duplicates: {summary['duplicates']}\n\n")
        f.write("## Missing values (top 10)\n")
        f.write(pd.Series(summary["missing"]).sort_values(ascending=False).head(10).to_markdown() + "\n")
        f.write("\n## Unique values per column (top 10)\n")
        f.write(pd.Series(summary["uniques"]).sort_values(ascending=False).head(10).to_markdown() + "\n")
        f.write("\n## Columns by type\n")
        f.write(f"Text: {len(text_cols)} | Numeric: {len(num_cols)} | Categorical: {len(cat_cols)} | Date: {len(date_cols)}\n")

    print(f"\n✅ Exploration report ready at: {os.path.abspath(REPORT_DIR)}")

if __name__ == "__main__":
    run()
