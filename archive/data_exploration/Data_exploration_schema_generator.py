import pandas as pd
import os
import random

# === CONFIG ===
EXCEL_FILE = os.path.join("original_data.xls")
SCHEMA_OUTPUT = os.path.join("Reports/schema_proposal.txt")
CATEGORIES_OUTPUT = os.path.join("Reports/categories_report.txt")
TOP_N_UNIQUES = 50
SAMPLE_COUNT = 5

# === FUNCTIONS ===

def infer_sql_type(series: pd.Series):
    """Infer basic SQLite-compatible type based on pandas dtype and content."""
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(series):
        return "REAL"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "DATE"
    else:
        # Try to detect dates in string form
        sample = series.dropna().astype(str).head(10)
        if sample.str.match(r"\d{4}-\d{2}-\d{2}|[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}").any():
            return "DATE"
        return "TEXT"


def analyze_column(col_name, series):
    """Analyze a column and return its SQL type and descriptive text."""
    sql_type = infer_sql_type(series)
    report = [
        f"📘 Column: {col_name}",
        f" - Non-null count: {series.count()}",
        f" - Missing values: {series.isna().sum()}",
        f" - Unique values: {series.nunique()}",
        f" - Proposed SQL type: {sql_type}",
    ]

    # Add top N frequent values
    top_values = series.value_counts(dropna=True).head(TOP_N_UNIQUES)
    if not top_values.empty:
        report.append(f" - Top {TOP_N_UNIQUES} unique values:")
        for val, count in top_values.items():
            val_str = str(val).replace("\n", " ")[:80]
            report.append(f"     {val_str} ({count})")

    # Add a few sample values
    non_null = series.dropna()
    if len(non_null) > 0:
        samples = list(non_null.head(SAMPLE_COUNT))
        samples += random.sample(list(non_null), min(SAMPLE_COUNT, len(non_null)))
        report.append(" - Sample values:")
        for val in samples:
            report.append(f"     {str(val)[:100]}")

    return "\n".join(report), sql_type


def load_excel(filepath):
    """Try reading .xls or .xlsx using the correct engine."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Excel file not found: {os.path.abspath(filepath)}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".xls":
        return pd.read_excel(filepath, engine="xlrd")
    elif ext == ".xlsx":
        return pd.read_excel(filepath, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# === MAIN ===

print("📂 Loading Excel file...")
df = load_excel(EXCEL_FILE)
print(f"✅ Loaded: {EXCEL_FILE}")
print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")

schema_lines = ["CREATE TABLE patient_feedback ("]
schema_lines.append("    id INTEGER PRIMARY KEY AUTOINCREMENT,")
report_lines = []
category_lines = []

for col in df.columns:
    analysis, sql_type = analyze_column(col, df[col])
    report_lines.append(analysis)
    report_lines.append("\n" + "-" * 60 + "\n")

    schema_lines.append(f'    "{col}" {sql_type},')

    # Collect category values if column seems categorical
    if df[col].nunique() <= 100:  # heuristic threshold
        unique_vals = df[col].dropna().unique().tolist()
        category_lines.append(f"\n--- {col} --- ({len(unique_vals)} unique values)\n")
        for val in unique_vals:
            category_lines.append(f" • {val}")

# Add embedding fields
embedding_fields = [
    ("embedding_text1", "TEXT"),
    ("embedding_text2", "TEXT"),
    ("embedding_text3", "TEXT"),
    ("embedding_text123", "TEXT"),
    ("embedding_text23", "TEXT"),
    ("embedding_model", "TEXT"),
]
for name, dtype in embedding_fields:
    schema_lines.append(f"    {name} {dtype},")

schema_lines[-1] = schema_lines[-1].rstrip(",")
schema_lines.append(");")

schema_sql = "\n".join(schema_lines)

# === SAVE OUTPUTS ===

full_report = (
    f"SCHEMA PROPOSAL REPORT\n{'='*60}\n\n"
    + "\n".join(report_lines)
    + "\n\nProposed SQLite Schema:\n"
    + "-"*60 + "\n"
    + schema_sql
)

with open(SCHEMA_OUTPUT, "w", encoding="utf-8") as f:
    f.write(full_report)

with open(CATEGORIES_OUTPUT, "w", encoding="utf-8") as f:
    f.write("\n".join(category_lines))

print(f"\n✅ Schema proposal saved to: {os.path.abspath(SCHEMA_OUTPUT)}")
print(f"✅ Categories report saved to: {os.path.abspath(CATEGORIES_OUTPUT)}")
print("\n🧩 Review both files, then we’ll finalize the SQLite DB creation.")
