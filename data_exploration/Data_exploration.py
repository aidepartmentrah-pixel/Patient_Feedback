import pandas as pd
import os
import random

# === CONFIG ===
EXCEL_FILE = "original_data.xls"  # The file is inside data_exploration
OUTPUT_FILE = "schema_proposal.txt"
TOP_N_UNIQUES = 32
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
        # check if looks like a date string
        sample = series.dropna().astype(str).head(10)
        date_like = sample.str.match(r"\d{4}-\d{2}-\d{2}|[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}")
        if date_like.any():
            return "DATE"
        return "TEXT"


def analyze_column(col_name, series):
    """Analyze a column and return a text report."""
    report = []
    report.append(f"📘 Column: {col_name}")
    report.append(f" - Non-null count: {series.count()}")
    report.append(f" - Missing values: {series.isna().sum()}")
    report.append(f" - Unique values: {series.nunique()}")

    sql_type = infer_sql_type(series)
    report.append(f" - Proposed SQL type: {sql_type}")

    # Top N most frequent values
    top_values = series.value_counts(dropna=True).head(TOP_N_UNIQUES)
    if not top_values.empty:
        report.append(f" - Top {TOP_N_UNIQUES} values:")
        for val, count in top_values.items():
            val_str = str(val).replace("\n", " ")[:80]  # truncate long text
            report.append(f"     {val_str} ({count})")

    # Samples
    non_null = series.dropna()
    if len(non_null) > 0:
        sample_head = list(non_null.head(SAMPLE_COUNT))
        sample_rand = random.sample(list(non_null), min(SAMPLE_COUNT, len(non_null)))
        samples = sample_head + sample_rand
        report.append(f" - Sample values:")
        for val in samples:
            report.append(f"     {str(val)[:100]}")

    return "\n".join(report), col_name, sql_type


def load_excel_safely(filepath):
    """Try loading Excel file with correct engine automatically."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Excel file not found: {os.path.abspath(filepath)}")

    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".xls":
            return pd.read_excel(filepath, engine="xlrd")
        elif ext == ".xlsx":
            return pd.read_excel(filepath, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load Excel file: {e}")


# === MAIN ===

df = load_excel_safely(EXCEL_FILE)
print(f"✅ Loaded Excel file: {EXCEL_FILE} with {len(df)} rows and {len(df.columns)} columns.\n")

report_lines = []
schema_lines = ["CREATE TABLE patient_feedback ("]
schema_lines.append("    id INTEGER PRIMARY KEY AUTOINCREMENT,")

for col in df.columns:
    analysis, col_name, sql_type = analyze_column(col, df[col])
    report_lines.append(analysis)
    schema_lines.append(f'    "{col_name}" {sql_type},')
    report_lines.append("\n" + "-" * 60 + "\n")

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

# Close schema
schema_lines[-1] = schema_lines[-1].rstrip(",")  # remove last comma
schema_lines.append(");")
schema_sql = "\n".join(schema_lines)

# Combine all text
full_report = (
    f"SCHEMA PROPOSAL REPORT\n{'='*60}\n\n"
    + "\n\n".join(report_lines)
    + "\n\nProposed SQLite Schema:\n"
    + "-"*60 + "\n"
    + schema_sql
)

# Print to console
print(full_report)

# Save to file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(full_report)

print(f"\n✅ Report saved to: {os.path.abspath(OUTPUT_FILE)}")
