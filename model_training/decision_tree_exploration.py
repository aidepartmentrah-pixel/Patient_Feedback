import pandas as pd
import sqlite3
from pathlib import Path

# ---------------------------------------------------------
# 1. CONNECT TO SQLITE DATABASE
# ---------------------------------------------------------
DB_PATH = Path(__file__).resolve().parent.parent / "models_directory" / "patient_feedback_ml.db"

conn = sqlite3.connect(DB_PATH)

# ---------------------------------------------------------
# 2. READ SOURCE TABLE
# ---------------------------------------------------------
query = """
SELECT domain, category, sub_category, classification_ar, 
       severity_level, stage, harm_level
FROM patient_feedback_encoded
"""

df = pd.read_sql_query(query, conn)

# ---------------------------------------------------------
# 3. BUILD GROUPED MAPPING TABLES
# ---------------------------------------------------------

# --- Domain → Category (grouped)
map_domain_category = (
    df.groupby('domain')['category']
    .apply(lambda x: sorted(set(x)))
    .reset_index()
)
map_domain_category['category_list'] = map_domain_category['category'].apply(
    lambda lst: ", ".join(str(i) for i in lst)
)
map_domain_category['count'] = map_domain_category['category'].apply(len)
map_domain_category = map_domain_category[['domain', 'category_list', 'count']]

# --- Category → Sub-Category (grouped)
map_category_subcategory = (
    df.groupby('category')['sub_category']
    .apply(lambda x: sorted(set(x)))
    .reset_index()
)
map_category_subcategory['sub_category_list'] = map_category_subcategory['sub_category'].apply(
    lambda lst: ", ".join(str(i) for i in lst)
)
map_category_subcategory['count'] = map_category_subcategory['sub_category'].apply(len)
map_category_subcategory = map_category_subcategory[['category', 'sub_category_list', 'count']]

# --- Sub-Category → classification_ar (grouped)
map_subcategory_class = (
    df.groupby('sub_category')['classification_ar']
    .apply(lambda x: sorted(set(x)))
    .reset_index()
)
map_subcategory_class['classification_ar_list'] = map_subcategory_class['classification_ar'].apply(
    lambda lst: ", ".join(str(i) for i in lst)
)
map_subcategory_class['count'] = map_subcategory_class['classification_ar'].apply(len)
map_subcategory_class = map_subcategory_class[['sub_category', 'classification_ar_list', 'count']]

# ---------------------------------------------------------
# 4. INVERSE MAPPINGS
# ---------------------------------------------------------

# --- Severity Level → classification_ar
map_severity_class = (
    df.groupby('severity_level')['classification_ar']
    .apply(lambda x: sorted(set(x)))
    .reset_index()
)
map_severity_class['classification_ar_list'] = map_severity_class['classification_ar'].apply(
    lambda lst: ", ".join(str(i) for i in lst)
)
map_severity_class['count'] = map_severity_class['classification_ar'].apply(len)
map_severity_class = map_severity_class[['severity_level', 'classification_ar_list', 'count']]

# --- Stage → classification_ar
map_stage_class = (
    df.groupby('stage')['classification_ar']
    .apply(lambda x: sorted(set(x)))
    .reset_index()
)
map_stage_class['classification_ar_list'] = map_stage_class['classification_ar'].apply(
    lambda lst: ", ".join(str(i) for i in lst)
)
map_stage_class['count'] = map_stage_class['classification_ar'].apply(len)
map_stage_class = map_stage_class[['stage', 'classification_ar_list', 'count']]

# --- Harm Level → classification_ar
map_harm_class = (
    df.groupby('harm_level')['classification_ar']
    .apply(lambda x: sorted(set(x)))
    .reset_index()
)
map_harm_class['classification_ar_list'] = map_harm_class['classification_ar'].apply(
    lambda lst: ", ".join(str(i) for i in lst)
)
map_harm_class['count'] = map_harm_class['classification_ar'].apply(len)
map_harm_class = map_harm_class[['harm_level', 'classification_ar_list', 'count']]

# ---------------------------------------------------------
# 5. WRITE EVERYTHING TO ONE EXCEL SHEET
# ---------------------------------------------------------
output_path = "mapping_hierarchy.xlsx"
sheet_name = "Mappings"

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    start_row = 0

    # Table 1 — Domain → Category
    map_domain_category.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)
    start_row += len(map_domain_category) + 3

    # Table 2 — Category → Sub-Category
    map_category_subcategory.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)
    start_row += len(map_category_subcategory) + 3

    # Table 3 — Sub-Category → Classification_AR
    map_subcategory_class.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)
    start_row += len(map_subcategory_class) + 3

    # Table 4 — Severity Level → Classification_AR (inverse)
    map_severity_class.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)
    start_row += len(map_severity_class) + 3

    # Table 5 — Stage → Classification_AR (inverse)
    map_stage_class.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)
    start_row += len(map_stage_class) + 3

    # Table 6 — Harm Level → Classification_AR (inverse)
    map_harm_class.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)

print("\n✔ Excel file created successfully:", output_path)
