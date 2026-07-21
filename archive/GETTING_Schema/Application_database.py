from backend.core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

# --------------------------------------------------
# Verify Connected Database
# --------------------------------------------------
db_name = cursor.execute("SELECT DB_NAME()").fetchone()[0]
print(f"Connected DB: {db_name}")

# --------------------------------------------------
# Get all user tables with schema (robust method)
# --------------------------------------------------
tables = cursor.execute("""
SELECT 
    s.name AS schema_name,
    t.name AS table_name
FROM sys.tables t
JOIN sys.schemas s ON t.schema_id = s.schema_id
ORDER BY s.name, t.name
""").fetchall()

output = []
output.append(f"DATABASE: {db_name}")
output.append("")

for schema_name, table_name in tables:

    output.append("=" * 60)
    output.append(f"TABLE: {schema_name}.{table_name}")
    output.append("=" * 60)
    output.append("COLUMNS:")

    # -----------------------------
    # Columns (schema-safe)
    # -----------------------------
    columns = cursor.execute("""
    SELECT 
        c.COLUMN_NAME,
        c.DATA_TYPE,
        c.CHARACTER_MAXIMUM_LENGTH,
        c.IS_NULLABLE,
        c.COLUMN_DEFAULT
    FROM INFORMATION_SCHEMA.COLUMNS c
    WHERE c.TABLE_NAME = ?
      AND c.TABLE_SCHEMA = ?
    ORDER BY c.ORDINAL_POSITION
    """, table_name, schema_name).fetchall()

    for col in columns:
        name, dtype, length, nullable, default = col
        line = f"  - {name}: {dtype}"
        if length and length > 0:
            line += f"({length})"
        if nullable == "NO":
            line += " (NOT NULL)"
        if default:
            line += f" (Default={default})"
        output.append(line)

    # -----------------------------
    # Primary Key (schema-safe)
    # -----------------------------
    pk = cursor.execute("""
    SELECT k.COLUMN_NAME
    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS t
    JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE k
        ON t.CONSTRAINT_NAME = k.CONSTRAINT_NAME
       AND t.TABLE_SCHEMA = k.TABLE_SCHEMA
    WHERE t.TABLE_NAME = ?
      AND t.TABLE_SCHEMA = ?
      AND t.CONSTRAINT_TYPE = 'PRIMARY KEY'
    ORDER BY k.ORDINAL_POSITION
    """, table_name, schema_name).fetchall()

    output.append("")
    output.append("PRIMARY KEY:")
    if pk:
        for (col,) in pk:
            output.append(f"  - {col}")
    else:
        output.append("  None")

    # -----------------------------
    # Foreign Keys (sys-based — more reliable)
    # -----------------------------
    fks = cursor.execute("""
    SELECT 
        pc.name AS column_name,
        rs.name AS ref_schema,
        rt.name AS ref_table,
        rc.name AS ref_column
    FROM sys.foreign_key_columns fkc
    JOIN sys.tables pt ON fkc.parent_object_id = pt.object_id
    JOIN sys.schemas ps ON pt.schema_id = ps.schema_id
    JOIN sys.columns pc 
        ON pc.object_id = pt.object_id
       AND pc.column_id = fkc.parent_column_id

    JOIN sys.tables rt ON fkc.referenced_object_id = rt.object_id
    JOIN sys.schemas rs ON rt.schema_id = rs.schema_id
    JOIN sys.columns rc
        ON rc.object_id = rt.object_id
       AND rc.column_id = fkc.referenced_column_id

    WHERE pt.name = ?
      AND ps.name = ?
    """, table_name, schema_name).fetchall()

    output.append("")
    output.append("FOREIGN KEYS:")
    if fks:
        for col, ref_schema, ref_table, ref_col in fks:
            output.append(f"  - {col} -> {ref_schema}.{ref_table}.{ref_col}")
    else:
        output.append("  None")

    output.append("")
    output.append("")

# --------------------------------------------------
# Write to file
# --------------------------------------------------
with open("database_schema.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Schema exported successfully.")
print("File: database_schema.txt")
