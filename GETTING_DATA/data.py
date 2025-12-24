from backend.core.database import get_connection

cursor = get_connection().cursor()

# --------------------------------------------------
# Get all tables
# --------------------------------------------------
tables = cursor.execute("""
SELECT TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'
ORDER BY TABLE_NAME
""").fetchall()

output = []

for (table_name,) in tables:

    output.append("=" * 60)
    output.append(f"TABLE: {table_name}")
    output.append("=" * 60)
    output.append("COLUMNS:")

    # -----------------------------
    # Columns
    # -----------------------------
    columns = cursor.execute("""
    SELECT 
        COLUMN_NAME,
        DATA_TYPE,
        CHARACTER_MAXIMUM_LENGTH,
        IS_NULLABLE,
        COLUMN_DEFAULT
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = ?
    ORDER BY ORDINAL_POSITION
    """, table_name).fetchall()

    for col in columns:
        name, dtype, length, nullable, default = col
        line = f"  - {name}: {dtype}"
        if length:
            line += f"({length})"
        if nullable == "NO":
            line += " (NOT NULL)"
        if default:
            line += f" (Default={default})"
        output.append(line)

    # -----------------------------
    # Primary Key
    # -----------------------------
    pk = cursor.execute("""
    SELECT k.COLUMN_NAME
    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS t
    JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE k
      ON t.CONSTRAINT_NAME = k.CONSTRAINT_NAME
    WHERE t.TABLE_NAME = ?
      AND t.CONSTRAINT_TYPE = 'PRIMARY KEY'
    """, table_name).fetchall()

    output.append("\nPRIMARY KEY:")
    if pk:
        for (col,) in pk:
            output.append(f"  - {col}")
    else:
        output.append("  None")

    # -----------------------------
    # Foreign Keys
    # -----------------------------
    fks = cursor.execute("""
    SELECT 
        k.COLUMN_NAME,
        ccu.TABLE_NAME AS REF_TABLE,
        ccu.COLUMN_NAME AS REF_COLUMN
    FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
    JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE k
        ON rc.CONSTRAINT_NAME = k.CONSTRAINT_NAME
    JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu
        ON rc.UNIQUE_CONSTRAINT_NAME = ccu.CONSTRAINT_NAME
    WHERE k.TABLE_NAME = ?
    """, table_name).fetchall()

    output.append("\nFOREIGN KEYS:")
    if fks:
        for col, ref_table, ref_col in fks:
            output.append(f"  - {col} -> {ref_table}.{ref_col}")
    else:
        output.append("  None")

    output.append("\n")

# --------------------------------------------------
# Write to file
# --------------------------------------------------
with open("database_schema.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Schema exported successfully.")
