import pyodbc
import os

# ============================================================
# CONFIGURATION (Edit these for your server)
# ============================================================

SERVER = "SocialMedia"
DATABASE = "master"
USE_SQL_AUTH = False
USERNAME = ""    # not needed for Windows Authentication
PASSWORD = ""    # not needed
OUTPUT_FILE = "schema_report.txt"     # e.g.: MyDatabase


# ============================================================
# BUILD CONNECTION STRING
# ============================================================

if USE_SQL_AUTH:
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        f"UID={USERNAME};"
        f"PWD={PASSWORD};"
    )
else:
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        "Trusted_Connection=yes;"
    )


# ============================================================
# HELPER: Write line to file
# ============================================================

def write_line(file, text=""):
    file.write(text + "\n")


# ============================================================
# MAIN LOGIC
# ============================================================

def extract_schema():
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        # ============================================================
        # 1. Get all tables
        # ============================================================
        cursor.execute("""
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
        """)

        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            write_line(f, f"============================================================")
            write_line(f, f"TABLE: {table}")
            write_line(f, f"============================================================")

            # ------------------------------------------------------------
            # 2. Columns
            # ------------------------------------------------------------
            cursor.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH,
                       IS_NULLABLE, COLUMN_DEFAULT
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = '{table}'
                ORDER BY ORDINAL_POSITION
            """)
            columns = cursor.fetchall()

            # Identity columns
            cursor.execute(f"""
                SELECT c.name
                FROM sys.identity_columns ic
                JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
                JOIN sys.tables t ON ic.object_id = t.object_id
                WHERE t.name = '{table}'
            """)
            identity_cols = {row[0] for row in cursor.fetchall()}

            write_line(f, "COLUMNS:")
            for col in columns:
                name, dtype, length, nullable, default = col

                # Clean type
                if length and length > 0:
                    dtype = f"{dtype}({length})"

                flags = []
                if name in identity_cols:
                    flags.append("Identity")
                if nullable == "NO":
                    flags.append("NOT NULL")
                if default:
                    flags.append(f"Default={default}")

                flags_text = f" ({', '.join(flags)})" if flags else ""

                write_line(f, f"  - {name}: {dtype}{flags_text}")

            write_line(f, "")

            # ------------------------------------------------------------
            # 3. Primary Keys
            # ------------------------------------------------------------
            cursor.execute(f"""
                SELECT c.COLUMN_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE c
                JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS t
                ON c.CONSTRAINT_NAME = t.CONSTRAINT_NAME
                WHERE t.TABLE_NAME = '{table}' AND t.CONSTRAINT_TYPE = 'PRIMARY KEY'
            """)
            pk_columns = [row[0] for row in cursor.fetchall()]

            write_line(f, "PRIMARY KEY:")
            if pk_columns:
                for pk in pk_columns:
                    write_line(f, f"  - {pk}")
            else:
                write_line(f, "  None")
            write_line(f, "")

            # ------------------------------------------------------------
            # 4. Foreign Keys
            # ------------------------------------------------------------
            cursor.execute(f"""
                SELECT 
                    fk.name AS FK_Name,
                    COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS ColumnName,
                    OBJECT_NAME(fkc.referenced_object_id) AS RefTable,
                    COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS RefColumn
                FROM sys.foreign_keys fk
                JOIN sys.foreign_key_columns fkc 
                    ON fk.object_id = fkc.constraint_object_id
                WHERE fk.parent_object_id = OBJECT_ID('{table}')
            """)
            fk_rows = cursor.fetchall()

            write_line(f, "FOREIGN KEYS:")
            if fk_rows:
                for fk in fk_rows:
                    write_line(
                        f,
                        f"  - {fk.ColumnName} -> {fk.RefTable}.{fk.RefColumn}"
                    )
            else:
                write_line(f, "  None")

            write_line(f, "\n\n")

    conn.close()
    print(f"Schema successfully written to {OUTPUT_FILE}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    extract_schema()
