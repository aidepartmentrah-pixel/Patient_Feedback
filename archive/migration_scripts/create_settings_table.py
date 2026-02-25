"""
Create APP_SystemSettings table for storing configuration parameters.
"""

import pyodbc

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=SOCIALMEDIA;'
    'DATABASE=IncidentManager;'
    'Trusted_Connection=yes;'
    'TrustServerCertificate=yes;'
)

cursor = conn.cursor()

print("Creating APP_SystemSettings table...")

# Create table
create_table_sql = """
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'APP_SystemSettings')
BEGIN
    CREATE TABLE dbo.APP_SystemSettings (
        SettingID INT IDENTITY(1,1) PRIMARY KEY,
        SettingKey NVARCHAR(100) NOT NULL UNIQUE,
        SettingValue NVARCHAR(MAX) NULL,
        SettingLabel NVARCHAR(200) NULL,
        SettingLabelAr NVARCHAR(200) NULL,
        SettingType NVARCHAR(50) NULL,  -- 'text', 'number', 'boolean', 'json'
        Description NVARCHAR(500) NULL,
        DescriptionAr NVARCHAR(500) NULL,
        IsActive BIT DEFAULT 1,
        CreatedAt DATETIME DEFAULT GETDATE(),
        UpdatedAt DATETIME DEFAULT GETDATE(),
        UpdatedBy INT NULL
    )
    
    PRINT 'Table APP_SystemSettings created successfully'
END
ELSE
BEGIN
    PRINT 'Table APP_SystemSettings already exists'
END
"""

cursor.execute(create_table_sql)
conn.commit()

print("\n✅ Table created successfully!")

# Insert some default settings
print("\nInserting default settings...")

default_settings = [
    ('max_file_size_mb', '10', 'Maximum File Size (MB)', 'الحد الأقصى لحجم الملف', 'number', 'Maximum upload file size in megabytes', 'الحد الأقصى لحجم رفع الملف بالميغابايت'),
    ('session_timeout_minutes', '30', 'Session Timeout (Minutes)', 'مهلة الجلسة (دقائق)', 'number', 'User session timeout in minutes', 'مهلة جلسة المستخدم بالدقائق'),
    ('enable_notifications', 'true', 'Enable Notifications', 'تفعيل الإشعارات', 'boolean', 'Enable system notifications', 'تفعيل إشعارات النظام'),
    ('default_language', 'ar', 'Default Language', 'اللغة الافتراضية', 'text', 'Default system language (ar/en)', 'اللغة الافتراضية للنظام'),
]

for key, value, label, label_ar, setting_type, desc, desc_ar in default_settings:
    try:
        cursor.execute("""
            IF NOT EXISTS (SELECT 1 FROM dbo.APP_SystemSettings WHERE SettingKey = ?)
            BEGIN
                INSERT INTO dbo.APP_SystemSettings 
                (SettingKey, SettingValue, SettingLabel, SettingLabelAr, SettingType, Description, DescriptionAr)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            END
        """, key, key, value, label, label_ar, setting_type, desc, desc_ar)
        conn.commit()
        print(f"  ✓ {label} ({key})")
    except Exception as e:
        print(f"  ✗ Failed to insert {key}: {e}")

print("\n✅ Default settings inserted!")

# Show all settings
print("\n=== Current Settings ===")
cursor.execute("SELECT SettingKey, SettingValue, SettingLabel, SettingType FROM dbo.APP_SystemSettings WHERE IsActive = 1")
for row in cursor.fetchall():
    print(f"  {row[0]:30} = {row[1]:15} ({row[3]})")

conn.close()
print("\n✅ Done!")
