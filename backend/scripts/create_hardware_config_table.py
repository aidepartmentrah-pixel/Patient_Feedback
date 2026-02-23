"""
Create APP_HardwareConfig table for deployment configuration.
Run this once to create the table.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import get_connection

def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    print('Creating APP_HardwareConfig table...')
    
    # Check if table exists
    cursor.execute("""
        SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_NAME = 'APP_HardwareConfig'
    """)
    if cursor.fetchone()[0] > 0:
        print('  Table already exists, dropping...')
        cursor.execute("DROP TABLE APP_HardwareConfig")
    
    # Create table
    cursor.execute("""
        CREATE TABLE APP_HardwareConfig (
            ConfigID INT IDENTITY(1,1) PRIMARY KEY,
            ConfigKey NVARCHAR(100) NOT NULL UNIQUE,
            ConfigValue NVARCHAR(500) NULL,
            ConfigType NVARCHAR(50) NOT NULL DEFAULT 'string',
            ConfigGroup NVARCHAR(50) NOT NULL,
            DisplayName NVARCHAR(200) NOT NULL,
            DisplayNameAr NVARCHAR(200) NULL,
            Description NVARCHAR(500) NULL,
            IsEncrypted BIT NOT NULL DEFAULT 0,
            IsEditable BIT NOT NULL DEFAULT 1,
            DisplayOrder INT NOT NULL DEFAULT 0,
            UpdatedAt DATETIME2 DEFAULT GETDATE(),
            UpdatedByUserID INT NULL
        )
    """)
    print('  Table created!')
    
    # Insert default configuration values
    print('Inserting default configuration...')
    
    default_configs = [
        # Database Configuration
        ('db_server', 'SOCIALMEDIA', 'string', 'database', 'Database Server', 'خادم قاعدة البيانات', 'SQL Server hostname or IP'),
        ('db_database', 'IncidentManager', 'string', 'database', 'Database Name', 'اسم قاعدة البيانات', 'Database name'),
        ('db_driver', 'ODBC Driver 17 for SQL Server', 'string', 'database', 'ODBC Driver', 'برنامج تشغيل ODBC', 'SQL Server ODBC driver'),
        ('db_use_windows_auth', 'true', 'bool', 'database', 'Use Windows Auth', 'مصادقة ويندوز', 'Use Windows domain authentication'),
        ('db_username', '', 'string', 'database', 'DB Username', 'اسم مستخدم قاعدة البيانات', 'SQL Server username (if not Windows auth)'),
        ('db_password', '', 'password', 'database', 'DB Password', 'كلمة مرور قاعدة البيانات', 'SQL Server password (encrypted)'),
        
        # External Views Configuration
        ('view_hr_employees', 'VW_HrEmployeeProfileView', 'string', 'views', 'HR Employees View', 'عرض موظفي الموارد البشرية', 'HR system employee view name'),
        ('view_patient_admission', 'VW_PatientAdmission', 'string', 'views', 'Patient Admission View', 'عرض قبول المرضى', 'HIS patient admission view name'),
        ('view_doctors', 'VW_Doctors', 'string', 'views', 'Doctors View', 'عرض الأطباء', 'HIS doctors view name'),
        
        # Network Configuration
        ('backend_api_url', 'http://localhost:8000', 'string', 'network', 'Backend API URL', 'رابط واجهة برمجة التطبيقات', 'Backend API base URL'),
        ('backend_host', '127.0.0.1', 'string', 'network', 'Backend Host', 'مضيف الخادم الخلفي', 'Backend server host (0.0.0.0 for network)'),
        ('backend_port', '8000', 'int', 'network', 'Backend Port', 'منفذ الخادم الخلفي', 'Backend server port'),
        ('cors_origins', 'http://localhost:3000,http://localhost:5173', 'string', 'network', 'CORS Origins', 'أصول CORS', 'Comma-separated list of allowed origins'),
        
        # SMTP Email Configuration
        ('smtp_enabled', 'false', 'bool', 'email', 'Enable Email', 'تفعيل البريد الإلكتروني', 'Enable email notifications'),
        ('smtp_host', 'smtp.hospital.local', 'string', 'email', 'SMTP Server', 'خادم SMTP', 'SMTP server hostname or IP'),
        ('smtp_port', '25', 'int', 'email', 'SMTP Port', 'منفذ SMTP', 'SMTP server port (25, 587, or 465)'),
        ('smtp_use_tls', 'false', 'bool', 'email', 'Use TLS', 'استخدام TLS', 'Enable TLS encryption'),
        ('smtp_use_ssl', 'false', 'bool', 'email', 'Use SSL', 'استخدام SSL', 'Enable SSL encryption (port 465)'),
        ('smtp_username', '', 'string', 'email', 'SMTP Username', 'اسم مستخدم SMTP', 'SMTP authentication username'),
        ('smtp_password', '', 'password', 'email', 'SMTP Password', 'كلمة مرور SMTP', 'SMTP authentication password (encrypted)'),
        ('sender_email', 'complaint-system@hospital.local', 'string', 'email', 'Sender Email', 'بريد المرسل', 'Email address for sending notifications'),
        ('sender_name', 'Hospital Complaint System', 'string', 'email', 'Sender Name', 'اسم المرسل', 'Display name for sent emails'),
        
        # Deployment Mode
        ('deployment_mode', 'offline', 'string', 'system', 'Deployment Mode', 'وضع النشر', 'offline or online'),
    ]
    
    for config in default_configs:
        cursor.execute("""
            INSERT INTO APP_HardwareConfig 
            (ConfigKey, ConfigValue, ConfigType, ConfigGroup, DisplayName, DisplayNameAr, Description, IsEncrypted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (config[0], config[1], config[2], config[3], config[4], config[5], config[6], 
              1 if config[2] == 'password' else 0))
        print(f'  + {config[0]}')
    
    conn.commit()
    print()
    print('Done! Hardware configuration table created with defaults.')
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM APP_HardwareConfig")
    count = cursor.fetchone()[0]
    print(f'Total configurations: {count}')
    
    cursor.close()
    conn.close()

if __name__ == '__main__':
    create_table()
