-- =====================================================
-- System Settings Table Creation Script
-- =====================================================
-- Purpose: Store global system configuration settings
-- that can be modified through the Settings page
-- =====================================================

-- Create the System Settings table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'APP_SystemSettings')
BEGIN
    CREATE TABLE APP_SystemSettings (
        SettingKey NVARCHAR(100) PRIMARY KEY,
        SettingValue NVARCHAR(MAX) NOT NULL,
        SettingType NVARCHAR(20) NOT NULL,  -- 'int', 'bool', 'string', 'json'
        Description NVARCHAR(500) NULL,
        UpdatedAt DATETIME NOT NULL DEFAULT GETDATE(),
        UpdatedByUserID INT NULL
    );
    
    PRINT 'Table APP_SystemSettings created successfully.';
END
ELSE
BEGIN
    PRINT 'Table APP_SystemSettings already exists.';
END
GO

-- Insert initial seed data
IF NOT EXISTS (SELECT * FROM APP_SystemSettings WHERE SettingKey = 'ComplaintDelayDays')
BEGIN
    INSERT INTO APP_SystemSettings (
        SettingKey,
        SettingValue,
        SettingType,
        Description,
        UpdatedAt,
        UpdatedByUserID
    )
    VALUES (
        'ComplaintDelayDays',
        '14',
        'int',
        N'After this many days, a complaint is considered delayed',
        GETDATE(),
        NULL
    );
    
    PRINT 'Initial seed data inserted: ComplaintDelayDays = 14';
END
ELSE
BEGIN
    PRINT 'Seed data already exists for ComplaintDelayDays';
END
GO

-- Verify the table and data
SELECT * FROM APP_SystemSettings;
GO
