-- =====================================================
-- External API Settings Table Creation Script
-- =====================================================
-- Purpose: Store runtime configuration for external REST API
-- integrations (currently: the Hospital Directory API), separate
-- from bootstrap database connection settings in db_settings.json.
--
-- The API key is stored ENCRYPTED (Fernet, via core/settings_encryption.py).
-- The encryption key itself lives only in the SETTINGS_ENCRYPTION_KEY
-- process environment variable — never in this table, never in this
-- database at all.
-- =====================================================

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'APP_ExternalApiSettings')
BEGIN
    CREATE TABLE APP_ExternalApiSettings (
        IntegrationName NVARCHAR(50) PRIMARY KEY,
        BaseUrl NVARCHAR(500) NULL,
        ApiKeyEncrypted NVARCHAR(MAX) NULL,
        TimeoutSeconds INT NOT NULL DEFAULT 10,
        VerifyTls BIT NOT NULL DEFAULT 1,
        Enabled BIT NOT NULL DEFAULT 0,
        LastTestStatus NVARCHAR(20) NULL,       -- 'SUCCESS' | 'FAILED'
        LastTestMessage NVARCHAR(1000) NULL,
        LastTestAt DATETIME NULL,
        UpdatedAt DATETIME NOT NULL DEFAULT GETDATE(),
        UpdatedByUserID INT NULL
    );

    PRINT 'Table APP_ExternalApiSettings created successfully.';
END
ELSE
BEGIN
    PRINT 'Table APP_ExternalApiSettings already exists.';
END
GO

-- Seed the single row this app expects to always exist, so application code
-- never has to handle an "INSERT if missing" case at runtime.
IF NOT EXISTS (SELECT * FROM APP_ExternalApiSettings WHERE IntegrationName = 'hospital_directory')
BEGIN
    INSERT INTO APP_ExternalApiSettings (
        IntegrationName, BaseUrl, ApiKeyEncrypted, TimeoutSeconds, VerifyTls, Enabled,
        LastTestStatus, LastTestMessage, LastTestAt, UpdatedAt, UpdatedByUserID
    )
    VALUES (
        'hospital_directory', NULL, NULL, 10, 1, 0,
        NULL, NULL, NULL, GETDATE(), NULL
    );

    PRINT 'Seed row inserted: IntegrationName = hospital_directory';
END
ELSE
BEGIN
    PRINT 'Seed row already exists for hospital_directory';
END
GO

-- Verify the table and data
SELECT IntegrationName, BaseUrl, TimeoutSeconds, VerifyTls, Enabled, LastTestStatus, LastTestAt
FROM APP_ExternalApiSettings;
GO
