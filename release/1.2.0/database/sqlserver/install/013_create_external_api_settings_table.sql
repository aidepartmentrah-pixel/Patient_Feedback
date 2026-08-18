-- Stores runtime configuration for external REST API integrations
-- (currently: the Hospital Directory API), separate from bootstrap database
-- connection settings in db_settings.json.
--
-- The API key is stored ENCRYPTED (Fernet, via core/settings_encryption.py).
-- The encryption key itself lives only in the SETTINGS_ENCRYPTION_KEY
-- process environment variable -- never in this table, never in this
-- database at all.
--
-- Mirrors backend/sql_scripts/create_external_api_settings_table.sql (kept
-- in place there too -- config_router.py and hospital_directory_client.py
-- both reference that path in their error messages). This copy is what
-- actually runs automatically via install_database.py's
-- sorted(INSTALL_DIR.glob("0*.sql")) discovery; the original was never wired
-- into any install pipeline, which is why APP_ExternalApiSettings had no
-- 'hospital_directory' row and every Hospital Directory API config endpoint
-- 500'd on a fresh install.

IF OBJECT_ID('dbo.APP_ExternalApiSettings', 'U') IS NULL
CREATE TABLE [dbo].[APP_ExternalApiSettings] (
    [IntegrationName]   nvarchar(50) NOT NULL PRIMARY KEY,
    [BaseUrl]           nvarchar(500) NULL,
    [ApiKeyEncrypted]   nvarchar(max) NULL,
    [TimeoutSeconds]    int NOT NULL DEFAULT (10),
    [VerifyTls]         bit NOT NULL DEFAULT (1),
    [Enabled]           bit NOT NULL DEFAULT (0),
    [LastTestStatus]    nvarchar(20) NULL,   -- 'SUCCESS' | 'FAILED'
    [LastTestMessage]   nvarchar(1000) NULL,
    [LastTestAt]        datetime NULL,
    [UpdatedAt]         datetime NOT NULL DEFAULT (getdate()),
    [UpdatedByUserID]   int NULL
);
GO

-- Seed the single row this app expects to always exist, so application code
-- never has to handle an "INSERT if missing" case at runtime.
IF NOT EXISTS (SELECT 1 FROM dbo.APP_ExternalApiSettings WHERE IntegrationName = 'hospital_directory')
INSERT INTO dbo.APP_ExternalApiSettings (
    IntegrationName, BaseUrl, ApiKeyEncrypted, TimeoutSeconds, VerifyTls, Enabled,
    LastTestStatus, LastTestMessage, LastTestAt, UpdatedAt, UpdatedByUserID
)
VALUES (
    'hospital_directory', NULL, NULL, 10, 1, 0,
    NULL, NULL, NULL, GETDATE(), NULL
);
GO
