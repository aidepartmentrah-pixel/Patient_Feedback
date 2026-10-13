-- ================================================================
-- MIGRATION: 003_app_user_password_export
-- ================================================================
-- Purpose: Safety net for databases installed before
--          dbo.APP_UserPasswordExport existed. That table holds the
--          reversible (Fernet-encrypted) copy of each user's password
--          that Users Management > Export Credentials prints.
--          Identical to the definition in install/002_create_schema.sql
--          and install/004_create_constraints.sql; a no-op on any
--          database that already has it.
--
-- Safe to re-run. Run it against the application database (the
-- one the backend's DB_DATABASE points at).
--
-- Date: 2026-09-23
-- ================================================================

IF OBJECT_ID('dbo.APP_UserPasswordExport', 'U') IS NULL
CREATE TABLE [dbo].[APP_UserPasswordExport] (
    [UserID] int NOT NULL,
    [EncryptedPassword] nvarchar(500) NOT NULL,
    [UpdatedAt] datetime2 NOT NULL DEFAULT (sysutcdatetime()),
    [UpdatedByUserID] int NULL,
    CONSTRAINT [PK_APP_UserPasswordExport] PRIMARY KEY CLUSTERED ([UserID])
);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_UserPasswordExport_User')
ALTER TABLE [dbo].[APP_UserPasswordExport] ADD CONSTRAINT [FK_APP_UserPasswordExport_User] FOREIGN KEY ([UserID]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM dbo.SchemaMigrationHistory WHERE MigrationName = '003_app_user_password_export')
    INSERT INTO dbo.SchemaMigrationHistory (MigrationName, Checksum, AppliedAt, AppliedBy, ApplicationVersion, Success)
    VALUES ('003_app_user_password_export', NULL, SYSUTCDATETIME(), SUSER_SNAME(), '1.2.1', 1);
GO
