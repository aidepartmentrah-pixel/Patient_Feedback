-- ================================================================
-- MIGRATION: 001_category_subcategory_ar_names
-- ================================================================
-- Purpose: Extend the Settings > Class Management tab to manage
--          Category and Sub-Category the same way Classification is
--          already managed (add, rename AR/EN, freeze/unfreeze).
--          dbo.APP_LOOKUP_CATEGORY and dbo.APP_LOOKUP_SUBCATEGORY
--          currently only store an English name and have no
--          active/inactive flag -- both are needed for that pattern.
--          Mirrors backend/database_migrations/
--          phase_category_subcategory_ar_names.sql for existing,
--          already-installed databases.
--
-- Backfill: the new Arabic-name columns start NULL on existing rows
--           -- there is no safe automatic English->Arabic
--           translation, so existing categories/subcategories show a
--           blank Arabic name until filled in through the new
--           Settings screen. IsActive defaults every existing row to
--           active (1), matching current behavior.
--
-- Date: 2026-09-09
-- ================================================================

USE IncidentManager;
GO

IF NOT EXISTS (
    SELECT * FROM sys.columns
    WHERE object_id = OBJECT_ID('dbo.APP_LOOKUP_CATEGORY')
    AND name = 'CategoryNameAr'
)
    ALTER TABLE dbo.APP_LOOKUP_CATEGORY ADD CategoryNameAr NVARCHAR(100) NULL;
GO

IF NOT EXISTS (
    SELECT * FROM sys.columns
    WHERE object_id = OBJECT_ID('dbo.APP_LOOKUP_CATEGORY')
    AND name = 'IsActive'
)
    ALTER TABLE dbo.APP_LOOKUP_CATEGORY ADD IsActive BIT NOT NULL DEFAULT 1;
GO

IF NOT EXISTS (
    SELECT * FROM sys.columns
    WHERE object_id = OBJECT_ID('dbo.APP_LOOKUP_SUBCATEGORY')
    AND name = 'SubCategoryNameAr'
)
    ALTER TABLE dbo.APP_LOOKUP_SUBCATEGORY ADD SubCategoryNameAr NVARCHAR(150) NULL;
GO

IF NOT EXISTS (
    SELECT * FROM sys.columns
    WHERE object_id = OBJECT_ID('dbo.APP_LOOKUP_SUBCATEGORY')
    AND name = 'IsActive'
)
    ALTER TABLE dbo.APP_LOOKUP_SUBCATEGORY ADD IsActive BIT NOT NULL DEFAULT 1;
GO

IF NOT EXISTS (SELECT 1 FROM dbo.SchemaMigrationHistory WHERE MigrationName = '001_category_subcategory_ar_names')
    INSERT INTO dbo.SchemaMigrationHistory (MigrationName, Checksum, AppliedAt, AppliedBy, ApplicationVersion, Success)
    VALUES ('001_category_subcategory_ar_names', NULL, SYSUTCDATETIME(), SUSER_SNAME(), '1.2.1', 1);
GO
