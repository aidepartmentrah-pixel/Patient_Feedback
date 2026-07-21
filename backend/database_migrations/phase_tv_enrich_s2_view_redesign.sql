-- ============================================================
-- TV-ENRICH-S2: Custom View Redesign — new Show* columns
-- Date: 2026-07-20
-- Safe to run multiple times (all checks are idempotent)
-- No data deleted. No existing columns altered.
-- ============================================================
--
-- Adds:
--   ShowRecordType — fixes a pre-existing gap: DataTable.js has
--     referenced this flag since it was written, but it never
--     existed in APP_CUSTOM_VIEWS, so the "Type" (Complaint/Notice)
--     column could never be turned on in any custom view.
--   ShowSectionEntry / ShowSectionDeadline
--   ShowDepartmentEntry / ShowDepartmentDeadline
--   ShowAdministrationEntry / ShowAdministrationDeadline
--     — surface APP_AdministrativeSubcase's existing
--     Section/Department/AdministrationEntryTimestamp and
--     Section/Department/AdministrationDeadlineAt columns, which
--     already exist in the DB but were never exposed to Table View.
-- ============================================================

USE IncidentManager;
GO

SET QUOTED_IDENTIFIER ON;
SET ANSI_NULLS ON;
GO

PRINT '============================================================';
PRINT 'TV-ENRICH-S2: Custom View Redesign — APP_CUSTOM_VIEWS';
PRINT '============================================================';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_CUSTOM_VIEWS') AND name = 'ShowRecordType')
BEGIN
    ALTER TABLE dbo.APP_CUSTOM_VIEWS
    ADD ShowRecordType BIT NOT NULL CONSTRAINT DF_CustomViews_ShowRecordType DEFAULT 0;
    PRINT '  ShowRecordType added';
END
ELSE
    PRINT '  ShowRecordType already exists. Skipping.';
GO

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_CUSTOM_VIEWS') AND name = 'ShowSectionEntry')
BEGIN
    ALTER TABLE dbo.APP_CUSTOM_VIEWS
    ADD ShowSectionEntry BIT NOT NULL CONSTRAINT DF_CustomViews_ShowSectionEntry DEFAULT 0;
    PRINT '  ShowSectionEntry added';
END
ELSE
    PRINT '  ShowSectionEntry already exists. Skipping.';
GO

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_CUSTOM_VIEWS') AND name = 'ShowSectionDeadline')
BEGIN
    ALTER TABLE dbo.APP_CUSTOM_VIEWS
    ADD ShowSectionDeadline BIT NOT NULL CONSTRAINT DF_CustomViews_ShowSectionDeadline DEFAULT 0;
    PRINT '  ShowSectionDeadline added';
END
ELSE
    PRINT '  ShowSectionDeadline already exists. Skipping.';
GO

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_CUSTOM_VIEWS') AND name = 'ShowDepartmentEntry')
BEGIN
    ALTER TABLE dbo.APP_CUSTOM_VIEWS
    ADD ShowDepartmentEntry BIT NOT NULL CONSTRAINT DF_CustomViews_ShowDepartmentEntry DEFAULT 0;
    PRINT '  ShowDepartmentEntry added';
END
ELSE
    PRINT '  ShowDepartmentEntry already exists. Skipping.';
GO

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_CUSTOM_VIEWS') AND name = 'ShowDepartmentDeadline')
BEGIN
    ALTER TABLE dbo.APP_CUSTOM_VIEWS
    ADD ShowDepartmentDeadline BIT NOT NULL CONSTRAINT DF_CustomViews_ShowDepartmentDeadline DEFAULT 0;
    PRINT '  ShowDepartmentDeadline added';
END
ELSE
    PRINT '  ShowDepartmentDeadline already exists. Skipping.';
GO

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_CUSTOM_VIEWS') AND name = 'ShowAdministrationEntry')
BEGIN
    ALTER TABLE dbo.APP_CUSTOM_VIEWS
    ADD ShowAdministrationEntry BIT NOT NULL CONSTRAINT DF_CustomViews_ShowAdministrationEntry DEFAULT 0;
    PRINT '  ShowAdministrationEntry added';
END
ELSE
    PRINT '  ShowAdministrationEntry already exists. Skipping.';
GO

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_CUSTOM_VIEWS') AND name = 'ShowAdministrationDeadline')
BEGIN
    ALTER TABLE dbo.APP_CUSTOM_VIEWS
    ADD ShowAdministrationDeadline BIT NOT NULL CONSTRAINT DF_CustomViews_ShowAdministrationDeadline DEFAULT 0;
    PRINT '  ShowAdministrationDeadline added';
END
ELSE
    PRINT '  ShowAdministrationDeadline already exists. Skipping.';
GO

SELECT name, TYPE_NAME(user_type_id) AS data_type, is_nullable
FROM sys.columns
WHERE object_id = OBJECT_ID('dbo.APP_CUSTOM_VIEWS')
  AND name IN ('ShowRecordType', 'ShowSectionEntry', 'ShowSectionDeadline', 'ShowDepartmentEntry', 'ShowDepartmentDeadline', 'ShowAdministrationEntry', 'ShowAdministrationDeadline')
ORDER BY column_id;

PRINT '';
PRINT '============================================================';
PRINT 'TV-ENRICH-S2: Migration complete.';
PRINT '============================================================';
GO

-- ============================================================
-- ROLLBACK SCRIPT (for reference only — do not execute)
-- ============================================================
/*
ALTER TABLE dbo.APP_CUSTOM_VIEWS DROP COLUMN ShowRecordType;
ALTER TABLE dbo.APP_CUSTOM_VIEWS DROP COLUMN ShowSectionEntry;
ALTER TABLE dbo.APP_CUSTOM_VIEWS DROP COLUMN ShowSectionDeadline;
ALTER TABLE dbo.APP_CUSTOM_VIEWS DROP COLUMN ShowDepartmentEntry;
ALTER TABLE dbo.APP_CUSTOM_VIEWS DROP COLUMN ShowDepartmentDeadline;
ALTER TABLE dbo.APP_CUSTOM_VIEWS DROP COLUMN ShowAdministrationEntry;
ALTER TABLE dbo.APP_CUSTOM_VIEWS DROP COLUMN ShowAdministrationDeadline;
*/
