-- ============================================================
-- INCDATE-S5: Table View — Incident Date / Publication Date columns
-- Date: 2026-07-16
-- Safe to run multiple times (all checks are idempotent)
-- No data deleted. No existing columns altered.
-- ============================================================
--
-- Adds two new Show* flags to APP_CUSTOM_VIEWS so users can include
-- Incident Date and Publication Date as columns in their custom Table
-- View layouts, alongside the existing ShowFeedbackRecievedDate /
-- ShowCreatedAt / ShowLastEdited flags.
-- ============================================================

USE IncidentManager;
GO

SET QUOTED_IDENTIFIER ON;
SET ANSI_NULLS ON;
GO

PRINT '============================================================';
PRINT 'INCDATE-S5: Table View columns — APP_CUSTOM_VIEWS';
PRINT '============================================================';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_CUSTOM_VIEWS') AND name = 'ShowIncidentDate')
BEGIN
    ALTER TABLE dbo.APP_CUSTOM_VIEWS
    ADD ShowIncidentDate BIT NOT NULL CONSTRAINT DF_CustomViews_ShowIncidentDate DEFAULT 0;
    PRINT '  ShowIncidentDate added';
END
ELSE
    PRINT '  ShowIncidentDate already exists. Skipping.';
GO

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_CUSTOM_VIEWS') AND name = 'ShowPublicationDate')
BEGIN
    ALTER TABLE dbo.APP_CUSTOM_VIEWS
    ADD ShowPublicationDate BIT NOT NULL CONSTRAINT DF_CustomViews_ShowPublicationDate DEFAULT 0;
    PRINT '  ShowPublicationDate added';
END
ELSE
    PRINT '  ShowPublicationDate already exists. Skipping.';
GO

SELECT name, TYPE_NAME(user_type_id) AS data_type, is_nullable
FROM sys.columns
WHERE object_id = OBJECT_ID('dbo.APP_CUSTOM_VIEWS')
  AND name IN ('ShowIncidentDate', 'ShowPublicationDate', 'ShowFeedbackRecievedDate', 'ShowLastEdited')
ORDER BY column_id;

PRINT '';
PRINT '============================================================';
PRINT 'INCDATE-S5: Migration complete.';
PRINT '============================================================';
GO

-- ============================================================
-- ROLLBACK SCRIPT (for reference only — do not execute)
-- ============================================================
/*
ALTER TABLE dbo.APP_CUSTOM_VIEWS DROP COLUMN ShowIncidentDate;
ALTER TABLE dbo.APP_CUSTOM_VIEWS DROP COLUMN ShowPublicationDate;
*/
