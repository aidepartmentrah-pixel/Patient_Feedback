-- ============================================================
-- TV-ENRICH-S1: Table View Data Enrichment — new Show* columns
-- Date: 2026-07-20
-- Safe to run multiple times (all checks are idempotent)
-- No data deleted. No existing columns altered.
-- ============================================================
--
-- Adds 4 new Show* flags to APP_CUSTOM_VIEWS so users can include
-- RCA Replies, Complaint Summary, and Customer Service (Patient
-- Services) Decision + its date as columns in Table View layouts.
-- ============================================================

USE IncidentManager;
GO

SET QUOTED_IDENTIFIER ON;
SET ANSI_NULLS ON;
GO

PRINT '============================================================';
PRINT 'TV-ENRICH-S1: Table View Data Enrichment — APP_CUSTOM_VIEWS';
PRINT '============================================================';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_CUSTOM_VIEWS') AND name = 'ShowRcaReplies')
BEGIN
    ALTER TABLE dbo.APP_CUSTOM_VIEWS
    ADD ShowRcaReplies BIT NOT NULL CONSTRAINT DF_CustomViews_ShowRcaReplies DEFAULT 0;
    PRINT '  ShowRcaReplies added';
END
ELSE
    PRINT '  ShowRcaReplies already exists. Skipping.';
GO

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_CUSTOM_VIEWS') AND name = 'ShowComplaintSummary')
BEGIN
    ALTER TABLE dbo.APP_CUSTOM_VIEWS
    ADD ShowComplaintSummary BIT NOT NULL CONSTRAINT DF_CustomViews_ShowComplaintSummary DEFAULT 0;
    PRINT '  ShowComplaintSummary added';
END
ELSE
    PRINT '  ShowComplaintSummary already exists. Skipping.';
GO

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_CUSTOM_VIEWS') AND name = 'ShowCustomerServiceDecision')
BEGIN
    ALTER TABLE dbo.APP_CUSTOM_VIEWS
    ADD ShowCustomerServiceDecision BIT NOT NULL CONSTRAINT DF_CustomViews_ShowCustomerServiceDecision DEFAULT 0;
    PRINT '  ShowCustomerServiceDecision added';
END
ELSE
    PRINT '  ShowCustomerServiceDecision already exists. Skipping.';
GO

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_CUSTOM_VIEWS') AND name = 'ShowCustomerServiceDecisionDate')
BEGIN
    ALTER TABLE dbo.APP_CUSTOM_VIEWS
    ADD ShowCustomerServiceDecisionDate BIT NOT NULL CONSTRAINT DF_CustomViews_ShowCSDecisionDate DEFAULT 0;
    PRINT '  ShowCustomerServiceDecisionDate added';
END
ELSE
    PRINT '  ShowCustomerServiceDecisionDate already exists. Skipping.';
GO

SELECT name, TYPE_NAME(user_type_id) AS data_type, is_nullable
FROM sys.columns
WHERE object_id = OBJECT_ID('dbo.APP_CUSTOM_VIEWS')
  AND name IN ('ShowRcaReplies', 'ShowComplaintSummary', 'ShowCustomerServiceDecision', 'ShowCustomerServiceDecisionDate')
ORDER BY column_id;

PRINT '';
PRINT '============================================================';
PRINT 'TV-ENRICH-S1: Migration complete.';
PRINT '============================================================';
GO

-- ============================================================
-- ROLLBACK SCRIPT (for reference only — do not execute)
-- ============================================================
/*
ALTER TABLE dbo.APP_CUSTOM_VIEWS DROP COLUMN ShowRcaReplies;
ALTER TABLE dbo.APP_CUSTOM_VIEWS DROP COLUMN ShowComplaintSummary;
ALTER TABLE dbo.APP_CUSTOM_VIEWS DROP COLUMN ShowCustomerServiceDecision;
ALTER TABLE dbo.APP_CUSTOM_VIEWS DROP COLUMN ShowCustomerServiceDecisionDate;
*/
