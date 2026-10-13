-- ================================================================
-- MIGRATION: 002_show_taken_action_in_review_views
-- ================================================================
-- Purpose: Table View's preset views (migrated from the old HCAT
--          system, see install/015 + seed/extract_custom_views.py)
--          never had the "Taken Action" column ticked, so the
--          Action Given by Section text is invisible in every preset
--          view. Ticks ShowTakenAction on the review-oriented presets.
--
--          Data-only and optional: the same result can be achieved
--          without SQL through Table View > Custom Views > Edit on
--          each view (Text & Responses > Taken Action).
--
-- Safe to re-run. Run it against the application database (the
-- one the backend's DB_DATABASE points at).
--
-- Date: 2026-09-23
-- ================================================================

UPDATE dbo.APP_CUSTOM_VIEWS
SET ShowTakenAction = 1
WHERE ViewName IN (
        N'Section Review',
        N'Department Review',
        N'Administration Review',
        N'Decision & Root Cause'
    )
  AND ShowTakenAction = 0;
GO

IF NOT EXISTS (SELECT 1 FROM dbo.SchemaMigrationHistory WHERE MigrationName = '002_show_taken_action_in_review_views')
    INSERT INTO dbo.SchemaMigrationHistory (MigrationName, Checksum, AppliedAt, AppliedBy, ApplicationVersion, Success)
    VALUES ('002_show_taken_action_in_review_views', NULL, SYSUTCDATETIME(), SUSER_SNAME(), '1.2.1', 1);
GO
