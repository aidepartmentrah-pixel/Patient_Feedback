-- ================================================================
-- PHASE 0: Database Schema Changes - Add RequiresExplanation
-- ================================================================
-- Purpose: Add RequiresExplanation column to APP_IncidentCase table
-- Date: 2026-01-19
-- 
-- This migration supports the new explanation workflow where:
-- - Red Flag/Never Event always requires explanation
-- - Ordinary complaints can optionally require explanation based on policy
-- ================================================================

USE IncidentManager;
GO

-- ================================================================
-- STEP 1: Add RequiresExplanation column
-- ================================================================
PRINT 'Step 1: Adding RequiresExplanation column to APP_IncidentCase...';

-- Check if column already exists
IF NOT EXISTS (
    SELECT 1 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA = 'dbo' 
      AND TABLE_NAME = 'APP_IncidentCase' 
      AND COLUMN_NAME = 'RequiresExplanation'
)
BEGIN
    ALTER TABLE dbo.APP_IncidentCase
    ADD RequiresExplanation BIT NOT NULL DEFAULT 0;
    
    PRINT '✓ RequiresExplanation column added successfully';
END
ELSE
BEGIN
    PRINT '⚠ RequiresExplanation column already exists, skipping...';
END
GO

-- ================================================================
-- STEP 2: Set all existing records to RequiresExplanation = 0
-- ================================================================
PRINT 'Step 2: Setting RequiresExplanation = 0 for all existing records...';

UPDATE dbo.APP_IncidentCase
SET RequiresExplanation = 0
WHERE RequiresExplanation IS NULL OR RequiresExplanation = 1;

DECLARE @UpdatedRows INT = @@ROWCOUNT;
PRINT '✓ Updated ' + CAST(@UpdatedRows AS NVARCHAR(10)) + ' records';
GO

-- ================================================================
-- STEP 3: Verify TakenAction field capacity
-- ================================================================
PRINT 'Step 3: Verifying TakenAction field capacity...';

SELECT 
    COLUMN_NAME,
    DATA_TYPE,
    CHARACTER_MAXIMUM_LENGTH,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'dbo'
  AND TABLE_NAME = 'APP_IncidentCase'
  AND COLUMN_NAME = 'TakenAction';

PRINT '✓ TakenAction field verification complete';
GO

-- ================================================================
-- STEP 4: Document lookup table IDs
-- ================================================================
PRINT 'Step 4: Documenting CaseStatus lookup values...';

SELECT 
    CaseStatusID,
    Code,
    Name,
    IsFinal,
    IsActive,
    DisplayOrder
FROM dbo.APP_LOOKUP_CASE_STATUS
ORDER BY DisplayOrder;

PRINT '';
PRINT 'Step 4b: Documenting ExplanationStatus lookup values...';

SELECT 
    StatusID,
    StatusName
FROM dbo.APP_LOOKUP_EXPLANATION_STATUS
ORDER BY StatusID;

PRINT '✓ Lookup tables documented';
GO

-- ================================================================
-- STEP 5: Verification Queries
-- ================================================================
PRINT 'Step 5: Running verification queries...';

-- Check schema change
PRINT 'Checking RequiresExplanation column...';
SELECT 
    COUNT(*) as TotalRecords,
    SUM(CASE WHEN RequiresExplanation = 0 THEN 1 ELSE 0 END) as NoExplanationNeeded,
    SUM(CASE WHEN RequiresExplanation = 1 THEN 1 ELSE 0 END) as ExplanationRequired
FROM dbo.APP_IncidentCase;

-- Check TakenAction usage
PRINT 'Checking TakenAction field usage...';
SELECT 
    COUNT(*) as TotalRecords,
    SUM(CASE WHEN TakenAction IS NULL THEN 1 ELSE 0 END) as NullTakenAction,
    SUM(CASE WHEN TakenAction IS NOT NULL AND TakenAction != '' THEN 1 ELSE 0 END) as HasTakenAction
FROM dbo.APP_IncidentCase;

PRINT '✓ All verification queries complete';
GO

-- ================================================================
-- ROLLBACK SCRIPT (for reference only - do not execute)
-- ================================================================
/*
-- To rollback this migration:
ALTER TABLE dbo.APP_IncidentCase
DROP COLUMN RequiresExplanation;
*/

PRINT '================================================================';
PRINT 'PHASE 0 MIGRATION COMPLETED SUCCESSFULLY';
PRINT '================================================================';
GO
