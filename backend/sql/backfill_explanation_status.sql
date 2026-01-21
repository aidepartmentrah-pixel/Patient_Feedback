-- ============================================================================
-- Data Migration: Backfill NULL ExplanationStatusID
-- ============================================================================
-- This script fixes legacy data where ExplanationStatusID is NULL by setting
-- appropriate values based on business logic:
--
-- 1. Red Flag (2) or Never Event (3) with NULL -> Set to Waiting (1)
-- 2. Ordinary with RequiresExplanation=1 and NULL -> Set to Waiting (1)
-- 3. Ordinary with RequiresExplanation=0 and NULL -> Set to No Explanation Needed (4)
--
-- This ensures historical data matches the FSM logic in insert_service.py
-- ============================================================================

USE IncidentManager;
GO

-- Start transaction for safety
BEGIN TRANSACTION;

DECLARE @UpdateCount INT = 0;

PRINT '============================================================================';
PRINT 'MIGRATION: Backfill NULL ExplanationStatusID';
PRINT '============================================================================';
PRINT '';

-- Show current state
PRINT 'BEFORE MIGRATION:';
PRINT '';
SELECT 
    CASE 
        WHEN ExplanationStatusID IS NULL THEN 'NULL'
        ELSE CAST(ExplanationStatusID AS VARCHAR)
    END AS ExplanationStatus,
    COUNT(*) AS CaseCount
FROM dbo.APP_IncidentCase
GROUP BY ExplanationStatusID
ORDER BY ExplanationStatusID;
PRINT '';

-- Count cases that need updating
DECLARE @NullCount INT;
SELECT @NullCount = COUNT(*)
FROM dbo.APP_IncidentCase
WHERE ExplanationStatusID IS NULL;

PRINT 'Total cases with NULL ExplanationStatusID: ' + CAST(@NullCount AS VARCHAR);
PRINT '';

-- Case 1: Red Flag or Never Event with NULL -> Waiting (1)
UPDATE dbo.APP_IncidentCase
SET ExplanationStatusID = 1  -- Waiting
WHERE ExplanationStatusID IS NULL
AND ClinicalRiskTypeID IN (2, 3);  -- Red Flag or Never Event

SET @UpdateCount = @@ROWCOUNT;
PRINT 'Updated ' + CAST(@UpdateCount AS VARCHAR) + ' Red Flag/Never Event cases to Waiting (1)';

-- Case 2: Ordinary with RequiresExplanation=1 and NULL -> Waiting (1)
UPDATE dbo.APP_IncidentCase
SET ExplanationStatusID = 1  -- Waiting
WHERE ExplanationStatusID IS NULL
AND ClinicalRiskTypeID NOT IN (2, 3)
AND RequiresExplanation = 1;

SET @UpdateCount = @@ROWCOUNT;
PRINT 'Updated ' + CAST(@UpdateCount AS VARCHAR) + ' ordinary cases (RequiresExplanation=1) to Waiting (1)';

-- Case 3: Ordinary with RequiresExplanation=0 and NULL -> No Explanation Needed (4)
UPDATE dbo.APP_IncidentCase
SET ExplanationStatusID = 4  -- No Explanation Needed
WHERE ExplanationStatusID IS NULL
AND ClinicalRiskTypeID NOT IN (2, 3)
AND (RequiresExplanation = 0 OR RequiresExplanation IS NULL);

SET @UpdateCount = @@ROWCOUNT;
PRINT 'Updated ' + CAST(@UpdateCount AS VARCHAR) + ' ordinary cases (RequiresExplanation=0) to No Explanation Needed (4)';

PRINT '';
PRINT 'AFTER MIGRATION:';
PRINT '';
SELECT 
    es.StatusName,
    COUNT(*) AS CaseCount
FROM dbo.APP_IncidentCase ic
LEFT JOIN dbo.APP_LOOKUP_EXPLANATION_STATUS es ON ic.ExplanationStatusID = es.StatusID
GROUP BY es.StatusName
ORDER BY es.StatusName;

-- Check for any remaining NULLs
DECLARE @RemainingNull INT;
SELECT @RemainingNull = COUNT(*)
FROM dbo.APP_IncidentCase
WHERE ExplanationStatusID IS NULL;

PRINT '';
PRINT 'Remaining NULL ExplanationStatusID: ' + CAST(@RemainingNull AS VARCHAR);
PRINT '';

IF @RemainingNull = 0
BEGIN
    PRINT '✅ SUCCESS: All NULL values have been backfilled';
    PRINT '';
    PRINT 'To apply changes, run: COMMIT TRANSACTION';
    PRINT 'To cancel changes, run: ROLLBACK TRANSACTION';
END
ELSE
BEGIN
    PRINT '⚠️ WARNING: ' + CAST(@RemainingNull AS VARCHAR) + ' NULL values remain';
    PRINT 'Review the data before committing.';
END

PRINT '';
PRINT '============================================================================';

-- Uncomment ONE of the following lines:
-- COMMIT TRANSACTION;   -- Apply changes
-- ROLLBACK TRANSACTION;  -- Cancel changes

-- Leave transaction open for manual review
PRINT '';
PRINT 'TRANSACTION IS OPEN - Review results and manually COMMIT or ROLLBACK';
