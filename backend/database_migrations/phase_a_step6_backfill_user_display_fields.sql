/*
================================================================================
PHASE A STEP 6: BACKFILL USER DISPLAY FIELDS
================================================================================

PURPOSE:
One-time script to populate DisplayName and DepartmentDisplayName for existing 
users who have NULL values in these columns.

BACKFILL RULES:
- DisplayName: If NULL → set to Username
- DepartmentDisplayName: If NULL → set to 'Unknown'

SAFETY:
- Idempotent (safe to run multiple times)
- Uses conditional WHERE clauses
- Only updates NULL values
- Does not modify other columns
- Does not delete rows
- Does not change constraints

USAGE:
Run this script once after deploying Phase A Step 1 schema changes.

================================================================================
*/

-- =============================================================================
-- BACKFILL DisplayName
-- =============================================================================
-- Set DisplayName = Username where DisplayName is NULL
UPDATE dbo.APP_Users
SET DisplayName = Username
WHERE DisplayName IS NULL;

-- =============================================================================
-- BACKFILL DepartmentDisplayName
-- =============================================================================
-- Set DepartmentDisplayName = 'Unknown' where DepartmentDisplayName is NULL
UPDATE dbo.APP_Users
SET DepartmentDisplayName = 'Unknown'
WHERE DepartmentDisplayName IS NULL;
