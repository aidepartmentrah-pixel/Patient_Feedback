# Phase 0: Database Schema Changes - Implementation Summary

## Overview
This phase adds the `RequiresExplanation` column to the `APP_IncidentCase` table to support the new explanation workflow.

## Status: Ready for Execution

---

## Files Created

### 1. Migration Script
**File:** `backend/database_migrations/phase0_add_requires_explanation.sql`

**What it does:**
- Adds `RequiresExplanation BIT NOT NULL DEFAULT 0` column
- Updates all existing records to `RequiresExplanation = 0`
- Verifies `TakenAction` field capacity
- Documents lookup table IDs for FSM implementation

**Safety features:**
- Checks if column already exists before adding
- Includes rollback script (commented)
- Comprehensive verification queries

### 2. Test Script
**File:** `backend/test_phase0_schema_changes.py`

**Test Coverage:**
1. ✅ Verify RequiresExplanation column exists
2. ✅ Verify all records have default value (0)
3. ✅ Verify TakenAction field capacity (NVARCHAR(MAX))
4. ✅ Document lookup table IDs
5. ✅ Test sample insert with RequiresExplanation

---

## Execution Steps

### Step 1: Run Migration (SQL Server Management Studio or Azure Data Studio)
```sql
-- Open: backend/database_migrations/phase0_add_requires_explanation.sql
-- Execute the entire script
```

### Step 2: Run Tests (Command Line)
```bash
cd "c:\Users\IT\Documents\GitHub Repository\Patient_Feedback"
python backend/test_phase0_schema_changes.py
```

Expected output: All 5 tests should pass ✓

---

## Schema Change Details

### New Column
```sql
ALTER TABLE dbo.APP_IncidentCase
ADD RequiresExplanation BIT NOT NULL DEFAULT 0;
```

**Purpose:** 
- `RequiresExplanation = 0` → Ordinary complaint, no explanation needed (default)
- `RequiresExplanation = 1` → Ordinary complaint that requires explanation

**Note:** Red Flag/Never Event cases will automatically be set to `RequiresExplanation = 1` via business logic (not enforced at DB level)

---

## Lookup Tables Reference

### Case Status IDs (APP_LOOKUP_CASE_STATUS)
```
ID  Code         Name         IsFinal
1   OPEN         Open         0
2   IN_PROGRESS  In Progress  0
3   CLOSED       Closed       1
```

### Explanation Status IDs (APP_LOOKUP_EXPLANATION_STATUS)
```
ID  StatusName
1   Waiting
2   Responded
3   Forcibly Closed
```

---

## FSM Logic (Reference for Next Phase)

### Three Paths at Insert:
1. **Red Flag/Never Event** → Always needs explanation
   - `RequiresExplanation = 1` (automatic)
   - `CaseStatusID = 1` (Open)
   - `ExplanationStatusID = 1` (Waiting)

2. **Ordinary + User checks "Requires Explanation"**
   - `RequiresExplanation = 1` (user decision)
   - `CaseStatusID = 1` (Open)
   - `ExplanationStatusID = 1` (Waiting)

3. **Ordinary + User does NOT check "Requires Explanation"**
   - `RequiresExplanation = 0` (default)
   - `CaseStatusID = 3` (Closed)
   - `ExplanationStatusID = NULL` (or ID for "No Explanation Needed")

---

## Rollback Instructions

If needed, to rollback this change:
```sql
ALTER TABLE dbo.APP_IncidentCase
DROP COLUMN RequiresExplanation;
```

**Warning:** This will permanently delete the column and its data.

---

## Next Phase Preview

**Phase 1: DB Layer - Read Operations**
Will create:
- `backend/api/db_layer/explanation_queries.py`
- Functions to query cases needing explanation
- Lookup helper functions

---

## Verification Checklist

After running migration and tests:
- [ ] Migration script executed without errors
- [ ] All 5 tests passed
- [ ] `RequiresExplanation` column visible in SQL Server
- [ ] All existing records have `RequiresExplanation = 0`
- [ ] `TakenAction` field is NVARCHAR(MAX)
- [ ] Lookup table IDs documented

---

## Questions Before Proceeding?

1. Do you want to execute the migration now?
2. Any concerns about the schema change?
3. Ready to proceed to Phase 1?
