# PHASE 3 FIX PLAN - Column Name Mismatch Resolution

**Date:** January 30, 2026  
**Issue:** Schema mismatch between expected and actual column names  
**Status:** ACTION PLAN READY

---

## PROBLEM IDENTIFIED

### Expected vs Actual Column Names

| Expected (Code) | Actual (Database) | Status |
|----------------|-------------------|--------|
| `IncidentID` | `IncidentRequestCaseID` | MISMATCH |
| `AssignedToRoleCode` | NOT EXISTS | MISSING |
| `CurrentStatusCode` | `Status` | MISMATCH |
| `TargetOrgUnitType` | NOT EXISTS | MISSING |

### Actual Schema (16 columns)
```
SubcaseID                    (PK)
CaseType                     (e.g., SEASONAL_REPORT_RESPONSE)
IncidentRequestCaseID        (FK to APP_IncidentCase)
SeasonalReportID             (FK to Seasonal Report)
TargetOrgUnitID              (Department/Unit ID)
Status                       (e.g., SUBMITTED_TO_SECTION)
SectionExplanationText       (Workflow text)
SectionRejectionText         (Workflow text)
DepartmentExplanationText    (Workflow text)
DepartmentRejectionText      (Workflow text)
AdministrationExplanationText (Workflow text)
AdministrationRejectionText  (Workflow text)
CreatedAt
CreatedByUserID
UpdatedAt
UpdatedByUserID
```

---

## ROOT CAUSE

**The table schema is DIFFERENT from what the code expects!**

This is not a bug - it's a **design mismatch**. The actual schema:
1. Uses `IncidentRequestCaseID` instead of `IncidentID`
2. Uses `Status` instead of `CurrentStatusCode`
3. Does NOT have `AssignedToRoleCode` or `TargetOrgUnitType`
4. Uses role-specific explanation/rejection text columns instead

---

## SOLUTION OPTIONS

### Option A: Update Code to Match Database (RECOMMENDED)
**What:** Update `administrative_subcase_db.py` to use actual column names  
**Pros:** No database changes, safer, faster  
**Cons:** Need to update db_layer and possibly service layer  
**Effort:** Medium (1-2 hours)

### Option B: Update Database to Match Code
**What:** Run ALTER TABLE to add/rename columns  
**Pros:** Code matches expectations  
**Cons:** Risky, might break existing data (SubcaseID 53)  
**Effort:** Medium-High (2-3 hours + testing)

### Option C: Hybrid Approach
**What:** Update code for existing columns, add missing ones to DB  
**Pros:** Best of both worlds  
**Cons:** More complex  
**Effort:** High (3-4 hours)

---

## RECOMMENDED PLAN: OPTION A (Update Code)

### Phase 1: Fix administrative_subcase_db.py

**Files to Update:**
1. `backend/api_v2/db_layer/administrative_subcase_db.py`

**Changes Required:**

#### 1.1 Update Column Mappings
```python
# OLD (Expected)
'incident_id' -> IncidentID
'assigned_to_role_code' -> AssignedToRoleCode
'current_status_code' -> CurrentStatusCode
'target_org_unit_type' -> TargetOrgUnitType

# NEW (Actual)
'incident_id' -> IncidentRequestCaseID
'assigned_to_role_code' -> (REMOVE - not in schema)
'current_status_code' -> Status
'target_org_unit_type' -> (REMOVE - not in schema)
```

#### 1.2 Update create_subcase() Function
```python
# OLD
INSERT INTO ... (IncidentID, AssignedToRoleCode, CurrentStatusCode, TargetOrgUnitType)

# NEW
INSERT INTO ... (IncidentRequestCaseID, Status)
# Remove AssignedToRoleCode, TargetOrgUnitType from INSERT
```

#### 1.3 Update Query Functions
```python
# OLD
SELECT IncidentID, CurrentStatusCode, AssignedToRoleCode, TargetOrgUnitType

# NEW
SELECT IncidentRequestCaseID AS incident_id, Status AS status
# Map back to expected names for API compatibility
```

---

### Phase 2: Update case_creation_service.py

**Files to Update:**
1. `backend/api_v2/services/case_creation_service.py`

**Changes Required:**

#### 2.1 Update _create_subcase() Calls
```python
# OLD
_create_subcase(
    target_org_unit_id=dept_id,
    created_by_user_id=user_id,
    initial_status='SUBMITTED_TO_SECTION'
)

# NEW
# Remove target_org_unit_type parameter (not in schema)
# Status is just 'Status' column now
```

---

### Phase 3: Update Test Files

**Files to Update:**
1. `test_step3_10_adapter_integration.py`
2. `diagnose_phase3_stack.py`

**Changes Required:**

#### 3.1 Update SQL Queries
```python
# OLD
SELECT IncidentID, AssignedToRoleCode, CurrentStatusCode, TargetOrgUnitType

# NEW
SELECT IncidentRequestCaseID, Status
```

#### 3.2 Update Field Access
```python
# OLD
sc.IncidentID
sc.AssignedToRoleCode
sc.CurrentStatusCode

# NEW
sc.IncidentRequestCaseID
sc.Status
```

---

### Phase 4: Missing Columns Discussion

**Question:** Do we need `AssignedToRoleCode` and `TargetOrgUnitType`?

#### Current Workarounds:
1. **AssignedToRoleCode:** 
   - Can derive from `Status` (SUBMITTED_TO_SECTION -> SECTION_ADMIN)
   - Or store separately in a new column later

2. **TargetOrgUnitType:**
   - Can lookup from `TargetOrgUnitID` (query org structure)
   - Or assume all are departments for now

---

## IMPLEMENTATION CHECKLIST

### Step 1: Backup Current Files
```bash
cp backend/api_v2/db_layer/administrative_subcase_db.py backend/api_v2/db_layer/administrative_subcase_db.py.backup
cp backend/api_v2/services/case_creation_service.py backend/api_v2/services/case_creation_service.py.backup
```

### Step 2: Update administrative_subcase_db.py
- [ ] Fix column names in create_subcase()
- [ ] Fix column names in get_subcase_by_id()
- [ ] Fix column names in get_subcases_by_incident()
- [ ] Fix column names in get_subcases_by_seasonal_report()
- [ ] Fix column names in update_subcase_status()
- [ ] Add column mapping for API compatibility

### Step 3: Update case_creation_service.py
- [ ] Remove target_org_unit_type parameter
- [ ] Update _create_subcase() function calls
- [ ] Test with existing SubcaseID 53

### Step 4: Update Tests
- [ ] Fix test_step3_10_adapter_integration.py
- [ ] Fix diagnose_phase3_stack.py
- [ ] Add new test for actual schema

### Step 5: Test Everything
- [ ] Test get_subcase_by_id(53) works
- [ ] Test creating new incident subcases
- [ ] Test creating new seasonal report subcases
- [ ] Run full diagnostic again

---

## ESTIMATED TIMELINE

- **Phase 1:** 30 minutes (update db_layer)
- **Phase 2:** 20 minutes (update service)
- **Phase 3:** 20 minutes (update tests)
- **Phase 4:** 10 minutes (discussion)
- **Testing:** 30 minutes
- **TOTAL:** ~2 hours

---

## SUCCESS CRITERIA

1. ✅ `diagnose_phase3_stack.py` shows no errors
2. ✅ `get_subcase_by_id(53)` returns correct data
3. ✅ Creating incident creates subcases successfully
4. ✅ Creating seasonal report creates subcases successfully
5. ✅ All tests pass without schema errors

---

## NEXT STEPS

**Do you want me to:**

1. **Option 1:** Start implementing the fixes immediately (I'll update the files one by one)
2. **Option 2:** Create a detailed code diff first for your review
3. **Option 3:** Focus on just the critical fix (db_layer only) and test
4. **Option 4:** Discuss if we should add the missing columns to the database instead

**Your choice?** 🤔
