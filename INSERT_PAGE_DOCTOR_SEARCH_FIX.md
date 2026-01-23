# Insert Page Doctor Search Fix

## Issue Description

**Problem**: Doctors created in the reserve table (`APP_RESERVE_DOCTOR`) were not appearing in the search results on the Insert Page, even though they were successfully created and stored in the database.

**Root Cause**: The `search_doctors()` function in `backend/api/services/search_service.py` was only querying the hospital view (`APP_VIEWTABLE_VW_DOCTORS`) and did not include the reserve table. This meant user-created doctors were invisible to the Insert Page search functionality.

## Affected Components

- **File**: `backend/api/services/search_service.py`
- **Functions**:
  1. `search_doctors()` - Used by Insert Page for doctor search
  2. `get_doctor_by_id()` - Used by Insert Page for doctor details

- **API Endpoint**: 
  - `GET /api/records/search/doctors` - Insert Page doctor search
  - `GET /api/records/doctor/{doctor_id}` - Insert Page doctor details

## Solution Applied

### 1. Updated `search_doctors()` Function

**Before**: Only searched `APP_VIEWTABLE_VW_DOCTORS` (hospital view)

**After**: Implemented UNION query pattern (same as `search_patients()`) to merge results from:
- `APP_VIEWTABLE_VW_DOCTORS` (hospital doctors)
- `APP_RESERVE_DOCTOR` (user-created doctors)

**Key Changes**:
- Added UNION ALL query to merge both data sources
- Added `source` field to response (`'hospital'` or `'reserve'`)
- Mapped reserve table fields to match hospital view structure:
  - `DoctorName` → `Name`
  - `Specialty` → `SpecialityName`
  - `NULL` → `SpecialityID` (reserve table doesn't have IDs)
  - `0` → `IsAdmitted` and `IsClinic` (reserve defaults)

### 2. Updated `get_doctor_by_id()` Function

**Before**: Only searched `APP_VIEWTABLE_VW_DOCTORS`

**After**: Implemented dual-source lookup pattern:
1. Check `APP_RESERVE_DOCTOR` first (prioritizes user-created doctors)
2. If not found, check `APP_VIEWTABLE_VW_DOCTORS`
3. Added `source` field to response

## Testing

Created test file: `test_insert_doctor_search_fix.py`

**Test Results**: ✅ ALL TESTS PASSED

The test verifies:
1. ✅ Reserve doctor can be created
2. ✅ Reserve doctor appears in Insert Page search results
3. ✅ Reserve doctor details can be retrieved by ID
4. ✅ `source` field correctly indicates 'reserve'

## Impact

### Before Fix
- ❌ Reserve doctors invisible in Insert Page search
- ❌ Cannot select user-created doctors for incident records
- ✅ Hospital doctors work fine

### After Fix
- ✅ Reserve doctors visible in Insert Page search
- ✅ Can select user-created doctors for incident records
- ✅ Hospital doctors still work fine
- ✅ `source` field indicates origin of each doctor

## Consistency with Other Features

This fix aligns the Insert Page doctor search with the existing patient search pattern:

| Feature | Hospital Source | Reserve Source | Status |
|---------|----------------|----------------|--------|
| Patient Search | `APP_VIEWTABLE_PATIENT_ADMISSION` | `APP_RESERVE_PATIENT` | ✅ Already working |
| **Doctor Search** | `APP_VIEWTABLE_VW_DOCTORS` | `APP_RESERVE_DOCTOR` | ✅ **NOW FIXED** |

## Related Documentation

- Patient search pattern: `search_patients()` in same file (already had UNION)
- Doctor API endpoints: `backend/api/routers/doctors_router.py`
- Doctor DB layer: `backend/api/db_layer/doctors_db.py` (uses UNION)
- Phase 6 Report: `PHASE6_COMPLETION_REPORT.md`

## Date Fixed
2026-01-21

## Notes

- The fix maintains backward compatibility
- No database schema changes required
- Frontend receives same response structure with additional `source` field
- All existing hospital doctor functionality remains unchanged
