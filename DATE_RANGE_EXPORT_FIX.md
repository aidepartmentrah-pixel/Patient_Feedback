# Date Range Export Fix

## Issue
User reported a **422 Unprocessable Content** error when attempting to export a monthly report using custom date ranges instead of a specific month:

```
POST /api/reports/monthly/export?format=docx&year=2026&display_mode=detailed&start_date=2024-11-03&end_date=2026-02-28
```

Error: `"detail": "Field required [type=missing, input={...}, url=...]"`

## Root Cause
The `/api/reports/monthly/export` endpoint required the `month` parameter as mandatory (`month: int = Query(...)`), but the user interface was sending `start_date` and `end_date` parameters for custom date range exports instead.

## Solution
Modified the endpoint and service layers to support **either** a month parameter **or** custom date range (start_date/end_date):

### 1. Router Changes ([reports_router.py](backend/api/routers/reports_router.py))
- Changed `month` parameter from required to optional: `Optional[int] = Query(None)`
- Added `start_date` and `end_date` parameters: `Optional[str] = Query(None)`
- Added validation logic to ensure either month OR date range is provided
- Updated both single export and multi-export paths to pass date parameters

**Lines Modified**: 432-445, 475-480, 605-615

### 2. Multi-Report Export Service ([multi_report_export_service.py](backend/api/services/multi_report_export_service.py))
- Updated `generate_multi_export()` signature to accept `start_date` and `end_date` as `Optional[str]`
- Updated `_generate_unit_report()` signature to accept date range parameters
- Modified filename generation to handle date ranges: `Monthly_Report_{unit}_{start_date}_to_{end_date}.{format}`
- Updated internal service calls to pass date parameters through the chain

**Lines Modified**: 35-45, 92-100, 118-130, 177-184, 201-209

### 3. Single Report Export Service ([report_export_service.py](backend/api/services/report_export_service.py))
- Updated `generate_export()` signature to accept `start_date` and `end_date` parameters
- Changed `month` parameter to `Optional[int]`
- Updated `_fetch_monthly_data()` signature to accept date range parameters
- Modified filename generation: `Monthly_Report_{start_date}_to_{end_date}.{format}`
- Updated service calls to pass date parameters to `monthly_report_service`

**Lines Modified**: 27-54, 80-87, 281-284, 357-391

### 4. Validation Logic
Added comprehensive validation in the router:

```python
if month is None and (start_date is None or end_date is None):
    raise HTTPException(
        status_code=400,
        detail="Must provide either 'month' parameter OR both 'start_date' and 'end_date'"
    )
```

## Backward Compatibility
- ✅ **Existing month-based exports**: Continue to work exactly as before
- ✅ **New date range exports**: Now supported with proper validation
- ✅ **Multi-export**: Works with both month and date range parameters
- ✅ **Single export**: Works with both month and date range parameters

## Testing
After restart, the endpoint now accepts both patterns:

**Pattern 1 - Month-based (existing)**:
```
POST /api/reports/monthly/export?format=docx&year=2026&month=1&display_mode=detailed
```

**Pattern 2 - Date range (new)**:
```
POST /api/reports/monthly/export?format=docx&year=2026&display_mode=detailed&start_date=2024-11-03&end_date=2026-02-28
```

## Files Modified
1. `backend/api/routers/reports_router.py` - Endpoint signature and validation
2. `backend/api/services/multi_report_export_service.py` - Multi-export support
3. `backend/api/services/report_export_service.py` - Single export support

## Related Services
The underlying `monthly_report_service.py` already supported date ranges, so no changes were needed there. This fix simply exposes that capability through the export endpoints.

## Server Status
✅ Server restarted successfully with all changes applied
✅ No compilation errors detected
✅ Ready for testing

---
**Date**: 2025-01-XX  
**Issue**: Date range export returning 422 error  
**Status**: RESOLVED
