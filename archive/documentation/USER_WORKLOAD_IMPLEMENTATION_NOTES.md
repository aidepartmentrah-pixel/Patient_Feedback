# User Workload Endpoint - Implementation Notes

## Discovery Summary

### Database Schema Analysis

**APP_Users:**
- UserID (int)
- Username (nvarchar)
- DisplayName (nvarchar) ✅
- DepartmentDisplayName (nvarchar) ✅
- ❌ No Email field
- ❌ No Phone field

**APP_AdministrativeSubcase:**
- SubcaseID
- Status (workflow state)
- TargetOrgUnitID
- UpdatedAt
- ❌ No AssignedToUserID field

**APP_SubcaseActionItem:**
- ActionItemID
- SubcaseID (FK to subcase)
- AssignedToUserID ✅ (FK to users)
- Status
- CompletedAt
- DueDate
- UpdatedAt

### Assignment Logic Decision

**Chosen Approach:** Track workload via **Action Items** assigned to users.

**Rationale:**
1. Subcases don't have explicit user assignments
2. Action items DO have `AssignedToUserID` field
3. Action items are the actual work units users must complete
4. This provides accurate, granular user workload tracking

### Workload Calculation Logic

**Open Work Items:**
- Action items where `CompletedAt IS NULL`
- Associated with subcases NOT in terminal states
- Terminal statuses: `ADMIN_APPROVED`, `SECTION_DENIED`, `FORCE_CLOSED`

**Oldest Item:**
- Track `MIN(UpdatedAt)` from action items per user
- Use action item `UpdatedAt` (not subcase UpdatedAt)

### Contact Info Fields

**Available:**
- `user_name`: DisplayName from APP_Users
- `user_role`: Derived from APP_UserRoleScope
- `primary_org_unit`: DepartmentDisplayName from APP_Users

**NOT Available:**
- ❌ Email: Column doesn't exist in APP_Users
- ❌ Phone: Column doesn't exist in APP_Users

**Response Schema Adjustment:**
```json
{
  "user_id": 123,
  "user_name": "Dr. John Smith",
  "user_role": "SECTION_ADMIN",
  "primary_org_unit": "Cardiology Section",
  "pending_count": 10,
  "oldest_item_days": 15
  // ❌ contact_info: NOT IMPLEMENTED (fields don't exist)
}
```

## Implementation Plan

### 1. DB Layer: `insight_db.py`
New function: `get_user_workload()`
- Query action items with `AssignedToUserID`
- JOIN to subcases to filter by non-terminal status
- JOIN to users for display name
- GROUP BY user
- Apply filters: org_unit_id, role, min_items
- Return sorted results

### 2. Service Layer: `insight_service.py`
New function: `get_user_workload()`
- Call DB layer
- Apply scope filtering via `allowed_unit_ids`
- No additional business logic needed

### 3. Router: `insight_router.py`
New endpoint: `GET /api/v2/insight/user-workload`
- Query params: org_unit_id, role, min_items, sort_by, sort_order
- Authorization: SOFTWARE_ADMIN, WORKER, COMPLAINT_SUPERVISOR
- Call service layer
- Return user workload list

### 4. Response Schema
No new Pydantic models needed - return list of dicts directly (matches other insight endpoints)

## Terminal Statuses Reference

From `get_stuck_subcases()`:
```python
Status NOT IN ('ADMIN_APPROVED', 'SECTION_DENIED', 'FORCE_CLOSED')
```

**Confirmed Terminal Statuses:**
- `ADMIN_APPROVED` - Final approval
- `SECTION_DENIED` - Section rejected responsibility
- `FORCE_CLOSED` - Administrative force close

## Next Steps

1. ✅ Create DB layer function
2. ✅ Create service layer function  
3. ✅ Add router endpoint
4. ✅ Create integration tests
5. ✅ Update documentation
