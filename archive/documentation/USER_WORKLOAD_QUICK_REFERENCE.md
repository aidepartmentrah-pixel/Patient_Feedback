# User Workload Endpoint - Quick Reference

## Endpoint

```
GET /api/v2/insight/user-workload
```

## Authorization

✅ Allowed:
- SOFTWARE_ADMIN
- WORKER
- COMPLAINT_SUPERVISOR

❌ Denied: All other roles

## Query Parameters

| Parameter | Type | Default | Options |
|-----------|------|---------|---------|
| `org_unit_id` | int | None | Any valid org unit ID |
| `role` | string | None | SECTION_ADMIN, DEPARTMENT_ADMIN, etc. |
| `min_items` | int | 1 | Any positive integer |
| `sort_by` | string | pending_count | pending_count, oldest_item, user_name |
| `sort_order` | string | desc | asc, desc |

## Response

```json
[
  {
    "user_id": 456,
    "user_name": "Dr. John Smith",
    "user_role": "SECTION_ADMIN",
    "primary_org_unit": "Cardiology Section",
    "pending_count": 10,
    "oldest_item_days": 15
  }
]
```

## Examples

### Default (all users)
```bash
GET /api/v2/insight/user-workload
```

### High workload users (≥10 items)
```bash
GET /api/v2/insight/user-workload?min_items=10
```

### Section admins only
```bash
GET /api/v2/insight/user-workload?role=SECTION_ADMIN
```

### Specific org unit
```bash
GET /api/v2/insight/user-workload?org_unit_id=5
```

### Sort by oldest item
```bash
GET /api/v2/insight/user-workload?sort_by=oldest_item&sort_order=desc
```

### Combined filters
```bash
GET /api/v2/insight/user-workload?role=SECTION_ADMIN&min_items=5&sort_by=pending_count
```

## How It Works

**Data Source:** `APP_SubcaseActionItem` (action items assigned to users)

**Workload Calculation:**
- Counts **incomplete** action items (`CompletedAt IS NULL`)
- Only for subcases **NOT in terminal states**:
  - ❌ ADMIN_APPROVED
  - ❌ SECTION_DENIED
  - ❌ FORCE_CLOSED
- Groups by user
- Applies scope filtering (`allowed_unit_ids`)

**Oldest Item:** Days since the oldest pending item was last updated

## Use Cases

### 1. Proactive Follow-Up
**Scenario:** Supervisor wants to call users with high workload

```bash
GET /api/v2/insight/user-workload?min_items=10
```

**Action:** Call top users to check if they need support

### 2. Workload Distribution
**Scenario:** Check if work is evenly distributed across section admins

```bash
GET /api/v2/insight/user-workload?role=SECTION_ADMIN&sort_by=pending_count
```

**Action:** Redistribute if one admin has significantly more items

### 3. Identify Bottlenecks
**Scenario:** Check which org unit has the most stuck items

```bash
GET /api/v2/insight/user-workload?sort_by=oldest_item&sort_order=desc
```

**Action:** Investigate users with oldest pending items

### 4. Monitor Department Performance
**Scenario:** Check workload for specific department

```bash
GET /api/v2/insight/user-workload?org_unit_id=8&min_items=1
```

**Action:** See all users in department with pending work

## Error Handling

### 403 Forbidden
```json
{
  "detail": "Insufficient permissions. Only admins and workers can view user workload."
}
```
**Reason:** User doesn't have required role

### 400 Bad Request
```json
{
  "detail": "Invalid sort_by value. Must be one of: pending_count, oldest_item, user_name"
}
```
**Reason:** Invalid query parameter

## Frontend Integration

```javascript
// src/api/insightApi.js
export async function getUserWorkload(filters = {}) {
  const params = {};
  if (filters.orgUnitId) params.org_unit_id = filters.orgUnitId;
  if (filters.role) params.role = filters.role;
  if (filters.minItems) params.min_items = filters.minItems;
  if (filters.sortBy) params.sort_by = filters.sortBy;
  if (filters.sortOrder) params.sort_order = filters.sortOrder;
  
  const res = await apiClient.get('/api/v2/insight/user-workload', { params });
  return res.data;
}
```

**Usage:**
```javascript
// Get all users with workload
const users = await getUserWorkload();

// Get section admins with ≥5 items
const busyAdmins = await getUserWorkload({
  role: 'SECTION_ADMIN',
  minItems: 5,
  sortBy: 'pending_count'
});
```

## Testing

**Test File:** `backend/test_user_workload_endpoint.py`

Run tests:
```bash
cd "c:\Users\IT\Documents\GitHub Repository\Patient_Feedback"
python backend/test_user_workload_endpoint.py
```

**Expected:** 11/12 tests passing ✅

## Implementation Status

✅ **COMPLETE AND TESTED**

- Database layer: `insight_db.py`
- Service layer: `insight_service.py`
- Router endpoint: `insight_router.py`
- Tests: `test_user_workload_endpoint.py`
- Documentation: Complete

**Ready for frontend integration.**

## Notes

⚠️ **Contact Info Not Available:**
- Email and Phone fields don't exist in APP_Users table
- Frontend cannot display contact details
- Users must use alternative communication methods

⚠️ **No Pagination:**
- Returns all matching users
- May need pagination if user volumes become large (100+)

⚠️ **Real-Time Data:**
- No caching implemented
- Always returns current workload state
- May add caching if performance becomes an issue

## Related Endpoints

- `GET /api/v2/insight/kpi-summary` - Overall KPI metrics
- `GET /api/v2/insight/stuck` - Stuck cases (case-centric view)
- `GET /api/v2/insight/distribution` - Distribution by dimension
- `GET /api/v2/insight/trend` - Time-series trends
