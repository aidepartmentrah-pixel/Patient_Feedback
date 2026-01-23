# Follow-Up Page API Documentation
## Updated: 2026-01-21

## ⚠️ IMPORTANT: Schema Limitations
The current `APP_ActionItem` table has limited fields. The following are **NOT** available:
- `DepartmentID` (returns `null`)
- `AssignedTo` (returns `null`)
- `Priority` (always returns `"medium"`)
- `Notes` (returns `null`)
- `LastUpdatedAt` / `LastUpdatedByUserID` (returns `CreatedAt` / `CreatedByUserID`)

These fields exist in the API response for future compatibility but are not stored in the database.

---

## Base URL
```
http://localhost:8000/api/follow-up
```

---

## 1. GET /actions - List All Follow-Up Actions

### URL
```
GET /api/follow-up/actions
```

### Query Parameters (Optional)
| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `status` | string | `pending`, `completed`, `all` | Filter by status (default: pending only) |
| `from_date` | string | `YYYY-MM-DD` | Filter actions due from this date |
| `to_date` | string | `YYYY-MM-DD` | Filter actions due until this date |
| `include_completed` | boolean | `true`, `false` | Include completed actions (default: false) |

**Note:** `priority` and `department` parameters are ignored (not in schema).

### Example Request
```http
GET /api/follow-up/actions?status=pending&from_date=2026-01-01&to_date=2026-12-31
```

### Response Structure
```json
{
  "actions": [
    {
      "id": 67,
      "actionTitle": "📊 Seasonal Performance Explanation",
      "actionDescription": "",
      "sourceType": "seasonal_explanation",  // incident_explanation | seasonal_explanation | manual
      "sourceId": "376",
      "departmentId": null,  // NOT AVAILABLE
      "assignedTo": null,    // NOT AVAILABLE
      "priority": "medium",  // ALWAYS "medium"
      "status": "pending",   // pending | completed
      "dueDate": "2026-01-25",
      "completedDate": null,
      "notes": null,         // NOT AVAILABLE
      "createdAt": "2026-01-21T11:30:42.923000",
      "createdByUserId": 1,
      "lastUpdatedAt": "2026-01-21T11:30:42.923000",  // Same as createdAt
      "lastUpdatedByUserId": 1,  // Same as createdByUserId
      "isOverdue": false,
      "daysRemaining": 4,
      "daysOverdue": 0
    }
  ],
  "total": 6,
  "statistics": {
    "actionsToTake": 6,
    "overdue": 0,
    "completed": 0
  }
}
```

---

## 2. GET /actions/{id} - Get Single Action

### URL
```
GET /api/follow-up/actions/{action_id}
```

### Example Request
```http
GET /api/follow-up/actions/67
```

### Response
Same structure as single action object from list endpoint.

---

## 3. PATCH /actions/{id} - Update Action

### URL
```
PATCH /api/follow-up/actions/{action_id}
```

### Request Body (All fields optional)
```json
{
  "dueDate": "2026-02-01",
  "status": "completed"  // pending | completed
}
```

**Note:** `assignedTo`, `priority`, `notes` are ignored (not in schema).

### Response
Returns updated action object.

---

## 4. POST /actions/{id}/complete - Mark Action as Completed

### URL
```
POST /api/follow-up/actions/{action_id}/complete
```

### Request Body (Optional)
```json
{
  "completedDate": "2026-01-21"
}
```

**Note:** `completionNotes` is ignored (Notes field not in schema).

### Response
Returns updated action with `status: "completed"`.

---

## 5. POST /actions/{id}/delay - Delay Action

### URL
```
POST /api/follow-up/actions/{action_id}/delay
```

### Request Body
```json
{
  "delayDays": 7
}
```

**Note:** `reason` parameter is ignored (Notes field not in schema).

### Response
Returns updated action with new `dueDate`.

---

## 6. POST /actions/{id}/reopen - Reopen Completed Action

### URL
```
POST /api/follow-up/actions/{action_id}/reopen
```

### Request Body
```json
{
  "reopenReason": "Additional work required",
  "newDueDate": "2026-02-15"  // Optional
}
```

**Note:** `reopenReason` is ignored (Notes field not in schema).

### Response
Returns updated action with `status: "pending"`.

---

## 7. GET /actions/{id}/history - Get Action History

### URL
```
GET /api/follow-up/actions/{action_id}/history
```

### Response
```json
[
  {
    "timestamp": "2026-01-21T11:30:42.923000",
    "userId": 1,
    "action": "Created",
    "details": "Action created: Seasonal Performance Explanation"
  },
  {
    "timestamp": "2026-01-25",
    "userId": 1,
    "action": "Completed",
    "details": "Action marked as completed"
  }
]
```

**Note:** History is minimal since Notes field is not available.

---

## 8. GET /calendar - Calendar View

### URL
```
GET /api/follow-up/calendar?month=2026-01
```

### Query Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `month` | string | No | Format: `YYYY-MM` (default: current month) |

### Response
```json
{
  "year": 2026,
  "month": 1,
  "calendar": {
    "2026-01-25": [
      {
        "id": 67,
        "actionTitle": "Seasonal Performance Explanation",
        "priority": "medium",
        "status": "pending",
        "departmentId": null,
        "assignedTo": null,
        "isOverdue": false
      }
    ],
    "2026-01-29": [
      {
        "id": 66,
        "actionTitle": "Red Flag Action",
        "priority": "medium",
        "status": "pending",
        "departmentId": null,
        "assignedTo": null,
        "isOverdue": false
      }
    ]
  }
}
```

---

## 9. POST /actions/bulk-complete - Bulk Complete Actions

### URL
```
POST /api/follow-up/actions/bulk-complete
```

### Request Body
```json
{
  "actionIds": [67, 66, 62],
  "completedDate": "2026-01-21"  // Optional
}
```

**Note:** `completionNotes` is ignored.

### Response
```json
{
  "successCount": 3,
  "failedCount": 0,
  "failedIds": []
}
```

---

## 10. POST /actions/bulk-delay - Bulk Delay Actions

### URL
```
POST /api/follow-up/actions/bulk-delay
```

### Request Body
```json
{
  "actionIds": [67, 66],
  "delayDays": 7
}
```

**Note:** `reason` parameter is ignored.

### Response
```json
{
  "successCount": 2,
  "failedCount": 0,
  "failedIds": []
}
```

---

## 11. POST /actions/bulk-update - Bulk Update Actions

### URL
```
POST /api/follow-up/actions/bulk-update
```

### Response
```json
{
  "successCount": 0,
  "failedCount": 2,
  "failedIds": [
    {"id": 67, "reason": "Bulk update not supported with current schema"},
    {"id": 66, "reason": "Bulk update not supported with current schema"}
  ]
}
```

**Note:** This endpoint is NOT SUPPORTED with current schema (no AssignedTo, Priority, DepartmentID fields).

---

## 12. POST /actions - Create New Action

### URL
```
POST /api/follow-up/actions
```

### Request Body
```json
{
  "actionTitle": "New Action",
  "actionDescription": "Description here",
  "incidentCaseId": 123,  // Optional, link to incident
  "seasonalReportId": null,  // Optional, link to seasonal report
  "dueDate": "2026-02-15"
}
```

**Note:** `departmentId`, `assignedTo`, `priority`, `notes` are ignored.

### Response
Returns created action object with generated ID.

---

## Frontend Recommendations

1. **Hide/Remove** these fields from UI:
   - `departmentId` filter
   - `assignedTo` field and filter
   - `priority` filter (always shows "medium")
   - `notes` field
   
2. **Simplify Status**:
   - Only show "Pending" and "Completed" (no "Delayed" state)
   
3. **Disable Bulk Update**:
   - Remove bulk update button (not supported)
   
4. **Calendar View**:
   - Works but won't show department/assigned-to info

5. **History**:
   - Shows minimal info (created + completed dates only)

---

## Testing URL
```
http://localhost:8000/api/follow-up/actions
```

Current response has 6 pending actions.
