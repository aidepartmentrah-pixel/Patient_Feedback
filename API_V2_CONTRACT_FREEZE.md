# STEP 3.5.6 — API v2 Contract Freeze

## Declaration

**API v2 contract is hereby FROZEN.**

This document defines the stable, authoritative API surface for Phase 3.5.

Frontend Phase 4 can proceed with full confidence that:
- Endpoint paths will not change
- Response shapes will not change
- Security behavior will not change
- Any modification requires a new phase

---

## API v2 Contract — Official Specification

### 1. Authentication

#### Get Current User

```
GET /api/auth/me
```

**Authentication:** Required (session-based)

**Response:** `200 OK`
```json
{
  "user": {
    "user_id": 1,
    "username": "john_doe",
    "is_active": true,
    "scopes": [
      {
        "role_code": "DEPARTMENT_ADMIN",
        "org_unit_id": 5,
        "org_unit_type": 2
      }
    ],
    "allowed_unit_ids": [5, 12, 18]
  }
}
```

**Key Fields:**
- `user_id`: Unique user identifier
- `username`: User's username
- `scopes[]`: User's role assignments (role + org unit)
- `allowed_unit_ids`: Phase 2.5 scope — organizational units user can access

**Authorization Contract:**
- **Backend is the single source of truth** for roles and scope
- Frontend **MUST NOT** infer permissions from role codes
- Frontend **MUST** rely on `allowed_unit_ids` for scope filtering
- Frontend **MUST** rely on `allowed_actions` in workflow responses

---

### 2. Workflow API — Inbox

#### Get Inbox (Role-Aware)

```
GET /api/v2/workflow/inbox
```

**Authentication:** Required

**Authorization:** Automatic (role-aware routing)
- Section admins see section-level inbox
- Department admins see department-level inbox
- Administration admins see administration-level inbox

**Response:** `200 OK`
```json
{
  "items": [
    {
      "subcase_id": 123,
      "case_type": "INCIDENT",
      "incident_id": 456,
      "seasonal_report_id": null,
      "target_org_unit_id": 5,
      "status": "SUBMITTED_TO_SECTION",
      "created_at": "2026-01-15T10:30:00Z",
      "allowed_actions": ["SUBMIT_RESPONSE", "REJECT"]
    }
  ]
}
```

**Backend Enforces:**
- Role logic (which inbox to show)
- Scope filtering (Phase 2.5 `allowed_unit_ids`)
- Status filtering (only actionable items)
- `allowed_actions` per item (role + status + ownership)

**Frontend Contract:**
- Frontend **MUST NOT** infer permissions
- Frontend **MUST** use `allowed_actions` to render UI buttons
- Frontend **MUST** handle 403 gracefully (scope violations)

---

### 3. Workflow API — Follow-Up

#### List Follow-Up Action Items

```
GET /api/v2/workflow/follow-up
```

**Authentication:** Required

**Authorization:** Automatic
- Returns action items assigned to user OR
- Returns action items user has privileged access to (via role)

**Response:** `200 OK`
```json
{
  "items": [
    {
      "action_item_id": 789,
      "subcase_id": 123,
      "title": "Investigate incident report",
      "description": "Review patient feedback and prepare response",
      "assigned_to_user_id": 1,
      "due_date": "2026-02-01",
      "status": "PENDING",
      "created_at": "2026-01-20T14:00:00Z"
    }
  ]
}
```

**Backend Enforces:**
- Assignment logic (assigned user OR privileged role)
- Scope filtering (Phase 2.5 `allowed_unit_ids`)

---

#### Start Action Item

```
POST /api/v2/workflow/follow-up/{action_item_id}/start
```

**Authentication:** Required

**Authorization:** User must be assigned OR have privileged role

**Request Body:** None

**Response:** `200 OK`
```json
{
  "success": true
}
```

**Error Response:** `403 Forbidden`
```json
{
  "detail": "User not authorized to start this action item"
}
```

**Backend Enforces:**
- Assignment validation
- Role validation
- Scope validation (Phase 2.5)

---

#### Complete Action Item

```
POST /api/v2/workflow/follow-up/{action_item_id}/complete
```

**Authentication:** Required

**Authorization:** User must be assigned OR have privileged role

**Request Body:** None

**Response:** `200 OK`
```json
{
  "success": true
}
```

**Error Response:** `403 Forbidden`
```json
{
  "detail": "User not authorized to complete this action item"
}
```

---

#### Delay Action Item

```
POST /api/v2/workflow/follow-up/{action_item_id}/delay
```

**Authentication:** Required

**Authorization:** User must be assigned OR have privileged role

**Request Body:** None

**Response:** `200 OK`
```json
{
  "success": true
}
```

**Error Response:** `403 Forbidden`
```json
{
  "detail": "User not authorized to delay this action item"
}
```

---

### 4. Workflow API — Case Actions

#### Unified Case Action Endpoint

```
POST /api/v2/workflow/case/{subcase_id}/act
```

**Authentication:** Required

**Authorization:** Dynamic (varies by action type)

**Request Body:**
```json
{
  "action": "SUBMIT_RESPONSE" | "REJECT" | "APPROVE" | "OVERRIDE" | "FORCE_CLOSE",
  "payload": {
    "explanation_text": "string (optional)",
    "rejection_text": "string (optional)",
    "action_items": [
      {
        "title": "string",
        "description": "string",
        "assigned_to_user_id": 123,
        "due_date": "2026-02-01"
      }
    ],
    "reason": "string (optional)"
  }
}
```

**Supported Actions:**

1. **SUBMIT_RESPONSE** (Section Admin)
   - Submits explanation + action items for a subcase
   - Payload: `explanation_text`, `action_items[]`

2. **REJECT** (Any level)
   - Rejects responsibility or response
   - Payload: `rejection_text`
   - Backend automatically determines level (section/dept/admin)

3. **APPROVE** (Department or Administration)
   - Approves a response
   - Payload: None
   - Backend automatically determines level

4. **OVERRIDE** (Department or Administration)
   - Overrides a response with new explanation
   - Payload: `explanation_text`, `action_items[]`
   - Backend automatically determines level

5. **FORCE_CLOSE** (Administration only)
   - Force closes a subcase
   - Payload: `reason`

**Response:** `200 OK`
```json
{
  "success": true
}
```

**Error Responses:**

- `400 Bad Request`: Unknown action or invalid payload
```json
{
  "detail": "Unknown action: INVALID_ACTION"
}
```

- `403 Forbidden`: Unauthorized for this action
```json
{
  "detail": "User not authorized to perform this action"
}
```

- `404 Not Found`: Subcase does not exist or not in scope
```json
{
  "detail": "Subcase not found or not accessible"
}
```

**Backend Enforces:**
- Role validation (correct role for action)
- Scope validation (Phase 2.5 `allowed_unit_ids`)
- Ownership validation (correct organizational unit)
- State machine validation (action valid for current status)

**Frontend Contract:**
- Frontend sends **intent only** via `action` field
- Frontend does **NOT** validate permissions client-side
- Frontend only reacts to `200` (success) or `403` (unauthorized)
- Backend owns all business logic and validation

---

## Explicit Exclusions

The following are **intentionally NOT included** in API v2:

❌ **Insight Endpoints**
- No `/api/v2/insights/*` endpoints
- No case summaries, counts, or aggregations
- No KPI endpoints
- No dashboard endpoints
- See [STEP_3_5_5_INSIGHT_DELAY_DECISION.md](STEP_3_5_5_INSIGHT_DELAY_DECISION.md)

❌ **Analytics Endpoints**
- No reporting endpoints
- No historical trend analysis

❌ **Legacy API v1 Workflow Endpoints**
- `/api/inbox/*` (deprecated)
- `/api/follow-up/*` (deprecated)
- See [STEP_3_5_0_API_SURFACE_AUDIT.md](STEP_3_5_0_API_SURFACE_AUDIT.md)

These exclusions are **by design**, not missing work.

---

## Contract Stability — Freeze Rules

After this freeze, the following changes are **PROHIBITED** without a new phase:

### ❌ PROHIBITED CHANGES

1. **Endpoint Path Renaming**
   - No changing `/inbox` to `/inbox-view`
   - No changing parameter names in paths

2. **Response Shape Changes**
   - No adding/removing required fields
   - No changing field types
   - No changing nesting structure
   - Optional fields may be added but not removed

3. **Semantic Behavior Changes**
   - No changing what `allowed_actions` means
   - No changing scope filtering logic
   - No changing role interpretation

4. **Security Contract Changes**
   - No relaxing authorization requirements
   - No removing scope filtering
   - No changing error response codes (200/403/404)

5. **Implicit Logic Movement**
   - No moving authorization from service to router
   - No moving scope filtering to frontend
   - Backend remains single source of truth

### ✅ ALLOWED CHANGES (Without Breaking Contract)

1. **Performance Optimizations**
   - Database query optimization
   - Caching improvements
   - Response time improvements

2. **Bug Fixes**
   - Fixing incorrect authorization logic
   - Fixing scope filtering bugs
   - Fixing state machine violations

3. **Internal Refactoring**
   - Renaming internal functions
   - Restructuring service layer
   - Improving code quality

4. **Adding Optional Fields**
   - New optional response fields (must be nullable)
   - New optional request parameters (must have defaults)

---

## Version Control

**API Version:** v2 (Frozen as of Phase 3.5.6)

**Freeze Date:** January 30, 2026

**Compatibility Promise:**
- All endpoints remain backward-compatible
- Breaking changes require `/api/v3` namespace
- Frontend can depend on this contract for Phase 4+

---

## Integration Contract — Frontend/Backend

### Frontend Responsibilities

1. **Respect Backend Authority**
   - Use `/api/auth/me` for user context
   - Use `allowed_actions` for UI rendering
   - Never infer permissions client-side

2. **Handle HTTP Status Codes**
   - `200` → Success, proceed
   - `403` → Unauthorized, show error
   - `404` → Not found, show error
   - `400` → Invalid request, show error

3. **Send Intent Only**
   - POST actions with simple `{ "action": "..." }` format
   - Let backend determine level/role/validation
   - Trust backend response

### Backend Responsibilities

1. **Single Source of Truth**
   - Owns role interpretation
   - Owns scope filtering (Phase 2.5)
   - Owns workflow state machine
   - Owns `allowed_actions` computation

2. **Security Enforcement**
   - All authorization in service layer
   - All scope filtering before queries
   - All state validation before mutations

3. **Stable Responses**
   - Consistent JSON shapes
   - Consistent error messages
   - Consistent HTTP status codes

---

## Stop Condition — Verification

This contract is considered **frozen and complete** when:

✅ **Frontend can proceed without backend inspection**
- All endpoint paths documented
- All request/response formats specified
- All error cases documented

✅ **Phase 4 prompts can be written with confidence**
- No need to "check what backend does"
- Clear contract for all interactions
- Stable foundation for frontend development

✅ **No implicit additions**
- All endpoints explicitly listed
- All exclusions explicitly stated
- All behavior explicitly documented

---

## Status

**🔒 API v2 CONTRACT FROZEN — PHASE 3.5.6 COMPLETE**

**Total Endpoints in API v2:** 6
- 1 Inbox endpoint
- 4 Follow-up endpoints
- 1 Case action endpoint

**Frontend Phase 4 is CLEAR TO PROCEED.**

---

## Next Steps

- **Phase 4:** Frontend implementation using this stable API
- **Phase 5:** Testing and validation
- **Phase 6:** Production deployment
- **Future:** Insight endpoints (after workflow validation)
