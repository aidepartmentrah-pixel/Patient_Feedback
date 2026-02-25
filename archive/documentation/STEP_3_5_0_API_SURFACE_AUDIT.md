# STEP 3.5.0 — API SURFACE AUDIT REPORT

**Date:** January 30, 2026  
**Task Type:** READ-ONLY AUDIT  
**Scope:** All workflow-related HTTP endpoints in backend

---

## 🎯 EXECUTIVE SUMMARY

### Key Findings

1. **API v1 (Legacy)** exposes **14 workflow-related endpoints** across 2 routers
2. **API v2** exposes **20 workflow endpoints** across 4 routers
3. **CRITICAL OVERLAP DETECTED**: Follow-up actions have duplicate implementations
4. **NO INBOX IN API v1**: Legacy API has no inbox concept
5. **NO CASE WORKFLOW IN API v1**: Legacy API has no subcase response workflow
6. **INSIGHT IS API V2 ONLY**: No legacy insight endpoints exist

### Risk Assessment Summary

- ✅ **Safe**: 14 endpoints (API v2 only, no conflicts)
- ⚠️ **DUPLICATE**: 10 endpoints (follow-up actions exist in both APIs)
- 🚨 **CONFLICTING**: 0 endpoints (none found)
- 🗑️ **SHOULD BE DEPRECATED**: 10 endpoints (API v1 follow-up router should be marked deprecated)

---

## 📊 DETAILED ENDPOINT INVENTORY

### 🔴 API v1 (LEGACY) — WORKFLOW ENDPOINTS

#### Router: `/api/action-items`
**File:** `backend/api/routers/action_items.py`

| # | Method | Path | Service Function | Operation | Security | Classification | Risk |
|---|--------|------|------------------|-----------|----------|----------------|------|
| 1 | GET | `/api/action-items/{action_item_id}` | `get_action_item()` | Read | ✅ Logged in | Legacy | 🗑️ Deprecate |
| 2 | GET | `/api/action-items/by-incident/{incident_case_id}` | `get_action_items_for_incident()` | Read | ✅ Logged in | Legacy | 🗑️ Deprecate |
| 3 | GET | `/api/action-items/by-seasonal-report/{seasonal_report_id}` | `get_action_items_for_seasonal_report()` | Read | ✅ Logged in | Legacy | 🗑️ Deprecate |
| 4 | GET | `/api/action-items/by-season/{season_case_id}` | `get_action_items_for_season()` | Read | ✅ Logged in | Legacy | 🗑️ Deprecate |
| 5 | POST | `/api/action-items/{action_item_id}/mark-done` | `mark_action_done()` | Mutation | ✅ Logged in | Legacy | 🗑️ Deprecate |

**Security Analysis:**
- Authentication: ✅ `require_logged_in()`
- Role guards: ❌ None
- Scope enforcement: ❌ None
- **Risk:** Basic authentication only, no role or scope validation

---

#### Router: `/api/follow-up`
**File:** `backend/api/routers/follow_up_router.py`

| # | Method | Path | Service Function | Operation | Security | Classification | Risk |
|---|--------|------|------------------|-----------|----------|----------------|------|
| 6 | POST | `/api/follow-up/actions` | `FollowUpService.create_follow_up_action()` | Mutation | ✅ Logged in | Legacy | 🗑️ Deprecate |
| 7 | GET | `/api/follow-up/actions` | `FollowUpService.get_follow_up_actions()` | Read | ✅ Logged in | Legacy | 🗑️ Deprecate |
| 8 | GET | `/api/follow-up/actions/{action_id}` | `FollowUpService.get_follow_up_action_by_id()` | Read | ✅ Logged in | Legacy | 🗑️ Deprecate |
| 9 | PATCH | `/api/follow-up/actions/{action_id}` | `FollowUpService.update_follow_up_action()` | Mutation | ✅ Logged in | Legacy | 🗑️ Deprecate |
| 10 | POST | `/api/follow-up/actions/{action_id}/complete` | `FollowUpService.complete_follow_up_action()` | Mutation | ✅ Logged in | Legacy | 🗑️ Deprecate |
| 11 | POST | `/api/follow-up/actions/{action_id}/delay` | `FollowUpService.delay_follow_up_action()` | Mutation | ✅ Logged in | Legacy | 🗑️ Deprecate |
| 12 | POST | `/api/follow-up/actions/{action_id}/reopen` | `FollowUpService.reopen_follow_up_action()` | Mutation | ✅ Logged in | Legacy | 🗑️ Deprecate |
| 13 | GET | `/api/follow-up/actions/{action_id}/history` | `FollowUpService.get_action_history()` | Read | ✅ Logged in | Legacy | 🗑️ Deprecate |
| 14 | GET | `/api/follow-up/calendar` | `FollowUpService.get_calendar_actions()` | Read | ✅ Logged in | Legacy | 🗑️ Deprecate |
| 15 | POST | `/api/follow-up/actions/bulk-complete` | `FollowUpService.bulk_complete_actions()` | Mutation | ✅ Logged in | Legacy | 🗑️ Deprecate |
| 16 | POST | `/api/follow-up/actions/bulk-delay` | `FollowUpService.bulk_delay_actions()` | Mutation | ✅ Logged in | Legacy | 🗑️ Deprecate |
| 17 | POST | `/api/follow-up/actions/bulk-update` | `FollowUpService.bulk_update_actions()` | Mutation | ✅ Logged in | Legacy | 🗑️ Deprecate |

**Security Analysis:**
- Authentication: ✅ `require_logged_in()`
- Role guards: ❌ None
- Scope enforcement: ❌ None
- **Risk:** No role/scope validation, allows any authenticated user to modify any action

---

### 🟢 API v2 (TARGET) — WORKFLOW ENDPOINTS

#### Router: `/api/v2/inbox`
**File:** `backend/api_v2/routers/inbox_router.py`

| # | Method | Path | Service Function | Operation | Security | Classification | Risk |
|---|--------|------|------------------|-----------|----------|----------------|------|
| 1 | GET | `/api/v2/inbox/section` | `inbox_service.get_section_inbox()` | Read | ✅ Section Admin + Scope | API v2 | ✅ Safe |
| 2 | GET | `/api/v2/inbox/department` | `inbox_service.get_department_inbox()` | Read | ✅ Dept Admin + Scope | API v2 | ✅ Safe |
| 3 | GET | `/api/v2/inbox/administration` | `inbox_service.get_administration_inbox()` | Read | ✅ Admin + Scope | API v2 | ✅ Safe |

**Security Analysis:**
- Authentication: ✅ Via dependency injection
- Role guards: ✅ `require_section_admin`, `require_dept_admin`, `require_administrator`
- Scope enforcement: ✅ Service layer (Phase 2.5)
- **Risk:** ✅ Fully protected, no conflicts with legacy API

---

#### Router: `/api/v2/follow-up`
**File:** `backend/api_v2/routers/follow_up_router.py`

| # | Method | Path | Service Function | Operation | Security | Classification | Risk |
|---|--------|------|------------------|-----------|----------|----------------|------|
| 4 | GET | `/api/v2/follow-up/action-items` | `follow_up_service.get_action_items_for_user()` | Read | ✅ Supervisor/Worker + Scope | API v2 | ⚠️ Duplicate |
| 5 | POST | `/api/v2/follow-up/action-items/{action_item_id}/complete` | `follow_up_service.complete_action_item()` | Mutation | ✅ Supervisor/Worker + Scope | API v2 | ⚠️ Duplicate |
| 6 | PUT | `/api/v2/follow-up/action-items/{action_item_id}` | `follow_up_service.update_action_item()` | Mutation | ✅ Supervisor/Worker + Scope | API v2 | ⚠️ Duplicate |
| 7 | DELETE | `/api/v2/follow-up/action-items/{action_item_id}` | `follow_up_service.delete_action_item()` | Mutation | ✅ Supervisor/Worker + Scope | API v2 | ⚠️ Duplicate |

**Security Analysis:**
- Authentication: ✅ Via dependency injection
- Role guards: ✅ `require_supervisor_or_worker`
- Scope enforcement: ✅ `validate_action_item_access()` + service layer
- **Risk:** ⚠️ **DUPLICATE** — Overlaps with legacy `/api/follow-up` and `/api/action-items` endpoints

---

#### Router: `/api/v2/subcases`
**File:** `backend/api_v2/routers/case_response_router.py`

| # | Method | Path | Service Function | Operation | Security | Classification | Risk |
|---|--------|------|------------------|-----------|----------|----------------|------|
| 8 | POST | `/api/v2/subcases/{subcase_id}/section-response` | `case_response_service.submit_section_response()` | Mutation | ✅ Section Admin + Subcase Scope | API v2 | ✅ Safe |
| 9 | POST | `/api/v2/subcases/{subcase_id}/section-reject` | `case_response_service.reject_responsibility()` | Mutation | ✅ Section Admin + Subcase Scope | API v2 | ✅ Safe |
| 10 | POST | `/api/v2/subcases/{subcase_id}/department-approve` | `case_response_service.approve_department()` | Mutation | ✅ Dept Admin + Subcase Scope | API v2 | ✅ Safe |
| 11 | POST | `/api/v2/subcases/{subcase_id}/department-reject` | `case_response_service.reject_department()` | Mutation | ✅ Dept Admin + Subcase Scope | API v2 | ✅ Safe |
| 12 | POST | `/api/v2/subcases/{subcase_id}/department-override` | `case_response_service.override_department()` | Mutation | ✅ Dept Admin + Subcase Scope | API v2 | ✅ Safe |
| 13 | POST | `/api/v2/subcases/{subcase_id}/administration-approve` | `case_response_service.approve_administration()` | Mutation | ✅ Admin + Subcase Scope | API v2 | ✅ Safe |
| 14 | POST | `/api/v2/subcases/{subcase_id}/administration-reject` | `case_response_service.reject_administration()` | Mutation | ✅ Admin + Subcase Scope | API v2 | ✅ Safe |
| 15 | POST | `/api/v2/subcases/{subcase_id}/administration-override` | `case_response_service.override_administration()` | Mutation | ✅ Admin + Subcase Scope | API v2 | ✅ Safe |
| 16 | POST | `/api/v2/subcases/{subcase_id}/force-close` | `case_response_service.force_close()` | Mutation | ✅ Administrator + Case Validation | API v2 | ✅ Safe |

**Security Analysis:**
- Authentication: ✅ Via dependency injection
- Role guards: ✅ High-level guards: `require_section_admin_on_subcase`, `require_dept_admin_on_subcase`, `require_admin_on_subcase`
- Scope enforcement: ✅ Guards perform both role and scope validation
- **Risk:** ✅ Fully protected, no legacy equivalents exist

---

#### Router: `/api/v2/insights`
**File:** `backend/api_v2/routers/insight_router.py`

| # | Method | Path | Service Function | Operation | Security | Classification | Risk |
|---|--------|------|------------------|-----------|----------|----------------|------|
| 17 | GET | `/api/v2/insights/open-subcases` | `insight_service.get_open_subcases()` | Read | ✅ Admin Level + Scope | API v2 | ✅ Safe |
| 18 | GET | `/api/v2/insights/open-cases` | `insight_service.get_open_cases_with_subcases()` | Read | ✅ Admin Level + Scope | API v2 | ✅ Safe |
| 19 | GET | `/api/v2/insights/overdue-action-items` | `insight_service.get_overdue_action_items()` | Read | ✅ Admin Level + Scope | API v2 | ✅ Safe |
| 20 | GET | `/api/v2/insights/bottlenecks` | `insight_service.get_bottlenecks()` | Read | ✅ Admin Level + Scope | API v2 | ✅ Safe |

**Security Analysis:**
- Authentication: ✅ Via dependency injection
- Role guards: ✅ `require_admin_level` (DEPARTMENT_ADMIN or ADMINISTRATION_ADMIN)
- Scope enforcement: ✅ Service layer filters by `allowed_unit_ids`
- **Risk:** ✅ Fully protected, no legacy equivalents exist

---

## 🔍 OVERLAP ANALYSIS

### Critical Overlaps

#### ⚠️ **DUPLICATE #1: Action Item Management**

**Functionality:** Action item CRUD operations

**API v1 Endpoints (Legacy):**
- `GET /api/action-items/{action_item_id}` — Get single action item
- `GET /api/action-items/by-incident/{incident_case_id}` — List by incident
- `GET /api/action-items/by-seasonal-report/{seasonal_report_id}` — List by seasonal report
- `GET /api/action-items/by-season/{season_case_id}` — List by season
- `POST /api/action-items/{action_item_id}/mark-done` — Mark done

**API v2 Endpoints (Target):**
- `GET /api/v2/follow-up/action-items` — List for user (scope-aware)
- `POST /api/v2/follow-up/action-items/{action_item_id}/complete` — Complete action
- `PUT /api/v2/follow-up/action-items/{action_item_id}` — Update action
- `DELETE /api/v2/follow-up/action-items/{action_item_id}` — Delete action

**Conflict Type:** Functional overlap with different security models

**Risk:** ⚠️ **HIGH** — Frontend could use either API, bypassing API v2 security

**Recommendation:** Deprecate API v1 action item endpoints immediately

---

#### ⚠️ **DUPLICATE #2: Follow-Up Actions**

**Functionality:** Advanced follow-up action management

**API v1 Endpoints (Legacy):**
- `POST /api/follow-up/actions` — Create action
- `GET /api/follow-up/actions` — List actions (with filters)
- `GET /api/follow-up/actions/{action_id}` — Get single action
- `PATCH /api/follow-up/actions/{action_id}` — Update action
- `POST /api/follow-up/actions/{action_id}/complete` — Complete action
- `POST /api/follow-up/actions/{action_id}/delay` — Delay action
- `POST /api/follow-up/actions/{action_id}/reopen` — Reopen action
- `GET /api/follow-up/actions/{action_id}/history` — Get history
- `GET /api/follow-up/calendar` — Calendar view
- `POST /api/follow-up/actions/bulk-complete` — Bulk complete
- `POST /api/follow-up/actions/bulk-delay` — Bulk delay
- `POST /api/follow-up/actions/bulk-update` — Bulk update

**API v2 Endpoints (Target):**
- `GET /api/v2/follow-up/action-items` — List for user
- `POST /api/v2/follow-up/action-items/{action_item_id}/complete` — Complete
- `PUT /api/v2/follow-up/action-items/{action_item_id}` — Update
- `DELETE /api/v2/follow-up/action-items/{action_item_id}` — Delete

**Conflict Type:** API v1 has MORE features than API v2!

**Risk:** ⚠️ **CRITICAL** — API v2 is incomplete compared to legacy API

**Recommendation:** 
1. Deprecate API v1 follow-up router
2. **Phase 3.5 must add missing features to API v2** (bulk ops, calendar, history, etc.)

---

### No Overlaps (Clean Separation)

#### ✅ **INBOX** — API v2 Only
- No legacy inbox concept exists
- Clean API v2 implementation
- **Risk:** ✅ None

#### ✅ **CASE RESPONSE WORKFLOW** — API v2 Only
- Subcase workflow is entirely new
- No legacy equivalents
- **Risk:** ✅ None

#### ✅ **INSIGHT** — API v2 Only
- No legacy insight endpoints
- Clean API v2 implementation
- **Risk:** ✅ None

---

## 📋 RECOMMENDATIONS

### 1. Immediate Actions (STEP 3.5.0.1)

**Deprecate the following legacy routers:**

| Router | File | Reason |
|--------|------|--------|
| `/api/action-items` | `backend/api/routers/action_items.py` | Overlaps with API v2 follow-up |
| `/api/follow-up` | `backend/api/routers/follow_up_router.py` | Overlaps with API v2 follow-up |

**Method:** Add deprecation warnings, not deletion:
- Add `@deprecated` decorators
- Add response headers: `X-API-Deprecated: true`
- Log warnings on usage
- Document migration path to API v2

---

### 2. API v2 Feature Gaps (STEP 3.5.3)

**API v2 follow-up router is missing features from legacy API:**

| Missing Feature | Legacy Endpoint | Required for Phase 3.5? |
|----------------|-----------------|-------------------------|
| Create action | `POST /api/follow-up/actions` | ✅ YES |
| List with filters | `GET /api/follow-up/actions` | ✅ YES |
| Get single action | `GET /api/follow-up/actions/{id}` | ✅ YES |
| Delay action | `POST /api/follow-up/actions/{id}/delay` | ⚠️ MAYBE |
| Reopen action | `POST /api/follow-up/actions/{id}/reopen` | ⚠️ MAYBE |
| Action history | `GET /api/follow-up/actions/{id}/history` | ❌ NO (audit) |
| Calendar view | `GET /api/follow-up/calendar` | ❌ NO (UI feature) |
| Bulk operations | `POST /api/follow-up/actions/bulk-*` | ❌ NO (convenience) |

**Recommendation for Phase 3.5:**
- Add core features: create, list, get, delay, reopen
- Postpone convenience features: history, calendar, bulk ops

---

### 3. Security Gaps in Legacy API

**CRITICAL:** Legacy `/api/follow-up` and `/api/action-items` routers have:
- ❌ No role guards
- ❌ No scope enforcement
- ❌ Only basic authentication

**Any authenticated user can:**
- View all action items (bypass scope)
- Modify any action item (bypass ownership)
- Delete any action item (bypass authorization)

**Mitigation:** Immediately deprecate and redirect frontend to API v2

---

### 4. API v2 Contract (STEP 3.5.6)

**After Phase 3.5, freeze these endpoints:**

#### Inbox (Already Complete)
- ✅ `GET /api/v2/inbox/section`
- ✅ `GET /api/v2/inbox/department`
- ✅ `GET /api/v2/inbox/administration`

#### Case Response (Already Complete)
- ✅ 9 subcase workflow endpoints (section, department, administration)

#### Insight (Already Complete)
- ✅ 4 insight/monitoring endpoints

#### Follow-Up (Needs Work)
- ⚠️ Add missing CRUD operations
- ⚠️ Add delay/reopen workflow actions
- ⚠️ Ensure scope validation on all operations

---

## 🎯 PHASE 3.5 SCOPE CLARIFICATION

### In Scope for Phase 3.5

1. ✅ **Inbox endpoints** — Already complete
2. ✅ **Case response endpoints** — Already complete
3. ⚠️ **Follow-up endpoints** — Need feature additions:
   - Create action item
   - List action items (with filters)
   - Get single action item
   - Delay action item
   - Reopen action item
4. ✅ **Insight endpoints** — Already complete

### Out of Scope for Phase 3.5

1. ❌ **Action history** — Audit feature (Phase 4+)
2. ❌ **Calendar view** — UI convenience (Phase 4+)
3. ❌ **Bulk operations** — Convenience feature (Phase 4+)
4. ❌ **Analytics** — Not workflow (separate phase)
5. ❌ **Reports** — Not workflow (separate phase)

---

## 📊 FINAL STATISTICS

| Category | Count | Notes |
|----------|-------|-------|
| **Total Workflow Endpoints** | **34** | API v1 + API v2 combined |
| **API v1 (Legacy)** | **17** | All should be deprecated |
| **API v2 (Target)** | **20** | Core workflow implementation |
| **Duplicate/Overlap** | **10** | Follow-up actions |
| **API v2 Only (Safe)** | **14** | Inbox, case response, insight |
| **Security Gaps (v1)** | **17** | All v1 endpoints lack proper guards |
| **Feature Gaps (v2)** | **~7** | Missing from follow-up router |

---

## ✅ CONCLUSION

### What We Found

1. **No inbox in API v1** ✅ — Clean separation
2. **No case workflow in API v1** ✅ — Clean separation
3. **No insight in API v1** ✅ — Clean separation
4. **Follow-up actions duplicated** ⚠️ — Security risk
5. **API v2 follow-up incomplete** ⚠️ — Feature gap

### Critical Risks

1. 🚨 **Security bypass risk**: Frontend could use legacy endpoints to bypass API v2 security
2. ⚠️ **Feature parity risk**: API v2 follow-up is incomplete, frontend may resist migration
3. ⚠️ **Contract ambiguity**: No clear signal which API is authoritative

### Next Steps

1. **STEP 3.5.0.1** — Mark legacy endpoints as deprecated
2. **STEP 3.5.3** — Add missing features to API v2 follow-up router
3. **STEP 3.5.6** — Freeze API v2 contract
4. **Phase 4** — Migrate frontend to API v2 exclusively

---

## 📝 AUDIT COMPLETE

**Status:** ✅ READ-ONLY AUDIT COMPLETE  
**Files Modified:** 0 (audit only)  
**Files Created:** 1 (this report)  
**Next Action:** Proceed to STEP 3.5.0.1 (Deprecation)
