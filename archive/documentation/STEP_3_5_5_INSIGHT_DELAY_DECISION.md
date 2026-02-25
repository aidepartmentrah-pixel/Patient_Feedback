# STEP 3.5.5 — Insight Delay Decision

## Decision Summary

**Insight endpoints are explicitly delayed from API v2.**

This is an intentional architectural decision, not missing work.

---

## Rationale

Insight is a **read-only analytical surface** that provides:
- Case summaries and counts
- KPI dashboards
- Open case aggregations
- Bottleneck detection
- Overdue action item monitoring

These capabilities depend on:

1. **Stable workflow state transitions** — Case lifecycle must be frozen and validated
2. **Stable scope enforcement** — Organizational access rules must be production-tested
3. **Frozen API contracts** — Endpoint paths and response shapes must be finalized
4. **Clear KPI semantics** — Business rules for "open", "overdue", "bottleneck" must be agreed upon

**These conditions are not yet met in Phase 3.5.**

---

## What Was Removed

The following files were removed from API v2:

- `backend/api_v2/routers/insight_router.py` — Insight router with 4 endpoints
- `backend/api_v2/services/insight_service.py` — Insight service layer

These endpoints were never registered in `main.py`, so no API surface was exposed.

---

## What is NOT Implemented

API v2 does **not** include:

- `/api/v2/insights/open-subcases` — List all open subcases
- `/api/v2/insights/open-cases` — Case-centric view with subcases
- `/api/v2/insights/overdue-action-items` — Overdue action items list
- `/api/v2/insights/bottlenecks` — Stuck subcases detection

No placeholder routers, fake data, or partial aggregations are exposed.

---

## API v2 Scope (Phase 3.5)

API v2 **only** includes:

✅ **Workflow API** (`/api/v2/workflow/*`)
- Inbox endpoint (role-aware)
- Follow-up endpoints (4 endpoints)
- Case action endpoint (unified action dispatcher)

These 6 endpoints constitute the **complete and frozen** API v2 surface for Phase 3.5.

---

## When Will Insight Be Implemented?

Insight will be implemented in a **future phase** after:

1. **Workflow contract freeze** — Current API v2 endpoints stabilize in production
2. **Scope engine validation** — Organizational access rules are verified under load
3. **Production usage feedback** — Real user workflows inform KPI requirements
4. **Business rule clarity** — Stakeholders agree on exact definitions of "open", "overdue", "stuck"

Estimated timeline: Post-Phase 4 (after legacy API v1 deprecation)

---

## Verification

Stop conditions met:

- ✅ No `insight_router.py` exists in API v2
- ✅ No Insight endpoints registered in `main.py`
- ✅ API v2 contract frozen without Insight (6 workflow endpoints only)

---

## Status

**STEP 3.5.5 COMPLETE — Insight formally delayed by design.**

Next step: STEP 3.5.6 — Freeze API v2 contract
