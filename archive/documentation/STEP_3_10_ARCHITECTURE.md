# STEP 3.10 — ADAPTER ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LEGACY API v1 FLOW                              │
│                      (UNCHANGED BEHAVIOR)                               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  INCIDENT CREATION PATH                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Frontend POST → /insert                                                │
│       ↓                                                                 │
│  insert_service.create_record(data)                                     │
│       ↓                                                                 │
│  ┌─────────────────────────────────────┐                               │
│  │ 1. Validate data                     │                               │
│  │ 2. Create incident record            │ ← LEGACY FLOW                 │
│  │ 3. Add target departments            │   (UNTOUCHED)                 │
│  │ 4. Add doctors                       │                               │
│  │ 5. ML hook (safe)                    │                               │
│  └─────────────────────────────────────┘                               │
│       ↓                                                                 │
│  ┌─────────────────────────────────────┐                               │
│  │ 🆕 API V2 ADAPTER HOOK               │                               │
│  │ ═══════════════════════════════════ │                               │
│  │ try:                                 │ ← ADDED IN STEP 3.10          │
│  │   create_subcases_for_incident()     │   NON-BLOCKING                │
│  │ except:                              │   SAFE                        │
│  │   log_warning()                      │   IDEMPOTENT                  │
│  └─────────────────────────────────────┘                               │
│       ↓                                                                 │
│  Return success to frontend             │                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  SEASONAL REPORT GENERATION PATH                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Frontend GET → /seasonal-report                                        │
│       ↓                                                                 │
│  seasonal_report_orchestrator.get_or_generate_seasonal_report()         │
│       ↓                                                                 │
│  seasonal_report_generator.generate_or_regenerate_report()              │
│       ↓                                                                 │
│  ┌─────────────────────────────────────┐                               │
│  │ 1. Aggregate data from incidents     │                               │
│  │ 2. Calculate domain totals           │ ← LEGACY FLOW                 │
│  │ 3. Evaluate policy compliance        │   (UNTOUCHED)                 │
│  │ 4. Insert/update report header       │                               │
│  │ 5. Insert classification stats       │                               │
│  │ 6. Insert policy snapshot            │                               │
│  └─────────────────────────────────────┘                               │
│       ↓                                                                 │
│  ┌─────────────────────────────────────┐                               │
│  │ 🆕 API V2 ADAPTER HOOK               │                               │
│  │ ═══════════════════════════════════ │                               │
│  │ try:                                 │ ← ADDED IN STEP 3.10          │
│  │   create_subcases_for_seasonal_      │   NON-BLOCKING                │
│  │   report()                           │   SAFE                        │
│  │ except:                              │   IDEMPOTENT                  │
│  │   log_warning()                      │                               │
│  └─────────────────────────────────────┘                               │
│       ↓                                                                 │
│  Return report to frontend              │                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          API V2 WORKFLOW                                │
│                     (TRIGGERED AUTOMATICALLY)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  case_creation_service.create_subcases_for_incident(incident_id)        │
│       ↓                                                                 │
│  ┌─────────────────────────────────────┐                               │
│  │ 1. Check if subcases already exist   │ ← IDEMPOTENCY                 │
│  │    → If yes, return (no-op)          │   CHECK                       │
│  └─────────────────────────────────────┘                               │
│       ↓                                                                 │
│  ┌─────────────────────────────────────┐                               │
│  │ 2. Query target departments          │                               │
│  │    FROM APP_IncidentCaseTarget       │                               │
│  │    Department                        │                               │
│  └─────────────────────────────────────┘                               │
│       ↓                                                                 │
│  ┌─────────────────────────────────────┐                               │
│  │ 3. For each target department:       │                               │
│  │    → Create Administrative_Subcase   │ ← NEW DATA                    │
│  │    → CaseType = INCIDENT_RESPONSE    │   CREATED                     │
│  │    → Status = SUBMITTED_TO_SECTION   │                               │
│  │    → AssignedToRole = SECTION_ADMIN  │                               │
│  └─────────────────────────────────────┘                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          API V2 WORKFLOW                                │
│                     (TRIGGERED AUTOMATICALLY)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  case_creation_service.create_subcases_for_seasonal_report(report_id)   │
│       ↓                                                                 │
│  ┌─────────────────────────────────────┐                               │
│  │ 1. Check if subcases already exist   │ ← IDEMPOTENCY                 │
│  │    → If yes, return (no-op)          │   CHECK                       │
│  └─────────────────────────────────────┘                               │
│       ↓                                                                 │
│  ┌─────────────────────────────────────┐                               │
│  │ 2. Query target org units            │                               │
│  │    (from policy violations)          │                               │
│  └─────────────────────────────────────┘                               │
│       ↓                                                                 │
│  ┌─────────────────────────────────────┐                               │
│  │ 3. For each violating org unit:      │                               │
│  │    → Create Administrative_Subcase   │ ← NEW DATA                    │
│  │    → CaseType = SEASONAL_REPORT_     │   CREATED                     │
│  │      RESPONSE                        │                               │
│  │    → Status = SUBMITTED_TO_SECTION   │                               │
│  │    → AssignedToRole = SECTION_ADMIN  │                               │
│  └─────────────────────────────────────┘                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

════════════════════════════════════════════════════════════════════════════

KEY DESIGN DECISIONS:

✅ SAFETY RULES FOLLOWED:
   - ❌ NO refactoring of legacy code
   - ❌ NO changes to business logic
   - ❌ NO changes to return values
   - ❌ NO changes to transactions
   - ✅ ONLY added adapter calls

✅ ADAPTER GUARANTEES:
   - Non-blocking: Failures don't break legacy flow
   - Idempotent: Can be called multiple times safely
   - Isolated: No side effects on legacy operations
   - Logged: All failures are logged for debugging

✅ RESULT:
   - Legacy API v1 → Works exactly as before
   - API v2 workflow → Triggered automatically
   - Data consistency → Both systems stay in sync
   - Zero downtime → Can deploy without service interruption

════════════════════════════════════════════════════════════════════════════

ADAPTER EXECUTION FLOW:

┌──────────────────────┐
│ Legacy Operation     │
│ Completes            │
│ Successfully         │
└──────────────────────┘
          ↓
┌──────────────────────┐
│ Adapter Hook         │
│ Triggered            │
└──────────────────────┘
          ↓
    ┌─────────┐
    │ Success?│
    └─────────┘
     ↙         ↘
   YES         NO
    ↓           ↓
┌────────┐  ┌────────┐
│Subcases│  │ Log    │
│Created │  │Warning │
└────────┘  └────────┘
    ↓           ↓
    └─────┬─────┘
          ↓
┌──────────────────────┐
│ Legacy Flow          │
│ Continues            │
│ (Unaffected)         │
└──────────────────────┘
          ↓
┌──────────────────────┐
│ Return Success       │
│ to Frontend          │
└──────────────────────┘

════════════════════════════════════════════════════════════════════════════
