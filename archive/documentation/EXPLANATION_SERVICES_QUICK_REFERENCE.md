# Explanation Services Implementation - Quick Reference

## Project Goal
Add explanation collection workflow for Red Flag, Never Event, and selected Ordinary complaints.

---

## Current Implementation Status

### ✅ PHASE 0: Database Schema Changes (READY TO EXECUTE)
- **Status:** Implementation complete, ready for execution
- **Files:** 
  - `backend/database_migrations/phase0_add_requires_explanation.sql`
  - `backend/test_phase0_schema_changes.py`
  - `backend/database_migrations/PHASE0_README.md`

**What it adds:**
- `RequiresExplanation` BIT column to APP_IncidentCase
- Test suite to verify changes

**Next Step:** Execute migration and run tests

---

## Implementation Phases Overview

| Phase | Name | Status | Dependencies |
|-------|------|--------|--------------|
| 0 | Database Schema | ✅ Ready | None |
| 1 | DB Layer - Read | ⏳ Pending | Phase 0 |
| 2 | DB Layer - Write | ⏳ Pending | Phase 1 |
| 3 | Service Layer | ⏳ Pending | Phase 2 |
| 4 | Insert Service Mod | ⏳ Pending | Phase 2 |
| 5 | Action Items Integration | ⏳ Pending | Phase 3 |
| 6 | API Router/Endpoints | ⏳ Pending | Phase 3, 5 |
| 7 | E2E Testing | ⏳ Pending | Phase 6 |

---

## Key Design Decisions Made

### 1. Data Storage
✅ **Decision:** Use existing `TakenAction` field for explanation text
- **Rationale:** Field already exists for this purpose, NVARCHAR(MAX) capacity
- **Alternative rejected:** Creating separate table (unnecessary complexity)

### 2. FSM Logic
✅ **Decision:** Three-path workflow at insert
```
1. Red Flag/Never Event → Always RequiresExplanation = 1
2. Ordinary + checked → RequiresExplanation = 1
3. Ordinary + unchecked → RequiresExplanation = 0 (stays closed)
```

### 3. Approval Workflow
✅ **Decision:** No approval needed
- Explanations are final when submitted

### 4. Action Items
✅ **Decision:** Multiple action items supported per case
- Can be created during explanation submission

---

## Database Schema Reference

### APP_IncidentCase (Modified)
```sql
-- New column added:
RequiresExplanation BIT NOT NULL DEFAULT 0

-- Existing columns used:
TakenAction NVARCHAR(MAX) NULL  -- Stores explanation text
CaseStatusID INT                -- FSM: Open(1), In Progress(2), Closed(3)
```

### Lookup Tables
```sql
-- APP_LOOKUP_CASE_STATUS
1 = OPEN
2 = IN_PROGRESS  
3 = CLOSED

-- APP_LOOKUP_EXPLANATION_STATUS
1 = Waiting
2 = Responded
3 = Forcibly Closed
```

---

## FSM State Transitions

```
INSERT:
  IF (IsRedFlag OR IsNeverEvent OR RequiresExplanation = 1)
    → CaseStatus = Open (1), ExplanationStatus = Waiting (1)
  ELSE
    → CaseStatus = Closed (3), ExplanationStatus = NULL

SUBMIT EXPLANATION:
  Open + Waiting → In Progress + Responded
  (Updates TakenAction field)

COMPLETE ALL ACTION ITEMS:
  In Progress + Responded → Closed + Responded

FORCE CLOSE:
  Any → Closed + Forcibly Closed (admin override)
```

---

## Files Structure (Planned)

```
backend/
├── database_migrations/
│   ├── phase0_add_requires_explanation.sql          ✅ Done
│   └── PHASE0_README.md                             ✅ Done
├── test_phase0_schema_changes.py                    ✅ Done
├── api/
│   ├── db_layer/
│   │   └── explanation_queries.py                   ⏳ Phase 1
│   ├── services/
│   │   ├── explanation_service.py                   ⏳ Phase 3
│   │   └── insert_service.py                        🔧 Phase 4 (modify)
│   └── routes/
│       └── explanation_routes.py                    ⏳ Phase 6
└── tests/
    ├── test_explanation_db_layer.py                 ⏳ Phase 1
    ├── test_explanation_service.py                  ⏳ Phase 3
    └── test_explanation_endpoints.py                ⏳ Phase 6
```

---

## API Endpoints (Planned - Phase 6)

```
GET  /api/explanations/pending
     → List cases needing explanation

GET  /api/explanations/{case_id}
     → Get case details + explanation status

POST /api/explanations/{case_id}
     → Submit explanation + create action items

PUT  /api/incident-case/{case_id}/requires-explanation
     → Admin: Toggle RequiresExplanation flag

GET  /api/explanations/statistics
     → Dashboard metrics
```

---

## Testing Strategy

### Phase 0: Schema Tests
- Column existence
- Default values
- Field capacity
- Insert/update operations

### Phase 1-2: DB Layer Tests
- Query accuracy
- FSM state transitions
- Data integrity
- Edge cases

### Phase 3-4: Service Layer Tests
- Business logic validation
- Permission checks
- Error handling
- Workflow enforcement

### Phase 6: API Tests
- Endpoint functionality
- Request/response validation
- Authentication
- Error responses

### Phase 7: E2E Tests
- Complete workflows
- Multi-user scenarios
- Integration points

---

## Commands Reference

### Execute Migration
```sql
-- In SQL Server Management Studio:
-- Open: backend/database_migrations/phase0_add_requires_explanation.sql
-- Execute (F5)
```

### Run Tests
```bash
# Phase 0 tests
python backend/test_phase0_schema_changes.py

# Future phase tests (example)
python backend/tests/test_explanation_db_layer.py
```

---

## Next Actions

1. **Execute Phase 0 migration**
   - Run SQL script
   - Verify with tests
   
2. **Confirm success**
   - All 5 tests pass
   - Document any issues

3. **Get approval to proceed**
   - Review implementation
   - Proceed to Phase 1

---

## Contact Points / Questions

- Database capacity concerns?
- FSM logic clarifications?
- Timeline expectations?
- Frontend integration requirements?
