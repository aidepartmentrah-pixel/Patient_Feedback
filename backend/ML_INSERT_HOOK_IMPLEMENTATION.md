"""
ML INSERT HOOK INTEGRATION - IMPLEMENTATION SUMMARY

Date: 2026-01-02
Status: ✓ COMPLETE

================================================================================
OVERVIEW
================================================================================

The create_record() service has been modified to call the ML insertion logic
after the main database insert succeeds, without affecting stability or return
behavior.

This change is purely additive — no existing logic was refactored or removed.

================================================================================
FILES MODIFIED
================================================================================

1. backend/api/services/insert_service.py
   - Added ML insert hook (lines 230-238)
   - Placed immediately after create_incident_case(payload)
   - Never blocks or affects main transaction

Files Created (pre-requisites):
- backend/config/ml_encoding_mapper.py (mapping loader)
- backend/ml_mapping/__init__.py (package init)
- backend/ml_mapping/ml_insert_adapter.py (insert logic)
- backend/test_ml_hook_integration.py (test suite)

================================================================================
IMPLEMENTATION DETAILS
================================================================================

Location: backend/api/services/insert_service.py (lines 230-238)

Code:
    # -------------------------------------------
    # ML INSERT HOOK (SAFE / NON-BLOCKING)
    # -------------------------------------------
    try:
        from backend.ml_mapping.ml_insert_adapter import add_to_ml_database
        add_to_ml_database(data)
    except Exception as e:
        # Log only — never interrupt main flow
        print(f"[ML INSERT WARNING] {str(e)}")

Placement:
    [BEFORE]
        new_id = create_incident_case(payload)
        # Related tables
        if data.get('target_department_ids'):
            ...

    [AFTER]
        new_id = create_incident_case(payload)
        
        # ML INSERT HOOK (NEW)
        try:
            from backend.ml_mapping.ml_insert_adapter import add_to_ml_database
            add_to_ml_database(data)
        except Exception as e:
            print(f"[ML INSERT WARNING] {str(e)}")
        
        # Related tables
        if data.get('target_department_ids'):
            ...

================================================================================
GUARANTEES
================================================================================

✓ Main system stability is preserved
  - Main DB insert always succeeds or fails on its own merits
  - ML layer cannot cause main insert to fail
  - Return value is never modified

✓ ML system is optional
  - ML DB can be missing, corrupted, or offline
  - Service continues normally
  - Only print() warnings logged

✓ Backward compatible
  - Existing API consumers see no change
  - Return structure unchanged
  - No new required fields

✓ Production safe
  - No retries
  - No async behavior
  - No exception propagation
  - No data modification

================================================================================
REQUIRED TESTING
================================================================================

Test Suite: backend/test_ml_hook_integration.py

Run with:
    cd backend
    python test_ml_hook_integration.py

Tests Included:

1. TEST 1: Normal Insert
   - Verify main DB insert succeeds
   - Verify ML insert runs
   - Check ML DB for new row
   ✓ Expected: Both succeed, API returns normal response

2. TEST 2: No Exceptions Propagate
   - Verify partial data is handled
   - Verify no unhandled exceptions escape
   ✓ Expected: Graceful validation, no crashes

3. TEST 3: Return Value Unchanged
   - Verify response structure is intact
   - Check all expected keys present
   ✓ Expected: Full response with success=true, record_id, etc.

Manual Testing Steps:

1. Insert a valid record via API:
   curl -X POST http://localhost:8000/api/records \
     -H "Content-Type: application/json" \
     -d '{
       "complaint_text": "Test complaint",
       "feedback_received_date": "2026-01-02",
       "issuing_department_id": 1,
       "domain_id": 1,
       "category_id": 1,
       "subcategory_id": 1,
       "classification_id": 1,
       "severity_id": 1,
       "stage_id": 1,
       "harm_id": 1,
       "patient_name": "Test Patient"
     }'

2. Check main DB:
   SELECT * FROM main_db.incidents WHERE id = <new_id>;
   ✓ Should see the record

3. Check ML DB:
   SELECT * FROM patient_feedback_ml.patient_feedback_encoded 
   WHERE record_id = <new_id>;
   ✓ Should see mapped record

4. Verify no console errors:
   [No "[ML INSERT WARNING]" messages should appear]

================================================================================
BEHAVIOR BY SCENARIO
================================================================================

Scenario 1: Normal Operation
├─ Main DB insert succeeds
├─ ML insert succeeds
├─ No console warnings
└─ API response: success=true ✓

Scenario 2: Main DB Fails
├─ Main DB insert fails
├─ ML insert never runs (exception caught earlier)
└─ API response: error message ✓

Scenario 3: ML DB Missing
├─ Main DB insert succeeds
├─ ML insert logs warning: [ML INSERT WARNING] Could not connect to ML database
└─ API response: success=true ✓

Scenario 4: ML Insert Fails (e.g., mapping error)
├─ Main DB insert succeeds
├─ ML insert logs warning: [ML INSERT WARNING] <error_message>
└─ API response: success=true ✓

Scenario 5: Partial/Invalid Data
├─ Main DB insert succeeds (with NULL fields)
├─ ML insert skips missing mapping fields
└─ API response: success=true ✓

================================================================================
MONITORING
================================================================================

Console Output Format:
    [ML INSERT WARNING] <error_message>

Warnings to Monitor:
    - "[ML INSERT WARNING] Could not connect to ML database at ..."
      → ML DB offline or path wrong
    
    - "[ML INSERT WARNING] Error during ML database insert: ..."
      → Insert failed (unique constraint, etc.)
    
    - "[ML INSERT WARNING] Failed to insert row into ML database: ..."
      → Individual row insert failed

Expected in Logs During Normal Operation:
    [None - no warnings expected]

================================================================================
INTEGRATION CHECKLIST
================================================================================

✓ Hook added to create_record()
✓ Hook placed correctly (after main insert, before return)
✓ Try/except wraps entire ML operation
✓ No exceptions propagate to main flow
✓ Return value unchanged
✓ Data passed as-is (no modification)
✓ ML database remains optional
✓ Test suite created
✓ Documentation complete

================================================================================
TROUBLESHOOTING
================================================================================

Problem: "[ML INSERT WARNING] ModuleNotFoundError: No module named 'backend.ml_mapping'"

Solution:
    1. Verify backend/ml_mapping/ directory exists
    2. Verify __init__.py exists in backend/ml_mapping/
    3. Verify ml_insert_adapter.py exists
    4. Check Python path includes project root

Problem: "[ML INSERT WARNING] Could not connect to ML database at ..."

Solution:
    1. Verify patient_feedback_ml.db exists at path shown
    2. Verify database is not corrupted (try opening with sqlite3)
    3. Verify file permissions allow read/write
    4. Check disk space

Problem: "[ML INSERT WARNING] Failed to insert row into ML database: ..."

Solution:
    1. Check ML database schema matches KNOWN_COLUMNS in adapter
    2. Verify mapping file Database_To_ML_Encoding.json is valid
    3. Check for duplicate records (unique constraints)
    4. Examine specific error message

Problem: API returns success but ML row not created

Solution:
    1. Check console for "[ML INSERT WARNING]" messages
    2. Verify ML database path is correct
    3. Run manual query on patient_feedback_encoded table
    4. Check database permissions

================================================================================
NEXT STEPS
================================================================================

1. Run test suite: python backend/test_ml_hook_integration.py
2. Verify console shows no warnings during normal operation
3. Monitor ML database for rows appearing after inserts
4. Set up alerts for "[ML INSERT WARNING]" in logs
5. Begin training pipeline runs with accumulated ML data

================================================================================
DESIGN RATIONALE
================================================================================

Why Try/Except?
    - ML system is secondary, never blocks main transaction
    - Graceful degradation if ML DB unavailable
    - Production stability guarantee

Why print() Instead of Logging?
    - Keeps code simple and dependency-free
    - Warnings appear in console immediately
    - No logging framework coupling

Why After create_incident_case()?
    - Main transaction already committed by db_layer
    - No risk of rollback affecting ML insert
    - Clean separation of concerns

Why Pass Original Data?
    - Mapping happens inside ML adapter
    - Main service doesn't know about ML encoding
    - Keeps concerns separated

Why No Retries?
    - Transient failures are rare for local SQLite
    - ML data can be backfilled if needed
    - Simplicity over complexity

================================================================================
END SUMMARY
================================================================================
"""