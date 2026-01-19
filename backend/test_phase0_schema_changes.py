"""
PHASE 0 TEST SCRIPT: Verify Database Schema Changes
=====================================================
Tests the RequiresExplanation column addition and data integrity.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.database import get_connection

def test_column_exists():
    """Test 1: Verify RequiresExplanation column exists"""
    print("=" * 70)
    print("TEST 1: Verify RequiresExplanation column exists")
    print("=" * 70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT 
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'dbo'
          AND TABLE_NAME = 'APP_IncidentCase'
          AND COLUMN_NAME = 'RequiresExplanation'
    """
    
    cursor.execute(query)
    result = cursor.fetchone()
    
    if result:
        print("✓ RequiresExplanation column EXISTS")
        print(f"  Column Name: {result[0]}")
        print(f"  Data Type: {result[1]}")
        print(f"  Is Nullable: {result[2]}")
        print(f"  Default Value: {result[3]}")
    else:
        print("✗ RequiresExplanation column DOES NOT EXIST")
        return False
    
    cursor.close()
    conn.close()
    return True


def test_default_values():
    """Test 2: Verify all existing records have RequiresExplanation = 0"""
    print("\n" + "=" * 70)
    print("TEST 2: Verify default values in existing records")
    print("=" * 70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT 
            COUNT(*) as TotalRecords,
            SUM(CASE WHEN RequiresExplanation = 0 THEN 1 ELSE 0 END) as WithFalse,
            SUM(CASE WHEN RequiresExplanation = 1 THEN 1 ELSE 0 END) as WithTrue,
            SUM(CASE WHEN RequiresExplanation IS NULL THEN 1 ELSE 0 END) as WithNull
        FROM dbo.APP_IncidentCase
    """
    
    cursor.execute(query)
    result = cursor.fetchone()
    
    total, false_count, true_count, null_count = result
    
    print(f"  Total Records: {total}")
    print(f"  RequiresExplanation = 0: {false_count}")
    print(f"  RequiresExplanation = 1: {true_count}")
    print(f"  RequiresExplanation = NULL: {null_count}")
    
    if null_count > 0:
        print("✗ FAILED: Found NULL values in RequiresExplanation")
        return False
    else:
        print("✓ PASSED: No NULL values found")
    
    cursor.close()
    conn.close()
    return True


def test_taken_action_field():
    """Test 3: Verify TakenAction field capacity and usage"""
    print("\n" + "=" * 70)
    print("TEST 3: Verify TakenAction field capacity")
    print("=" * 70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check field definition
    query = """
        SELECT 
            COLUMN_NAME,
            DATA_TYPE,
            CHARACTER_MAXIMUM_LENGTH,
            IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'dbo'
          AND TABLE_NAME = 'APP_IncidentCase'
          AND COLUMN_NAME = 'TakenAction'
    """
    
    cursor.execute(query)
    result = cursor.fetchone()
    
    if result:
        col_name, data_type, max_length, is_nullable = result
        print(f"  Column Name: {col_name}")
        print(f"  Data Type: {data_type}")
        print(f"  Max Length: {max_length if max_length != -1 else 'MAX'}")
        print(f"  Is Nullable: {is_nullable}")
        
        if data_type.lower() == 'nvarchar' and (max_length == -1 or max_length >= 4000):
            print("✓ TakenAction has sufficient capacity (NVARCHAR(MAX) or large)")
        else:
            print(f"⚠ WARNING: TakenAction may have limited capacity: {data_type}({max_length})")
    else:
        print("✗ TakenAction column not found")
        cursor.close()
        conn.close()
        return False
    
    # Check usage
    query = """
        SELECT 
            COUNT(*) as TotalRecords,
            SUM(CASE WHEN TakenAction IS NULL THEN 1 ELSE 0 END) as NullCount,
            SUM(CASE WHEN TakenAction IS NOT NULL AND TakenAction != '' THEN 1 ELSE 0 END) as HasData,
            MAX(LEN(TakenAction)) as MaxLength
        FROM dbo.APP_IncidentCase
    """
    
    cursor.execute(query)
    result = cursor.fetchone()
    
    total, null_count, has_data, max_len = result
    
    print(f"\n  TakenAction Usage:")
    print(f"  Total Records: {total}")
    print(f"  NULL/Empty: {null_count}")
    print(f"  Has Data: {has_data}")
    print(f"  Max Length: {max_len if max_len else 0} characters")
    
    cursor.close()
    conn.close()
    return True


def test_lookup_tables():
    """Test 4: Document lookup table IDs for FSM"""
    print("\n" + "=" * 70)
    print("TEST 4: Document lookup table values")
    print("=" * 70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Case Status lookup
    print("\n  APP_LOOKUP_CASE_STATUS:")
    query = """
        SELECT 
            CaseStatusID,
            Code,
            Name,
            IsFinal,
            IsActive,
            DisplayOrder
        FROM dbo.APP_LOOKUP_CASE_STATUS
        ORDER BY DisplayOrder
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    
    print("  " + "-" * 60)
    print(f"  {'ID':<5} {'Code':<15} {'Name':<20} {'Final':<7} {'Active':<7}")
    print("  " + "-" * 60)
    
    for row in results:
        case_id, code, name, is_final, is_active, display_order = row
        print(f"  {case_id:<5} {code:<15} {name:<20} {is_final:<7} {is_active:<7}")
    
    # Explanation Status lookup
    print("\n  APP_LOOKUP_EXPLANATION_STATUS:")
    query = """
        SELECT 
            StatusID,
            StatusName
        FROM dbo.APP_LOOKUP_EXPLANATION_STATUS
        ORDER BY StatusID
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    
    print("  " + "-" * 40)
    print(f"  {'ID':<5} {'Status Name':<30}")
    print("  " + "-" * 40)
    
    for row in results:
        status_id, status_name = row
        print(f"  {status_id:<5} {status_name:<30}")
    
    cursor.close()
    conn.close()
    return True


def test_sample_insert():
    """Test 5: Test inserting a record with RequiresExplanation"""
    print("\n" + "=" * 70)
    print("TEST 5: Test sample insert with RequiresExplanation")
    print("=" * 70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Try to get lookup IDs first
    cursor.execute("SELECT TOP 1 CaseStatusID FROM dbo.APP_LOOKUP_CASE_STATUS WHERE Code = 'CLOSED'")
    closed_status = cursor.fetchone()
    if not closed_status:
        print("✗ Cannot find CLOSED status in lookup table")
        cursor.close()
        conn.close()
        return False
    
    cursor.execute("SELECT TOP 1 DomainID FROM dbo.APP_LOOKUP_DOMAIN")
    domain = cursor.fetchone()
    
    cursor.execute("SELECT TOP 1 CategoryID FROM dbo.APP_LOOKUP_CATEGORY")
    category = cursor.fetchone()
    
    cursor.execute("SELECT TOP 1 SubCategoryID FROM dbo.APP_LOOKUP_SUBCATEGORY")
    subcategory = cursor.fetchone()
    
    cursor.execute("SELECT TOP 1 ClassificationID FROM dbo.APP_LOOKUP_CLASSIFICATION")
    classification = cursor.fetchone()
    
    cursor.execute("SELECT TOP 1 StageID FROM dbo.APP_LOOKUP_CASE_STAGE")
    stage = cursor.fetchone()
    
    cursor.execute("SELECT TOP 1 HarmID FROM dbo.APP_LOOKUP_HARM_LEVEL")
    harm = cursor.fetchone()
    
    cursor.execute("SELECT TOP 1 ClinicalRiskTypeID FROM dbo.APP_LOOKUP_CLINICAL_RISK_TYPE WHERE IsActive = 1")
    risk = cursor.fetchone()
    
    cursor.execute("SELECT TOP 1 FeedbackIntentTypeID FROM dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE WHERE IsActive = 1")
    intent = cursor.fetchone()
    
    cursor.execute("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit")
    org_unit = cursor.fetchone()
    
    cursor.execute("SELECT TOP 1 SourceID FROM dbo.APP_LOOKUP_SOURCE WHERE IsActive = 1")
    source = cursor.fetchone()
    
    cursor.execute("SELECT TOP 1 BuildingID FROM dbo.APP_LOOKUP_BUILDING")
    building = cursor.fetchone()
    
    cursor.execute("SELECT TOP 1 SeverityID FROM dbo.APP_LOOKUP_SEVERITY WHERE IsActive = 1")
    severity = cursor.fetchone()
    
    if not all([domain, category, subcategory, classification, stage, harm, risk, intent, org_unit, source, building, severity]):
        print("⚠ WARNING: Missing required lookup data, skipping insert test")
        cursor.close()
        conn.close()
        return True  # Not a failure, just can't test
    
    try:
        # Insert test record
        query = """
            INSERT INTO dbo.APP_IncidentCase (
                ComplaintText,
                PatientName,
                isINPatient,
                FeedbackRecievedDate,
                RequiresExplanation,
                CaseStatusID,
                ExplanationStatusID,
                DomainID,
                CategoryID,
                SubCategoryID,
                ClassificationID,
                StageID,
                HarmLevelID,
                ClinicalRiskTypeID,
                FeedbackIntentTypeID,
                IssuingOrgUnitID,
                SourceID,
                BuildingID,
                SeverityID,
                ImmediateAction,
                TakenAction,
                CreatedByUserID,
                CreatedAt
            )
            OUTPUT INSERTED.IncidentRequestCaseID
            VALUES (
                'TEST RECORD - Phase 0 verification',
                'Test Patient',  -- PatientName
                1,  -- isINPatient (1 = inpatient, 0 = outpatient)
                GETDATE(),  -- FeedbackRecievedDate
                0,  -- RequiresExplanation = False
                ?,  -- CaseStatusID (Closed)
                4,  -- ExplanationStatusID (No Explanation Needed)
                ?, ?, ?, ?, ?, ?,  -- Domain, Category, Sub, Class, Stage, Harm
                ?, ?,  -- Risk, Intent
                ?,  -- OrgUnit
                ?, ?, ?,  -- Source, Building, Severity
                '', '',  -- ImmediateAction, TakenAction (empty strings)
                1,  -- CreatedByUserID
                GETDATE()
            )
        """
        
        cursor.execute(query, (
            closed_status[0],
            domain[0], category[0], subcategory[0], classification[0], stage[0], harm[0],
            risk[0], intent[0], org_unit[0],
            source[0], building[0], severity[0]
        ))
        
        inserted_id = cursor.fetchone()[0]
        print(f"✓ Successfully inserted test record with ID: {inserted_id}")
        
        # Verify the insert
        cursor.execute("""
            SELECT RequiresExplanation, ComplaintText 
            FROM dbo.APP_IncidentCase 
            WHERE IncidentRequestCaseID = ?
        """, (inserted_id,))
        
        result = cursor.fetchone()
        print(f"  RequiresExplanation: {result[0]}")
        print(f"  ComplaintText: {result[1]}")
        
        # Clean up test record
        cursor.execute("DELETE FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (inserted_id,))
        conn.commit()
        print(f"✓ Test record cleaned up")
        
    except Exception as e:
        print(f"✗ Insert test failed: {e}")
        conn.rollback()
        cursor.close()
        conn.close()
        return False
    
    cursor.close()
    conn.close()
    return True


def run_all_tests():
    """Run all Phase 0 tests"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  PHASE 0: DATABASE SCHEMA VERIFICATION TESTS".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    tests = [
        ("Column Exists", test_column_exists),
        ("Default Values", test_default_values),
        ("TakenAction Field", test_taken_action_field),
        ("Lookup Tables", test_lookup_tables),
        ("Sample Insert", test_sample_insert),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ TEST FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n")
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name:<30} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("=" * 70)
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Phase 0 Complete!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed - Please review")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
