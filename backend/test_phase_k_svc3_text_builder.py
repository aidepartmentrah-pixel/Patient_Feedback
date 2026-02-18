"""
PHASE K — SVC3 — MIGRATION TEXT BUILDER TESTS

Comprehensive unit tests for migration_text_builder module.

Tests:
1. complaint_content merge (case + request)
2. immediate_action from first action
3. actions_taken from remaining actions
4. empty inputs produce empty strings
5. partial fields (missing data)
6. determinism (same input = same output)
7. datetime format validation
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from api.services.migration_text_builder import (
    build_migration_texts,
    build_complaint_content,
    build_immediate_action,
    build_actions_taken,
    non_empty,
    join_single,
    join_double
)


def print_header(text):
    """Print formatted test section header"""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print('=' * 80)


def print_test(test_name, passed, message=""):
    """Print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} — {test_name}")
    if message:
        print(f"   {message}")


def test_helper_functions():
    """TEST 0: Helper functions"""
    print_header("TEST 0: HELPER FUNCTIONS")
    
    try:
        # Test non_empty
        result = non_empty(["hello", None, "", "world"])
        correct = result == ["hello", "world"]
        print_test("non_empty filters None and empty", correct, f"Result: {result}")
        
        # Test join_single
        result = join_single(["Line 1", "Line 2"])
        correct = result == "Line 1\nLine 2"
        print_test("join_single uses \\n", correct)
        
        # Test join_double
        result = join_double(["Block 1", "Block 2"])
        correct = result == "Block 1\n\nBlock 2"
        print_test("join_double uses \\n\\n", correct)
        
        return True
        
    except Exception as e:
        print_test("Helper functions", False, str(e))
        return False


def test_complaint_content_merge():
    """TEST 1: complaint_content merge"""
    print_header("TEST 1: COMPLAINT CONTENT MERGE")
    
    try:
        case = {"Description": "Case text"}
        request = {"Note": "Requester text"}
        actions = []
        
        result = build_complaint_content(case, request)
        
        print(f"\n📝 Output:\n{result}\n")
        
        # Check structure
        has_case_label = "[Case Description]" in result
        has_case_text = "Case text" in result
        has_req_label = "[Requester Note]" in result
        has_req_text = "Requester text" in result
        
        print_test("Contains [Case Description]", has_case_label)
        print_test("Contains case text", has_case_text)
        print_test("Contains [Requester Note]", has_req_label)
        print_test("Contains requester text", has_req_text)
        
        # Check double newline separation
        has_double_newline = "\n\n" in result
        print_test("Uses double newline separator", has_double_newline)
        
        # Check order (Case before Requester)
        case_pos = result.find("[Case Description]")
        req_pos = result.find("[Requester Note]")
        correct_order = case_pos < req_pos
        print_test("Case before Requester", correct_order)
        
        return (has_case_label and has_case_text and has_req_label and 
                has_req_text and has_double_newline and correct_order)
        
    except Exception as e:
        print_test("Complaint content merge", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_immediate_action_from_first():
    """TEST 2: immediate_action from first action"""
    print_header("TEST 2: IMMEDIATE ACTION FROM FIRST")
    
    try:
        actions = [
            {
                "Description": "Action desc",
                "SectionNote": "Section note text",
                "SelectionNote": "Selection note text",
                "ProblemReason": "Problem reason text",
                "DateAndTimeCreated": "2025-11-27 10:00:00"
            },
            {
                "Description": "Second action (should be ignored)"
            }
        ]
        
        result = build_immediate_action(actions)
        
        print(f"\n📝 Output:\n{result}\n")
        
        # Check all expected labels present
        has_action_desc = "[Action Description]" in result
        has_section = "[Section Note]" in result
        has_selection = "[Selection Note]" in result
        has_problem = "[Problem Reason]" in result
        
        print_test("Contains [Action Description]", has_action_desc)
        print_test("Contains [Section Note]", has_section)
        print_test("Contains [Selection Note]", has_selection)
        print_test("Contains [Problem Reason]", has_problem)
        
        # Check content present
        has_content = "Action desc" in result
        print_test("Contains action description text", has_content)
        
        # Check second action NOT included
        no_second = "Second action" not in result
        print_test("Second action not included", no_second)
        
        # Check double newline separation
        has_double_newline = "\n\n" in result
        print_test("Uses double newline separator", has_double_newline)
        
        return (has_action_desc and has_section and has_selection and 
                has_problem and has_content and no_second and has_double_newline)
        
    except Exception as e:
        print_test("Immediate action", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_actions_taken_from_remaining():
    """TEST 3: actions_taken from remaining actions"""
    print_header("TEST 3: ACTIONS TAKEN FROM REMAINING")
    
    try:
        actions = [
            {
                "Description": "First action (should be skipped)",
                "DateAndTimeCreated": "2025-11-27 10:00:00"
            },
            {
                "Description": "Second action desc",
                "Note": "Second action note",
                "DepartmentNote": "Dept note",
                "DateAndTimeCreated": "2025-11-27 12:00:00"
            },
            {
                "Description": "Third action desc",
                "SectionNote": "Section note",
                "GoverningPolicies": "Policy text",
                "DateAndTimeCreated": "2025-11-28 09:30:00"
            }
        ]
        
        result = build_actions_taken(actions)
        
        print(f"\n📝 Output:\n{result}\n")
        
        # Check first action NOT included
        no_first = "First action (should be skipped)" not in result
        print_test("First action excluded", no_first)
        
        # Check second action included
        has_second = "Second action desc" in result
        print_test("Second action included", has_second)
        
        # Check third action included
        has_third = "Third action desc" in result
        print_test("Third action included", has_third)
        
        # Check date headers present
        has_date_header = "[Action —" in result
        print_test("Has date header format", has_date_header)
        
        # Check specific date formats
        has_date1 = "2025-11-27 12:00" in result
        has_date2 = "2025-11-28 09:30" in result
        print_test("Second action date correct", has_date1)
        print_test("Third action date correct", has_date2)
        
        # Check field labels present
        has_desc_label = "Description:" in result
        has_note_label = "Note:" in result
        has_policies_label = "Policies:" in result
        print_test("Has Description label", has_desc_label)
        print_test("Has Note label", has_note_label)
        print_test("Has Policies label", has_policies_label)
        
        # Check double newline separation between actions
        double_newline_count = result.count("\n\n")
        # Should have at least 1 double newline between the 2 actions
        has_separation = double_newline_count >= 1
        print_test("Actions separated by double newline", has_separation, 
                   f"Found {double_newline_count} double newlines")
        
        return (no_first and has_second and has_third and has_date_header and 
                has_date1 and has_date2 and has_desc_label and has_separation)
        
    except Exception as e:
        print_test("Actions taken", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_empty_inputs():
    """TEST 4: Empty inputs produce empty strings"""
    print_header("TEST 4: EMPTY INPUTS")
    
    try:
        case = {"Description": None}
        request = {"Note": None}
        actions = []
        
        result = build_migration_texts(case, request, actions)
        
        # All should be empty strings, not None
        complaint_is_str = isinstance(result["complaint_content"], str)
        immediate_is_str = isinstance(result["immediate_action"], str)
        actions_is_str = isinstance(result["actions_taken"], str)
        
        print_test("complaint_content is str", complaint_is_str)
        print_test("immediate_action is str", immediate_is_str)
        print_test("actions_taken is str", actions_is_str)
        
        complaint_empty = result["complaint_content"] == ""
        immediate_empty = result["immediate_action"] == ""
        actions_empty = result["actions_taken"] == ""
        
        print_test("complaint_content is empty", complaint_empty, 
                   f"Value: '{result['complaint_content']}'")
        print_test("immediate_action is empty", immediate_empty, 
                   f"Value: '{result['immediate_action']}'")
        print_test("actions_taken is empty", actions_empty, 
                   f"Value: '{result['actions_taken']}'")
        
        return (complaint_is_str and immediate_is_str and actions_is_str and
                complaint_empty and immediate_empty and actions_empty)
        
    except Exception as e:
        print_test("Empty inputs", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_partial_fields():
    """TEST 5: Partial fields (missing data)"""
    print_header("TEST 5: PARTIAL FIELDS")
    
    try:
        # Action with only some fields populated
        actions = [
            {
                "Description": "Has description",
                "SectionNote": None,  # Missing
                "SelectionNote": "",  # Empty
                "ProblemReason": "Has problem"  # Present
                # Other fields missing
            }
        ]
        
        result = build_immediate_action(actions)
        
        print(f"\n📝 Output:\n{result}\n")
        
        # Should have labels only for populated fields
        has_desc = "[Action Description]" in result
        has_problem = "[Problem Reason]" in result
        print_test("Has description label", has_desc)
        print_test("Has problem label", has_problem)
        
        # Should NOT have labels for empty/missing fields
        no_section = "[Section Note]" not in result
        no_selection = "[Selection Note]" not in result
        print_test("No section note label (was None)", no_section)
        print_test("No selection note label (was empty)", no_selection)
        
        # Check content is present
        has_desc_text = "Has description" in result
        has_problem_text = "Has problem" in result
        print_test("Description text present", has_desc_text)
        print_test("Problem text present", has_problem_text)
        
        return (has_desc and has_problem and no_section and no_selection and
                has_desc_text and has_problem_text)
        
    except Exception as e:
        print_test("Partial fields", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_determinism():
    """TEST 6: Determinism (identical inputs = identical outputs)"""
    print_header("TEST 6: DETERMINISM")
    
    try:
        case = {"Description": "Test case description"}
        request = {"Note": "Test request note"}
        actions = [
            {
                "Description": "Action 1",
                "Note": "Note 1",
                "DateAndTimeCreated": "2025-11-27 10:00:00"
            },
            {
                "Description": "Action 2",
                "SectionNote": "Section note",
                "DateAndTimeCreated": "2025-11-27 11:00:00"
            }
        ]
        
        # Call function twice
        result1 = build_migration_texts(case, request, actions)
        result2 = build_migration_texts(case, request, actions)
        
        # Check all three fields are identical
        complaint_same = result1["complaint_content"] == result2["complaint_content"]
        immediate_same = result1["immediate_action"] == result2["immediate_action"]
        actions_same = result1["actions_taken"] == result2["actions_taken"]
        
        print_test("complaint_content deterministic", complaint_same)
        print_test("immediate_action deterministic", immediate_same)
        print_test("actions_taken deterministic", actions_same)
        
        # Byte-level equality check
        all_identical = (complaint_same and immediate_same and actions_same)
        print_test("All outputs byte-identical", all_identical)
        
        return all_identical
        
    except Exception as e:
        print_test("Determinism", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_datetime_format():
    """TEST 7: Datetime format validation"""
    print_header("TEST 7: DATETIME FORMAT")
    
    try:
        actions = [
            {
                "Description": "First (skipped)",
                "DateAndTimeCreated": "2025-11-27 10:00:00"
            },
            {
                "Description": "Second action",
                "DateAndTimeCreated": "2025-12-15 14:30:45"
            }
        ]
        
        result = build_actions_taken(actions)
        
        print(f"\n📝 Output:\n{result}\n")
        
        # Check format is [Action — YYYY-MM-DD HH:MM]
        has_header = "[Action — 2025-12-15 14:30]" in result
        print_test("Datetime format correct", has_header, 
                   "Expected: [Action — 2025-12-15 14:30]")
        
        # Check seconds are truncated
        no_seconds = "14:30:45" not in result
        print_test("Seconds truncated", no_seconds)
        
        # Check exact format match
        import re
        pattern = r'\[Action — \d{4}-\d{2}-\d{2} \d{2}:\d{2}\]'
        matches = re.findall(pattern, result)
        has_match = len(matches) > 0
        print_test("Matches regex pattern", has_match, 
                   f"Found {len(matches)} match(es)")
        
        return has_header and no_seconds and has_match
        
    except Exception as e:
        print_test("Datetime format", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_integration_full():
    """TEST 8: Full integration test"""
    print_header("TEST 8: FULL INTEGRATION")
    
    try:
        case = {
            "Description": "Patient reported pain in left arm after procedure"
        }
        
        request = {
            "Note": "Family member called to report concern about care quality"
        }
        
        actions = [
            {
                "Description": "Initial assessment completed",
                "SectionNote": "Nursing team reviewed",
                "DateAndTimeCreated": "2025-11-27 10:00:00"
            },
            {
                "Description": "Doctor consultation arranged",
                "Note": "Orthopedic specialist contacted",
                "DateAndTimeCreated": "2025-11-27 14:00:00"
            },
            {
                "Description": "Follow-up appointment scheduled",
                "DepartmentNote": "Outpatient clinic",
                "GoverningPolicies": "Standard follow-up protocol",
                "DateAndTimeCreated": "2025-11-28 09:00:00"
            }
        ]
        
        result = build_migration_texts(case, request, actions)
        
        print("\n📋 COMPLAINT CONTENT:")
        print("-" * 80)
        print(result["complaint_content"])
        
        print("\n📋 IMMEDIATE ACTION:")
        print("-" * 80)
        print(result["immediate_action"])
        
        print("\n📋 ACTIONS TAKEN:")
        print("-" * 80)
        print(result["actions_taken"])
        
        # Verify all three fields are non-empty
        has_complaint = len(result["complaint_content"]) > 0
        has_immediate = len(result["immediate_action"]) > 0
        has_actions = len(result["actions_taken"]) > 0
        
        print("\n")
        print_test("complaint_content populated", has_complaint, 
                   f"{len(result['complaint_content'])} chars")
        print_test("immediate_action populated", has_immediate, 
                   f"{len(result['immediate_action'])} chars")
        print_test("actions_taken populated", has_actions, 
                   f"{len(result['actions_taken'])} chars")
        
        # Verify complaint has both case and request
        complaint_complete = ("[Case Description]" in result["complaint_content"] and
                             "[Requester Note]" in result["complaint_content"])
        print_test("Complaint has both sources", complaint_complete)
        
        # Verify immediate has first action only
        immediate_correct = ("Initial assessment" in result["immediate_action"] and
                            "Doctor consultation" not in result["immediate_action"])
        print_test("Immediate action correct", immediate_correct)
        
        # Verify actions_taken has 2nd and 3rd only
        actions_correct = ("Initial assessment" not in result["actions_taken"] and
                          "Doctor consultation" in result["actions_taken"] and
                          "Follow-up appointment" in result["actions_taken"])
        print_test("Actions taken correct", actions_correct)
        
        return (has_complaint and has_immediate and has_actions and
                complaint_complete and immediate_correct and actions_correct)
        
    except Exception as e:
        print_test("Full integration", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print_header("PHASE K — SVC3 — MIGRATION TEXT BUILDER TESTS")
    print("Unit tests for pure text transformation functions")
    
    results = []
    
    results.append(("Helper Functions", test_helper_functions()))
    results.append(("Complaint Content Merge", test_complaint_content_merge()))
    results.append(("Immediate Action", test_immediate_action_from_first()))
    results.append(("Actions Taken", test_actions_taken_from_remaining()))
    results.append(("Empty Inputs", test_empty_inputs()))
    results.append(("Partial Fields", test_partial_fields()))
    results.append(("Determinism", test_determinism()))
    results.append(("Datetime Format", test_datetime_format()))
    results.append(("Full Integration", test_integration_full()))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} — {test_name}")
    
    print(f"\n{'=' * 80}")
    print(f"TOTAL: {passed}/{total} tests passed")
    print('=' * 80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED — K-SVC-3 COMPLETE")
        return True
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
