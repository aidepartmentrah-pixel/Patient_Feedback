"""
STEP 3.14 MASTER TEST RUNNER — Case Response Service

Executes all test suites for case_response_service.py in sequence:
1. Prompt 1: Section-level workflow actions
2. Prompt 2: Department & Administration workflow actions
3. Prompt 3: Force close & full lifecycle tests

Run this file to validate the complete STEP 3.14 implementation.
"""

import sys
import os
import subprocess

# Force UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def run_test_file(test_file):
    """Run a test file and return success status."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {test_file}")
    print('='*80)
    
    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=False,
        text=True
    )
    
    return result.returncode == 0


if __name__ == "__main__":
    print("\n" + "="*80)
    print("STEP 3.14 — MASTER TEST SUITE")
    print("Complete validation of case_response_service.py")
    print("="*80)
    
    test_files = [
        'test_step3_14_prompt1_section_workflow.py',
        'test_step3_14_prompt2_dept_admin_workflow.py',
        'test_step3_14_prompt3_force_close_lifecycle.py',
    ]
    
    results = {}
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"\n❌ ERROR: Test file not found: {test_file}")
            results[test_file] = False
            continue
        
        success = run_test_file(test_file)
        results[test_file] = success
    
    # Summary
    print("\n\n" + "="*80)
    print("MASTER TEST SUITE SUMMARY")
    print("="*80)
    
    for test_file, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {test_file}")
    
    total = len(results)
    passed = sum(1 for s in results.values() if s)
    failed = total - passed
    
    print(f"\n{'='*80}")
    print(f"Total Test Suites: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print('='*80)
    
    if failed == 0:
        print("\n🎉🎉🎉 ALL TEST SUITES PASSED! 🎉🎉🎉")
        print("\n✨ STEP 3.14 is COMPLETE and VALIDATED!")
        print("✨ case_response_service.py is production-ready!")
        print("\nYou can now proceed to the next step with confidence.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {failed} test suite(s) failed.")
        print("Please review the errors above and fix the implementation.")
        sys.exit(1)
