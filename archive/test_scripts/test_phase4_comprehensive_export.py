"""
Phase 4: Comprehensive Export Testing
Tests all combinations of monthly report exports with date ranges and numeric mode
"""

import requests
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_export(test_name, params, expected_content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
    """Test a single export endpoint"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"Params: {params}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/reports/monthly/export",
            params=params,
            timeout=60
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            content_length = len(response.content)
            print(f"[SUCCESS] Content-Type: {content_type}, Size: {content_length} bytes")
            
            # Verify content type
            if expected_content_type in content_type or content_type in expected_content_type:
                print(f"[OK] Content type matches expected")
            else:
                print(f"[WARN] Content type mismatch: expected {expected_content_type}, got {content_type}")
            
            return True
        else:
            print(f"[FAILED]")
            try:
                error_detail = response.json()
                print(f"Error: {error_detail}")
            except:
                print(f"Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"[EXCEPTION]: {str(e)[:200]}")
        return False


def run_all_tests():
    """Run comprehensive test matrix"""
    
    print("\n" + "="*80)
    print("PHASE 4: COMPREHENSIVE MONTHLY EXPORT TESTING")
    print("="*80)
    
    results = {
        "passed": [],
        "failed": []
    }
    
    # Test Matrix Configuration
    test_cases = [
        # ===================================================================
        # SINGLE EXPORT - HOSPITAL LEVEL
        # ===================================================================
        {
            "name": "Hospital - Numeric - Month - DOCX",
            "params": {
                "year": 2026,
                "month": 1,
                "format": "docx",
                "display_mode": "numeric"
            }
        },
        {
            "name": "Hospital - Numeric - Date Range - DOCX",
            "params": {
                "year": 2026,
                "start_date": "2024-11-03",
                "end_date": "2026-02-28",
                "format": "docx",
                "display_mode": "numeric"
            }
        },
        {
            "name": "Hospital - Detailed - Month - DOCX",
            "params": {
                "year": 2026,
                "month": 1,
                "format": "docx",
                "display_mode": "detailed"
            }
        },
        {
            "name": "Hospital - Detailed - Date Range - DOCX",
            "params": {
                "year": 2026,
                "start_date": "2024-11-03",
                "end_date": "2026-02-28",
                "format": "docx",
                "display_mode": "detailed"
            }
        },
        
        # ===================================================================
        # SINGLE EXPORT - ALL ADMINISTRATIONS (with breakdown)
        # ===================================================================
        {
            "name": "All Administrations - Numeric - Month - DOCX",
            "params": {
                "year": 2026,
                "month": 1,
                "format": "docx",
                "display_mode": "numeric",
                "administration_ids": "all"
            }
        },
        {
            "name": "All Administrations - Numeric - Date Range - DOCX",
            "params": {
                "year": 2026,
                "start_date": "2024-11-03",
                "end_date": "2026-02-28",
                "format": "docx",
                "display_mode": "numeric",
                "administration_ids": "all"
            }
        },
        
        # ===================================================================
        # SINGLE EXPORT - SPECIFIC ADMINISTRATION
        # ===================================================================
        {
            "name": "Specific Administration - Numeric - Month - DOCX",
            "params": {
                "year": 2026,
                "month": 1,
                "format": "docx",
                "display_mode": "numeric",
                "administration_ids": "3"
            }
        },
        {
            "name": "Specific Administration - Numeric - Date Range - DOCX",
            "params": {
                "year": 2026,
                "start_date": "2024-11-03",
                "end_date": "2026-02-28",
                "format": "docx",
                "display_mode": "numeric",
                "administration_ids": "3"
            }
        },
        {
            "name": "Specific Administration - Detailed - Month - DOCX",
            "params": {
                "year": 2026,
                "month": 1,
                "format": "docx",
                "display_mode": "detailed",
                "administration_ids": "3"
            }
        },
        
        # ===================================================================
        # SINGLE EXPORT - ALL DEPARTMENTS (with breakdown)
        # ===================================================================
        {
            "name": "All Departments - Numeric - Month - DOCX",
            "params": {
                "year": 2026,
                "month": 1,
                "format": "docx",
                "display_mode": "numeric",
                "department_ids": "all"
            }
        },
        
        # ===================================================================
        # SINGLE EXPORT - SPECIFIC DEPARTMENT
        # ===================================================================
        {
            "name": "Specific Department - Numeric - Month - DOCX",
            "params": {
                "year": 2026,
                "month": 1,
                "format": "docx",
                "display_mode": "numeric",
                "department_ids": "28"
            }
        },
        
        # ===================================================================
        # MULTI EXPORT - ALL ADMINISTRATIONS (ZIP with multiple files)
        # ===================================================================
        {
            "name": "Multi-Export - All Administrations - Numeric - Month",
            "params": {
                "year": 2026,
                "month": 1,
                "format": "docx",
                "display_mode": "numeric",
                "report_level": "administration"
            },
            "expected_content": "application/zip"
        },
        {
            "name": "Multi-Export - All Administrations - Numeric - Date Range",
            "params": {
                "year": 2026,
                "start_date": "2024-11-03",
                "end_date": "2026-02-28",
                "format": "docx",
                "display_mode": "numeric",
                "report_level": "administration"
            },
            "expected_content": "application/zip"
        },
        {
            "name": "Multi-Export - All Administrations - Detailed - Month",
            "params": {
                "year": 2026,
                "month": 1,
                "format": "docx",
                "display_mode": "detailed",
                "report_level": "administration"
            },
            "expected_content": "application/zip"
        },
        
        # ===================================================================
        # MULTI EXPORT - ALL DEPARTMENTS (ZIP with multiple files)
        # ===================================================================
        {
            "name": "Multi-Export - All Departments - Numeric - Month",
            "params": {
                "year": 2026,
                "month": 1,
                "format": "docx",
                "display_mode": "numeric",
                "report_level": "department"
            },
            "expected_content": "application/zip"
        },
        {
            "name": "Multi-Export - All Departments - Detailed - Month",
            "params": {
                "year": 2026,
                "month": 1,
                "format": "docx",
                "display_mode": "detailed",
                "report_level": "department"
            },
            "expected_content": "application/zip"
        },
        
        # ===================================================================
        # CSV/XLSX EXPORTS
        # ===================================================================
        {
            "name": "Hospital - Numeric - Month - CSV",
            "params": {
                "year": 2026,
                "month": 1,
                "format": "csv",
                "display_mode": "numeric"
            },
            "expected_content": "text/csv"
        },
        {
            "name": "Hospital - Numeric - Month - XLSX",
            "params": {
                "year": 2026,
                "month": 1,
                "format": "xlsx",
                "display_mode": "numeric"
            },
            "expected_content": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }
    ]
    
    # Run all tests
    total = len(test_cases)
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{total}]")
        
        expected_content = test_case.get("expected_content", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        
        success = test_export(
            test_case["name"],
            test_case["params"],
            expected_content
        )
        
        if success:
            results["passed"].append(test_case["name"])
        else:
            results["failed"].append(test_case["name"])
        
        # Small delay between tests
        time.sleep(0.5)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {total}")
    print(f"[OK] Passed: {len(results['passed'])}")
    print(f"[FAIL] Failed: {len(results['failed'])}")
    print(f"Success Rate: {len(results['passed'])/total*100:.1f}%")
    
    if results["failed"]:
        print("\n[FAIL] Failed Tests:")
        for test_name in results["failed"]:
            print(f"  - {test_name}")
    
    if len(results["passed"]) == total:
        print("\n[SUCCESS] ALL TESTS PASSED!")
    
    print("="*80)


if __name__ == "__main__":
    print(f"Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    run_all_tests()
    print(f"\nFinished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
