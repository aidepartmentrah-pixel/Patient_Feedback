"""
Test Seasonal Report Export Generation
Tests Word and PDF generation for all organizational levels:
- Hospital (single report)
- Administrations (multi-export ZIP)
- Departments (multi-export ZIP)
- Sections (multi-export ZIP)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import requests
import json
from datetime import datetime

# Base URL
BASE_URL = "http://localhost:8000"

# Test parameters
TEST_YEAR = 2026
TEST_PERIOD = "Q1"  # Q1 2026 has 10 incidents

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_export(test_name, params, expected_type="file"):
    """Test a seasonal export endpoint."""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(f"Parameters: {json.dumps(params, indent=2)}")
    
    try:
        url = f"{BASE_URL}/api/reports/seasonal/export"
        response = requests.post(url, params=params, timeout=60)
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        print(f"Content-Length: {len(response.content)} bytes")
        
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            
            if 'application/zip' in content_type:
                print(f"✅ SUCCESS - ZIP file received")
                print(f"   ZIP size: {len(response.content):,} bytes")
                
                # Save ZIP for inspection
                filename = f"test_seasonal_{params.get('orgunit_type')}_{params.get('format')}.zip"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"   Saved to: {filename}")
                
            elif 'application/pdf' in content_type:
                print(f"✅ SUCCESS - PDF file received")
                print(f"   PDF size: {len(response.content):,} bytes")
                
                filename = f"test_seasonal_hospital_{params.get('format')}.pdf"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"   Saved to: {filename}")
                
            elif 'officedocument.wordprocessingml' in content_type:
                print(f"✅ SUCCESS - Word file received")
                print(f"   DOCX size: {len(response.content):,} bytes")
                
                filename = f"test_seasonal_hospital_{params.get('format')}.docx"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"   Saved to: {filename}")
                
            elif 'text/csv' in content_type:
                print(f"✅ SUCCESS - CSV file received")
                print(f"   CSV size: {len(response.content):,} bytes")
                
            elif 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' in content_type:
                print(f"✅ SUCCESS - Excel file received")
                print(f"   XLSX size: {len(response.content):,} bytes")
                
            else:
                print(f"⚠️ UNEXPECTED CONTENT TYPE: {content_type}")
                
            return True
            
        else:
            print(f"❌ FAILED")
            try:
                error_data = response.json()
                print(f"   Error: {json.dumps(error_data, indent=2)}")
            except:
                print(f"   Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("  SEASONAL REPORT EXPORT TEST SUITE")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Period: {TEST_PERIOD} {TEST_YEAR}")
    print(f"Server: {BASE_URL}")
    
    results = {}
    
    # ========================================================================
    # TEST 1: HOSPITAL LEVEL - SINGLE REPORT (DOCX)
    # ========================================================================
    print_section("TEST 1: Hospital Level - Single Report (DOCX)")
    results['hospital_docx'] = test_export(
        "Hospital - Word Document",
        {
            'year': TEST_YEAR,
            'period': TEST_PERIOD,
            'orgunit_id': 1,
            'orgunit_type': 0,  # Hospital level
            'format': 'docx',
            'language': 'en'
        },
        expected_type="file"
    )
    
    # ========================================================================
    # TEST 2: HOSPITAL LEVEL - SINGLE REPORT (PDF)
    # ========================================================================
    print_section("TEST 2: Hospital Level - Single Report (PDF)")
    results['hospital_pdf'] = test_export(
        "Hospital - PDF Document",
        {
            'year': TEST_YEAR,
            'period': TEST_PERIOD,
            'orgunit_id': 1,
            'orgunit_type': 0,  # Hospital level
            'format': 'pdf',
            'language': 'en'
        },
        expected_type="file"
    )
    
    # ========================================================================
    # TEST 3: ALL ADMINISTRATIONS - MULTI-EXPORT (DOCX)
    # ========================================================================
    print_section("TEST 3: All Administrations - Multi-Export ZIP (DOCX)")
    results['admin_docx'] = test_export(
        "Administrations - Word Documents ZIP",
        {
            'year': TEST_YEAR,
            'period': TEST_PERIOD,
            'orgunit_id': 1,
            'orgunit_type': 1,  # Administration level
            'format': 'docx',
            'language': 'en'
        },
        expected_type="zip"
    )
    
    # ========================================================================
    # TEST 4: ALL ADMINISTRATIONS - MULTI-EXPORT (PDF)
    # ========================================================================
    print_section("TEST 4: All Administrations - Multi-Export ZIP (PDF)")
    results['admin_pdf'] = test_export(
        "Administrations - PDF Documents ZIP",
        {
            'year': TEST_YEAR,
            'period': TEST_PERIOD,
            'orgunit_id': 1,
            'orgunit_type': 1,  # Administration level
            'format': 'pdf',
            'language': 'en'
        },
        expected_type="zip"
    )
    
    # ========================================================================
    # TEST 5: ALL DEPARTMENTS - MULTI-EXPORT (DOCX)
    # ========================================================================
    print_section("TEST 5: All Departments - Multi-Export ZIP (DOCX)")
    results['dept_docx'] = test_export(
        "Departments - Word Documents ZIP",
        {
            'year': TEST_YEAR,
            'period': TEST_PERIOD,
            'orgunit_id': 1,
            'orgunit_type': 2,  # Department level
            'format': 'docx',
            'language': 'en'
        },
        expected_type="zip"
    )
    
    # ========================================================================
    # TEST 6: ALL DEPARTMENTS - MULTI-EXPORT (PDF)
    # ========================================================================
    print_section("TEST 6: All Departments - Multi-Export ZIP (PDF)")
    results['dept_pdf'] = test_export(
        "Departments - PDF Documents ZIP",
        {
            'year': TEST_YEAR,
            'period': TEST_PERIOD,
            'orgunit_id': 1,
            'orgunit_type': 2,  # Department level
            'format': 'pdf',
            'language': 'en'
        },
        expected_type="zip"
    )
    
    # ========================================================================
    # TEST 7: ALL SECTIONS - MULTI-EXPORT (DOCX)
    # ========================================================================
    print_section("TEST 7: All Sections - Multi-Export ZIP (DOCX)")
    results['section_docx'] = test_export(
        "Sections - Word Documents ZIP",
        {
            'year': TEST_YEAR,
            'period': TEST_PERIOD,
            'orgunit_id': 1,
            'orgunit_type': 3,  # Section level
            'format': 'docx',
            'language': 'en'
        },
        expected_type="zip"
    )
    
    # ========================================================================
    # TEST 8: ALL SECTIONS - MULTI-EXPORT (PDF)
    # ========================================================================
    print_section("TEST 8: All Sections - Multi-Export ZIP (PDF)")
    results['section_pdf'] = test_export(
        "Sections - PDF Documents ZIP",
        {
            'year': TEST_YEAR,
            'period': TEST_PERIOD,
            'orgunit_id': 1,
            'orgunit_type': 3,  # Section level
            'format': 'pdf',
            'language': 'en'
        },
        expected_type="zip"
    )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "-"*80)
    print(f"Total: {passed}/{total} tests passed ({(passed/total*100):.1f}%)")
    print("="*80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
    
    print("\n" + "="*80 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
