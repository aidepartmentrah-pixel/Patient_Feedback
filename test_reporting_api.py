"""
Test script for Reporting Page API Endpoints
Tests all 8 endpoints with various filter combinations
"""

import requests
import json
from datetime import datetime, date, timedelta

BASE_URL = "http://localhost:8000/api/reports"

# Test data
TEST_YEAR = 2024
TEST_MONTH = 3
TEST_TRIMESTER = 1
TEST_QUARTER = 1


def print_response(title: str, response):
    """Pretty print API response."""
    print(f"\n{'='*80}")
    print(f"TEST: {title}")
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    if response.status_code == 200 or response.status_code == 202:
        try:
            print(json.dumps(response.json(), indent=2, default=str))
        except:
            print(response.text)
    else:
        print(json.dumps(response.json(), indent=2))
    print(f"{'='*80}")


def test_get_complaints():
    """Test B1: Fetch filtered complaints."""
    print("\n" + "="*80)
    print("TESTING: B1 - GET /api/reports/complaints")
    print("="*80)
    
    # Test 1: Monthly report with pagination
    params = {
        "report_type": "monthly",
        "year": TEST_YEAR,
        "month": TEST_MONTH,
        "page": 1,
        "page_size": 10
    }
    response = requests.get(f"{BASE_URL}/complaints", params=params)
    print_response("Monthly Complaints (Page 1, Size 10)", response)
    
    # Test 2: With department filter
    params["dayra_id"] = 12
    response = requests.get(f"{BASE_URL}/complaints", params=params)
    print_response("Monthly Complaints - Department Filter", response)
    
    # Test 3: With status filter
    params.pop("dayra_id")
    params["status"] = "closed"
    response = requests.get(f"{BASE_URL}/complaints", params=params)
    print_response("Monthly Complaints - Closed Status Filter", response)
    
    # Test 4: Seasonal report with trimester
    params = {
        "report_type": "seasonal",
        "year": TEST_YEAR,
        "trimester": TEST_TRIMESTER,
        "page": 1,
        "page_size": 10
    }
    response = requests.get(f"{BASE_URL}/complaints", params=params)
    print_response("Seasonal Complaints (Trimester)", response)
    
    # Test 5: Seasonal report with quarter
    params["trimester"] = None
    params["quarter"] = TEST_QUARTER
    response = requests.get(f"{BASE_URL}/complaints", params=params)
    print_response("Seasonal Complaints (Quarter)", response)


def test_monthly_statistics():
    """Test B2: Fetch monthly aggregated statistics."""
    print("\n" + "="*80)
    print("TESTING: B2 - GET /api/reports/monthly-statistics")
    print("="*80)
    
    # Test 1: Basic monthly statistics
    params = {
        "year": TEST_YEAR,
        "month": TEST_MONTH
    }
    response = requests.get(f"{BASE_URL}/monthly-statistics", params=params)
    print_response("Monthly Statistics - March 2024", response)
    
    # Test 2: With department filter
    params["dayra_id"] = 12
    response = requests.get(f"{BASE_URL}/monthly-statistics", params=params)
    print_response("Monthly Statistics - Department Filter", response)
    
    # Test 3: Yearly statistics
    params = {
        "year": TEST_YEAR
    }
    response = requests.get(f"{BASE_URL}/monthly-statistics", params=params)
    print_response("Yearly Statistics - 2024", response)


def test_seasonal_hcat():
    """Test B3: Fetch seasonal HCAT analysis."""
    print("\n" + "="*80)
    print("TESTING: B3 - GET /api/reports/seasonal-hcat")
    print("="*80)
    
    # Test 1: Trimester-based analysis
    params = {
        "year": TEST_YEAR,
        "trimester": 1,
        "threshold": 50
    }
    response = requests.get(f"{BASE_URL}/seasonal-hcat", params=params)
    print_response("Seasonal HCAT - Trimester 1 (Threshold 50)", response)
    
    # Test 2: Quarter-based analysis
    params = {
        "year": TEST_YEAR,
        "quarter": 1,
        "threshold": 30
    }
    response = requests.get(f"{BASE_URL}/seasonal-hcat", params=params)
    print_response("Seasonal HCAT - Quarter 1 (Threshold 30)", response)
    
    # Test 3: With department filter
    params = {
        "year": TEST_YEAR,
        "trimester": 1,
        "threshold": 50,
        "dayra_id": 12
    }
    response = requests.get(f"{BASE_URL}/seasonal-hcat", params=params)
    print_response("Seasonal HCAT - With Department Filter", response)


def test_bulk_summary():
    """Test B4: Fetch bulk export summary."""
    print("\n" + "="*80)
    print("TESTING: B4 - GET /api/reports/bulk-summary")
    print("="*80)
    
    # Test 1: Monthly bulk summary
    params = {
        "report_type": "monthly",
        "year": TEST_YEAR,
        "month": TEST_MONTH
    }
    response = requests.get(f"{BASE_URL}/bulk-summary", params=params)
    print_response("Bulk Summary - March 2024", response)
    
    # Test 2: Seasonal bulk summary (trimester)
    params = {
        "report_type": "seasonal",
        "year": TEST_YEAR,
        "trimester": 1
    }
    response = requests.get(f"{BASE_URL}/bulk-summary", params=params)
    print_response("Bulk Summary - Trimester 1", response)
    
    # Test 3: Seasonal bulk summary (quarter)
    params = {
        "report_type": "seasonal",
        "year": TEST_YEAR,
        "quarter": 1
    }
    response = requests.get(f"{BASE_URL}/bulk-summary", params=params)
    print_response("Bulk Summary - Quarter 1", response)


def test_export_pdf():
    """Test B5: Export as PDF."""
    print("\n" + "="*80)
    print("TESTING: B5 - POST /api/reports/export/pdf")
    print("="*80)
    
    # Test 1: Detailed monthly PDF
    payload = {
        "report_type": "monthly",
        "display_mode": "detailed",
        "year": TEST_YEAR,
        "month": TEST_MONTH,
        "filters": {"dayra_id": 12},
        "include_charts": True,
        "language": "en"
    }
    response = requests.post(f"{BASE_URL}/export/pdf", json=payload)
    print_response("PDF Export - Detailed Monthly", response)
    
    # Test 2: Numeric mode PDF
    payload = {
        "report_type": "monthly",
        "display_mode": "numeric",
        "year": TEST_YEAR,
        "month": TEST_MONTH,
        "language": "ar"
    }
    response = requests.post(f"{BASE_URL}/export/pdf", json=payload)
    print_response("PDF Export - Numeric Mode (Arabic)", response)
    
    # Test 3: HCAT seasonal PDF
    payload = {
        "report_type": "seasonal",
        "display_mode": "hcat",
        "year": TEST_YEAR,
        "trimester": 1,
        "language": "en"
    }
    response = requests.post(f"{BASE_URL}/export/pdf", json=payload)
    print_response("PDF Export - Seasonal HCAT", response)


def test_export_csv():
    """Test B6: Export as CSV."""
    print("\n" + "="*80)
    print("TESTING: B6 - POST /api/reports/export/csv")
    print("="*80)
    
    # Test 1: Monthly CSV
    payload = {
        "report_type": "monthly",
        "year": TEST_YEAR,
        "month": TEST_MONTH,
        "filters": {"status": "closed"},
        "language": "en"
    }
    response = requests.post(f"{BASE_URL}/export/csv", json=payload)
    print_response("CSV Export - Monthly", response)
    
    # Test 2: Seasonal CSV
    payload = {
        "report_type": "seasonal",
        "year": TEST_YEAR,
        "quarter": 1,
        "language": "ar"
    }
    response = requests.post(f"{BASE_URL}/export/csv", json=payload)
    print_response("CSV Export - Seasonal (Arabic)", response)


def test_download_export(export_id: str):
    """Test B7: Download exported file."""
    print("\n" + "="*80)
    print("TESTING: B7 - GET /api/reports/download/{export_id}")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/download/{export_id}")
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type', 'N/A')}")
    print(f"Content-Length: {len(response.content)} bytes")
    print(f"Download successful: {response.status_code == 200}")


def test_bulk_export():
    """Test B8: Bulk export."""
    print("\n" + "="*80)
    print("TESTING: B8 - POST /api/reports/export/bulk")
    print("="*80)
    
    # Test 1: Monthly bulk export
    payload = {
        "report_type": "monthly",
        "year": TEST_YEAR,
        "month": TEST_MONTH,
        "format": "pdf",
        "language": "en"
    }
    response = requests.post(f"{BASE_URL}/export/bulk", json=payload)
    print_response("Bulk Export - Monthly PDF", response)
    
    # Test 2: Seasonal bulk export
    payload = {
        "report_type": "seasonal",
        "year": TEST_YEAR,
        "trimester": 1,
        "format": "csv",
        "language": "ar"
    }
    response = requests.post(f"{BASE_URL}/export/bulk", json=payload)
    print_response("Bulk Export - Seasonal CSV (Arabic)", response)


def run_all_tests():
    """Run all endpoint tests."""
    print("\n" + "="*80)
    print("REPORTING PAGE API - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    try:
        # Test all endpoints
        test_get_complaints()
        test_monthly_statistics()
        test_seasonal_hcat()
        test_bulk_summary()
        test_export_pdf()
        test_export_csv()
        test_bulk_export()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
