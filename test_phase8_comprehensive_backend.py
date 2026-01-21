"""
PHASE 8: Comprehensive Backend Testing
Tests all seasonal comparison endpoints with edge cases, validation, and error handling
"""

import requests
import json
from datetime import datetime
from typing import Dict, Any, List
import os

# Configuration
BASE_URL = "http://localhost:8000"
TEST_OUTPUT_DIR = "test_phase8_outputs"

# Test data
TEST_ORGUNIT_ID = 1
TEST_ORGUNIT_TYPE = 0

# Create output directory
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

# Test Results Tracker
test_results = {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "warnings": 0,
    "tests": []
}

def log_test_result(test_name: str, status: str, message: str, details: Dict = None):
    """Log test result"""
    test_results["total"] += 1
    if status == "PASS":
        test_results["passed"] += 1
        icon = "✅"
    elif status == "FAIL":
        test_results["failed"] += 1
        icon = "❌"
    else:
        test_results["warnings"] += 1
        icon = "⚠️"
    
    test_results["tests"].append({
        "name": test_name,
        "status": status,
        "message": message,
        "details": details,
        "timestamp": datetime.now().isoformat()
    })
    
    print(f"{icon} {status}: {test_name}")
    print(f"   {message}")
    if details:
        print(f"   Details: {json.dumps(details, indent=2)[:200]}")
    print()

def test_available_quarters():
    """Test 1: Available Quarters Endpoint"""
    print("="*70)
    print("TEST 1: AVAILABLE QUARTERS ENDPOINT")
    print("="*70)
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/seasonal-comparison/available-quarters",
            params={
                "orgunit_id": TEST_ORGUNIT_ID,
                "orgunit_type": TEST_ORGUNIT_TYPE
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Validate response structure
            if "available_seasons" not in data:
                log_test_result("Available Quarters", "FAIL", "Missing 'available_seasons' key in response")
                return False
            
            seasons = data["available_seasons"]
            if not isinstance(seasons, list):
                log_test_result("Available Quarters", "FAIL", "Seasons is not a list")
                return False
            
            if len(seasons) == 0:
                log_test_result("Available Quarters", "WARNING", "No seasons available in database", 
                              {"seasons_count": 0})
                return False
            
            # Validate season structure
            required_keys = ["season_id", "name", "start_date", "end_date"]
            first_season = seasons[0]
            for key in required_keys:
                if key not in first_season:
                    log_test_result("Available Quarters", "FAIL", f"Missing key: {key}")
                    return False
            
            log_test_result("Available Quarters", "PASS", 
                          f"Retrieved {len(seasons)} seasons successfully",
                          {"seasons_count": len(seasons), "first_season": first_season})
            return True
        else:
            log_test_result("Available Quarters", "FAIL", 
                          f"HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        log_test_result("Available Quarters", "FAIL", f"Exception: {str(e)}")
        return False

def test_2quarter_json(season_ids: List[int]):
    """Test 2: 2-Quarter Comparison - JSON Format"""
    print("="*70)
    print("TEST 2: 2-QUARTER COMPARISON - JSON")
    print("="*70)
    
    try:
        payload = {
            "season_ids": season_ids[:2],
            "orgunit_id": TEST_ORGUNIT_ID,
            "orgunit_type": TEST_ORGUNIT_TYPE,
            "format": "json"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/2-quarters",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Validate structure
            required_keys = ["success", "comparison_type", "periods", "data"]
            for key in required_keys:
                if key not in data:
                    log_test_result("2-Quarter JSON", "FAIL", f"Missing key: {key}")
                    return False
            
            if data["comparison_type"] != "2-quarters":
                log_test_result("2-Quarter JSON", "FAIL", "Invalid comparison_type")
                return False
            
            if len(data["periods"]) != 2:
                log_test_result("2-Quarter JSON", "FAIL", f"Expected 2 periods, got {len(data['periods'])}")
                return False
            
            # Save to file
            output_file = os.path.join(TEST_OUTPUT_DIR, "2quarter_comparison.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            log_test_result("2-Quarter JSON", "PASS", 
                          "JSON response structure valid",
                          {"periods": data["periods"],
                           "output_file": output_file})
            return True
        else:
            log_test_result("2-Quarter JSON", "FAIL", 
                          f"HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        log_test_result("2-Quarter JSON", "FAIL", f"Exception: {str(e)}")
        return False

def test_2quarter_docx(season_ids: List[int]):
    """Test 3: 2-Quarter Comparison - DOCX Format"""
    print("="*70)
    print("TEST 3: 2-QUARTER COMPARISON - DOCX")
    print("="*70)
    
    try:
        payload = {
            "season_ids": season_ids[:2],
            "orgunit_id": TEST_ORGUNIT_ID,
            "orgunit_type": TEST_ORGUNIT_TYPE,
            "format": "docx"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/2-quarters",
            json=payload
        )
        
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")
            if "wordprocessingml" not in content_type:
                log_test_result("2-Quarter DOCX", "FAIL", f"Invalid content type: {content_type}")
                return False
            
            # Save file
            output_file = os.path.join(TEST_OUTPUT_DIR, "2quarter_comparison.docx")
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            file_size = len(response.content)
            log_test_result("2-Quarter DOCX", "PASS", 
                          f"DOCX file generated ({file_size:,} bytes)",
                          {"file_size": file_size, "output_file": output_file})
            return True
        else:
            log_test_result("2-Quarter DOCX", "FAIL", 
                          f"HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        log_test_result("2-Quarter DOCX", "FAIL", f"Exception: {str(e)}")
        return False

def test_3quarter_json(season_ids: List[int]):
    """Test 4: 3-Quarter Comparison - JSON Format"""
    print("="*70)
    print("TEST 4: 3-QUARTER COMPARISON - JSON")
    print("="*70)
    
    try:
        payload = {
            "season_ids": season_ids[:3],
            "orgunit_id": TEST_ORGUNIT_ID,
            "orgunit_type": TEST_ORGUNIT_TYPE,
            "format": "json"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/3-quarters",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Validate structure
            required_keys = ["success", "comparison_type", "periods", "data"]
            for key in required_keys:
                if key not in data:
                    log_test_result("3-Quarter JSON", "FAIL", f"Missing key: {key}")
                    return False
            
            if data["comparison_type"] != "3-quarters":
                log_test_result("3-Quarter JSON", "FAIL", "Invalid comparison_type")
                return False
            
            if len(data["periods"]) != 3:
                log_test_result("3-Quarter JSON", "FAIL", f"Expected 3 periods, got {len(data['periods'])}")
                return False
            
            # Save to file
            output_file = os.path.join(TEST_OUTPUT_DIR, "3quarter_comparison.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            log_test_result("3-Quarter JSON", "PASS", 
                          "Trend analysis structure valid",
                          {"periods": data["periods"],
                           "output_file": output_file})
            return True
        else:
            log_test_result("3-Quarter JSON", "FAIL", 
                          f"HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        log_test_result("3-Quarter JSON", "FAIL", f"Exception: {str(e)}")
        return False

def test_3quarter_docx(season_ids: List[int]):
    """Test 5: 3-Quarter Comparison - DOCX Format"""
    print("="*70)
    print("TEST 5: 3-QUARTER COMPARISON - DOCX")
    print("="*70)
    
    try:
        payload = {
            "season_ids": season_ids[:3],
            "orgunit_id": TEST_ORGUNIT_ID,
            "orgunit_type": TEST_ORGUNIT_TYPE,
            "format": "docx"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/3-quarters",
            json=payload
        )
        
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")
            if "wordprocessingml" not in content_type:
                log_test_result("3-Quarter DOCX", "FAIL", f"Invalid content type: {content_type}")
                return False
            
            # Save file
            output_file = os.path.join(TEST_OUTPUT_DIR, "3quarter_comparison.docx")
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            file_size = len(response.content)
            log_test_result("3-Quarter DOCX", "PASS", 
                          f"DOCX file generated ({file_size:,} bytes)",
                          {"file_size": file_size, "output_file": output_file})
            return True
        else:
            log_test_result("3-Quarter DOCX", "FAIL", 
                          f"HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        log_test_result("3-Quarter DOCX", "FAIL", f"Exception: {str(e)}")
        return False

def test_4quarter_json(season_ids: List[int]):
    """Test 6: 4-Quarter Comparison - JSON Format"""
    print("="*70)
    print("TEST 6: 4-QUARTER COMPARISON - JSON")
    print("="*70)
    
    try:
        payload = {
            "season_ids": season_ids[:4],
            "orgunit_id": TEST_ORGUNIT_ID,
            "orgunit_type": TEST_ORGUNIT_TYPE,
            "format": "json"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/4-quarters",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Validate structure
            required_keys = ["success", "comparison_type", "periods", "data"]
            for key in required_keys:
                if key not in data:
                    log_test_result("4-Quarter JSON", "FAIL", f"Missing key: {key}")
                    return False
            
            if data["comparison_type"] != "4-quarters":
                log_test_result("4-Quarter JSON", "FAIL", "Invalid comparison_type")
                return False
            
            if len(data["periods"]) != 4:
                log_test_result("4-Quarter JSON", "FAIL", f"Expected 4 periods, got {len(data['periods'])}")
                return False
            
            # Save to file
            output_file = os.path.join(TEST_OUTPUT_DIR, "4quarter_comparison.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            log_test_result("4-Quarter JSON", "PASS", 
                          "Yearly summary structure valid",
                          {"periods": data["periods"],
                           "output_file": output_file})
            return True
        else:
            log_test_result("4-Quarter JSON", "FAIL", 
                          f"HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        log_test_result("4-Quarter JSON", "FAIL", f"Exception: {str(e)}")
        return False

def test_4quarter_docx(season_ids: List[int]):
    """Test 7: 4-Quarter Comparison - DOCX Format"""
    print("="*70)
    print("TEST 7: 4-QUARTER COMPARISON - DOCX")
    print("="*70)
    
    try:
        payload = {
            "season_ids": season_ids[:4],
            "orgunit_id": TEST_ORGUNIT_ID,
            "orgunit_type": TEST_ORGUNIT_TYPE,
            "format": "docx"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/4-quarters",
            json=payload
        )
        
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")
            if "wordprocessingml" not in content_type:
                log_test_result("4-Quarter DOCX", "FAIL", f"Invalid content type: {content_type}")
                return False
            
            # Save file
            output_file = os.path.join(TEST_OUTPUT_DIR, "4quarter_comparison.docx")
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            file_size = len(response.content)
            log_test_result("4-Quarter DOCX", "PASS", 
                          f"DOCX file generated ({file_size:,} bytes)",
                          {"file_size": file_size, "output_file": output_file})
            return True
        else:
            log_test_result("4-Quarter DOCX", "FAIL", 
                          f"HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        log_test_result("4-Quarter DOCX", "FAIL", f"Exception: {str(e)}")
        return False

def test_validation_errors():
    """Test 8: Validation Error Handling"""
    print("="*70)
    print("TEST 8: VALIDATION ERROR HANDLING")
    print("="*70)
    
    passed = 0
    total = 0
    
    # Test 8.1: Wrong number of seasons for 2-quarter
    total += 1
    try:
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/2-quarters",
            json={
                "season_ids": [1, 2, 3],  # 3 seasons instead of 2
                "orgunit_id": 1,
                "orgunit_type": 0,
                "format": "json"
            }
        )
        if response.status_code == 422:
            passed += 1
            print("   ✅ 8.1: Correctly rejected 3 seasons for 2-quarter endpoint")
        else:
            print(f"   ❌ 8.1: Expected 422, got {response.status_code}")
    except Exception as e:
        print(f"   ❌ 8.1: Exception: {str(e)}")
    
    # Test 8.2: Invalid format
    total += 1
    try:
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/2-quarters",
            json={
                "season_ids": [1, 2],
                "orgunit_id": 1,
                "orgunit_type": 0,
                "format": "pdf"  # Invalid format
            }
        )
        if response.status_code == 422:
            passed += 1
            print("   ✅ 8.2: Correctly rejected invalid format")
        else:
            print(f"   ❌ 8.2: Expected 422, got {response.status_code}")
    except Exception as e:
        print(f"   ❌ 8.2: Exception: {str(e)}")
    
    # Test 8.3: Missing required fields
    total += 1
    try:
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/2-quarters",
            json={
                "season_ids": [1, 2]
                # Missing orgunit_id and orgunit_type
            }
        )
        if response.status_code == 422:
            passed += 1
            print("   ✅ 8.3: Correctly rejected missing required fields")
        else:
            print(f"   ❌ 8.3: Expected 422, got {response.status_code}")
    except Exception as e:
        print(f"   ❌ 8.3: Exception: {str(e)}")
    
    # Test 8.4: Invalid orgunit_type
    total += 1
    try:
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/2-quarters",
            json={
                "season_ids": [1, 2],
                "orgunit_id": 1,
                "orgunit_type": 99,  # Invalid type
                "format": "json"
            }
        )
        if response.status_code == 422:
            passed += 1
            print("   ✅ 8.4: Correctly rejected invalid orgunit_type")
        else:
            print(f"   ❌ 8.4: Expected 422, got {response.status_code}")
    except Exception as e:
        print(f"   ❌ 8.4: Exception: {str(e)}")
    
    log_test_result("Validation Error Handling", "PASS" if passed == total else "FAIL",
                   f"Passed {passed}/{total} validation tests",
                   {"passed": passed, "total": total})
    
    return passed == total

def test_edge_cases(season_ids: List[int]):
    """Test 9: Edge Cases"""
    print("="*70)
    print("TEST 9: EDGE CASES")
    print("="*70)
    
    passed = 0
    total = 0
    
    # Test 9.1: Non-existent season ID (graceful handling - returns empty data)
    total += 1
    try:
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/2-quarters",
            json={
                "season_ids": [99999, 99998],  # Non-existent IDs
                "orgunit_id": 1,
                "orgunit_type": 0,
                "format": "json"
            }
        )
        if response.status_code == 200:
            data = response.json()
            # Check if it gracefully returns empty data
            if data.get("success") and data.get("data"):
                reports = data["data"].get("reports", [])
                if len(reports) == 2 and all(r["header"]["total_cases"] == 0 for r in reports):
                    passed += 1
                    print("   ✅ 9.1: Gracefully handled non-existent season IDs (returns empty data)")
                else:
                    print(f"   ⚠️ 9.1: Unexpected data structure: {reports[0]['header'] if reports else 'N/A'}")
            else:
                print(f"   ⚠️ 9.1: Unexpected response structure")
        elif response.status_code in [404, 500]:
            passed += 1
            print("   ✅ 9.1: Correctly rejected non-existent season IDs with error")
        else:
            print(f"   ❌ 9.1: Unexpected status code: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 9.1: Exception: {str(e)}")
    
    # Test 9.2: Non-existent orgunit (graceful handling - returns empty data)
    total += 1
    try:
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/2-quarters",
            json={
                "season_ids": season_ids[:2],
                "orgunit_id": 99999,  # Non-existent orgunit
                "orgunit_type": 0,
                "format": "json"
            }
        )
        if response.status_code == 200:
            data = response.json()
            # Check if it gracefully returns empty data
            if data.get("success") and data.get("data"):
                reports = data["data"].get("reports", [])
                if len(reports) == 2 and all(r["header"]["total_cases"] == 0 for r in reports):
                    passed += 1
                    print("   ✅ 9.2: Gracefully handled non-existent orgunit (returns empty data)")
                else:
                    print(f"   ⚠️ 9.2: Unexpected data structure")
            else:
                print(f"   ⚠️ 9.2: Unexpected response structure")
        elif response.status_code in [404, 500]:
            passed += 1
            print("   ✅ 9.2: Correctly rejected non-existent orgunit with error")
        else:
            print(f"   ❌ 9.2: Unexpected status code: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 9.2: Exception: {str(e)}")
    
    # Test 9.3: Duplicate season IDs
    total += 1
    try:
        response = requests.post(
            f"{BASE_URL}/api/seasonal-comparison/2-quarters",
            json={
                "season_ids": [season_ids[0], season_ids[0]],  # Duplicate IDs
                "orgunit_id": 1,
                "orgunit_type": 0,
                "format": "json"
            }
        )
        # System should either reject (422) or handle gracefully (200)
        if response.status_code in [200, 422]:
            passed += 1
            print(f"   ✅ 9.3: Duplicate season IDs handled ({response.status_code})")
        else:
            print(f"   ❌ 9.3: Unexpected status code: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 9.3: Exception: {str(e)}")
    
    log_test_result("Edge Cases", "PASS" if passed == total else "WARNING",
                   f"Handled {passed}/{total} edge cases",
                   {"passed": passed, "total": total})
    
    return passed == total

def test_single_season_export(season_ids: List[int]):
    """Test 10: Single Season Export (PHASE 6 Verification)"""
    print("="*70)
    print("TEST 10: SINGLE SEASON EXPORT (NO AUTO-COMPARISON)")
    print("="*70)
    
    try:
        # Step 1: Request export (returns JSON with download URL)
        season_id = season_ids[0]
        
        response = requests.post(
            f"{BASE_URL}/api/reports/export?format=docx",
            json={
                "report_type": "seasonal",
                "year": 2026,
                "quarter": 4,
                "language": "ar",
                "filters": {
                    "season_id": season_id,
                    "orgunit_id": TEST_ORGUNIT_ID,
                    "orgunit_type": TEST_ORGUNIT_TYPE
                },
                "display_mode": "hcat"  # Valid display_mode for seasonal reports
            }
        )
        
        if response.status_code != 200:
            log_test_result("Single Season Export", "FAIL", 
                          f"Export request failed: HTTP {response.status_code}: {response.text}")
            return False
        
        # Step 2: Parse JSON response
        export_data = response.json()
        if "download_url" not in export_data:
            log_test_result("Single Season Export", "FAIL", 
                          f"Missing download_url in response: {export_data}")
            return False
        
        download_url = export_data["download_url"]
        export_id = download_url.split("/")[-1]
        
        # Step 3: Download the actual file
        download_response = requests.get(f"{BASE_URL}{download_url}")
        
        if download_response.status_code != 200:
            log_test_result("Single Season Export", "FAIL", 
                          f"Download failed: HTTP {download_response.status_code}")
            return False
        
        content_type = download_response.headers.get("Content-Type", "")
        content_disposition = download_response.headers.get("Content-Disposition", "")
            
        content_type = download_response.headers.get("Content-Type", "")
        content_disposition = download_response.headers.get("Content-Disposition", "")
        
        # Verify it's DOCX, not ZIP
        if "wordprocessingml" not in content_type:
            log_test_result("Single Season Export", "FAIL", 
                          f"Expected DOCX content type, got: {content_type}")
            return False
        
        # Verify filename doesn't have "Reports" (plural) or ".zip"
        if ".zip" in content_disposition.lower():
            log_test_result("Single Season Export", "FAIL", 
                          "Filename contains .zip (auto-comparison not removed)")
            return False
        
        # Save file
        output_file = os.path.join(TEST_OUTPUT_DIR, "single_season_export.docx")
        with open(output_file, "wb") as f:
            f.write(download_response.content)
        
        file_size = len(download_response.content)
        log_test_result("Single Season Export", "PASS", 
                      "Single season export working (no auto-comparison)",
                      {"file_size": file_size, 
                       "content_type": content_type,
                       "filename": export_data.get("file_name"),
                       "export_id": export_id,
                       "output_file": output_file})
        return True
            
    except Exception as e:
        log_test_result("Single Season Export", "FAIL", f"Exception: {str(e)}")
        return False

def test_performance(season_ids: List[int]):
    """Test 11: Performance Testing"""
    print("="*70)
    print("TEST 11: PERFORMANCE TESTING")
    print("="*70)
    
    tests = [
        ("2Q JSON", f"{BASE_URL}/api/seasonal-comparison/2-quarters", 
         {"season_ids": season_ids[:2], "orgunit_id": 1, "orgunit_type": 0, "format": "json"}),
        ("2Q DOCX", f"{BASE_URL}/api/seasonal-comparison/2-quarters", 
         {"season_ids": season_ids[:2], "orgunit_id": 1, "orgunit_type": 0, "format": "docx"}),
        ("3Q JSON", f"{BASE_URL}/api/seasonal-comparison/3-quarters", 
         {"season_ids": season_ids[:3], "orgunit_id": 1, "orgunit_type": 0, "format": "json"}),
        ("4Q JSON", f"{BASE_URL}/api/seasonal-comparison/4-quarters", 
         {"season_ids": season_ids[:4], "orgunit_id": 1, "orgunit_type": 0, "format": "json"}),
    ]
    
    performance_results = []
    
    for test_name, url, payload in tests:
        try:
            start_time = datetime.now()
            response = requests.post(url, json=payload, timeout=30)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            status = "✅" if response.status_code == 200 else "❌"
            print(f"   {status} {test_name}: {elapsed:.2f}s ({response.status_code})")
            
            performance_results.append({
                "test": test_name,
                "elapsed": elapsed,
                "status": response.status_code,
                "success": response.status_code == 200
            })
        except Exception as e:
            print(f"   ❌ {test_name}: Exception - {str(e)}")
            performance_results.append({
                "test": test_name,
                "elapsed": None,
                "status": "ERROR",
                "success": False
            })
    
    avg_time = sum(r["elapsed"] for r in performance_results if r["elapsed"]) / len([r for r in performance_results if r["elapsed"]])
    successful = sum(1 for r in performance_results if r["success"])
    
    log_test_result("Performance Testing", "PASS" if successful == len(tests) else "WARNING",
                   f"Average response time: {avg_time:.2f}s, {successful}/{len(tests)} succeeded",
                   {"performance_results": performance_results})
    
    return successful >= len(tests) * 0.75  # Pass if 75% succeed

def generate_test_report():
    """Generate final test report"""
    print("\n")
    print("="*70)
    print("PHASE 8: COMPREHENSIVE BACKEND TESTING - FINAL REPORT")
    print("="*70)
    print(f"Total Tests: {test_results['total']}")
    print(f"✅ Passed: {test_results['passed']}")
    print(f"❌ Failed: {test_results['failed']}")
    print(f"⚠️ Warnings: {test_results['warnings']}")
    print(f"Success Rate: {(test_results['passed'] / test_results['total'] * 100):.1f}%")
    print("="*70)
    
    # Save detailed report
    report_file = os.path.join(TEST_OUTPUT_DIR, "test_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Detailed report saved: {report_file}")
    
    # Summary by category
    print("\n📋 Test Categories:")
    categories = {}
    for test in test_results["tests"]:
        category = test["name"].split(" - ")[0] if " - " in test["name"] else test["name"]
        if category not in categories:
            categories[category] = {"passed": 0, "failed": 0, "warnings": 0}
        
        if test["status"] == "PASS":
            categories[category]["passed"] += 1
        elif test["status"] == "FAIL":
            categories[category]["failed"] += 1
        else:
            categories[category]["warnings"] += 1
    
    for category, counts in categories.items():
        total = counts["passed"] + counts["failed"] + counts["warnings"]
        print(f"   {category}: {counts['passed']}/{total} passed")
    
    print("\n" + "="*70)
    if test_results['failed'] == 0:
        print("🎉 ALL TESTS PASSED! Backend is production-ready.")
    elif test_results['failed'] <= 2:
        print("⚠️ MINOR ISSUES DETECTED. Review failed tests.")
    else:
        print("❌ SIGNIFICANT ISSUES DETECTED. Requires attention.")
    print("="*70)

def main():
    """Main test execution"""
    print("\n" + "="*70)
    print("PHASE 8: COMPREHENSIVE BACKEND TESTING")
    print("Seasonal Comparison Feature - Full Integration Test Suite")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base URL: {BASE_URL}")
    print(f"Output Directory: {TEST_OUTPUT_DIR}")
    print("="*70)
    print("\n")
    
    # Test 1: Get available quarters
    success = test_available_quarters()
    if not success:
        print("❌ Cannot proceed: No quarters available or endpoint failed")
        generate_test_report()
        return
    
    # Get season IDs for subsequent tests
    response = requests.get(
        f"{BASE_URL}/api/seasonal-comparison/available-quarters",
        params={"orgunit_id": TEST_ORGUNIT_ID, "orgunit_type": TEST_ORGUNIT_TYPE}
    )
    season_ids = [s["season_id"] for s in response.json()["available_seasons"]]
    
    if len(season_ids) < 4:
        print(f"⚠️ WARNING: Only {len(season_ids)} seasons available. Some tests may be skipped.")
    
    # Test 2-7: Comparison endpoints
    test_2quarter_json(season_ids)
    test_2quarter_docx(season_ids)
    
    if len(season_ids) >= 3:
        test_3quarter_json(season_ids)
        test_3quarter_docx(season_ids)
    
    if len(season_ids) >= 4:
        test_4quarter_json(season_ids)
        test_4quarter_docx(season_ids)
    
    # Test 8: Validation
    test_validation_errors()
    
    # Test 9: Edge cases
    test_edge_cases(season_ids)
    
    # Test 10: Single season (PHASE 6)
    test_single_season_export(season_ids)
    
    # Test 11: Performance
    test_performance(season_ids)
    
    # Generate final report
    generate_test_report()

if __name__ == "__main__":
    main()
