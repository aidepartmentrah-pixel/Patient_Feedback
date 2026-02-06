"""
Test for MODULE 5.7 - Markdown Credential Export
Tests admin endpoint for exporting credentials as Markdown.

⚠️ TEST ONLY — This endpoint should be disabled in production

Run from backend directory:
    python test_module5_7_markdown_export.py
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from main import app

# Create test client
client = TestClient(app)


def clear_sessions():
    """Clear all test client sessions."""
    client.cookies.clear()


def login_as_admin():
    """Helper to login as software_admin."""
    response = client.post(
        "/api/auth/login",
        json={"username": "software_admin", "password": "admin123"}
    )
    return response


# ==================== TESTS ====================

def test_markdown_without_login():
    """Test that endpoint requires authentication."""
    print("\n" + "="*70)
    print("TEST 1: Markdown Export - Without Login")
    print("="*70)
    
    clear_sessions()
    
    response = client.get("/api/admin/testing/user-credentials-markdown")
    
    if response.status_code == 401:
        print("✅ PASS: Returns 401 Unauthorized when not logged in")
    else:
        print(f"❌ FAIL: Expected 401, got {response.status_code}")
        print(f"Response: {response.text[:200]}")


def test_markdown_with_admin():
    """Test successful markdown export with admin user."""
    print("\n" + "="*70)
    print("TEST 2: Markdown Export - With Admin Login")
    print("="*70)
    
    clear_sessions()
    
    # Login as admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed with {login_response.status_code}")
        return
    
    print("✓ Logged in as software_admin")
    
    # Get markdown export
    response = client.get("/api/admin/testing/user-credentials-markdown")
    
    if response.status_code == 200:
        print(f"✅ PASS: Markdown export successful")
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if 'text/markdown' in content_type:
            print(f"   ✓ Content-Type: {content_type}")
        else:
            print(f"   ⚠️  Content-Type: {content_type} (expected text/markdown)")
        
        # Check content
        markdown_content = response.text
        print(f"   ✓ Content length: {len(markdown_content)} bytes")
        
        # Verify markdown structure
        required_elements = [
            "# User Credentials",
            "| Username | Role | Org Unit | Active | Password |",
            "|----------|------|----------|--------|----------|"
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in markdown_content:
                missing_elements.append(element)
        
        if not missing_elements:
            print(f"   ✓ All required markdown elements present")
        else:
            print(f"   ⚠️  Missing elements: {missing_elements}")
        
        # Count rows (lines starting with |)
        table_rows = [line for line in markdown_content.split('\n') if line.startswith('|') and 'Username' not in line and '---' not in line]
        print(f"   ✓ Table rows: {len(table_rows)}")
        
        # Show sample output
        print(f"\n   Sample markdown output (first 10 lines):")
        for i, line in enumerate(markdown_content.split('\n')[:10]):
            print(f"   {line}")
        
        # Check for security warnings
        if "⚠️" in markdown_content and "WARNING" in markdown_content:
            print(f"\n   ✓ Security warnings present")
        
    else:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")
        print(f"Response: {response.text[:500]}")


def test_markdown_contains_test_users():
    """Test that known test users appear in markdown."""
    print("\n" + "="*70)
    print("TEST 3: Verify Test Users in Markdown")
    print("="*70)
    
    clear_sessions()
    
    # Login as admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed")
        return
    
    print("✓ Logged in as software_admin")
    
    # Get markdown
    response = client.get("/api/admin/testing/user-credentials-markdown")
    
    if response.status_code != 200:
        print(f"❌ FAIL: Markdown export failed")
        return
    
    markdown_content = response.text
    
    # Check for known test users
    known_users = ['software_admin', 'worker', 'supervisor']
    found_users = []
    
    print(f"Checking for known test users in markdown:")
    for user in known_users:
        if user in markdown_content:
            found_users.append(user)
            print(f"   ✓ Found: {user}")
        else:
            print(f"   ⚠️  Not found: {user}")
    
    if found_users:
        print(f"\n✅ PASS: Found {len(found_users)} known test user(s)")
    else:
        print(f"\n⚠️  No known test users found (may need bulk user creation)")


def test_non_admin_access():
    """Test that non-admin users cannot export markdown."""
    print("\n" + "="*70)
    print("TEST 4: Non-Admin Access Denial")
    print("="*70)
    
    clear_sessions()
    
    # Login as worker (not admin)
    login_response = client.post(
        "/api/auth/login",
        json={"username": "worker", "password": "worker123"}
    )
    
    if login_response.status_code != 200:
        print(f"⚠️  SKIP: Worker login failed, cannot test")
        return
    
    print("✓ Logged in as worker")
    
    # Try to export markdown
    response = client.get("/api/admin/testing/user-credentials-markdown")
    
    if response.status_code == 403:
        print("✅ PASS: Returns 403 Forbidden for non-admin user")
    else:
        print(f"❌ FAIL: Expected 403, got {response.status_code}")
        print(f"Response: {response.text[:200]}")


def test_markdown_format_valid():
    """Test that markdown is properly formatted."""
    print("\n" + "="*70)
    print("TEST 5: Markdown Format Validation")
    print("="*70)
    
    clear_sessions()
    
    # Login as admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed")
        return
    
    print("✓ Logged in as software_admin")
    
    # Get markdown
    response = client.get("/api/admin/testing/user-credentials-markdown")
    
    if response.status_code != 200:
        print(f"❌ FAIL: Markdown export failed")
        return
    
    markdown_content = response.text
    lines = markdown_content.split('\n')
    
    # Validate structure
    print(f"Validating markdown structure:")
    
    # Check for header
    if any("# User Credentials" in line for line in lines):
        print(f"   ✓ Markdown header present")
    
    # Check for table header
    table_header_found = False
    separator_found = False
    for i, line in enumerate(lines):
        if "| Username | Role | Org Unit | Active | Password |" in line:
            table_header_found = True
            # Check next line for separator
            if i + 1 < len(lines) and "|-------" in lines[i + 1]:
                separator_found = True
    
    if table_header_found:
        print(f"   ✓ Table header present")
    if separator_found:
        print(f"   ✓ Table separator present")
    
    # Count table rows (lines with 5 pipes)
    table_rows = [line for line in lines if line.count('|') >= 5 and 'Username' not in line and '---' not in line]
    print(f"   ✓ Valid table rows: {len(table_rows)}")
    
    # Check footer
    if "Total Users:" in markdown_content:
        print(f"   ✓ Footer with total count present")
    
    if "SECURITY" in markdown_content or "WARNING" in markdown_content:
        print(f"   ✓ Security notice present")
    
    if table_header_found and separator_found:
        print(f"\n✅ PASS: Markdown format is valid")
    else:
        print(f"\n❌ FAIL: Markdown format issues detected")


# ==================== MAIN RUNNER ====================

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("MODULE 5.7 - MARKDOWN CREDENTIAL EXPORT TESTS (TEST ONLY)")
    print("="*70)
    print("⚠️  WARNING: This endpoint should be disabled in production")
    
    test_markdown_without_login()
    test_markdown_with_admin()
    test_markdown_contains_test_users()
    test_non_admin_access()
    test_markdown_format_valid()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
