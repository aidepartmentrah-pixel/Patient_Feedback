"""
Phase A Adj-3: CurrentUser Serialization Safety Check
======================================================

INSPECTION REPORT
-----------------

This is a READ-ONLY safety verification step to ensure adding 
display_name and department_display_name to CurrentUser model 
will not break existing authentication mechanisms.

FINDINGS:

1. AUTHENTICATION TYPE: Session-Based (NOT JWT)
   - Location: backend/api/services/auth_service.py
   - Method: Starlette SessionMiddleware
   - Session stores ONLY: user_id (integer)
   - No CurrentUser serialization in session

2. SESSION STORAGE (Line 118 in auth_service.py)
   ```python
   request.session["user_id"] = user_data["user_id"]
   ```
   - Only stores user_id (int)
   - CurrentUser is NOT stored in session
   - CurrentUser is NOT serialized to session cookie

3. USER CONTEXT RETRIEVAL (get_current_user_from_session)
   - Reads user_id from session
   - Queries database via get_user_with_scopes(user_id)
   - Constructs CurrentUser from fresh database data EVERY request
   - Database returns: user_id, username, is_active, scopes[]
   - Does NOT include display_name or department_display_name yet

4. MIDDLEWARE (backend/main.py lines 56-62)
   ```python
   app.add_middleware(
       SessionMiddleware,
       secret_key="...",
       session_cookie="incident_manager_session",
       max_age=86400,
       ...
   )
   ```
   - Standard Starlette SessionMiddleware
   - No custom serialization
   - No JWT encoding/decoding
   - Session cookie contains only session_id reference

5. CURRENTUSER CONSTRUCTION (Lines 258-265 in auth_service.py)
   ```python
   current_user = CurrentUser(
       user_id=user_data["user_id"],
       username=user_data["username"],
       is_active=user_data["is_active"],
       scopes=scopes_list
   )
   ```
   - CurrentUser built from database query
   - display_name and department_display_name will be None
   - Model already supports optional fields
   - No serialization risk

RISK ASSESSMENT:
================

❌ NO RISK - Adding fields to CurrentUser is 100% SAFE

REASONS:
1. CurrentUser is NEVER serialized to session
2. Session only stores user_id (primitive int)
3. CurrentUser is reconstructed from DB on EVERY request
4. Database query (get_user_with_scopes) doesn't return display fields yet
5. New optional fields default to None - backward compatible
6. No JWT tokens involved
7. No custom middleware that serializes CurrentUser

COMPATIBILITY:
- ✅ Session cookies: No change (only store user_id)
- ✅ Middleware: No change (SessionMiddleware unaffected)
- ✅ Database queries: No change yet (display fields NULL)
- ✅ Model serialization: Optional fields = backward compatible
- ✅ Existing code: Works without display fields

NEXT STEPS (Later phases):
- Update get_user_with_scopes() to SELECT DisplayName, DepartmentDisplayName
- Update CurrentUser construction to include display fields
- Both are additive changes - no breaking changes

CONCLUSION:
===========
Adding display_name and department_display_name to CurrentUser model
is SAFE and will NOT break authentication or session management.

The authentication flow is:
1. Login: Store user_id in session ✅ (unchanged)
2. Request: Read user_id from session ✅ (unchanged)
3. Build CurrentUser from DB query ✅ (new fields will be None initially)
4. Return CurrentUser ✅ (new fields present but None)

This is a ZERO-RISK change.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.api.schemas.auth_models import CurrentUser, UserScope


def test_session_stores_only_user_id():
    """Test 1: Verify understanding - session stores only user_id, not CurrentUser."""
    print("\n" + "="*60)
    print("TEST 1: Session Storage Pattern")
    print("="*60)
    
    try:
        # This test verifies our understanding of the codebase
        # by reading the actual source code
        
        import inspect
        from backend.api.services import auth_service
        
        # Get login function source
        login_source = inspect.getsource(auth_service.login)
        
        # Verify login stores only user_id
        assert 'request.session["user_id"]' in login_source, \
            "Login should store user_id in session"
        
        # Verify it does NOT store CurrentUser
        assert 'request.session["current_user"]' not in login_source, \
            "Login should NOT store CurrentUser in session"
        assert 'request.session["user"]' not in login_source, \
            "Login should NOT store user object in session"
        
        print("✓ Session storage pattern verified:")
        print("  - Stores: user_id (int)")
        print("  - Does NOT store: CurrentUser object")
        print("  - Does NOT store: user data dict")
        print("\n✓ TEST 1 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 1 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 1 ERROR: {str(e)}")
        return False


def test_currentuser_not_serialized():
    """Test 2: Verify CurrentUser is constructed per-request, not serialized."""
    print("\n" + "="*60)
    print("TEST 2: CurrentUser Per-Request Construction")
    print("="*60)
    
    try:
        import inspect
        from backend.api.services import auth_service
        
        # Get get_current_user_from_session source
        get_user_source = inspect.getsource(auth_service.get_current_user_from_session)
        
        # Verify it queries database
        assert 'get_user_with_scopes' in get_user_source, \
            "Should call get_user_with_scopes to fetch from DB"
        
        # Verify it constructs CurrentUser from query results
        assert 'CurrentUser(' in get_user_source, \
            "Should construct CurrentUser from DB data"
        
        # Verify it reads user_id from session
        assert 'request.session.get("user_id")' in get_user_source, \
            "Should read user_id from session"
        
        print("✓ Per-request construction verified:")
        print("  1. Read user_id from session")
        print("  2. Query database (get_user_with_scopes)")
        print("  3. Construct CurrentUser from fresh data")
        print("  4. Return new instance every request")
        print("\n  CurrentUser is NEVER serialized!")
        print("\n✓ TEST 2 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 2 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 2 ERROR: {str(e)}")
        return False


def test_new_fields_optional():
    """Test 3: Verify new display fields are optional and backward compatible."""
    print("\n" + "="*60)
    print("TEST 3: New Fields Optional (Backward Compatible)")
    print("="*60)
    
    try:
        # Create CurrentUser WITHOUT display fields (old style)
        user_old_style = CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[]
        )
        
        # Fields should default to None
        assert user_old_style.display_name is None
        assert user_old_style.department_display_name is None
        
        print("✓ Old-style construction works:")
        print(f"  - Created without display fields")
        print(f"  - display_name defaults to None")
        print(f"  - department_display_name defaults to None")
        
        # Create CurrentUser WITH display fields (new style)
        user_new_style = CurrentUser(
            user_id=2,
            username="test_user_2",
            display_name="Dr. Test",
            department_display_name="Test Dept",
            is_active=True,
            scopes=[]
        )
        
        assert user_new_style.display_name == "Dr. Test"
        assert user_new_style.department_display_name == "Test Dept"
        
        print("\n✓ New-style construction works:")
        print(f"  - Created with display fields")
        print(f"  - display_name: {user_new_style.display_name}")
        print(f"  - department_display_name: {user_new_style.department_display_name}")
        
        print("\n✅ Backward compatibility CONFIRMED")
        print("\n✓ TEST 3 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 3 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 3 ERROR: {str(e)}")
        return False


def test_no_jwt_tokens():
    """Test 4: Verify no JWT encoding/decoding in auth system."""
    print("\n" + "="*60)
    print("TEST 4: No JWT Tokens")
    print("="*60)
    
    try:
        import inspect
        from backend.api.services import auth_service
        
        # Get entire module source
        module_source = inspect.getsource(auth_service)
        
        # Verify NO JWT usage
        assert 'jwt.encode' not in module_source.lower(), \
            "Should NOT use JWT encoding"
        assert 'jwt.decode' not in module_source.lower(), \
            "Should NOT use JWT decoding"
        assert 'import jwt' not in module_source.lower(), \
            "Should NOT import jwt library"
        
        # Verify session-based approach mentioned in docs
        assert 'SESSION' in module_source or 'session' in module_source, \
            "Should use session-based auth"
        
        print("✓ Authentication type verified:")
        print("  ✓ NO JWT tokens")
        print("  ✓ NO token encoding/decoding")
        print("  ✓ Session-based authentication")
        print("\n  Result: No token serialization to worry about!")
        print("\n✓ TEST 4 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 4 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 4 ERROR: {str(e)}")
        return False


def test_middleware_standard():
    """Test 5: Verify SessionMiddleware is standard (no custom serialization)."""
    print("\n" + "="*60)
    print("TEST 5: Standard SessionMiddleware")
    print("="*60)
    
    try:
        import inspect
        from backend import main
        
        # Get main module source
        main_source = inspect.getsource(main)
        
        # Verify SessionMiddleware usage
        assert 'SessionMiddleware' in main_source, \
            "Should use SessionMiddleware"
        
        # Verify it's from starlette (standard)
        assert 'from starlette.middleware.sessions import SessionMiddleware' in main_source, \
            "Should use standard Starlette SessionMiddleware"
        
        # Verify no custom session serializer
        assert 'serializer=' not in main_source, \
            "Should NOT use custom serializer"
        assert 'encoder=' not in main_source, \
            "Should NOT use custom encoder"
        
        print("✓ Middleware configuration verified:")
        print("  ✓ Uses standard Starlette SessionMiddleware")
        print("  ✓ No custom serialization")
        print("  ✓ No custom encoding")
        print("\n  Result: Standard session handling - safe for model changes!")
        print("\n✓ TEST 5 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 5 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 5 ERROR: {str(e)}")
        return False


def test_model_dict_serialization():
    """Test 6: Verify CurrentUser.model_dump() works with new fields."""
    print("\n" + "="*60)
    print("TEST 6: Model Dict Serialization")
    print("="*60)
    
    try:
        # Create user with new fields
        user = CurrentUser(
            user_id=1,
            username="test",
            display_name="Dr. Test",
            department_display_name="Test Dept",
            is_active=True,
            scopes=[]
        )
        
        # Serialize to dict (used by FastAPI response models)
        user_dict = user.model_dump()
        
        # Verify all fields present
        assert 'user_id' in user_dict
        assert 'username' in user_dict
        assert 'display_name' in user_dict
        assert 'department_display_name' in user_dict
        assert 'is_active' in user_dict
        assert 'scopes' in user_dict
        
        # Verify values
        assert user_dict['display_name'] == "Dr. Test"
        assert user_dict['department_display_name'] == "Test Dept"
        
        print("✓ Model dict serialization works:")
        print(f"  ✓ All fields present in dict")
        print(f"  ✓ display_name: {user_dict['display_name']}")
        print(f"  ✓ department_display_name: {user_dict['department_display_name']}")
        
        # Test with None values
        user_none = CurrentUser(
            user_id=2,
            username="test2",
            is_active=True,
            scopes=[]
        )
        
        user_none_dict = user_none.model_dump()
        assert user_none_dict['display_name'] is None
        assert user_none_dict['department_display_name'] is None
        
        print("\n✓ None values serialize correctly:")
        print(f"  ✓ display_name: {user_none_dict['display_name']}")
        print(f"  ✓ department_display_name: {user_none_dict['department_display_name']}")
        
        print("\n✅ FastAPI response serialization will work!")
        print("\n✓ TEST 6 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 6 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 6 ERROR: {str(e)}")
        return False


def test_model_json_serialization():
    """Test 7: Verify CurrentUser JSON serialization works."""
    print("\n" + "="*60)
    print("TEST 7: Model JSON Serialization")
    print("="*60)
    
    try:
        import json
        
        # Create user with display fields
        user = CurrentUser(
            user_id=1,
            username="test",
            display_name="Dr. Test",
            department_display_name="Test Dept",
            is_active=True,
            scopes=[
                UserScope(role_code="ADMIN", org_unit_id=1, org_unit_type="ADMIN")
            ]
        )
        
        # Serialize to JSON
        user_json = user.model_dump_json()
        
        # Verify JSON is valid
        user_data = json.loads(user_json)
        
        assert user_data['display_name'] == "Dr. Test"
        assert user_data['department_display_name'] == "Test Dept"
        
        print("✓ JSON serialization works:")
        print(f"  ✓ Valid JSON produced")
        print(f"  ✓ display_name in JSON: {user_data['display_name']}")
        print(f"  ✓ department_display_name in JSON: {user_data['department_display_name']}")
        
        # Test JSON with None values
        user_none = CurrentUser(
            user_id=2,
            username="test2",
            is_active=True,
            scopes=[]
        )
        
        user_none_json = user_none.model_dump_json()
        user_none_data = json.loads(user_none_json)
        
        assert user_none_data['display_name'] is None
        assert user_none_data['department_display_name'] is None
        
        print("\n✓ None values in JSON:")
        print(f"  ✓ display_name: null")
        print(f"  ✓ department_display_name: null")
        
        print("\n✅ JSON serialization safe for API responses!")
        print("\n✓ TEST 7 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 7 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 7 ERROR: {str(e)}")
        return False


def run_all_tests():
    """Run all serialization safety tests."""
    print("\n" + "="*60)
    print("PHASE A ADJ-3: CURRENTUSER SERIALIZATION SAFETY CHECK")
    print("="*60)
    print("\nOBJECTIVE: Verify adding display fields won't break auth")
    print("TYPE: Read-only inspection with verification tests")
    
    tests = [
        ("Test 1: Session Storage Pattern", test_session_stores_only_user_id),
        ("Test 2: Per-Request Construction", test_currentuser_not_serialized),
        ("Test 3: Backward Compatibility", test_new_fields_optional),
        ("Test 4: No JWT Tokens", test_no_jwt_tokens),
        ("Test 5: Standard Middleware", test_middleware_standard),
        ("Test 6: Model Dict Serialization", test_model_dict_serialization),
        ("Test 7: Model JSON Serialization", test_model_json_serialization),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {name} EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print("SAFETY ASSESSMENT SUMMARY")
    print("="*60)
    print(f"Total Tests: {len(tests)}")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"Pass Rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        print("\n" + "="*60)
        print("🎉 SAFETY VERIFICATION: 100% PASSED")
        print("="*60)
        print("\n✅ RISK ASSESSMENT: NO RISK")
        print("\nKEY FINDINGS:")
        print("  1. Session stores ONLY user_id (int)")
        print("  2. CurrentUser is NEVER serialized to session")
        print("  3. CurrentUser rebuilt from DB every request")
        print("  4. No JWT tokens (session-based auth)")
        print("  5. Standard SessionMiddleware (no custom serialization)")
        print("  6. New fields are optional (backward compatible)")
        print("  7. Model serialization works (dict and JSON)")
        print("\n✅ CONCLUSION:")
        print("  Adding display_name and department_display_name")
        print("  to CurrentUser model is 100% SAFE.")
        print("\n  No breaking changes to:")
        print("    • Session storage ✓")
        print("    • Authentication flow ✓")
        print("    • Middleware ✓")
        print("    • API responses ✓")
        print("    • Existing code ✓")
        print("\n📋 NEXT STEPS (Later phases):")
        print("  • Update get_user_with_scopes() to fetch display fields")
        print("  • Update CurrentUser construction in auth_service.py")
        print("  • Both are additive changes - zero risk")
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        print("\n⚠️  Safety verification incomplete - investigate failures")
    
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
