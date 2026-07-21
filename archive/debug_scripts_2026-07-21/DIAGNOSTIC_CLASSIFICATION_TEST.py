"""
DIAGNOSTIC TEST: Classification Failure Investigation

This script tests the classification pipeline to isolate:
1. Whether failure depends on input text (not role - proven role-independent)
2. Which predictor module throws the error
3. Exact exception type (KeyError, ValueError, etc.)

Run from: Patient_Feedback/backend
Command: python DIAGNOSTIC_CLASSIFICATION_TEST.py
"""

import sys
import os
from pathlib import Path

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

print(f"Workspace root: {workspace_root}")
print("=" * 70)


# ============================================================
# TEST 1: Direct Classification Pipeline Test
# ============================================================

def test_direct_classification():
    """
    Test the classify_feedback function directly with various inputs.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Direct Classification Pipeline Test")
    print("=" * 70)
    
    from models_directory.Classification_Models.package_models import classify_feedback
    
    # Test texts - use variety to trigger different domains
    test_texts = [
        # Original test text
        ("Test 1: Basic complaint", "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج"),
        
        # Administrative/Management domain text
        ("Test 2: Administrative", "مشكلة في إجراءات التسجيل والاستقبال"),
        
        # Relational domain text 
        ("Test 3: Relational", "الممرضة لم تتواصل بشكل جيد مع المريض"),
        
        # Clinical domain text
        ("Test 4: Clinical", "هناك خطأ في وصف الدواء للمريض"),
        
        # Simple short text
        ("Test 5: Short text", "شكوى"),
        
        # Longer text
        ("Test 6: Long text", "المريض يعاني من مشاكل متعددة في الرعاية الصحية بما في ذلك التأخير في التشخيص وعدم التواصل الجيد من قبل الطاقم الطبي وكذلك مشاكل في النظافة والإدارة"),
    ]
    
    results = []
    
    for label, text in test_texts:
        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"Text: {text[:80]}...")
        print(f"{'='*60}")
        
        try:
            result = classify_feedback(
                patient_text=text,
                text_2="",
                text_3="",
                Print=True  # Enable verbose output
            )
            print(f"\n✅ SUCCESS: {label}")
            print(f"   Domain: {result.get('domain')} ({result.get('domain_id')})")
            print(f"   Category: {result.get('category')} ({result.get('category_id')})")
            print(f"   Subcategory: {result.get('sub_category')} ({result.get('sub_category_id')})")
            results.append((label, "SUCCESS", result))
            
        except KeyError as e:
            print(f"\n❌ KEYERROR: {label}")
            print(f"   Key: {e}")
            print(f"   Type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            results.append((label, "KEYERROR", str(e)))
            
        except Exception as e:
            print(f"\n❌ EXCEPTION: {label}")
            print(f"   Type: {type(e).__name__}")
            print(f"   Message: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((label, type(e).__name__, str(e)))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    success_count = sum(1 for r in results if r[1] == "SUCCESS")
    failure_count = len(results) - success_count
    
    print(f"Total: {len(results)} | Success: {success_count} | Failed: {failure_count}")
    
    for label, status, detail in results:
        if status == "SUCCESS":
            print(f"  ✅ {label}: Domain={detail.get('domain_id')}, Cat={detail.get('category_id')}")
        else:
            print(f"  ❌ {label}: {status} - {detail}")
    
    return results


# ============================================================
# TEST 2: Isolated Hierarchical Predictor Test
# ============================================================

def test_hierarchical_predictor():
    """
    Test just the hierarchical predictor with a sample embedding.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Hierarchical Predictor Test")
    print("=" * 70)
    
    import numpy as np
    from models_directory.Classification_Models.Stage.modular_functions import get_embedding
    from models_directory.Classification_Models.Hierarchical_Classification_Model.hierarchical_predictor import (
        hierarchical_predict_embeddings,
        predict_domain_embedding
    )
    
    test_texts = [
        "المريض يشكو من ألم شديد في البطن",
        "مشكلة في إجراءات التسجيل",
        "التواصل مع الممرضة كان سيئا",
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n--- Text {i+1}: {text[:50]}... ---")
        
        try:
            # Get embedding
            raw_emb = get_embedding(text)
            embedding = np.frombuffer(raw_emb, dtype=np.float32)
            print(f"Embedding shape: {embedding.shape}")
            
            # Test domain prediction first
            print("\nDomain prediction:")
            domain_result = predict_domain_embedding(embedding)
            print(f"  Domain predictions: {domain_result}")
            
            # Full hierarchical prediction
            print("\nFull hierarchical prediction:")
            result = hierarchical_predict_embeddings(embedding)
            print(f"  ✅ Result: {result}")
            
        except Exception as e:
            print(f"  ❌ Exception: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()


# ============================================================
# TEST 3: XGB Model Label Range Check
# ============================================================

def test_xgb_label_ranges():
    """
    Test XGB models to see what class indices they can produce.
    """
    print("\n" + "=" * 70)
    print("TEST 3: XGB Model Label Range Check")
    print("=" * 70)
    
    import numpy as np
    import joblib
    from xgboost import XGBClassifier
    
    # Check domain 2 category model
    base_dir = workspace_root / "models_directory" / "Classification_Models" / "Hierarchical_Classification_Model" / "category" / "domain_2" / "vocab_models"
    
    print(f"\nChecking domain_2 category XGB model at: {base_dir}")
    
    try:
        xgb = XGBClassifier()
        xgb.load_model(str(base_dir / "xgb_category_domain2.json"))
        
        # Print model info
        print(f"  n_classes: {xgb.n_classes_}")
        print(f"  classes: {xgb.classes_ if hasattr(xgb, 'classes_') else 'N/A'}")
        
        # The temp_to_label map in the predictor
        temp_to_label = {0: 2, 1: 3}
        print(f"  temp_to_label map: {temp_to_label}")
        print(f"  Expected n_classes: {len(temp_to_label)}")
        
        if xgb.n_classes_ != len(temp_to_label):
            print(f"  ⚠️ WARNING: n_classes ({xgb.n_classes_}) != len(temp_to_label) ({len(temp_to_label)})")
            print(f"  This could cause KeyError if XGB outputs class index >= {len(temp_to_label)}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Check other domain models too
    for domain in [1, 3]:
        base_dir = workspace_root / "models_directory" / "Classification_Models" / "Hierarchical_Classification_Model" / "category" / f"domain_{domain}" / "vocab_models"
        
        try:
            xgb = XGBClassifier()
            xgb.load_model(str(base_dir / f"xgb_category_domain{domain}.json"))
            print(f"\nDomain {domain} category XGB: n_classes = {xgb.n_classes_}")
        except Exception as e:
            print(f"\nDomain {domain}: Error - {e}")


# ============================================================
# TEST 4: HTTP Endpoint Test (requires running server)
# ============================================================

def test_http_endpoint():
    """
    Test the HTTP endpoint with different users.
    Requires the server to be running on localhost:8000
    """
    print("\n" + "=" * 70)
    print("TEST 4: HTTP Endpoint Test")
    print("=" * 70)
    
    import requests
    
    BASE_URL = "http://localhost:8000"
    
    # Define test users
    test_users = [
        ("software_admin", "admin123"),
        ("supervisor", "super123"),
        ("worker", "worker123"),
    ]
    
    test_payload = {
        "text": "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج",
        "explain": True
    }
    
    print(f"\nTest payload: {test_payload}")
    
    for username, password in test_users:
        print(f"\n--- Testing user: {username} ---")
        
        session = requests.Session()
        
        try:
            # Login
            login_resp = session.post(
                f"{BASE_URL}/api/auth/login",
                json={"username": username, "password": password}
            )
            
            if login_resp.status_code != 200:
                print(f"  Login failed: {login_resp.status_code} - {login_resp.text}")
                continue
            
            print(f"  Login: ✅ Success")
            
            # Call classification endpoint
            classify_resp = session.post(
                f"{BASE_URL}/api/classification/classify",
                json=test_payload
            )
            
            print(f"  Classification: Status {classify_resp.status_code}")
            
            if classify_resp.status_code == 200:
                result = classify_resp.json()
                print(f"    ✅ Success")
                if "classifications" in result:
                    cls = result["classifications"]
                    print(f"    Domain: {cls.get('domain')} ({cls.get('domain_id')})")
                    print(f"    Category: {cls.get('category')} ({cls.get('category_id')})")
            else:
                print(f"    ❌ Failed: {classify_resp.text[:200]}")
                
        except requests.exceptions.ConnectionError:
            print(f"  ❌ Connection error - is the server running on {BASE_URL}?")
            break
        except Exception as e:
            print(f"  ❌ Exception: {e}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DIAGNOSTIC: Classification Failure Investigation")
    print("=" * 70)
    
    print("\nRunning tests...\n")
    
    # Run direct classification test
    test_direct_classification()
    
    # Run hierarchical predictor test
    test_hierarchical_predictor()
    
    # Run XGB label range check
    test_xgb_label_ranges()
    
    # Optionally run HTTP test (comment out if server not running)
    # test_http_endpoint()
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("\nCheck the output above to identify:")
    print("1. Which texts cause failures")
    print("2. Which predictor module throws the error")
    print("3. The exact exception type and message")
    print("\nIf you see 'KeyError: 2', it means the XGB model is outputting")
    print("class index 2 but temp_to_label only has keys {0, 1}")
