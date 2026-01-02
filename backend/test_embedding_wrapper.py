#!/usr/bin/env python
"""
Test Embedding Wrapper Integration

Tests the new add_corrected_record_to_ml() function to ensure:
1. Embeddings are generated correctly
2. Data is enriched with embedding fields
3. ML database insertion succeeds
4. Graceful handling of missing/empty fields
"""

import sys
import os
from datetime import datetime

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_embedding_imports():
    """Test 1: Verify embedding functions are available"""
    print("\n" + "=" * 70)
    print("TEST 1: EMBEDDING FUNCTIONS AVAILABLE")
    print("=" * 70)
    
    try:
        from models_directory.Classification_Models.Stage.modular_functions import (
            split_arabic_text_into_sentences,
            get_embedding,
            get_embedding_list,
            l2_normalize
        )
        print("✓ All embedding functions imported successfully")
        print("  - split_arabic_text_into_sentences")
        print("  - get_embedding")
        print("  - get_embedding_list")
        print("  - l2_normalize")
        return True
    except Exception as e:
        print(f"✗ Embedding functions import failed: {e}")
        return False


def test_wrapper_import():
    """Test 2: Verify wrapper function is available"""
    print("\n" + "=" * 70)
    print("TEST 2: WRAPPER FUNCTION AVAILABLE")
    print("=" * 70)
    
    try:
        from backend.ml_mapping import add_corrected_record_to_ml
        print("✓ Wrapper function imported successfully")
        print("  - add_corrected_record_to_ml()")
        return True
    except Exception as e:
        print(f"✗ Wrapper import failed: {e}")
        return False


def test_embedding_generation():
    """Test 3: Generate embeddings from Arabic text"""
    print("\n" + "=" * 70)
    print("TEST 3: EMBEDDING GENERATION")
    print("=" * 70)
    
    try:
        from models_directory.Classification_Models.Stage.modular_functions import (
            split_arabic_text_into_sentences,
            get_embedding,
            get_embedding_list
        )
        
        # Test data
        complaint = "المريض يشكو من آلام حادة في الظهر منذ أسبوع"
        immediate = "تم إعطاء مسكن ألم"
        taken = "تم إحالة المريض للتخصص"
        
        # Generate embeddings
        emb1 = get_embedding(complaint)
        emb2 = get_embedding(immediate)
        emb3 = get_embedding(taken)
        
        print(f"✓ Individual embeddings generated")
        print(f"  - embedding_text1: {len(emb1)} bytes (float32)")
        print(f"  - embedding_text2: {len(emb2)} bytes (float32)")
        print(f"  - embedding_text3: {len(emb3)} bytes (float32)")
        
        # Combination embeddings
        text123 = f"{complaint} {immediate} {taken}"
        emb123 = get_embedding(text123)
        print(f"✓ Combination embeddings generated")
        print(f"  - embedding_text123: {len(emb123)} bytes (float32)")
        
        # Sentence embeddings
        sentences = split_arabic_text_into_sentences(complaint, max_sentences=6)
        print(f"✓ Sentences extracted: {len(sentences)} sentences")
        for i, sent in enumerate(sentences, 1):
            print(f"  - Sentence {i}: {sent[:50]}...")
        
        sentence_embs = get_embedding_list(sentences)
        print(f"✓ Sentence embeddings generated: {len(sentence_embs)} embeddings")
        for i, emb in enumerate(sentence_embs, 1):
            print(f"  - sentence_{i}_embedding: {len(emb)} bytes (float32)")
        
        return True
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wrapper_with_arabic_text():
    """Test 4: Test wrapper with complete Arabic data"""
    print("\n" + "=" * 70)
    print("TEST 4: WRAPPER WITH ARABIC TEXT")
    print("=" * 70)
    
    try:
        from backend.ml_mapping import add_corrected_record_to_ml
        
        data = {
            'record_id': 999,  # Test record
            'patient_full_name': 'أحمد محمد',
            'complaint_text': 'المريض يشكو من آلام حادة في الظهر منذ أسبوع مع صعوبة الحركة',
            'immediate_action': 'تم إعطاء مسكن ألم وتطبيق كمادات دافئة',
            'taken_action': 'تم إحالة المريض لقسم العلاج الطبيعي للمتابعة',
            'feedback_received_date': datetime.now().strftime('%Y-%m-%d'),
            'domain_id': 1,
            'category_id': 1,
            'subcategory_id': 1,
            'severity_id': 1,
            'stage_id': 1,
            'harm_id': 1,
        }
        
        print("Input data:")
        print(f"  - record_id: {data['record_id']}")
        print(f"  - patient_name: {data['patient_full_name']}")
        print(f"  - complaint_text: {data['complaint_text'][:50]}...")
        print(f"  - immediate_action: {data['immediate_action'][:50]}...")
        print(f"  - taken_action: {data['taken_action'][:50]}...")
        
        print("\nCalling add_corrected_record_to_ml()...")
        add_corrected_record_to_ml(data)
        
        print("✓ Wrapper executed successfully")
        print("\n[VERIFY IN ML DATABASE]")
        print("SELECT * FROM patient_feedback_encoded WHERE record_id = 999;")
        print("\nCheck for:")
        print("  - embedding_text1 (BLOB, not NULL)")
        print("  - embedding_text2 (BLOB, not NULL)")
        print("  - embedding_text3 (BLOB, not NULL)")
        print("  - embedding_text123 (BLOB, not NULL)")
        print("  - sentence_1_embedding through sentence_6_embedding (BLOBs)")
        
        return True
    except Exception as e:
        print(f"✗ Wrapper execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wrapper_with_partial_data():
    """Test 5: Test wrapper with missing/empty fields"""
    print("\n" + "=" * 70)
    print("TEST 5: WRAPPER WITH PARTIAL DATA")
    print("=" * 70)
    
    try:
        from backend.ml_mapping import add_corrected_record_to_ml
        
        data = {
            'record_id': 998,  # Test record
            'patient_full_name': 'محمد علي',
            'complaint_text': 'شكوى عامة',  # Very short
            'immediate_action': None,  # Missing
            'taken_action': '',  # Empty
            'feedback_received_date': datetime.now().strftime('%Y-%m-%d'),
            'domain_id': 1,
            'category_id': 1,
            'subcategory_id': 1,
            'severity_id': 1,
            'stage_id': 1,
            'harm_id': 1,
        }
        
        print("Input data (partial/empty):")
        print(f"  - record_id: {data['record_id']}")
        print(f"  - complaint_text: '{data['complaint_text']}'")
        print(f"  - immediate_action: {data['immediate_action']}")
        print(f"  - taken_action: '{data['taken_action']}'")
        
        print("\nCalling add_corrected_record_to_ml()...")
        add_corrected_record_to_ml(data)
        
        print("✓ Wrapper handled partial data gracefully")
        print("\n[VERIFY IN ML DATABASE]")
        print("SELECT * FROM patient_feedback_encoded WHERE record_id = 998;")
        print("\nExpected:")
        print("  - Some embeddings may be NULL (for missing/empty fields)")
        print("  - Record should still be inserted")
        
        return True
    except Exception as e:
        print(f"✗ Partial data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wrapper_with_no_text():
    """Test 6: Test wrapper with completely empty text fields"""
    print("\n" + "=" * 70)
    print("TEST 6: WRAPPER WITH NO TEXT")
    print("=" * 70)
    
    try:
        from backend.ml_mapping import add_corrected_record_to_ml
        
        data = {
            'record_id': 997,  # Test record
            'patient_full_name': 'علي محمد',
            'complaint_text': None,
            'immediate_action': None,
            'taken_action': None,
            'feedback_received_date': datetime.now().strftime('%Y-%m-%d'),
            'domain_id': 1,
            'category_id': 1,
            'subcategory_id': 1,
            'severity_id': 1,
            'stage_id': 1,
            'harm_id': 1,
        }
        
        print("Input data (no text fields):")
        print(f"  - record_id: {data['record_id']}")
        print(f"  - All text fields: None")
        
        print("\nCalling add_corrected_record_to_ml()...")
        add_corrected_record_to_ml(data)
        
        print("✓ Wrapper handled no-text scenario gracefully")
        print("\n[VERIFY IN ML DATABASE]")
        print("SELECT * FROM patient_feedback_encoded WHERE record_id = 997;")
        print("\nExpected:")
        print("  - All embedding fields: NULL")
        print("  - Record inserted with mapped fields only")
        
        return True
    except Exception as e:
        print(f"✗ No-text test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 70)
    print("EMBEDDING WRAPPER INTEGRATION TESTS")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("Embedding Imports", test_embedding_imports()))
    results.append(("Wrapper Import", test_wrapper_import()))
    
    if not results[0][1]:  # If embedding imports failed, skip others
        print("\n⚠️  Skipping remaining tests - embedding functions unavailable")
    else:
        results.append(("Embedding Generation", test_embedding_generation()))
        results.append(("Wrapper with Arabic", test_wrapper_with_arabic_text()))
        results.append(("Wrapper Partial Data", test_wrapper_with_partial_data()))
        results.append(("Wrapper No Text", test_wrapper_with_no_text()))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        print("\n📊 NEXT STEPS:")
        print("1. Query ML database for test records (record_id 997-999)")
        print("2. Verify embedding columns contain data")
        print("3. Run training pipeline with accumulated ML data")
        print("4. Monitor for [Embedding Warning] messages in console")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
