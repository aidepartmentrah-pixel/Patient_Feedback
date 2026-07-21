"""
CLASSIFICATION FAILURE DIAGNOSTIC
=================================
This script tests each stage of the classification pipeline to identify
where the "CLASSIFICATION_FAILED: 2" error originates.

Save this output for troubleshooting across sessions.
"""

import sys
import os
import traceback

# Setup paths
sys.path.insert(0, r"C:\Users\Administrator\Documents\GitHub\Patient_Feedback")
os.chdir(r"C:\Users\Administrator\Documents\GitHub\Patient_Feedback")

# Test text
TEST_TEXT = "مريض يشكو من ألم في الرأس"

def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def diagnostic_report():
    results = {}
    
    print_header("CLASSIFICATION DIAGNOSTIC REPORT")
    
    # =========================================================
    # STEP 1: Check MPNet Model Path
    # =========================================================
    print_header("STEP 1: MPNet Model Check")
    
    MODEL_PATH = r"C:\Users\Administrator\Documents\GitHub\Patient_Feedback\models_directory\Classification_Models\model_storage\mpnet_embeddings"
    
    print(f"Model path: {MODEL_PATH}")
    print(f"Path exists: {os.path.exists(MODEL_PATH)}")
    
    if os.path.exists(MODEL_PATH):
        files = os.listdir(MODEL_PATH)
        print(f"Files in model directory: {files}")
        results["mpnet_exists"] = True
    else:
        print("ERROR: MPNet model directory not found!")
        results["mpnet_exists"] = False
        
    # =========================================================
    # STEP 2: Test Embedding Generation
    # =========================================================
    print_header("STEP 2: Embedding Generation")
    
    try:
        from models_directory.Classification_Models.Stage.modular_functions import get_embedding
        
        print(f"Test text: {TEST_TEXT}")
        embedding_bytes = get_embedding(TEST_TEXT, Troubleshoot=True)
        
        import numpy as np
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding dtype: {embedding.dtype}")
        print(f"Embedding sample (first 5): {embedding[:5]}")
        results["embedding_success"] = True
        
    except Exception as e:
        print(f"EMBEDDING ERROR: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        results["embedding_success"] = False
        results["embedding_error"] = str(e)
        
    # =========================================================
    # STEP 3: Test Hierarchical Prediction
    # =========================================================
    print_header("STEP 3: Hierarchical Prediction (Domain/Category/Subcategory)")
    
    try:
        from models_directory.Classification_Models.Hierarchical_Classification_Model.hierarchical_predictor import hierarchical_predict_embeddings
        
        result = hierarchical_predict_embeddings(embedding)
        print(f"Domain ID: {result.get('domain')}")
        print(f"Category ID: {result.get('category')}")
        print(f"Subcategory ID: {result.get('subcategory')}")
        results["hierarchical_success"] = True
        results["domain_id"] = result.get('domain')
        results["category_id"] = result.get('category')
        results["subcategory_id"] = result.get('subcategory')
        
    except Exception as e:
        print(f"HIERARCHICAL ERROR: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        results["hierarchical_success"] = False
        results["hierarchical_error"] = str(e)
        
    # =========================================================
    # STEP 4: Test Severity Prediction
    # =========================================================
    print_header("STEP 4: Severity Prediction")
    
    try:
        from models_directory.Classification_Models.Severity_level.predict_severity import predict_severity_from_embedding
        
        severity_id = predict_severity_from_embedding(embedding)
        print(f"Severity ID: {severity_id}")
        results["severity_success"] = True
        results["severity_id"] = severity_id
        
    except Exception as e:
        print(f"SEVERITY ERROR: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        results["severity_success"] = False
        results["severity_error"] = str(e)
        
    # =========================================================
    # STEP 5: Test Stage Prediction
    # =========================================================
    print_header("STEP 5: Stage Prediction")
    
    try:
        from models_directory.Classification_Models.Stage.model_package import classify_stage_Score_Based
        
        stage_result = classify_stage_Score_Based(TEST_TEXT, Print=True)
        print(f"Stage result: {stage_result}")
        results["stage_success"] = True
        results["stage_result"] = stage_result
        
    except Exception as e:
        print(f"STAGE ERROR: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        results["stage_success"] = False
        results["stage_error"] = str(e)
        
    # =========================================================
    # STEP 6: Test Harm Prediction
    # =========================================================
    print_header("STEP 6: Harm Prediction")
    
    try:
        from models_directory.Classification_Models.Harm_level.predict_harm import predict_harm_from_embedding
        
        harm_result = predict_harm_from_embedding(embedding)
        print(f"Harm result: {harm_result}")
        results["harm_success"] = True
        results["harm_result"] = harm_result
        
    except Exception as e:
        print(f"HARM ERROR: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        results["harm_success"] = False
        results["harm_error"] = str(e)
        
    # =========================================================
    # STEP 7: Test Feedback Type Prediction
    # =========================================================
    print_header("STEP 7: Feedback Type Prediction")
    
    try:
        from models_directory.Classification_Models.feedback_type.predict_feedback_type import predict_feedback_type_from_embedding
        
        feedback_type = predict_feedback_type_from_embedding(embedding)
        print(f"Feedback type: {feedback_type}")
        results["feedback_type_success"] = True
        results["feedback_type"] = feedback_type
        
    except Exception as e:
        print(f"FEEDBACK TYPE ERROR: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        results["feedback_type_success"] = False
        results["feedback_type_error"] = str(e)
        
    # =========================================================
    # STEP 8: Test Improvement Opportunity Prediction
    # =========================================================
    print_header("STEP 8: Improvement Opportunity Prediction")
    
    try:
        from models_directory.Classification_Models.improvement_opportunity_type.predict_improvement import predict_improvement_from_embedding
        
        improvement = predict_improvement_from_embedding(embedding)
        print(f"Improvement opportunity: {improvement}")
        results["improvement_success"] = True
        results["improvement"] = improvement
        
    except Exception as e:
        print(f"IMPROVEMENT ERROR: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        results["improvement_success"] = False
        results["improvement_error"] = str(e)
        
    # =========================================================
    # STEP 9: Test Classification EN Prediction
    # =========================================================
    print_header("STEP 9: Classification EN Prediction")
    
    try:
        from models_directory.Classification_Models.Classification_En.predict_classification_en import predict_classification_en_from_embedding
        
        classification_en = predict_classification_en_from_embedding(embedding)
        print(f"Classification EN: {classification_en}")
        results["classification_en_success"] = True
        results["classification_en"] = classification_en
        
    except Exception as e:
        print(f"CLASSIFICATION EN ERROR: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        results["classification_en_success"] = False
        results["classification_en_error"] = str(e)
        
    # =========================================================
    # STEP 10: Full Pipeline Test
    # =========================================================
    print_header("STEP 10: Full Pipeline (classify_feedback)")
    
    try:
        from models_directory.Classification_Models.package_models import classify_feedback
        
        result = classify_feedback(TEST_TEXT, "", "", Print=True)
        print(f"\nFull result keys: {result.keys()}")
        results["full_pipeline_success"] = True
        results["full_result"] = result
        
    except Exception as e:
        print(f"FULL PIPELINE ERROR: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        results["full_pipeline_success"] = False
        results["full_pipeline_error"] = str(e)
        
    # =========================================================
    # SUMMARY
    # =========================================================
    print_header("DIAGNOSTIC SUMMARY")
    
    checks = [
        ("MPNet Model Exists", results.get("mpnet_exists")),
        ("Embedding Generation", results.get("embedding_success")),
        ("Hierarchical Prediction", results.get("hierarchical_success")),
        ("Severity Prediction", results.get("severity_success")),
        ("Stage Prediction", results.get("stage_success")),
        ("Harm Prediction", results.get("harm_success")),
        ("Feedback Type Prediction", results.get("feedback_type_success")),
        ("Improvement Prediction", results.get("improvement_success")),
        ("Classification EN Prediction", results.get("classification_en_success")),
        ("Full Pipeline", results.get("full_pipeline_success")),
    ]
    
    all_pass = True
    for name, status in checks:
        icon = "✓" if status else "✗"
        print(f"  {icon} {name}: {'PASS' if status else 'FAIL'}")
        if not status:
            all_pass = False
            error_key = name.lower().replace(" ", "_") + "_error"
            if error_key in results:
                print(f"      Error: {results[error_key]}")
    
    print("\n" + "=" * 60)
    if all_pass:
        print(" ALL CHECKS PASSED ")
    else:
        print(" SOME CHECKS FAILED - Review errors above ")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    diagnostic_report()
