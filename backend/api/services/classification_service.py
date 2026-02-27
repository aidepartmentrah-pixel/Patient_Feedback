"""
Classification Service
Handles Arabic text classification for patient feedback using AI models.
"""
import sys
import uuid
from pathlib import Path

# Add workspace root to path to import models_directory
# From: backend/api/services/classification_service.py
# To: Patient_Feedback/ (4 levels up)
workspace_root = Path(__file__).resolve().parent.parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

def _get_classifier():
    """Lazy import classifier to avoid heavy startup downloads on reload."""
    from models_directory.Classification_Models.package_models import classify_feedback
    return classify_feedback


def _get_valid_classifications(subcategory_id: int) -> list[int]:
    """Get valid classification IDs for a subcategory."""
    try:
        from api.db_layer.lookups import get_classifications
        classifications = get_classifications(subcategory_id)
        return [c.get("ClassificationID") for c in classifications if c.get("ClassificationID")]
    except Exception:
        return []


def classify_text(text: str, explain: bool = True) -> dict:
    """
    Classify patient feedback text into multiple categories.
    
    Args:
        text: Arabic patient feedback text
        explain: Whether to include explanation for classifications
        
    Returns:
        Dictionary with classification results for all 8 categories:
        - domain
        - category
        - subcategory
        - classification
        - severity_level
        - stage
        - harm_level
        - improvement_opportunity_type
    """
    # === TEMPORARY DEBUG INSTRUMENTATION (cross-request isolation test) ===
    request_debug_id = str(uuid.uuid4())
    print("====================================")
    print("CLASSIFY REQUEST ID:", request_debug_id)
    print("INPUT TEXT:", text)
    print("INPUT HASH:", hash(text))
    print("====================================")
    # === END TEMPORARY DEBUG ===
    
    print(f"\n{'='*60}")
    print(f"[CLASSIFY] === DIAGNOSTIC START ===")
    print(f"[CLASSIFY] Text length: {len(text) if text else 0}")
    print(f"[CLASSIFY] Text preview: {text[:100] if text else 'EMPTY'}...")
    print(f"[CLASSIFY] Explain mode: {explain}")
    
    if not text or not text.strip():
        print("[CLASSIFY] ERROR: Empty text received")
        return {
            "success": False,
            "error": "EMPTY_TEXT",
            "message": "Text is required for classification",
            "message_ar": "النص مطلوب للتصنيف"
        }
    
    try:
        # Run classification model
        # classify_feedback expects: patient_text, text_2, text_3, Print
        print("[CLASSIFY] Getting classifier...")
        classifier = _get_classifier()
        print("[CLASSIFY] Running classification (Print=True for diagnostics)...")
        classification_result = classifier(
            patient_text=text,
            text_2="",
            text_3="",
            Print=True  # Enable diagnostic output from model
        )
        print(f"[CLASSIFY] Full result: {classification_result}")
        
        # Validate classification_en_id against subcategory
        sub_category_id = classification_result.get("sub_category_id")
        classification_en_id = classification_result.get("classification_en_id")
        
        if sub_category_id and classification_en_id:
            valid_ids = _get_valid_classifications(sub_category_id)
            if valid_ids and classification_en_id not in valid_ids:
                # Pick first valid classification for this subcategory
                classification_result["classification_en_id"] = valid_ids[0]
                classification_result["classification_en"] = f"Classification {valid_ids[0]}"
        
        result = {
            "success": True,
            "text": text,
            "classifications": classification_result
        }
        
        # === TEMPORARY DEBUG INSTRUMENTATION (cross-request isolation test) ===
        print("====================================")
        print("CLASSIFY REQUEST ID:", request_debug_id)
        print("OUTPUT RESULT:", result)
        print("RESULT OBJECT ID:", id(result))
        print("====================================")
        # === END TEMPORARY DEBUG ===
        
        return result
        
    except Exception as e:
        import traceback
        print(f"\n{'='*60}")
        print(f"[CLASSIFY] !!! EXCEPTION CAUGHT !!!")
        print(f"[CLASSIFY] Exception type: {type(e).__name__}")
        print(f"[CLASSIFY] Exception message: {str(e)}")
        print(f"[CLASSIFY] Exception repr: {repr(e)}")
        print(f"[CLASSIFY] Text that caused failure: {text[:200] if text else 'NONE'}")
        print(f"{'='*60}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        return {
            "success": False,
            "error": "CLASSIFICATION_FAILED",
            "message": f"Classification failed: {type(e).__name__}: {str(e)}",
            "message_ar": f"فشل التصنيف: {str(e)}"
        }


def classify_batch(texts: list[str], explain: bool = False) -> dict:
    """
    Classify multiple texts in batch.
    
    Args:
        texts: List of Arabic patient feedback texts
        explain: Whether to include explanation for classifications
        
    Returns:
        Dictionary with results for all texts
    """
    
    if not texts or len(texts) == 0:
        return {
            "success": False,
            "error": "EMPTY_BATCH",
            "message": "At least one text is required",
            "message_ar": "مطلوب نص واحد على الأقل"
        }
    
    results = []
    failed_count = 0
    
    for idx, text in enumerate(texts):
        result = classify_text(text, explain=explain)
        results.append({
            "index": idx,
            "text": text,
            "result": result
        })
        
        if not result.get("success", False):
            failed_count += 1
    
    return {
        "success": True,
        "total": len(texts),
        "successful": len(texts) - failed_count,
        "failed": failed_count,
        "results": results
    }



