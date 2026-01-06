"""
Classification Service
Handles Arabic text classification for patient feedback using AI models.
"""
import sys
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
    
    if not text or not text.strip():
        return {
            "success": False,
            "error": "EMPTY_TEXT",
            "message": "Text is required for classification",
            "message_ar": "النص مطلوب للتصنيف"
        }
    
    try:
        # Run classification model
        # classify_feedback expects: patient_text, text_2, text_3, Print
        classifier = _get_classifier()
        classification_result = classifier(
            patient_text=text,
            text_2="",
            text_3="",
            Print=False
        )
        
        return {
            "success": True,
            "text": text,
            "classifications": classification_result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": "CLASSIFICATION_FAILED",
            "message": f"Classification failed: {str(e)}",
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



