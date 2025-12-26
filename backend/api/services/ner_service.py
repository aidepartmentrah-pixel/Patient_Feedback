"""
NER (Named Entity Recognition) Service
Extracts named entities from Arabic patient feedback text.
"""
import sys
from pathlib import Path

# Add workspace root to path to import models_directory
# From: backend/api/services/ner_service.py
# To: Patient_Feedback/ (4 levels up)
workspace_root = Path(__file__).resolve().parent.parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
from models_directory.NER_Model.solution_gliner import extract_names_gliner_arabic


def extract_entities(text: str) -> dict:
    """
    Extract named entities from Arabic text using GLiNER model.
    
    Args:
        text: Arabic patient feedback text
        
    Returns:
        Dictionary with extracted entities (patient names, doctor names, locations, etc.)
    """
    
    if not text or not text.strip():
        return {
            "success": False,
            "error": "EMPTY_TEXT",
            "message": "Text is required for NER",
            "message_ar": "النص مطلوب لاستخراج الكيانات"
        }
    
    try:
        # Run NER model
        ner_result = extract_names_gliner_arabic(text)
        
        return {
            "success": True,
            "text": text,
            "entities": ner_result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": "NER_FAILED",
            "message": f"NER extraction failed: {str(e)}",
            "message_ar": f"فشل استخراج الكيانات: {str(e)}"
        }


def extract_entities_batch(texts: list[str]) -> dict:
    """
    Extract entities from multiple texts in batch.
    
    Args:
        texts: List of Arabic patient feedback texts
        
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
        result = extract_entities(text)
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
