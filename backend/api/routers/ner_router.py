"""
NER (Named Entity Recognition) Router
API endpoints for extracting named entities from Arabic text.
"""

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional

from ..services.ner_service import extract_entities, extract_entities_batch


router = APIRouter(prefix="/api/ner", tags=["NER"])


# ==================== REQUEST/RESPONSE MODELS ====================

class NERRequest(BaseModel):
    """Request model for single text NER."""
    text: str = Field(..., min_length=1, description="Arabic patient feedback text")


class BatchNERRequest(BaseModel):
    """Request model for batch NER."""
    texts: list[str] = Field(..., min_items=1, max_items=100, description="List of Arabic feedback texts")


# ==================== ENDPOINTS ====================

@router.post("/extract")
async def extract_named_entities(request: NERRequest = Body(...)):
    """
    Extract named entities from Arabic patient feedback text.
    
    **Extracts entities such as:**
    - Patient names (أسماء المرضى)
    - Doctor names (أسماء الأطباء)
    - Hospital departments (الأقسام الطبية)
    - Locations (المواقع)
    - Medical conditions (الحالات الطبية)
    - Medications (الأدوية)
    - Dates and times
    
    **Example Request:**
    ```json
    {
      "text": "المريض أحمد محمد يشكو من ألم في البطن وتم فحصه بواسطة الدكتور خالد في قسم الطوارئ"
    }
    ```
    
    **Returns:**
    - Extracted entities with their types and positions
    """
    
    try:
        result = extract_entities(request.text)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": result.get("error", "NER_FAILED"),
                    "message": result.get("message", "NER extraction failed"),
                    "message_ar": result.get("message_ar", "فشل استخراج الكيانات")
                }
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"An error occurred: {str(e)}",
                "message_ar": f"حدث خطأ: {str(e)}"
            }
        )


@router.post("/extract-batch")
async def extract_entities_from_batch(request: BatchNERRequest = Body(...)):
    """
    Extract entities from multiple texts in batch (up to 100 texts).
    
    **Example Request:**
    ```json
    {
      "texts": [
        "المريض أحمد يشكو من ألم",
        "الدكتور خالد في قسم الطوارئ",
        "تم إعطاء الباراسيتامول للمريض"
      ]
    }
    ```
    
    **Returns:**
    - Results for all texts
    - Success/failure count
    """
    
    try:
        result = extract_entities_batch(request.texts)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": result.get("error", "BATCH_FAILED"),
                    "message": result.get("message", "Batch NER failed"),
                    "message_ar": result.get("message_ar", "فشل استخراج الكيانات الجماعي")
                }
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"An error occurred: {str(e)}",
                "message_ar": f"حدث خطأ: {str(e)}"
            }
        )


@router.get("/test")
async def test_ner():
    """
    Test endpoint to verify NER service is working.
    
    Returns a sample NER result.
    """
    
    sample_text = "المريض أحمد محمد يشكو من ألم في البطن وتم فحصه بواسطة الدكتور خالد في قسم الطوارئ"
    
    result = extract_entities(sample_text)
    
    return {
        "status": "operational",
        "service": "ner",
        "sample_result": result
    }
