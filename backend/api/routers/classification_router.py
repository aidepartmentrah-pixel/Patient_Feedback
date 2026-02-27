"""
Classification Router
API endpoints for text classification using AI models.
"""

from fastapi import APIRouter, HTTPException, Body, Depends
from pydantic import BaseModel, Field
from typing import Optional

from core.constants.roles import SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR, WORKER
from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_role
from ..services.classification_service import classify_text, classify_batch


router = APIRouter(prefix="/api/classification", tags=["Classification"])


# ==================== REQUEST/RESPONSE MODELS ====================

class ClassificationRequest(BaseModel):
    """Request model for single text classification."""
    text: str = Field(..., min_length=1, description="Arabic patient feedback text")
    explain: bool = Field(True, description="Include explanation for classifications")


class BatchClassificationRequest(BaseModel):
    """Request model for batch text classification."""
    texts: list[str] = Field(..., min_items=1, max_items=100, description="List of Arabic feedback texts")
    explain: bool = Field(False, description="Include explanation for classifications")


# ==================== ENDPOINTS ====================

@router.post("/classify")
async def classify_feedback_text(
    request: ClassificationRequest = Body(...),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Classify patient feedback text into multiple categories.
    
    Authorization: SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR, WORKER
    
    **Classifies text into 8 categories:**
    1. Domain (المجال)
    2. Category (التصنيف)
    3. SubCategory (التصنيف الفرعي)
    4. Classification (التصنيف الجديد)
    5. Severity Level (مستوى الخطورة)
    6. Stage (المرحلة)
    7. Harm Level (مستوى الضرر)
    8. Improvement Opportunity Type (نوع فرصة التحسين)
    9. Classification EN (التصنيف الإنجليزي)
    
    **Example Request:**
    ```json
    {
      "text": "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج",
      "explain": true
    }
    ```
    
    **Returns:**
    - Classification results for all 9 categories (including Classification EN)
    - Confidence scores
    - Optional explanations
    """
    # Authorization guard
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR, WORKER])
    
    try:
        result = classify_text(request.text, explain=request.explain)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": result.get("error", "CLASSIFICATION_FAILED"),
                    "message": result.get("message", "Classification failed"),
                    "message_ar": result.get("message_ar", "فشل التصنيف")
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


@router.post("/classify-batch")
async def classify_batch_texts(
    request: BatchClassificationRequest = Body(...),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Classify multiple texts in batch (up to 100 texts).
    
    Authorization: SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR, WORKER
    
    **Example Request:**
    ```json
    {
      "texts": [
        "المريض يشكو من ألم شديد",
        "تأخر في تقديم العلاج",
        "الطاقم الطبي محترم جداً"
      ],
      "explain": false
    }
    ```
    
    **Returns:**
    - Results for all texts
    - Success/failure count
    """
    # Authorization guard
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR, WORKER])
    
    try:
        result = classify_batch(request.texts, explain=request.explain)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": result.get("error", "BATCH_FAILED"),
                    "message": result.get("message", "Batch classification failed"),
                    "message_ar": result.get("message_ar", "فشل التصنيف الجماعي")
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
async def test_classification():
    """
    Test endpoint to verify classification service is working.
    
    Returns a sample classification result.
    """
    
    sample_text = "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج"
    
    result = classify_text(sample_text, explain=True)
    
    return {
        "status": "operational",
        "service": "classification",
        "sample_result": result
    }
