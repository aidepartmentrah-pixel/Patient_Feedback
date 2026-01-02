"""
Reference Data Router
API endpoints for fetching dropdown/lookup reference data.
"""

from fastapi import APIRouter, Query
from typing import Optional

from ..services.reference_service import (
    get_departments,
    get_sources,
    get_domains,
    get_categories,
    get_subcategories,
    get_classifications,
    get_severity_levels,
    get_stages,
    get_harm_levels,
  get_explanation_statuses,
    get_clinical_risk_types,
    get_feedback_intent_types,
    get_buildings,
    get_all_reference_data
)


router = APIRouter(prefix="/api/reference", tags=["Reference Data"])


# ==================== ENDPOINTS ====================

@router.get("/departments")
async def get_departments_endpoint():
    """
    Get all departments for issuing/target department dropdowns.
    
    **Returns:**
    ```json
    {
      "departments": [
        { "id": 1, "name_en": "Emergency Department", "name_ar": "قسم الطوارئ" },
        { "id": 2, "name_en": "ICU", "name_ar": "وحدة العناية المركزة" }
      ]
    }
    ```
    """
    return get_departments()


@router.get("/sources")
async def get_sources_endpoint():
    """
    Get all feedback sources.
    
    **Returns:**
    ```json
    {
      "sources": [
        { "id": 1, "name": "Phone" },
        { "id": 2, "name": "Email" },
        { "id": 3, "name": "In Person" }
      ]
    }
    ```
    """
    return get_sources()


@router.get("/domains")
async def get_domains_endpoint():
    """
    Get all domains (top-level classification).
    
    **Returns:**
    ```json
    {
      "domains": [
        { "id": 1, "name_en": "Clinical", "name_ar": "سريري" },
        { "id": 2, "name_en": "Management", "name_ar": "إداري" }
      ]
    }
    ```
    """
    return get_domains()


@router.get("/categories")
async def get_categories_endpoint(
    domain_id: Optional[int] = Query(None, description="Filter by domain ID")
):
    """
    Get categories, optionally filtered by domain.
    
    **Query Parameters:**
    - `domain_id` (optional): Filter categories by domain
    
    **Examples:**
    - `/api/reference/categories` - Get all categories
    - `/api/reference/categories?domain_id=1` - Get categories for domain 1
    
    **Returns:**
    ```json
    {
      "categories": [
        { "id": 12, "domain_id": 1, "name_en": "Medication Error", "name_ar": "خطأ في الدواء" },
        { "id": 13, "domain_id": 1, "name_en": "Delayed Diagnosis", "name_ar": "تأخر في التشخيص" }
      ]
    }
    ```
    """
    return get_categories(domain_id)


@router.get("/subcategories")
async def get_subcategories_endpoint(
    category_id: Optional[int] = Query(None, description="Filter by category ID")
):
    """
    Get subcategories, optionally filtered by category.
    
    **Query Parameters:**
    - `category_id` (optional): Filter subcategories by category
    
    **Examples:**
    - `/api/reference/subcategories` - Get all subcategories
    - `/api/reference/subcategories?category_id=12` - Get subcategories for category 12
    
    **Returns:**
    ```json
    {
      "subcategories": [
        { "id": 45, "category_id": 12, "name_en": "Wrong Dosage", "name_ar": "جرعة خاطئة" },
        { "id": 46, "category_id": 12, "name_en": "Wrong Medication", "name_ar": "دواء خاطئ" }
      ]
    }
    ```
    """
    return get_subcategories(category_id)


@router.get("/classifications")
async def get_classifications_endpoint(
    subcategory_id: Optional[int] = Query(None, description="Filter by subcategory ID")
):
    """
    Get classifications (most specific level), optionally filtered by subcategory.
    
    **Query Parameters:**
    - `subcategory_id` (optional): Filter classifications by subcategory
    
    **Examples:**
    - `/api/reference/classifications` - Get all classifications
    - `/api/reference/classifications?subcategory_id=45` - Get classifications for subcategory 45
    
    **Returns:**
    ```json
    {
      "classifications": [
        { "id": 102, "subcategory_id": 45, "name_en": "Prescription Error", "name_ar": "خطأ في الوصفة" },
        { "id": 103, "subcategory_id": 45, "name_en": "Administration Error", "name_ar": "خطأ في التطبيق" }
      ]
    }
    ```
    """
    return get_classifications(subcategory_id)


@router.get("/severity-levels")
async def get_severity_levels_endpoint():
    """
    Get all severity levels.
    
    **Returns:**
    ```json
    {
      "severity_levels": [
        { "id": 1, "name_en": "Low", "name_ar": "منخفض" },
        { "id": 2, "name_en": "Medium", "name_ar": "متوسط" },
        { "id": 3, "name_en": "High", "name_ar": "عالي" }
      ]
    }
    ```
    """
    return get_severity_levels()


@router.get("/stages")
async def get_stages_endpoint():
    """
    Get all care stages.
    
    **Returns:**
    ```json
    {
      "stages": [
        { "id": 1, "name_en": "Admission", "name_ar": "القبول" },
        { "id": 2, "name_en": "Care", "name_ar": "الرعاية" },
        { "id": 3, "name_en": "Discharge", "name_ar": "الخروج" }
      ]
    }
    ```
    """
    return get_stages()


@router.get("/harm-levels")
async def get_harm_levels_endpoint():
    """
    Get all harm levels.
    
    **Returns:**
    ```json
    {
      "harm_levels": [
        { "id": 1, "name_en": "No Harm", "name_ar": "لا ضرر" },
        { "id": 2, "name_en": "Minor", "name_ar": "طفيف" },
        { "id": 3, "name_en": "Major", "name_ar": "كبير" },
        { "id": 4, "name_en": "Severe", "name_ar": "شديد" }
      ]
    }
    ```
    """
    return get_harm_levels()


@router.get("/buildings")
async def get_buildings_endpoint():
    """
    Get all buildings (e.g., RAH=1, BCI=2).
    """
    return get_buildings()


@router.get("/explanation-statuses")
async def get_explanation_statuses_endpoint():
    """
    Get all explanation statuses (for Insert Page dropdown).
    """
    return get_explanation_statuses()


@router.get("/clinical-risk-types")
async def get_clinical_risk_types_endpoint():
    """
    Get all clinical risk types.
    
    **Returns:**
    ```json
    {
      "clinical_risk_types": [
        { "id": 1, "name": "High Risk", "name_ar": "خطر عالي" },
        { "id": 2, "name": "Medium Risk", "name_ar": "خطر متوسط" }
      ]
    }
    ```
    """
    return get_clinical_risk_types()


@router.get("/feedback-intent-types")
async def get_feedback_intent_types_endpoint():
    """
    Get all feedback intent types.
    
    **Returns:**
    ```json
    {
      "feedback_intent_types": [
        { "id": 1, "name": "Complaint", "name_ar": "شكوى" },
        { "id": 2, "name": "Suggestion", "name_ar": "اقتراح" }
      ]
    }
    ```
    """
    return get_feedback_intent_types()


@router.get("/all")
async def get_all_reference_data_endpoint():
    """
    Get all reference data in a single request (useful for initialization).
    
    **Returns:**
    Combined response with all reference data:
    - departments
    - sources
    - domains
    - categories (all)
    - subcategories (all)
    - classifications (all)
    - severity_levels
    - stages
    - harm_levels
    - clinical_risk_types
    - feedback_intent_types
    
    **Note:** This may be a large response. Consider caching on frontend.
    Categories/subcategories/classifications can be filtered client-side.
    """
    return get_all_reference_data()


@router.get("/test")
async def test_reference_endpoint():
    """
    Test endpoint to verify reference data service is operational.
    """
    return {
        "status": "operational",
        "service": "reference_data",
        "message": "Reference data service is running",
        "endpoints": [
            "/api/reference/departments",
            "/api/reference/sources",
            "/api/reference/domains",
            "/api/reference/categories",
            "/api/reference/subcategories",
            "/api/reference/classifications",
            "/api/reference/severity-levels",
            "/api/reference/stages",
            "/api/reference/harm-levels",
            "/api/reference/explanation-statuses",
            "/api/reference/clinical-risk-types",
            "/api/reference/feedback-intent-types",
            "/api/reference/buildings",
            "/api/reference/all"
        ]
    }
