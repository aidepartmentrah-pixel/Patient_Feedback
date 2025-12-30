"""
Reference Data Service
Fetches reference/lookup data for dropdowns and selectors.
"""

from typing import List, Dict, Any, Optional
from core.database import get_connection


def get_departments() -> Dict[str, Any]:
    """Get all departments (AdminsrationUnit)."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT UniqueID, Name
            FROM AdminsrationUnit
            WHERE Frozen = 0
            ORDER BY Name
        """)
        
        departments = []
        for row in cursor.fetchall():
            departments.append({
                "id": row.UniqueID,
                "name_en": row.Name,
                "name_ar": row.Name
            })
        
        return {"departments": departments}
        
    except Exception as e:
        return {
            "departments": [],
            "error": f"Failed to fetch departments: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_sources() -> Dict[str, Any]:
    """Get all feedback sources."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT SourceID, SourceName, SourceNameAr
            FROM APP_LOOKUP_SOURCE
            WHERE IsActive = 1
            ORDER BY DisplayOrder
        """)
        
        sources = []
        for row in cursor.fetchall():
            sources.append({
                "id": row.SourceID,
                "name": row.SourceName,
                "name_ar": row.SourceNameAr
            })
        
        return {"sources": sources}
        
    except Exception as e:
        return {
            "sources": [],
            "error": f"Failed to fetch sources: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_domains() -> Dict[str, Any]:
    """Get all domains."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DomainID, DomainName, DomainCode
            FROM APP_LOOKUP_DOMAIN
            ORDER BY DomainOrder
        """)
        
        domains = []
        for row in cursor.fetchall():
            domains.append({
                "id": row.DomainID,
                "name_en": row.DomainName,
                "name_ar": row.DomainName,
                "code": row.DomainCode
            })
        
        return {"domains": domains}
        
    except Exception as e:
        return {
            "domains": [],
            "error": f"Failed to fetch domains: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_categories(domain_id: Optional[int] = None) -> Dict[str, Any]:
    """Get categories, optionally filtered by domain."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        if domain_id:
            cursor.execute("""
                SELECT CategoryID, DomainID, CategoryName
                FROM APP_LOOKUP_CATEGORY
                WHERE DomainID = ?
                ORDER BY CategoryOrder
            """, (domain_id,))
        else:
            cursor.execute("""
                SELECT CategoryID, DomainID, CategoryName
                FROM APP_LOOKUP_CATEGORY
                ORDER BY CategoryOrder
            """)
        
        categories = []
        for row in cursor.fetchall():
            categories.append({
                "id": row.CategoryID,
                "domain_id": row.DomainID,
                "name_en": row.CategoryName,
                "name_ar": row.CategoryName
            })
        
        return {"categories": categories}
        
    except Exception as e:
        return {
            "categories": [],
            "error": f"Failed to fetch categories: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_subcategories(category_id: Optional[int] = None) -> Dict[str, Any]:
    """Get subcategories, optionally filtered by category."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        if category_id:
            cursor.execute("""
                SELECT SubCategoryID, CategoryID, SubCategoryName
                FROM APP_LOOKUP_SUBCATEGORY
                WHERE CategoryID = ?
                ORDER BY SubCategoryName
            """, (category_id,))
        else:
            cursor.execute("""
                SELECT SubCategoryID, CategoryID, SubCategoryName
                FROM APP_LOOKUP_SUBCATEGORY
                ORDER BY SubCategoryName
            """)
        
        subcategories = []
        for row in cursor.fetchall():
            subcategories.append({
                "id": row.SubCategoryID,
                "category_id": row.CategoryID,
                "name_en": row.SubCategoryName,
                "name_ar": row.SubCategoryName
            })
        
        return {"subcategories": subcategories}
        
    except Exception as e:
        return {
            "subcategories": [],
            "error": f"Failed to fetch subcategories: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_classifications(subcategory_id: Optional[int] = None) -> Dict[str, Any]:
    """Get classifications, optionally filtered by subcategory."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        if subcategory_id:
            cursor.execute("""
                SELECT ClassificationID, SubCategoryID, Classification_EN, Classification_AR
                FROM APP_LOOKUP_CLASSIFICATION
                WHERE SubCategoryID = ?
                ORDER BY Classification_EN
            """, (subcategory_id,))
        else:
            cursor.execute("""
                SELECT ClassificationID, SubCategoryID, Classification_EN, Classification_AR
                FROM APP_LOOKUP_CLASSIFICATION
                ORDER BY Classification_EN
            """)
        
        classifications = []
        for row in cursor.fetchall():
            classifications.append({
                "id": row.ClassificationID,
                "subcategory_id": row.SubCategoryID,
                "name_en": row.Classification_EN or row.Classification_AR,
                "name_ar": row.Classification_AR
            })
        
        return {"classifications": classifications}
        
    except Exception as e:
        return {
            "classifications": [],
            "error": f"Failed to fetch classifications: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_severity_levels() -> Dict[str, Any]:
    """Get all severity levels."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT SeverityID, SeverityName, SeverityCode
            FROM APP_LOOKUP_SEVERITY
            WHERE IsActive = 1
            ORDER BY SeverityOrder
        """)
        
        severity_levels = []
        for row in cursor.fetchall():
            severity_levels.append({
                "id": row.SeverityID,
                "name_en": row.SeverityName,
                "name_ar": row.SeverityName,
                "code": row.SeverityCode
            })
        
        return {"severity_levels": severity_levels}
        
    except Exception as e:
        return {
            "severity_levels": [],
            "error": f"Failed to fetch severity levels: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_stages() -> Dict[str, Any]:
    """Get all care stages."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT StageID, StageName
            FROM APP_LOOKUP_CASE_STAGE
            ORDER BY StageOrder
        """)
        
        stages = []
        for row in cursor.fetchall():
            stages.append({
                "id": row.StageID,
                "name_en": row.StageName,
                "name_ar": row.StageName
            })
        
        return {"stages": stages}
        
    except Exception as e:
        return {
            "stages": [],
            "error": f"Failed to fetch stages: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_harm_levels() -> Dict[str, Any]:
    """Get all harm levels."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT HarmID, HarmLevel
            FROM APP_LOOKUP_HARM_LEVEL
            ORDER BY SeverityOrder
        """)
        
        harm_levels = []
        for row in cursor.fetchall():
            harm_levels.append({
                "id": row.HarmID,
                "name_en": row.HarmLevel,
                "name_ar": row.HarmLevel
            })
        
        return {"harm_levels": harm_levels}
        
    except Exception as e:
        return {
            "harm_levels": [],
            "error": f"Failed to fetch harm levels: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_clinical_risk_types() -> Dict[str, Any]:
    """Get all clinical risk types."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT ClinicalRiskTypeID, Code, Name, IsActive, DisplayOrder
            FROM APP_LOOKUP_CLINICAL_RISK_TYPE
            WHERE IsActive = 1
            ORDER BY DisplayOrder
        """)
        
        clinical_risk_types = []
        for row in cursor.fetchall():
            clinical_risk_types.append({
                "id": row.ClinicalRiskTypeID,
                "name": row.Name,
                "name_ar": row.Name,
                "code": row.Code if hasattr(row, 'Code') else None
            })
        
        return {"clinical_risk_types": clinical_risk_types}
        
    except Exception as e:
        return {
            "clinical_risk_types": [],
            "error": f"Failed to fetch clinical risk types: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_feedback_intent_types() -> Dict[str, Any]:
    """Get all feedback intent types."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT FeedbackIntentTypeID, Code, NameAr, NameEn, IsActive, DisplayOrder
            FROM APP_LOOKUP_FEEDBACK_INTENT_TYPE
            WHERE IsActive = 1
            ORDER BY DisplayOrder
        """)
        
        feedback_intent_types = []
        for row in cursor.fetchall():
            feedback_intent_types.append({
                "id": row.FeedbackIntentTypeID,
                "name": row.NameAr,
                "name_ar": row.NameAr,
                "name_en": row.NameEn if hasattr(row, 'NameEn') else None,
                "code": row.Code if hasattr(row, 'Code') else None
            })
        
        return {"feedback_intent_types": feedback_intent_types}
        
    except Exception as e:
        return {
            "feedback_intent_types": [],
            "error": f"Failed to fetch feedback intent types: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_all_reference_data() -> Dict[str, Any]:
    """Get all reference data in a single call."""
    return {
        **get_departments(),
        **get_sources(),
        **get_domains(),
        **get_categories(),
        **get_subcategories(),
        **get_classifications(),
        **get_severity_levels(),
        **get_stages(),
        **get_harm_levels(),
        **get_clinical_risk_types(),
        **get_feedback_intent_types()
    }
