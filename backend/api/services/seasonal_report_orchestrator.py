"""
Seasonal Report Orchestrator
Service layer orchestration for seasonal report generation and retrieval.
"""

import logging
import time
from typing import Dict, Any

from backend.api.services.seasonal_report_generator import SeasonalReportGenerator
from backend.api.services.seasonal_report_query_service import SeasonalReportQueryService


logger = logging.getLogger("seasonal_report")


def get_or_generate_seasonal_report(
    season_id: int,
    orgunit_id: int,
    orgunit_type: int,
    user_id: int
) -> Dict[str, Any]:
    """
    Generate (or regenerate) a seasonal report and return the full report object.
    
    This function orchestrates the generation of a seasonal report by:
    1. Generating/regenerating the report using SeasonalReportGenerator
    2. Fetching the complete report data using SeasonalReportQueryService
    3. Returning the full report object to the caller
    
    Args:
        season_id: ID of the season (from APP_LOOKUP_SEASON)
        orgunit_id: Organizational unit ID
        orgunit_type: Organizational unit type (1=Department, 2=Building, etc.)
        user_id: User ID who triggered the generation
    
    Returns:
        Complete seasonal report dictionary with all fields and statistics
    
    Raises:
        RuntimeError: If generation or fetch fails
    """
    start = time.time()
    logger.info(f"[SEASONAL] Start generation: season={season_id}, org={orgunit_id}, type={orgunit_type}")
    
    try:
        # Step 1: Generate or regenerate the seasonal report
        generator = SeasonalReportGenerator()
        seasonal_report_id = generator.generate_or_regenerate_report(
            season_id=season_id,
            orgunit_id=orgunit_id,
            orgunit_type=orgunit_type,
            generated_by_user_id=user_id
        )
        
    except Exception as e:
        logger.exception("[SEASONAL] Failed")
        raise RuntimeError(f"Seasonal generation failed: {str(e)}")
    
    try:
        # Step 2: Fetch the complete report data
        query_service = SeasonalReportQueryService()
        report = query_service.get_by_id(report_id=seasonal_report_id)
        
        if not report:
            logger.exception("[SEASONAL] Failed")
            raise RuntimeError(
                f"Seasonal report not found after generation (ID={seasonal_report_id})"
            )
        
        elapsed = time.time() - start
        logger.info(f"[SEASONAL] Done in {elapsed:.2f}s")
        
        return report
        
    except RuntimeError:
        # Re-raise RuntimeError without wrapping
        raise
    except Exception as e:
        logger.exception("[SEASONAL] Failed")
        raise RuntimeError(f"Seasonal fetch failed: {str(e)}")
