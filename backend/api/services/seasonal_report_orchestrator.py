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
        generation_result = generator.generate_or_regenerate_report(
            season_id=season_id,
            orgunit_id=orgunit_id,
            orgunit_type=orgunit_type,
            generated_by_user_id=user_id
        )
        
        seasonal_report_id = generation_result['seasonal_report_id']
        
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


def get_or_generate_comparative_seasonal_reports(
    season_id: int,
    orgunit_id: int,
    orgunit_type: int,
    user_id: int
) -> Dict[str, Any]:
    """
    Generate (or regenerate) both current and previous seasonal reports for comparison.
    
    This function orchestrates the generation of a comparative report by:
    1. Generating/fetching the current season report
    2. Determining the previous season
    3. Generating/fetching the previous season report (even if it has zero data)
    4. Returning both reports for comparison
    
    Args:
        season_id: ID of the CURRENT season (from APP_LOOKUP_SEASON)
        orgunit_id: Organizational unit ID
        orgunit_type: Organizational unit type (1=Department, 2=Building, etc.)
        user_id: User ID who triggered the generation
    
    Returns:
        Dictionary with:
        {
            'current_report': Dict with current season data,
            'previous_report': Dict with previous season data (may have zero cases),
            'has_previous': bool indicating if previous season exists
        }
    
    Raises:
        RuntimeError: If generation or fetch fails
    """
    from backend.api.db_layer.seasonal_report import get_previous_season
    
    start = time.time()
    logger.info(f"[COMPARATIVE] Start: current_season={season_id}, org={orgunit_id}")
    
    try:
        # Step 1: Generate current season report
        current_report = get_or_generate_seasonal_report(
            season_id=season_id,
            orgunit_id=orgunit_id,
            orgunit_type=orgunit_type,
            user_id=user_id
        )
        
        # Step 2: Get previous season ID
        previous_season_id = get_previous_season(season_id)
        
        if previous_season_id is None:
            logger.info(f"[COMPARATIVE] No previous season found for season_id={season_id}")
            # Return current report with empty previous report structure
            previous_report = {
                'header': {
                    'period': 'N/A',
                    'orgunit_id': orgunit_id,
                    'orgunit_type': orgunit_type,
                    'orgunit_name': current_report['header'].get('orgunit_name', 'N/A'),
                    'total_cases': 0,
                    'low_severity_count': 0,
                    'medium_severity_count': 0,
                    'high_severity_count': 0,
                    'clinical_domain_count': 0,
                    'management_domain_count': 0,
                    'relational_domain_count': 0,
                    'is_compliant': True,
                    'violated_rules': None
                },
                'classification_stats': [],
                'policy_snapshot': {}
            }
        else:
            # Step 3: Generate/fetch previous season report
            logger.info(f"[COMPARATIVE] Generating previous season: {previous_season_id}")
            try:
                previous_report = get_or_generate_seasonal_report(
                    season_id=previous_season_id,
                    orgunit_id=orgunit_id,
                    orgunit_type=orgunit_type,
                    user_id=user_id
                )
            except Exception as e:
                logger.warning(f"[COMPARATIVE] Could not fetch previous report: {e}")
                # If previous season exists but report fails, create empty structure
                previous_report = {
                    'header': {
                        'period': f'Season-{previous_season_id}',
                        'orgunit_id': orgunit_id,
                        'orgunit_type': orgunit_type,
                        'orgunit_name': current_report['header'].get('orgunit_name', 'N/A'),
                        'total_cases': 0,
                        'low_severity_count': 0,
                        'medium_severity_count': 0,
                        'high_severity_count': 0,
                        'clinical_domain_count': 0,
                        'management_domain_count': 0,
                        'relational_domain_count': 0,
                        'is_compliant': True,
                        'violated_rules': None
                    },
                    'classification_stats': [],
                    'policy_snapshot': {}
                }
        
        elapsed = time.time() - start
        logger.info(f"[COMPARATIVE] Done in {elapsed:.2f}s")
        
        return {
            'current_report': current_report,
            'previous_report': previous_report,
            'has_previous': previous_season_id is not None
        }
        
    except RuntimeError:
        raise
    except Exception as e:
        logger.exception("[COMPARATIVE] Failed")
        raise RuntimeError(f"Comparative report generation failed: {str(e)}")

