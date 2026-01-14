"""
Seasonal Report Query Service
Read-only service for fetching materialized seasonal reports.
"""

from typing import Dict, Any, Optional
from backend.api.db_layer.seasonal_report import (
    get_full_seasonal_report,
    get_seasonal_report_keys_by_id
)


class SeasonalReportQueryService:
    """
    Read-only query service for materialized seasonal reports.
    Never recomputes - only fetches stored reports.
    """
    
    def get_by_season_orgunit(
        self,
        season_id: int,
        orgunit_id: int,
        orgunit_type: int
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch complete seasonal report by unique key.
        
        Args:
            season_id: Season identifier
            orgunit_id: Organizational unit identifier
            orgunit_type: Type of organizational unit
        
        Returns:
            Complete report object or None if not found:
            {
                'header': {
                    'seasonal_report_id': int,
                    'season_id': int,
                    'orgunit_id': int,
                    'orgunit_type': int,
                    'total_cases': int,
                    'low_severity_count': int,
                    'medium_severity_count': int,
                    'high_severity_count': int,
                    'clinical_domain_count': int,
                    'management_domain_count': int,
                    'relational_domain_count': int,
                    'is_compliant': bool,
                    'violated_rules': str | None,
                    'explanation_status_id': int,
                    'created_by_user_id': int,
                    'created_at': str
                },
                'classification_stats': [
                    {
                        'classification_id': int,
                        'total_count': int,
                        'low_count': int,
                        'medium_count': int,
                        'high_count': int,
                        'preventive_yes_count': int,
                        'preventive_no_count': int
                    },
                    ...
                ],
                'policy_snapshot': {
                    'orgunit_id': int,
                    'orgunit_type': int,
                    'max_allowed_cases': int,
                    'max_clinical_domain': int,
                    'max_management_domain': int,
                    'max_relational_domain': int,
                    'require_explanation_above_threshold': bool,
                    'escalation_enabled': bool,
                    'escalation_threshold_percentage': int
                }
            }
        """
        return get_full_seasonal_report(
            season_id=season_id,
            orgunit_id=orgunit_id,
            orgunit_type=orgunit_type
        )
    
    def get_by_id(
        self,
        report_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch complete seasonal report by direct report ID.
        
        This method first retrieves the header keys to get the unique identifiers,
        then fetches the complete report using those keys.
        
        Args:
            report_id: Seasonal report identifier (SeasonalReportID)
        
        Returns:
            Complete report object or None if not found.
            Same structure as get_by_season_orgunit().
        """
        # Step 1: Get the unique keys for this report
        keys = get_seasonal_report_keys_by_id(report_id)
        
        if not keys:
            return None
        
        # Step 2: Fetch the full report using those keys
        return get_full_seasonal_report(
            season_id=keys['season_id'],
            orgunit_id=keys['orgunit_id'],
            orgunit_type=keys['orgunit_type']
        )
