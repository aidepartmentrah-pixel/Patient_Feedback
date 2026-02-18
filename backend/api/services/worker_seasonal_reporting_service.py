"""
D-B7: Worker Seasonal Report Builder

Pure data aggregation module for seasonal worker reports.
Calls existing worker services (NOT DB layer directly) per architecture rules.
This prepares worker seasonal payload for report export layer.

Returns structured dict payload - NO Word document generation here.
NO router imports. NO direct DB calls.
"""

from typing import Dict, Any
from datetime import date
from .worker_reporting_service import WorkerReportingService
from .performance_scoring import compute_performance_score


class WorkerSeasonalReportingService:
    """
    Builds seasonal report data for workers by aggregating existing services.
    
    Pure business logic - no DB calls, no Word generation.
    """
    
    @staticmethod
    def build_worker_seasonal_report_data(
        employee_id: int,
        season_start: date,
        season_end: date
    ) -> Dict[str, Any]:
        """
        Build seasonal report data for a worker.
        
        This function prepares a worker seasonal payload for report export layer.
        It aggregates data from existing worker reporting service without directly
        querying the database.
        
        Calls:
        - get_worker_profile: Gets worker identity and metrics for date range
        - compute_performance_score: Computes performance scoring from metrics
        
        Args:
            employee_id: Unique employee identifier
            season_start: Start date of reporting period
            season_end: End date of reporting period
            
        Returns:
            Dict with structure:
            {
                'worker_identity': {
                    'employee_id': int,
                    'full_name': str,
                    'job_title': str,
                    'department_id': int,
                    'section_id': int,
                    'administration_id': int
                },
                'period': {
                    'start': date,
                    'end': date
                },
                'metrics': {
                    'total_incidents': int,
                    'high_severity': int,
                    'medium_severity': int,
                    'low_severity': int,
                    'good_feedback_count': int,
                    'bad_feedback_count': int,
                    'neutral_feedback_count': int,
                    'total_action_items': int,
                    'completed_action_items': int,
                    'overdue_action_items': int,
                    'explanation_accepted_count': int,
                    'explanation_rejected_count': int
                },
                'incidents_details': list,
                'performance': {
                    'score': int,
                    'praise_level': str,
                    'risk_level': str,
                    'flags': list
                }
            }
            
        Raises:
            ValueError: If worker not found
            Exception: For data retrieval errors
        """
        # 1. Get worker profile with metrics for the season
        profile_response = WorkerReportingService.get_worker_profile(
            employee_id=employee_id,
            date_from=season_start,
            date_to=season_end
        )
        
        # 2. Extract identity block from Pydantic model
        worker_identity = {
            'employee_id': profile_response.worker.employee_id,
            'full_name': profile_response.worker.full_name,
            'job_title': profile_response.worker.job_title,
            'department_id': profile_response.worker.department_id,
            'section_id': profile_response.worker.section_id,
            'administration_id': profile_response.worker.administration_id
        }
        
        # 3. Extract metrics block from Pydantic model
        metrics = {
            'total_incidents': profile_response.metrics.total_incidents,
            'high_severity': profile_response.metrics.high_severity,
            'medium_severity': profile_response.metrics.medium_severity,
            'low_severity': profile_response.metrics.low_severity,
            'good_feedback_count': profile_response.metrics.good_feedback_count,
            'bad_feedback_count': profile_response.metrics.bad_feedback_count,
            'neutral_feedback_count': profile_response.metrics.neutral_feedback_count,
            'total_action_items': profile_response.metrics.total_action_items,
            'completed_action_items': profile_response.metrics.completed_action_items,
            'overdue_action_items': profile_response.metrics.overdue_action_items,
            'explanation_accepted_count': profile_response.metrics.explanation_accepted_count,
            'explanation_rejected_count': profile_response.metrics.explanation_rejected_count
        }
        
        # 3b. Extract incidents detail list
        incidents_details = profile_response.incidents if profile_response.incidents else []
        
        # 4. Compute performance score using scoring module
        performance_result = compute_performance_score(
            total_incidents=metrics['total_incidents'],
            overdue_actions=metrics['overdue_action_items'],
            rejected_explanations=metrics['explanation_rejected_count'],
            completed_actions=metrics['completed_action_items']
        )
        
        # 5. Build performance dict from score result
        performance = {
            'score': performance_result.score,
            'praise_level': performance_result.praise_level,
            'risk_level': performance_result.risk_level,
            'flags': performance_result.flags
        }
        
        # 6. Build and return final dict structure
        return {
            'worker_identity': worker_identity,
            'period': {
                'start': season_start,
                'end': season_end
            },
            'metrics': metrics,
            'incidents_details': incidents_details,
            'performance': performance
        }
