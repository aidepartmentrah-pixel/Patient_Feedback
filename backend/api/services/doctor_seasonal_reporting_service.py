"""
D-B6: Doctor Seasonal Report Builder

Pure data aggregation module for seasonal doctor reports.
Calls existing doctor services (NOT DB layer directly) per architecture rules.
Returns structured dict payload for later Word generator reuse.

Architecture:
- This service → doctors_service → doctors_db
- NO Word generation here
- NO scoring yet (comes in later integration step)
"""

from typing import Dict, Any
from datetime import date
from collections import Counter
from .doctors_service import DoctorService


class DoctorSeasonalReportingService:
    """
    Builds seasonal report data for doctors by aggregating existing services.
    
    Pure business logic - no DB calls, no Word generation, no scoring.
    """
    
    @staticmethod
    def build_doctor_seasonal_report_data(
        doctor_id: int,
        season_start: date,
        season_end: date
    ) -> Dict[str, Any]:
        """
        Build seasonal report data for a doctor.
        
        This function prepares a data payload for later Word generator reuse.
        It aggregates data from existing doctor services without directly querying the database.
        
        Calls:
        - get_doctor_profile: Gets doctor identity information
        - get_doctor_statistics: Gets aggregated metrics for date range
        - get_doctor_incidents: Gets incident details for top categories computation
        
        Args:
            doctor_id: Employee ID of the doctor
            season_start: Start date of reporting period
            season_end: End date of reporting period
            
        Returns:
            Dict with structure:
            {
                'doctor_identity': {
                    'id': int,
                    'name': str,
                    'specialty': str
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
                    'red_flags': int
                },
                'incidents_summary': {
                    'count': int,
                    'top_categories': list  # Top 5 categories by frequency
                }
            }
            
        Raises:
            Exception: If doctor not found or data retrieval fails
        """
        # Convert dates to ISO string format for DoctorService calls
        # Handle both date objects and string inputs
        if isinstance(season_start, str):
            date_from = season_start
        else:
            date_from = season_start.isoformat()
        
        if isinstance(season_end, str):
            date_to = season_end
        else:
            date_to = season_end.isoformat()
        
        # 1. Get doctor identity from profile service
        profile = DoctorService.get_doctor_profile(doctor_id)
        
        # 2. Get statistics for the season date range
        statistics_result = DoctorService.get_doctor_statistics(
            doctor_id=doctor_id,
            from_date=date_from,
            to_date=date_to
        )
        stats = statistics_result.get('statistics', {})
        
        # 3. Get all incidents for the season to compute top categories
        incidents_result = DoctorService.get_doctor_incidents(
            doctor_id=doctor_id,
            from_date=date_from,
            to_date=date_to,
            severity=None,  # Get all severities
            limit=1000,  # Large limit to get all incidents
            offset=0
        )
        incidents_list = incidents_result.get('incidents', [])
        
        # 4. Compute top 5 categories from incidents list
        categories = [inc.get('category', 'Unknown') for inc in incidents_list]
        category_counts = Counter(categories)
        top_categories = [{'category': cat, 'count': count} 
                         for cat, count in category_counts.most_common(5)]
        
        # 5. Build and return data payload
        return {
            'doctor_identity': {
                'id': profile.get('employee_id') or profile.get('id') or doctor_id,
                'name': profile.get('name_en') or profile.get('name', 'Unknown'),
                'specialty': profile.get('specialty', 'Unknown')
            },
            'period': {
                'start': season_start,
                'end': season_end
            },
            'metrics': {
                'total_incidents': stats.get('total', 0),
                'high_severity': stats.get('high', 0),
                'medium_severity': stats.get('medium', 0),
                'low_severity': stats.get('low', 0),
                'red_flags': stats.get('red_flags', 0),
                'good_feedback_count': stats.get('good_feedback', 0),
                'bad_feedback_count': stats.get('bad_feedback', 0),
                'neutral_feedback_count': stats.get('neutral_feedback', 0)
            },
            'incidents_summary': {
                'count': len(incidents_list),
                'top_categories': top_categories
            },
            'incidents_details': incidents_list  # Add full incident details for Word table
        }

