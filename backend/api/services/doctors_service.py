"""
Doctor Service Layer
Business logic for doctor profiles, statistics, and analytics.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re
from ..db_layer import doctors_db


class DoctorService:
    """Service for doctor-related operations."""
    
    @staticmethod
    def create_doctor(
        doctor_name: str,
        specialty: Optional[str] = None,
        is_active: bool = True,
        source_system: str = 'MANUAL'
    ) -> Dict[str, Any]:
        """
        Create a new doctor in the reserve table.
        
        Validates input and creates doctor record.
        
        Args:
            doctor_name: Doctor's full name (required, 3-200 chars)
            specialty: Medical specialty (optional, max 200 chars)
            is_active: Active status (default: True)
            source_system: Source identifier (default: 'MANUAL')
        
        Returns:
            Dict with success status and created doctor data
            
        Raises:
            ValueError: If validation fails
            Exception: For database errors
        """
        try:
            # ============================================
            # VALIDATION
            # ============================================
            
            # 1. Doctor name validation
            if not doctor_name or not doctor_name.strip():
                raise ValueError("Doctor name is required")
            
            doctor_name = doctor_name.strip()
            
            if len(doctor_name) < 3:
                raise ValueError("Doctor name must be at least 3 characters")
            
            if len(doctor_name) > 200:
                raise ValueError("Doctor name cannot exceed 200 characters")
            
            # 2. Specialty validation (if provided)
            if specialty:
                specialty = specialty.strip()
                if len(specialty) > 200:
                    raise ValueError("Specialty cannot exceed 200 characters")
            
            # 3. Source system validation (if provided)
            if source_system:
                source_system = source_system.strip()
                if len(source_system) > 100:
                    raise ValueError("Source system cannot exceed 100 characters")
            
            # ============================================
            # CREATE DOCTOR
            # ============================================
            
            doctor = doctors_db.create_doctor(
                doctor_name=doctor_name,
                specialty=specialty if specialty else None,
                is_active=is_active,
                source_system=source_system
            )
            
            # ============================================
            # SUCCESS RESPONSE
            # ============================================
            
            return {
                'success': True,
                'message': f"Doctor '{doctor_name}' created successfully",
                'message_ar': f"تم إنشاء الطبيب '{doctor_name}' بنجاح",
                'doctor': doctor
            }
            
        except ValueError as ve:
            # Validation errors
            raise ValueError(str(ve))
        except Exception as e:
            # Database or other errors
            error_msg = str(e)
            if "already exists" in error_msg.lower():
                raise ValueError(f"Doctor with name '{doctor_name}' already exists")
            raise Exception(f"Failed to create doctor: {error_msg}")
    
    @staticmethod
    def search_doctors(
        query: Optional[str] = None,
        department: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Search for doctors.
        
        Returns:
            Dict with doctors array and total count
        """
        try:
            doctors = doctors_db.search_doctors(
                query=query,
                department=department,
                status=status,
                limit=limit
            )
            
            return {
                'doctors': doctors,
                'total': len(doctors)
            }
        except Exception as e:
            raise Exception(f"Doctor search failed: {str(e)}")
    
    @staticmethod
    def get_doctor_profile(doctor_id: int) -> Dict[str, Any]:
        """
        Get full doctor profile.
        
        Returns:
            Doctor profile dict
            
        Raises:
            ValueError if doctor not found
        """
        try:
            profile = doctors_db.get_doctor_profile(doctor_id)
            
            if not profile:
                raise ValueError(f"Doctor {doctor_id} not found")
            
            return profile
        except Exception as e:
            raise Exception(f"Failed to fetch doctor profile: {str(e)}")
    
    @staticmethod
    def get_doctor_statistics(
        doctor_id: int,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated incident statistics for a doctor.
        
        Validates date range and defaults to last 6 months if not provided.
        
        Returns:
            Dict with statistics and period info
            
        Raises:
            ValueError if dates invalid
        """
        try:
            # Validate doctor exists
            profile = doctors_db.get_doctor_profile(doctor_id)
            if not profile:
                raise ValueError(f"Doctor {doctor_id} not found")
            
            # Parse and validate dates
            if to_date:
                try:
                    to_date_obj = datetime.strptime(to_date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError("Invalid to_date format (use YYYY-MM-DD)")
            else:
                to_date_obj = datetime.now()
                to_date = to_date_obj.strftime('%Y-%m-%d')
            
            if from_date:
                try:
                    from_date_obj = datetime.strptime(from_date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError("Invalid from_date format (use YYYY-MM-DD)")
            else:
                from_date_obj = to_date_obj - timedelta(days=180)
                from_date = from_date_obj.strftime('%Y-%m-%d')
            
            if from_date_obj > to_date_obj:
                raise ValueError("from_date cannot be after to_date")
            
            # Fetch statistics
            stats = doctors_db.get_doctor_statistics(
                doctor_id=doctor_id,
                from_date=from_date,
                to_date=to_date
            )
            
            return {
                'statistics': stats,
                'period': {
                    'from': from_date,
                    'to': to_date
                }
            }
        except Exception as e:
            raise Exception(f"Failed to fetch statistics: {str(e)}")
    
    @staticmethod
    def get_doctor_analytics(
        doctor_id: int,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get analytics data including category breakdown and monthly trend.
        
        Returns:
            Dict with categoryBreakdown, monthlyTrend, and period
        """
        try:
            # Validate doctor exists
            profile = doctors_db.get_doctor_profile(doctor_id)
            if not profile:
                raise ValueError(f"Doctor {doctor_id} not found")
            
            # Parse and validate dates
            if to_date:
                try:
                    to_date_obj = datetime.strptime(to_date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError("Invalid to_date format (use YYYY-MM-DD)")
            else:
                to_date_obj = datetime.now()
                to_date = to_date_obj.strftime('%Y-%m-%d')
            
            if from_date:
                try:
                    from_date_obj = datetime.strptime(from_date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError("Invalid from_date format (use YYYY-MM-DD)")
            else:
                from_date_obj = to_date_obj - timedelta(days=180)
                from_date = from_date_obj.strftime('%Y-%m-%d')
            
            if from_date_obj > to_date_obj:
                raise ValueError("from_date cannot be after to_date")
            
            # Fetch analytics data
            category_breakdown = doctors_db.get_doctor_category_breakdown(
                doctor_id=doctor_id,
                from_date=from_date,
                to_date=to_date
            )
            
            monthly_trend = doctors_db.get_doctor_monthly_trend(
                doctor_id=doctor_id,
                from_date=from_date,
                to_date=to_date
            )
            
            # Zero-fill months if needed
            monthly_trend = DoctorService._zero_fill_months(
                monthly_trend, from_date_obj, to_date_obj
            )
            
            return {
                'categoryBreakdown': category_breakdown,
                'monthlyTrend': monthly_trend,
                'period': {
                    'from': from_date,
                    'to': to_date
                }
            }
        except Exception as e:
            raise Exception(f"Failed to fetch analytics: {str(e)}")
    
    @staticmethod
    def get_doctor_incidents(
        doctor_id: int,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        red_flags_only: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get paginated incidents for a doctor.
        
        Returns:
            Dict with incidents array and pagination info
        """
        try:
            # Validate doctor exists
            profile = doctors_db.get_doctor_profile(doctor_id)
            if not profile:
                raise ValueError(f"Doctor {doctor_id} not found")
            
            # Validate severity if provided
            if severity:
                if severity not in ['HIGH', 'MEDIUM', 'LOW']:
                    raise ValueError(f"Invalid severity: {severity}")
            
            # Validate status if provided
            if status:
                if status not in ['OPEN', 'UNDER_REVIEW', 'CLOSED']:
                    raise ValueError(f"Invalid status: {status}")
            
            # Parse and validate dates
            if to_date:
                try:
                    datetime.strptime(to_date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError("Invalid to_date format (use YYYY-MM-DD)")
            
            if from_date:
                try:
                    datetime.strptime(from_date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError("Invalid from_date format (use YYYY-MM-DD)")
            
            # Fetch incidents
            result = doctors_db.get_doctor_incidents(
                doctor_id=doctor_id,
                from_date=from_date,
                to_date=to_date,
                severity=severity,
                status=status,
                red_flags_only=red_flags_only,
                limit=limit,
                offset=offset
            )
            
            return result
        except Exception as e:
            raise Exception(f"Failed to fetch incidents: {str(e)}")
    
    @staticmethod
    def get_doctor_analytics(
        doctor_id: int,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get doctor analytics combining category breakdown and monthly trends.
        
        Returns:
            Dict with categoryBreakdown, monthlyTrend, period
            
        Raises:
            ValueError: If doctor not found or invalid date format
        """
        try:
            # Validate doctor exists
            profile = doctors_db.get_doctor_profile(doctor_id)
            if not profile:
                raise ValueError(f"Doctor with ID {doctor_id} not found")
            
            # Parse and default dates
            today = datetime.now()
            default_from = today - timedelta(days=180)
            
            if from_date:
                try:
                    parsed_from = datetime.strptime(from_date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError(f"Invalid from_date format. Use YYYY-MM-DD")
            else:
                parsed_from = default_from
            
            if to_date:
                try:
                    parsed_to = datetime.strptime(to_date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError(f"Invalid to_date format. Use YYYY-MM-DD")
            else:
                parsed_to = today
            
            # Validate date range
            if parsed_from > parsed_to:
                raise ValueError("from_date must be before or equal to to_date")
            
            # Fetch category breakdown and monthly trend
            category_breakdown = doctors_db.get_doctor_category_breakdown(
                doctor_id, parsed_from, parsed_to
            )
            
            monthly_trend = doctors_db.get_doctor_monthly_trend(
                doctor_id, parsed_from, parsed_to
            )
            
            # Zero-fill months for complete trend
            zero_filled_trend = DoctorService._zero_fill_months(
                monthly_trend, parsed_from, parsed_to
            )
            
            return {
                'categoryBreakdown': category_breakdown,
                'monthlyTrend': zero_filled_trend,
                'period': {
                    'from': parsed_from.strftime('%Y-%m-%d'),
                    'to': parsed_to.strftime('%Y-%m-%d')
                }
            }
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Failed to fetch analytics: {str(e)}")
    
    @staticmethod
    def _zero_fill_months(
        monthly_trend: List[Dict],
        from_date: datetime,
        to_date: datetime
    ) -> List[Dict]:
        """
        Fill in missing months with zero counts.
        
        Ensures all months in the date range are represented.
        """
        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create a map of existing months
        existing = {item['month']: item['count'] for item in monthly_trend}
        
        # Generate all months in range
        result = []
        current = from_date.replace(day=1)
        
        while current <= to_date:
            month_label = month_names[current.month]
            
            # Include year if spanning multiple years
            from_year = from_date.year
            to_year = to_date.year
            if from_year != to_year:
                month_label = f"{month_label} {current.year}"
            
            count = existing.get(month_label, 0)
            result.append({'month': month_label, 'count': count})
            
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return result
    
    @staticmethod
    def get_doctor_full_report(
        doctor_id: int,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        red_flags_only: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get comprehensive doctor report combining all data.
        
        Single call returns: profile, statistics, analytics, incidents.
        
        Returns:
            Dict with profile, statistics, analytics, incidents sections
            
        Raises:
            ValueError: If doctor not found or invalid parameters
        """
        try:
            # Validate doctor exists
            profile = doctors_db.get_doctor_profile(doctor_id)
            if not profile:
                raise ValueError(f"Doctor with ID {doctor_id} not found")
            
            # Get all components
            profile_data = DoctorService.get_doctor_profile(doctor_id)
            
            statistics_data = DoctorService.get_doctor_statistics(
                doctor_id=doctor_id,
                from_date=from_date,
                to_date=to_date
            )
            
            analytics_data = DoctorService.get_doctor_analytics(
                doctor_id=doctor_id,
                from_date=from_date,
                to_date=to_date
            )
            
            incidents_data = DoctorService.get_doctor_incidents(
                doctor_id=doctor_id,
                from_date=from_date,
                to_date=to_date,
                severity=severity,
                status=status,
                red_flags_only=red_flags_only,
                limit=limit,
                offset=offset
            )
            
            return {
                'profile': profile_data,
                'statistics': statistics_data.get('statistics'),
                'analytics': {
                    'categoryBreakdown': analytics_data.get('categoryBreakdown'),
                    'monthlyTrend': analytics_data.get('monthlyTrend')
                },
                'incidents': incidents_data.get('incidents'),
                'incidentsPagination': {
                    'total': incidents_data.get('total'),
                    'limit': incidents_data.get('limit'),
                    'offset': incidents_data.get('offset')
                },
                'period': statistics_data.get('period')
            }
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Failed to generate full report: {str(e)}")
