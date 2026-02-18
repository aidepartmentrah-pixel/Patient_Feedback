"""
Doctor Service Layer
Business logic for doctor profiles, statistics, and analytics.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import csv
import io
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
    def get_reserve_doctors(limit: int = 100) -> Dict[str, Any]:
        """
        Get all doctors from reserve table only.
        
        Returns only user-created doctors (not from hospital system).
        
        Args:
            limit: Max results
        
        Returns:
            Dict with reserve doctors array and total count
        """
        try:
            doctors = doctors_db.get_reserve_doctors(limit=limit)
            
            return {
                'doctors': doctors,
                'total': len(doctors),
                'message': f"Found {len(doctors)} reserve doctor(s)",
                'message_ar': f"تم العثور على {len(doctors)} طبيب في الجدول الاحتياطي"
            }
        except Exception as e:
            raise Exception(f"Failed to fetch reserve doctors: {str(e)}")
    
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
        import logging
        logger = logging.getLogger("doctor_full_report")
        
        try:
            # Validate doctor exists
            profile = doctors_db.get_doctor_profile(doctor_id)
            if not profile:
                logger.warning(f"Doctor with ID {doctor_id} not found in any table.")
                raise ValueError(f"Doctor with ID {doctor_id} not found")
            
            # Get all components with robust error handling
            profile_data = None
            statistics_data = {'statistics': {}, 'period': None}
            analytics_data = {'categoryBreakdown': {}, 'monthlyTrend': {}}
            incidents_data = {'incidents': [], 'total': 0, 'limit': limit, 'offset': offset}
            
            try:
                profile_data = DoctorService.get_doctor_profile(doctor_id)
            except Exception as e:
                logger.error(f"[doctor_id={doctor_id}] Profile error: {e}")
                profile_data = profile  # fallback to raw profile
            
            try:
                statistics_data = DoctorService.get_doctor_statistics(
                    doctor_id=doctor_id,
                    from_date=from_date,
                    to_date=to_date
                )
            except Exception as e:
                logger.error(f"[doctor_id={doctor_id}] Statistics error: {e}")
            
            try:
                analytics_data = DoctorService.get_doctor_analytics(
                    doctor_id=doctor_id,
                    from_date=from_date,
                    to_date=to_date
                )
            except Exception as e:
                logger.error(f"[doctor_id={doctor_id}] Analytics error: {e}")
            
            try:
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
            except Exception as e:
                logger.error(f"[doctor_id={doctor_id}] Incidents error: {e}")
            
            return {
                'profile': profile_data,
                'statistics': statistics_data.get('statistics', {}),
                'analytics': {
                    'categoryBreakdown': analytics_data.get('categoryBreakdown', {}),
                    'monthlyTrend': analytics_data.get('monthlyTrend', {})
                },
                'incidents': incidents_data.get('incidents', []),
                'incidentsPagination': {
                    'total': incidents_data.get('total', 0),
                    'limit': incidents_data.get('limit', limit),
                    'offset': incidents_data.get('offset', offset)
                },
                'period': statistics_data.get('period', None)
            }
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"[doctor_id={doctor_id}] Full report failed: {str(e)}")
            raise Exception(f"Failed to generate full report: {str(e)}")


# ==================== FULL HISTORY (Combined — mirrors patient) ====================

def get_doctor_full_history_service(
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
    Get doctor profile and incidents in single response.
    Mirrors get_patient_full_history_service for UI consistency.
    
    Returns:
        Dict with profile, metrics, items, and meta (V2 schema compatible)
    """
    try:
        # Get profile
        profile = doctors_db.get_doctor_profile(doctor_id)
        if not profile:
            raise ValueError(f"Doctor with ID {doctor_id} not found")
        
        # Get incidents
        incidents_data = doctors_db.get_doctor_incidents(
            doctor_id=doctor_id,
            from_date=from_date,
            to_date=to_date,
            severity=severity,
            status=status,
            red_flags_only=red_flags_only,
            limit=limit,
            offset=offset
        )
        
        # Get aggregated metrics
        metrics = doctors_db.get_doctor_metrics(
            doctor_id=doctor_id,
            from_date=from_date,
            to_date=to_date
        )
        
        # Return normalized V2 schema format (same shape as patient)
        return {
            "profile": profile,
            "metrics": metrics,
            "items": incidents_data.get('incidents', []),
            "meta": {
                "entity_type": "doctor",
                "entity_id": doctor_id,
                "period_from": from_date,
                "period_to": to_date,
                "total": incidents_data.get('total', 0),
                "limit": limit,
                "offset": offset
            }
        }
    
    except ValueError:
        raise
    except Exception as e:
        raise Exception(f"Failed to get full doctor history: {str(e)}")


# ==================== EXPORT ====================

def export_doctor_history_service(
    doctor_id: int,
    format_type: str = "json",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    include_profile: bool = True
) -> Dict[str, Any]:
    """
    Generate doctor history export in CSV, JSON, or Word format.
    Mirrors export_patient_history_service.
    
    Args:
        doctor_id: Doctor ID
        format_type: "csv", "json", or "word"
        from_date: Optional start date filter
        to_date: Optional end date filter
        include_profile: Include doctor profile
    
    Returns:
        For CSV: Dict with filename and csv content
        For Word: Dict with filename and word bytes
        For JSON: JSON export dict
    """
    try:
        # Get export data
        export_data = doctors_db.get_doctor_incidents_for_export(
            doctor_id=doctor_id,
            from_date=from_date,
            to_date=to_date,
            include_profile=include_profile
        )
        
        if format_type == "csv":
            return _generate_doctor_csv_export(export_data, doctor_id)
        elif format_type == "word":
            return _generate_doctor_word_export(export_data, doctor_id)
        else:
            return export_data
    
    except Exception as e:
        raise Exception(f"Failed to export doctor history: {str(e)}")


def _generate_doctor_csv_export(export_data: Dict[str, Any], doctor_id: int) -> Dict[str, Any]:
    """Generate CSV export from doctor export data."""
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write doctor profile header
        if export_data.get('doctor'):
            doc = export_data['doctor']
            writer.writerow(['DOCTOR PROFILE'])
            writer.writerow(['Doctor ID', doc.get('doctor_id')])
            writer.writerow(['Full Name', doc.get('full_name')])
            writer.writerow(['Specialty', doc.get('specialty', '')])
            writer.writerow(['Status', doc.get('status', '')])
            writer.writerow(['Total Incidents', doc.get('total_incidents', 0)])
            writer.writerow([])
        
        # Write incidents
        writer.writerow(['INCIDENT HISTORY'])
        
        if export_data.get('incidents'):
            headers = list(export_data['incidents'][0].keys())
            writer.writerow(headers)
            
            for incident in export_data['incidents']:
                writer.writerow(incident.values())
        
        csv_content = output.getvalue()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"doctor_{doctor_id}_history_{timestamp}.csv"
        
        return {
            "filename": filename,
            "content": csv_content,
            "content_type": "text/csv"
        }
    
    except Exception as e:
        raise Exception(f"Failed to generate CSV: {str(e)}")


def _generate_doctor_word_export(export_data: Dict[str, Any], doctor_id: int) -> Dict[str, Any]:
    """Generate Word document export from doctor export data."""
    try:
        from .seasonal_word_adapter import SeasonalWordAdapter
        
        # Adapt export_data to the structure expected by the word generator
        # Map to the doctor seasonal format
        doctor_info = export_data.get('doctor', {}) or {}
        incidents = export_data.get('incidents', [])
        
        word_data = {
            'doctor_identity': {
                'doctor_id': doctor_info.get('doctor_id', doctor_id),
                'name': doctor_info.get('full_name', f'Doctor {doctor_id}'),
                'specialty': doctor_info.get('specialty', ''),
            },
            'period': {
                'from': export_data.get('export_date', ''),
                'to': export_data.get('export_date', ''),
                'label': 'Export Report'
            },
            'metrics': {
                'total_incidents': doctor_info.get('total_incidents', len(incidents)),
                'high_severity': sum(1 for i in incidents if i.get('Severity') == 'High'),
                'medium_severity': sum(1 for i in incidents if i.get('Severity') == 'Medium'),
                'low_severity': sum(1 for i in incidents if i.get('Severity') == 'Low'),
                'red_flags': sum(1 for i in incidents if i.get('RedFlag') == 'Yes'),
            },
            'performance': {
                'top_categories': [],
            },
            'incidents': incidents
        }
        
        word_bytes = SeasonalWordAdapter.generate_doctor_seasonal_word(word_data)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"doctor_{doctor_id}_report_{timestamp}.docx"
        
        return {
            "filename": filename,
            "content": word_bytes,
            "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        }
    
    except Exception as e:
        raise Exception(f"Failed to generate Word document: {str(e)}")
