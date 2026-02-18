"""
Worker Reporting Service Layer
Business logic for worker profile and performance metrics aggregation.

This module provides service functions for retrieving worker profiles with
aggregated performance metrics. It orchestrates calls to the database layer
and assembles response models.

Note:
    - No permission/scope checking here (done at router layer)
    - No performance scoring logic (separate module in D-B5)
    - Pure aggregation from multiple DB sources
"""

from typing import Optional, Dict, Any, List
from datetime import date, datetime
import csv
import io

from ..db_layer import worker_reporting_db
from ..schemas.worker_reporting_schema import (
    WorkerProfileResponse,
    WorkerIdentityBlock,
    WorkerMetricBlock
)


class WorkerReportingService:
    """Service for worker reporting and profile operations."""
    
    @staticmethod
    def get_worker_profile(
        employee_id: int,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None
    ) -> WorkerProfileResponse:
        """
        Retrieve worker profile with aggregated performance metrics.
        
        This function orchestrates multiple database queries to assemble a complete
        worker profile including identity information from HR and performance metrics
        aggregated from incidents, action items, and explanation workflows.
        
        Aggregation Steps:
            1. Fetch worker identity from HR employee view
            2. Count incidents involving worker (filtered by date range)
            3. Count action items with completion status breakdown
            4. Count explanation statuses (accepted vs rejected)
            5. Assemble all metrics into response model
        
        Note:
            - No performance scoring is calculated here (see scoring module D-B5)
            - Date range filters are optional (None = all-time metrics)
            - Raises ValueError if worker not found in HR system
            - All counts default to 0 if no data exists
        
        Args:
            employee_id: Unique employee identifier from HR system
            date_from: Optional start date for metrics aggregation period
            date_to: Optional end date for metrics aggregation period
        
        Returns:
            WorkerProfileResponse with identity block, metrics block, and date range
        
        Raises:
            ValueError: If worker identity not found in HR employee view
            Exception: For database connection or query errors
        
        Example:
            >>> from datetime import date
            >>> profile = WorkerReportingService.get_worker_profile(
            ...     employee_id=12345,
            ...     date_from=date(2025, 1, 1),
            ...     date_to=date(2025, 12, 31)
            ... )
            >>> print(f"{profile.worker.full_name}: {profile.metrics.total_incidents} incidents")
        """
        try:
            # ============================================
            # STEP 1: GET WORKER IDENTITY
            # ============================================
            
            worker_identity = worker_reporting_db.get_worker_identity(employee_id)
            
            if not worker_identity:
                raise ValueError(f"Worker not found: employee_id={employee_id}")
            
            # ============================================
            # STEP 2: COUNT INCIDENTS (via junction table)
            # ============================================
            
            total_incidents = worker_reporting_db.count_worker_incidents(
                employee_id=employee_id,
                date_from=date_from,
                date_to=date_to
            )
            
            # ============================================
            # STEP 3: SEVERITY BREAKDOWN
            # ============================================
            
            severity_counts = worker_reporting_db.count_worker_incidents_by_severity(
                employee_id=employee_id,
                date_from=date_from,
                date_to=date_to
            )
            
            # ============================================
            # STEP 4: INTENT CLASSIFICATION (Good/Bad/Neutral)
            # ============================================
            
            intent_counts = worker_reporting_db.count_worker_incidents_by_intent(
                employee_id=employee_id,
                date_from=date_from,
                date_to=date_to
            )
            
            # ============================================
            # STEP 5: DETAILED INCIDENTS LIST
            # ============================================
            
            incidents_detail = worker_reporting_db.get_worker_incidents_detail(
                employee_id=employee_id,
                date_from=date_from,
                date_to=date_to
            )
            
            # ============================================
            # STEP 6: COUNT ACTION ITEMS
            # ============================================
            
            action_item_counts = worker_reporting_db.count_worker_action_items(
                employee_id=employee_id,
                date_from=date_from,
                date_to=date_to
            )
            
            # ============================================
            # STEP 7: COUNT EXPLANATION STATUSES
            # ============================================
            
            explanation_statuses = worker_reporting_db.count_worker_explanation_status(
                employee_id=employee_id,
                status_id=None,
                date_from=date_from,
                date_to=date_to
            )
            
            explanation_accepted_count = explanation_statuses.get(3, 0)
            explanation_rejected_count = explanation_statuses.get(4, 0)
            
            # ============================================
            # STEP 8: BUILD RESPONSE MODEL
            # ============================================
            
            worker_block = WorkerIdentityBlock(
                employee_id=worker_identity['employee_id'],
                full_name=worker_identity['full_name'],
                job_title=worker_identity.get('job_title'),
                department_id=worker_identity.get('department_id'),
                section_id=worker_identity.get('section_id'),
                administration_id=worker_identity.get('administration_id'),
                is_active=worker_identity.get('is_active')
            )
            
            metrics_block = WorkerMetricBlock(
                total_incidents=total_incidents,
                high_severity=severity_counts.get('high', 0),
                medium_severity=severity_counts.get('medium', 0),
                low_severity=severity_counts.get('low', 0),
                good_feedback_count=intent_counts.get('good', 0),
                bad_feedback_count=intent_counts.get('bad', 0),
                neutral_feedback_count=intent_counts.get('neutral', 0),
                total_action_items=action_item_counts.get('total', 0),
                completed_action_items=action_item_counts.get('completed', 0),
                overdue_action_items=action_item_counts.get('overdue', 0),
                explanation_rejected_count=explanation_rejected_count,
                explanation_accepted_count=explanation_accepted_count
            )
            
            response = WorkerProfileResponse(
                worker=worker_block,
                metrics=metrics_block,
                incidents=incidents_detail,
                period_from=date_from,
                period_to=date_to
            )
            
            return response
            
        except ValueError as ve:
            # Re-raise validation errors (worker not found)
            raise ve
        except Exception as e:
            # Wrap database or other errors
            raise Exception(f"Failed to retrieve worker profile: {str(e)}")


# ==================== FULL HISTORY (Combined — mirrors patient/doctor) ====================

def get_worker_full_history_service(
    employee_id: int,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Get worker profile and incidents in single response.
    Mirrors get_patient_full_history_service / get_doctor_full_history_service.
    
    Returns:
        Dict with profile, metrics, items, and meta (V2 schema compatible)
    """
    try:
        # Get worker identity
        identity = worker_reporting_db.get_worker_identity(employee_id)
        if not identity:
            raise ValueError(f"Worker with employee_id {employee_id} not found")
        
        # Get incidents list
        incidents = worker_reporting_db.get_worker_incidents_detail(
            employee_id=employee_id,
            date_from=date_from,
            date_to=date_to,
            limit=limit
        )
        
        # Get aggregated metrics
        metrics = worker_reporting_db.get_worker_metrics(
            employee_id=employee_id,
            date_from=date_from,
            date_to=date_to
        )
        
        # Get action item counts
        action_counts = worker_reporting_db.count_worker_action_items(
            employee_id=employee_id,
            date_from=date_from,
            date_to=date_to
        )
        metrics['action_items'] = action_counts
        
        # Return normalized V2 schema format (same shape as patient/doctor)
        return {
            "profile": identity,
            "metrics": metrics,
            "items": incidents[offset:offset + limit] if offset > 0 else incidents[:limit],
            "meta": {
                "entity_type": "worker",
                "entity_id": employee_id,
                "period_from": date_from.isoformat() if date_from else None,
                "period_to": date_to.isoformat() if date_to else None,
                "total": len(incidents),
                "limit": limit,
                "offset": offset
            }
        }
    
    except ValueError:
        raise
    except Exception as e:
        raise Exception(f"Failed to get full worker history: {str(e)}")


# ==================== EXPORT ====================

def export_worker_history_service(
    employee_id: int,
    format_type: str = "json",
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    include_profile: bool = True
) -> Dict[str, Any]:
    """
    Generate worker history export in CSV, JSON, or Word format.
    Mirrors export_patient_history_service / export_doctor_history_service.
    
    Args:
        employee_id: Employee ID
        format_type: "csv", "json", or "word"
        date_from: Optional start date filter
        date_to: Optional end date filter
        include_profile: Include worker profile
    
    Returns:
        For CSV: Dict with filename and csv content
        For Word: Dict with filename and word bytes
        For JSON: JSON export dict
    """
    try:
        # Get export data
        export_data = worker_reporting_db.get_worker_incidents_for_export(
            employee_id=employee_id,
            date_from=date_from,
            date_to=date_to,
            include_profile=include_profile
        )
        
        if format_type == "csv":
            return _generate_worker_csv_export(export_data, employee_id)
        elif format_type == "word":
            return _generate_worker_word_export(export_data, employee_id)
        else:
            return export_data
    
    except Exception as e:
        raise Exception(f"Failed to export worker history: {str(e)}")


def _generate_worker_csv_export(export_data: Dict[str, Any], employee_id: int) -> Dict[str, Any]:
    """Generate CSV export from worker export data."""
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write worker profile header
        if export_data.get('worker'):
            w = export_data['worker']
            writer.writerow(['WORKER PROFILE'])
            writer.writerow(['Employee ID', w.get('employee_id')])
            writer.writerow(['Full Name', w.get('full_name')])
            writer.writerow(['Job Title', w.get('job_title', '')])
            writer.writerow(['Department ID', w.get('department_id', '')])
            writer.writerow(['Section ID', w.get('section_id', '')])
            writer.writerow(['Status', 'Active' if w.get('is_active') else 'Inactive'])
            writer.writerow(['Total Incidents', w.get('total_incidents', 0)])
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
        filename = f"worker_{employee_id}_history_{timestamp}.csv"
        
        return {
            "filename": filename,
            "content": csv_content,
            "content_type": "text/csv"
        }
    
    except Exception as e:
        raise Exception(f"Failed to generate CSV: {str(e)}")


def _generate_worker_word_export(export_data: Dict[str, Any], employee_id: int) -> Dict[str, Any]:
    """Generate Word document export from worker export data."""
    try:
        from .seasonal_word_adapter import SeasonalWordAdapter
        
        worker_info = export_data.get('worker', {}) or {}
        incidents = export_data.get('incidents', [])
        
        # Build data structure expected by generate_worker_seasonal_word
        word_data = {
            'worker_identity': {
                'employee_id': worker_info.get('employee_id', employee_id),
                'full_name': worker_info.get('full_name', f'Worker {employee_id}'),
                'job_title': worker_info.get('job_title', ''),
                'department_id': worker_info.get('department_id', ''),
                'section_id': worker_info.get('section_id', ''),
            },
            'period': {
                'from': export_data.get('export_date', ''),
                'to': export_data.get('export_date', ''),
                'label': 'Export Report'
            },
            'metrics': {
                'total_incidents': worker_info.get('total_incidents', len(incidents)),
                'high_severity': sum(1 for i in incidents if i.get('Severity') == 'High'),
                'medium_severity': sum(1 for i in incidents if i.get('Severity') == 'Medium'),
                'low_severity': sum(1 for i in incidents if i.get('Severity') == 'Low'),
                'good_feedback_count': sum(1 for i in incidents if i.get('Classification') == 'Good'),
                'bad_feedback_count': sum(1 for i in incidents if i.get('Classification') == 'Bad'),
                'neutral_feedback_count': sum(1 for i in incidents if i.get('Classification') == 'Neutral'),
                'total_action_items': 0,
                'completed_action_items': 0,
                'overdue_action_items': 0,
                'explanation_accepted_count': 0,
                'explanation_rejected_count': 0,
            },
            'incidents_details': [
                {
                    'date': i.get('Date', ''),
                    'patient_name': i.get('PatientName', ''),
                    'category': i.get('Category', ''),
                    'severity': i.get('Severity', ''),
                    'classification': i.get('Classification', ''),
                    'status': i.get('Status', ''),
                    'id': i.get('RecordID', ''),
                }
                for i in incidents
            ],
            'performance': {
                'score': 0,
                'label': 'N/A',
            },
        }
        
        word_bytes = SeasonalWordAdapter.generate_worker_seasonal_word(word_data)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"worker_{employee_id}_report_{timestamp}.docx"
        
        return {
            "filename": filename,
            "content": word_bytes,
            "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        }
    
    except Exception as e:
        raise Exception(f"Failed to generate Word document: {str(e)}")
