"""
Monthly Report Service
Handles all business logic for monthly reports (detailed & numeric).
"""

from datetime import date, timedelta
from typing import Dict, Any, Optional, Literal

from ..db_layer.reports_db import get_filtered_complaints, get_monthly_statistics


class MonthlyReportService:
    """
    Service responsible for building monthly reports.
    This layer isolates report logic from routers and DB.
    """

    def generate_monthly_report(
        self,
        year: int,
        month: Optional[int],
        start_date: Optional[str],
        end_date: Optional[str],
        mode: Literal["detailed", "numeric"],
        scope: Optional[str],
        administration_ids: Optional[str],
        department_ids: Optional[str],
        section_ids: Optional[str],
    ) -> Dict[str, Any]:
        """
        Generate a monthly report (dispatcher method).
        
        This method acts as a unified entry point for monthly reports, converting
        router parameters into the internal filter format and routing to the
        appropriate report generation method.
        
        Args:
            year: Report year (required)
            month: Report month (1-12), required if start_date/end_date not provided
            start_date: Custom range start date (ISO format string)
            end_date: Custom range end date (ISO format string)
            mode: Report mode - "detailed" for paginated cases or "numeric" for aggregated stats
            scope: Organizational scope filter (optional)
            administration_ids: Comma-separated administration IDs (optional)
            department_ids: Comma-separated department IDs (optional)
            section_ids: Comma-separated section IDs (optional)
        
        Returns:
            Report dictionary structure (depends on mode)
        
        Raises:
            ValueError: If year is missing, mode is invalid, or date logic is inconsistent
        """
        # Validate required parameters
        if not year:
            raise ValueError("Year is required")
        
        if mode not in ["detailed", "numeric"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'detailed' or 'numeric'")
        
        # Validate date logic: either month OR (start_date AND end_date)
        if start_date or end_date:
            if not (start_date and end_date):
                raise ValueError("Both start_date and end_date must be provided for custom range")
        elif not month:
            raise ValueError("Either month or (start_date and end_date) must be provided")
        
        # Build filters dictionary
        filters: Dict[str, Any] = {
            "year": year
        }
        
        # Add month or date range
        if month:
            filters["month"] = month
        
        if start_date and end_date:
            # Convert string dates to date objects
            try:
                filters["start_date"] = date.fromisoformat(start_date)
                filters["end_date"] = date.fromisoformat(end_date)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid date format: {e}. Use ISO format (YYYY-MM-DD)")
        
        # Map organizational unit filters
        # Note: The existing service uses building_id, idara_id (administration),
        # dayra_id (department), qism_id (section)
        
        if administration_ids:
            # Parse first ID if comma-separated (simple approach)
            admin_id_list = [int(x.strip()) for x in administration_ids.split(",") if x.strip()]
            if admin_id_list:
                filters["idara_id"] = admin_id_list[0]
        
        if department_ids:
            dept_id_list = [int(x.strip()) for x in department_ids.split(",") if x.strip()]
            if dept_id_list:
                filters["dayra_id"] = dept_id_list[0]
        
        if section_ids:
            section_id_list = [int(x.strip()) for x in section_ids.split(",") if x.strip()]
            if section_id_list:
                filters["qism_id"] = section_id_list[0]
        
        # Handle scope parameter (if needed for future enhancements)
        if scope:
            filters["scope"] = scope
        
        # Route to appropriate method based on mode
        if mode == "detailed":
            return self.get_detailed_monthly_report(filters)
        else:  # mode == "numeric"
            return self.get_numeric_monthly_report(filters)

    def get_detailed_monthly_report(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the detailed monthly report from IncidentCase.
        
        This function retrieves paginated complaint data with optional filtering
        by organizational unit (building, idara, dayra, qism) and complaint attributes
        (domain, category, severity, status).
        
        Args:
            filters: Dictionary containing:
                - year: int (required)
                - month: int (required, 1-12)
                - start_date: Optional[date]
                - end_date: Optional[date]
                - building_id: Optional[int]
                - idara_id: Optional[int]
                - dayra_id: Optional[int]
                - qism_id: Optional[int]
                - domain_id: Optional[int]
                - category_id: Optional[int]
                - severity_id: Optional[int]
                - status: Optional[str]
                - page: int (default 1)
                - page_size: int (default 50)
        
        Returns:
            Dictionary containing:
                - complaints: List of complaint records
                - pagination: Pagination metadata
                - period: Period label and date range
        """
        # Extract parameters from filters
        year = filters.get("year")
        month = filters.get("month")
        start_date = filters.get("start_date")
        end_date = filters.get("end_date")
        building_id = filters.get("building_id")
        idara_id = filters.get("idara_id")
        dayra_id = filters.get("dayra_id")
        qism_id = filters.get("qism_id")
        domain_id = filters.get("domain_id")
        category_id = filters.get("category_id")
        severity_id = filters.get("severity_id")
        status = filters.get("status")
        page = filters.get("page", 1)
        page_size = filters.get("page_size", 50)
        
        # Calculate period dates
        if start_date and end_date:
            period_start = start_date
            period_end = end_date
            label = f"Custom Range {start_date} to {end_date}"
            label_ar = f"نطاق مخصص {start_date} إلى {end_date}"
        else:
            if not month or month < 1 or month > 12:
                raise ValueError("Month required and must be 1-12")
            
            # First day of month
            period_start = date(year, month, 1)
            # Last day of month
            if month == 12:
                period_end = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                period_end = date(year, month + 1, 1) - timedelta(days=1)
            
            months = {
                1: ("January", "يناير"),
                2: ("February", "فبراير"),
                3: ("March", "مارس"),
                4: ("April", "أبريل"),
                5: ("May", "مايو"),
                6: ("June", "يونيو"),
                7: ("July", "يوليو"),
                8: ("August", "أغسطس"),
                9: ("September", "سبتمبر"),
                10: ("October", "أكتوبر"),
                11: ("November", "نوفمبر"),
                12: ("December", "ديسمبر")
            }
            
            label = f"{months[month][0]} {year}"
            label_ar = f"{months[month][1]} {year}"
        
        # Fetch complaints
        complaints, total_records = get_filtered_complaints(
            year=year,
            month=month,
            start_date=period_start,
            end_date=period_end,
            building_id=building_id,
            idara_id=idara_id,
            dayra_id=dayra_id,
            qism_id=qism_id,
            domain_id=domain_id,
            category_id=category_id,
            severity_id=severity_id,
            status=status,
            page=page,
            page_size=page_size
        )
        
        total_pages = (total_records + page_size - 1) // page_size
        
        return {
            "complaints": complaints,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_records": total_records,
                "total_pages": total_pages
            },
            "period": {
                "label": label,
                "label_ar": label_ar,
                "start_date": period_start.isoformat(),
                "end_date": period_end.isoformat()
            }
        }


    def get_numeric_monthly_report(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the numeric monthly aggregated report.
        
        This function fetches aggregated monthly statistics grouped by domain, category,
        severity, and department. Used for high-level reporting and analytics.
        
        Args:
            filters: Dictionary containing:
                - year: int (required)
                - month: Optional[int] (1-12)
                - start_date: Optional[date]
                - end_date: Optional[date]
                - building_id: Optional[int]
                - idara_id: Optional[int]
                - dayra_id: Optional[int]
                - qism_id: Optional[int]
        
        Returns:
            Dictionary containing:
                - period: Period metadata and labels
                - summary: Overall statistics
                - by_domain: Grouped by complaint domain
                - by_category: Grouped by complaint category
                - by_severity: Grouped by severity level
                - by_department: Grouped by organizational department
        """
        # Extract parameters from filters
        year = filters.get("year")
        month = filters.get("month")
        start_date = filters.get("start_date")
        end_date = filters.get("end_date")
        building_id = filters.get("building_id")
        idara_id = filters.get("idara_id")
        dayra_id = filters.get("dayra_id")
        qism_id = filters.get("qism_id")
        
        # Calculate period dates
        if start_date and end_date:
            period_start = start_date
            period_end = end_date
            label = f"Custom Range {start_date} to {end_date}"
            label_ar = f"نطاق مخصص {start_date} إلى {end_date}"
        else:
            if not month or month < 1 or month > 12:
                raise ValueError("Month required and must be 1-12")
            
            # First day of month
            period_start = date(year, month, 1)
            # Last day of month
            if month == 12:
                period_end = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                period_end = date(year, month + 1, 1) - timedelta(days=1)
            
            months = {
                1: ("January", "يناير"),
                2: ("February", "فبراير"),
                3: ("March", "مارس"),
                4: ("April", "أبريل"),
                5: ("May", "مايو"),
                6: ("June", "يونيو"),
                7: ("July", "يوليو"),
                8: ("August", "أغسطس"),
                9: ("September", "سبتمبر"),
                10: ("October", "أكتوبر"),
                11: ("November", "نوفمبر"),
                12: ("December", "ديسمبر")
            }
            
            label = f"{months[month][0]} {year}"
            label_ar = f"{months[month][1]} {year}"
        
        # Fetch statistics
        stats = get_monthly_statistics(
            year=year,
            month=month,
            start_date=period_start,
            end_date=period_end,
            building_id=building_id,
            idara_id=idara_id,
            dayra_id=dayra_id,
            qism_id=qism_id
        )
        
        return {
            "period": {
                "year": year,
                "month": month,
                "label": label,
                "label_ar": label_ar,
                "start_date": period_start.isoformat(),
                "end_date": period_end.isoformat()
            },
            "summary": stats["summary"],
            "by_domain": stats["by_domain"],
            "by_category": stats["by_category"],
            "by_severity": stats["by_severity"],
            "by_department": stats["by_department"]
        }


# Singleton instance (same pattern as other services)
monthly_report_service = MonthlyReportService()
