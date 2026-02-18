"""  
Monthly Report Service
Handles all business logic for monthly reports (detailed & numeric).
"""

from datetime import date, timedelta
from typing import Dict, Any, Optional, Literal

from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_any_unit_in_scope
from ..db_layer.reports_db import get_filtered_complaints, get_monthly_statistics


class MonthlyReportService:
    """
    Service responsible for building monthly reports.
    This layer isolates report logic from routers and DB.
    """

    def generate_monthly_report(
        self,
        current_user: CurrentUser,
        year: int,
        month: Optional[int],
        start_date: Optional[str],
        end_date: Optional[str],
        mode: Literal["detailed", "numeric"],
        scope: Optional[str],
        administration_ids: Optional[str],
        department_ids: Optional[str],
        section_ids: Optional[str],
        page: int = 1,
        page_size: int = 50,
        group_by: str = "section"
    ) -> Dict[str, Any]:
        """
        Generate a monthly report (dispatcher method).
        
        Phase 2.5.7: Enforces organizational scope using current_user.allowed_unit_ids.
        Client-provided org unit IDs are validated but data is filtered by allowed scope.
        
        This method acts as a unified entry point for monthly reports, converting
        router parameters into the internal filter format and routing to the
        appropriate report generation method.
        
        Args:
            current_user: Authenticated user with allowed org units
            year: Report year (required)
            month: Report month (1-12), required if start_date/end_date not provided
            start_date: Custom range start date (ISO format string)
            end_date: Custom range end date (ISO format string)
            mode: Report mode - "detailed" for paginated cases or "numeric" for aggregated stats
            scope: Organizational scope filter (optional)
            administration_ids: Comma-separated administration IDs (optional, validated)
            department_ids: Comma-separated department IDs (optional, validated)
            section_ids: Comma-separated section IDs (optional, validated)
            page: Page number for pagination (default 1)
            page_size: Number of records per page (default 50, use 9999 for exports)
        
        Returns:
            Report dictionary structure (depends on mode)
        
        Raises:
            ValueError: If year is missing, mode is invalid, or date logic is inconsistent
            HTTPException: If user requests data outside their scope (403)
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
            "year": year,
            "page": page,
            "page_size": page_size
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
        
        # Phase 2.5.7: Validate any client-provided org unit IDs are in scope
        # Collect all requested unit IDs for validation
        requested_unit_ids = []
        
        if administration_ids:
            admin_id_list = [int(x.strip()) for x in administration_ids.split(",") if x.strip()]
            requested_unit_ids.extend(admin_id_list)
        
        if department_ids:
            dept_id_list = [int(x.strip()) for x in department_ids.split(",") if x.strip()]
            requested_unit_ids.extend(dept_id_list)
        
        if section_ids:
            section_id_list = [int(x.strip()) for x in section_ids.split(",") if x.strip()]
            requested_unit_ids.extend(section_id_list)
        
        # Validate: All requested units must be in user's scope
        if requested_unit_ids:
            require_any_unit_in_scope(current_user, requested_unit_ids)
        
        # Security boundary: always use the full user scope for IssuingOrgUnitID filter
        filters["allowed_unit_ids"] = list(current_user.allowed_unit_ids)
        
        # Target department filter: when specific sections/depts/admins are requested,
        # filter by TARGET departments (APP_IncidentCaseTargetDepartment) — this is the
        # dimension that determines which section a complaint is ABOUT, not who filed it.
        # Expand requested IDs to include all descendants (e.g., admin → all its sections)
        if requested_unit_ids:
            from ..db_layer.reports_db import debug_expand_org_units
            expanded_target_ids = debug_expand_org_units(requested_unit_ids)
            filters["target_unit_ids"] = expanded_target_ids
        
        # Handle scope parameter (if needed for future enhancements)
        if scope:
            filters["scope"] = scope
        
        # group_by: Controls the aggregation level for by_department breakdown
        # Valid values: "section" (default), "department", "administration"
        filters["group_by"] = group_by if group_by in ("section", "department", "administration") else "section"
        
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
                - page_size: int (default 50, use 9999 for exports to get all records)
        
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
        allowed_unit_ids = filters.get("allowed_unit_ids", [])
        domain_id = filters.get("domain_id")
        category_id = filters.get("category_id")
        severity_id = filters.get("severity_id")
        status = filters.get("status")
        page = filters.get("page", 1)
        page_size = filters.get("page_size", 50)  # Default 50 for UI, exports can override with 9999
        
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
        target_unit_ids = filters.get("target_unit_ids")
        complaints, total_records = get_filtered_complaints(
            year=year,
            month=month,
            start_date=period_start,
            end_date=period_end,
            allowed_unit_ids=allowed_unit_ids,
            target_unit_ids=target_unit_ids,
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
        allowed_unit_ids = filters.get("allowed_unit_ids", [])
        
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
        
        # group_by controls the aggregation level for by_department
        group_by = filters.get("group_by", "section")
        
        # Fetch statistics
        target_unit_ids = filters.get("target_unit_ids")
        stats = get_monthly_statistics(
            year=year,
            month=month,
            start_date=period_start,
            end_date=period_end,
            allowed_unit_ids=allowed_unit_ids,
            target_unit_ids=target_unit_ids,
            group_by=group_by
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
