"""
Reports Service Layer
Handles business logic, data aggregation, filtering, and export preparation.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO, StringIO
import csv
import json
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.table import WD_ROW_HEIGHT_RULE


try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor, Cm, Mm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.section import WD_ORIENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    PYTHON_DOCX_AVAILABLE = True
except ImportError:
    PYTHON_DOCX_AVAILABLE = False

from ..db_layer.reports_db import (
    get_filtered_complaints,
    get_monthly_statistics,
    get_seasonal_hcat,
    get_bulk_summary
)


class ReportsService:
    """Service class for report operations."""
    
    @staticmethod
    def get_period_dates(
        report_type: str,
        year: int,
        month: Optional[int] = None,
        trimester: Optional[int] = None,
        quarter: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Tuple[date, date, str, str]:
        """
        Calculate period start/end dates and labels.
        
        Returns:
            Tuple of (start_date, end_date, label, label_ar)
        """
        if start_date and end_date:
            label = f"Custom Range {start_date} to {end_date}"
            label_ar = f"نطاق مخصص {start_date} إلى {end_date}"
            return start_date, end_date, label, label_ar
        
        if report_type == "monthly":
            if not month or month < 1 or month > 12:
                raise ValueError("Month required and must be 1-12")
            
            # First day of month
            start = date(year, month, 1)
            # Last day of month
            if month == 12:
                end = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                end = date(year, month + 1, 1) - timedelta(days=1)
            
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
            
            en_label = f"{months[month][0]} {year}"
            ar_label = f"{months[month][1]} {year}"
            return start, end, en_label, ar_label
        
        elif report_type == "seasonal":
            if trimester:
                # Trimester: 4-month periods
                if trimester == 1:
                    start = date(year, 1, 1)
                    end = date(year, 4, 30)
                    label = f"Trimester 1 - {year}"
                    label_ar = f"الفصل الأول - {year}"
                elif trimester == 2:
                    start = date(year, 5, 1)
                    end = date(year, 8, 31)
                    label = f"Trimester 2 - {year}"
                    label_ar = f"الفصل الثاني - {year}"
                elif trimester == 3:
                    start = date(year, 9, 1)
                    end = date(year, 12, 31)
                    label = f"Trimester 3 - {year}"
                    label_ar = f"الفصل الثالث - {year}"
                else:
                    raise ValueError("Trimester must be 1, 2, or 3")
                return start, end, label, label_ar
            
            elif quarter:
                # Quarter: 3-month periods
                if quarter == 1:
                    start = date(year, 1, 1)
                    end = date(year, 3, 31)
                    label = f"Q1 - {year}"
                    label_ar = f"الربع الأول - {year}"
                elif quarter == 2:
                    start = date(year, 4, 1)
                    end = date(year, 6, 30)
                    label = f"Q2 - {year}"
                    label_ar = f"الربع الثاني - {year}"
                elif quarter == 3:
                    start = date(year, 7, 1)
                    end = date(year, 9, 30)
                    label = f"Q3 - {year}"
                    label_ar = f"الربع الثالث - {year}"
                elif quarter == 4:
                    start = date(year, 10, 1)
                    end = date(year, 12, 31)
                    label = f"Q4 - {year}"
                    label_ar = f"الربع الرابع - {year}"
                else:
                    raise ValueError("Quarter must be 1, 2, 3, or 4")
                return start, end, label, label_ar
            else:
                raise ValueError("Trimester or Quarter required for seasonal report")
        
        else:
            raise ValueError("report_type must be 'monthly' or 'seasonal'")
    
    @staticmethod
    def get_filtered_complaints_with_pagination(
        report_type: str,
        year: int,
        month: Optional[int] = None,
        trimester: Optional[int] = None,
        quarter: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        building_id: Optional[int] = None,
        idara_id: Optional[int] = None,
        dayra_id: Optional[int] = None,
        qism_id: Optional[int] = None,
        domain_id: Optional[int] = None,
        category_id: Optional[int] = None,
        severity_id: Optional[int] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Dict[str, Any]:
        """Fetch complaints with pagination for detailed monthly view."""
        
        # Get period dates
        period_start, period_end, label, label_ar = ReportsService.get_period_dates(
            report_type, year, month, trimester, quarter, start_date, end_date
        )
        
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
    
    @staticmethod
    def get_monthly_statistics_report(
        year: int,
        month: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        building_id: Optional[int] = None,
        idara_id: Optional[int] = None,
        dayra_id: Optional[int] = None,
        qism_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Fetch aggregated monthly statistics."""
        
        # Get period dates
        period_start, period_end, label, label_ar = ReportsService.get_period_dates(
            "monthly", year, month, None, None, start_date, end_date
        )
        
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
    
    @staticmethod
    def get_seasonal_hcat_report(
        year: int,
        trimester: Optional[int] = None,
        quarter: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        threshold: Optional[int] = None,
        building_id: Optional[int] = None,
        idara_id: Optional[int] = None,
        dayra_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Fetch seasonal HCAT analysis with threshold evaluation."""
        
        # Get period dates
        period_start, period_end, label, label_ar = ReportsService.get_period_dates(
            "seasonal", year, None, trimester, quarter, start_date, end_date
        )
        
        # Use default threshold if not provided
        if threshold is None:
            threshold = 50
        
        # Fetch HCAT analysis
        hcat_data = get_seasonal_hcat(
            year=year,
            start_date=period_start,
            end_date=period_end,
            threshold=threshold,
            building_id=building_id,
            idara_id=idara_id,
            dayra_id=dayra_id
        )
        
        return {
            "period": {
                "year": year,
                "trimester": trimester,
                "quarter": quarter,
                "label": label,
                "label_ar": label_ar,
                "start_date": period_start.isoformat(),
                "end_date": period_end.isoformat()
            },
            "threshold": {
                "value": threshold,
                "source": "user_input"
            },
            "total_complaints": hcat_data["total_complaints"],
            "domains": hcat_data["domains"],
            "exceeding_count": hcat_data["exceeding_count"],
            "within_threshold_count": hcat_data["within_threshold_count"]
        }
    
    @staticmethod
    def get_bulk_summary_report(
        report_type: str,
        year: int,
        month: Optional[int] = None,
        trimester: Optional[int] = None,
        quarter: Optional[int] = None,
        building_id: Optional[int] = None,
        idara_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Fetch bulk export summary per department."""
        
        # Get period dates
        period_start, period_end, label, label_ar = ReportsService.get_period_dates(
            report_type, year, month, trimester, quarter
        )
        
        # Fetch summaries
        departments = get_bulk_summary(
            year=year,
            month=month,
            start_date=period_start,
            end_date=period_end,
            building_id=building_id,
            idara_id=idara_id
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
            "departments": departments,
            "total_departments": len(departments),
            "grand_total_complaints": sum(d["total_complaints"] for d in departments)
        }
    
    @staticmethod
    def generate_csv_export(
        report_data: List[Dict[str, Any]],
        filename: str,
        language: str = "en"
    ) -> bytes:
        """Generate CSV from report data."""
        
        csv_buffer = StringIO()
        
        if not report_data:
            return b""
        
        # Get fieldnames from first record
        fieldnames = list(report_data[0].keys())
        
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in report_data:
            writer.writerow(row)
        
        return csv_buffer.getvalue().encode('utf-8')
    
    @staticmethod
    def generate_xlsx_export(
        report_data: List[Dict[str, Any]],
        filename: str,
        language: str = "en"
    ) -> bytes:
        """
        Generate professional Excel XLSX file from report data using openpyxl.
        Formatted for official hospital monthly reports with RTL support.
        
        Args:
            report_data: List of dictionaries (rows) - FULL complaint DTO
            filename: Target filename
            language: Language code (en or ar)
        
        Returns:
            bytes: Real Excel .xlsx file content
        """
        if not OPENPYXL_AVAILABLE:
            raise ImportError(
                "openpyxl is required for Excel export. "
                "Install with: pip install openpyxl"
            )
        
        from openpyxl.styles import Alignment, Border, Side
        
        if not report_data:
            # Return empty workbook
            wb = Workbook()
            ws = wb.active
            ws.title = "Report"
            excel_buffer = BytesIO()
            wb.save(excel_buffer)
            excel_buffer.seek(0)
            return excel_buffer.getvalue()
        
        # Create workbook and active sheet
        wb = Workbook()
        ws = wb.active
        ws.title = "تقرير فرص التحسين الشهري"  # Official sheet name
        
        # Set RTL direction
        ws.sheet_view.rightToLeft = True
        
        # Define official columns (RTL order - right to left)
        # Format: (header, field, width, is_vertical)
        columns = [
            ("تاريخ تلقي الملاحظة", "received_date", 4, True),           # 1 - Vertical
            ("الرقم", "id", 4, True),                                     # 2 - Vertical
            ("P. Full Name", "patient_name", 6, True),                    # 3 - Vertical
            ("قسم الصادر", "section_name", 6, True),            # 4 - Section (actual issuing unit)
            ("الإدارة", "administration_name", 5, True),                # 5 - Administration (top level)
            ("القسم المعني", "department_name", 6, True),           # 6 - Department (middle level)
            ("المصدر", "source_name", 5, True),                           # 7 - Vertical
            ("النوع", "feedback_intent_type_name", 5, True),              # 8 - Vertical
            ("Domain", "domain_name", 6, True),                           # 9 - Vertical
            ("Category", "category_name", 6, True),                       # 10 - Vertical
            ("Sub-Category", "subcategory_name", 6, True),                # 11 - Vertical
            ("Target Departments", "target_departments_display", 8, True),    # 12 - Vertical
            ("classification in Arabic", "classification_name", 7, True), # 13 - Vertical
            ("classification in English", "classification_name", 7, True),# 14 - Vertical
            ("محتوى الشكوى (Raw Content)", "complaint_text", 60, False), # 14 - VERY WIDE (8x)
            ("Immediate Action (خدمات المرضى+القسم)", "immediate_action", 35, False), # 15 - WIDE (4x)
            ("الإجراءات المتخذة (القسم/الدائرة/الإدارة)", "taken_action", 28, False), # 16 - WIDE (3x)
            ("Severity", "severity_name", 5, True),                       # 17 - Vertical
            ("Stage", "stage_name", 5, True),                             # 18 - Vertical
            ("Harm", "harm_level", 4, True)                               # 19 - Vertical
        ]
        
        # Extract metadata from data
        start_date = "—"
        end_date = "—"
        idara_name = "—"
        dayra_name = "—"
        qism_name = "—"
        
        if report_data:
            first_record = report_data[0]
            last_record = report_data[-1] if len(report_data) > 1 else first_record
            
            # Extract dates
            if first_record.get("received_date"):
                start_date = first_record["received_date"]
                if isinstance(start_date, (datetime, date)):
                    start_date = start_date.strftime("%Y-%m-%d") if isinstance(start_date, datetime) else start_date.isoformat()
            
            if last_record.get("received_date"):
                end_date = last_record["received_date"]
                if isinstance(end_date, (datetime, date)):
                    end_date = end_date.strftime("%Y-%m-%d") if isinstance(end_date, datetime) else end_date.isoformat()
            
            # Extract department info
            idara_name = first_record.get("administration_name", "—")
            dayra_name = first_record.get("department_name", "—")
            qism_name = first_record.get("section_name", "—")
        
        # Define styles
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # Light green hospital color for header
        header_fill = PatternFill(start_color="B4E7CE", end_color="B4E7CE", fill_type="solid")
        header_font = Font(bold=True, color="000000", size=9)
        title_font = Font(bold=True, size=12)
        subtitle_font = Font(bold=True, size=11)
        
        current_row = 1
        num_columns = len(columns)
        
        # Row 1: Date range header
        date_text = f"الشهر المعني: من {start_date} إلى {end_date}"
        ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=num_columns)
        cell = ws.cell(row=current_row, column=1, value=date_text)
        cell.font = title_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        current_row += 1
        
        # Row 2: Department info header
        dept_text = f"الإدارة: {idara_name}      الدائرة: {dayra_name}      القسم المعني: {qism_name}"
        ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=num_columns)
        cell = ws.cell(row=current_row, column=1, value=dept_text)
        cell.font = subtitle_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        current_row += 1
        
        # Empty row
        current_row += 1
        
        # Row for column headers
        header_row = current_row
        for col_idx, (header_name, _, _, is_vertical) in enumerate(columns, start=1):
            cell = ws.cell(row=header_row, column=col_idx, value=header_name)
            cell.fill = header_fill
            cell.font = header_font
            cell.border = thin_border
            
            if is_vertical:
                # Vertical text: rotate 90 degrees, center align
                cell.alignment = Alignment(
                    horizontal="center", 
                    vertical="center", 
                    wrap_text=True,
                    text_rotation=90
                )
            else:
                # Horizontal text: center align, wrap
                cell.alignment = Alignment(
                    horizontal="center", 
                    vertical="center", 
                    wrap_text=True
                )
        
        # Set row height for header to accommodate vertical text
        ws.row_dimensions[header_row].height = 100
        
        current_row += 1
        
        # Data rows
        for row_data in report_data:
            for col_idx, (header_name, field_name, _, is_vertical) in enumerate(columns, start=1):
                # Special handling for target departments: concatenate all department names
                if field_name == "target_departments_display":
                    target_depts = row_data.get("target_departments", [])
                    if target_depts and isinstance(target_depts, list):
                        dept_names = [dept.get("department_name", "") for dept in target_depts if dept.get("department_name")]
                        value = ", ".join(dept_names) if dept_names else "—"
                    else:
                        value = "—"
                else:
                    # Get value from data
                    value = row_data.get(field_name, "")
                
                # Convert date/datetime to string
                if isinstance(value, datetime):
                    value = value.strftime("%Y-%m-%d")
                elif isinstance(value, date):
                    value = value.isoformat()
                
                # Handle None
                if value is None:
                    value = ""
                
                # Write cell
                cell = ws.cell(row=current_row, column=col_idx, value=value)
                cell.border = thin_border
                
                # Apply alignment based on column type
                if is_vertical:
                    # Vertical columns: centered
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                else:
                    # Wide columns: wrapped, top-aligned
                    cell.alignment = Alignment(wrap_text=True, vertical="top", horizontal="right")
            
            current_row += 1
        
        # Set column widths USING INDEX (not column_letter from cells to avoid MergedCell bug)
        from openpyxl.utils import get_column_letter
        for col_idx, (_, _, width, _) in enumerate(columns, start=1):
            col_letter = get_column_letter(col_idx)
            ws.column_dimensions[col_letter].width = width
        
        # Add footer signature block
        # Leave 2 empty rows
        current_row += 2
        
        # Signature block structure (4 rows x 4 columns spanning all columns)
        sig_start_row = current_row
        
        # Row 1: إسم مسؤول العملية
        ws.cell(row=current_row, column=1, value="إسم مسؤول العملية:..........")
        ws.cell(row=current_row, column=5, value="التوقيع:..............................")
        ws.cell(row=current_row, column=10, value="التاريخ:.....................")
        ws.cell(row=current_row, column=14, value="خاص خدمات المرضى-")
        
        # Row 2: إسم رئيس الدائرة
        current_row += 1
        ws.cell(row=current_row, column=1, value="إسم رئيس الدائرة:.............")
        ws.cell(row=current_row, column=5, value="التوقيع:..............................")
        ws.cell(row=current_row, column=10, value="التاريخ:.....................")
        ws.cell(row=current_row, column=14, value="الإسم: ....................")
        
        # Row 3: تاريخ الإستلام
        current_row += 1
        ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=num_columns)
        ws.cell(row=current_row, column=1, value="تاريخ الإستلام: ....................")
        
        # Row 4: إسم مدير الإدارة
        current_row += 1
        ws.cell(row=current_row, column=1, value="إسم مدير الإدارة:...............")
        ws.cell(row=current_row, column=5, value="التوقيع:..............................")
        ws.cell(row=current_row, column=10, value="التاريخ:.....................")
        ws.cell(row=current_row, column=14, value="التوقيع: ....................")
        
        # Apply borders and alignment to signature block
        for row_idx in range(sig_start_row, current_row + 1):
            for col_idx in range(1, num_columns + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                if cell.value:  # Only style cells with content
                    cell.border = thin_border
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                    cell.font = Font(size=10)
        
        # Add footer quote
        current_row += 2
        ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=num_columns)
        quote_cell = ws.cell(row=current_row, column=1, 
                            value='"نؤمن أن الإبتكار لا يكون فقط في التقنيات، بل في أسلوب الخدمة والتواصل والتعاطف… فلنبتكر معًا تجربة ذات أثر طيب"')
        quote_cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        quote_cell.font = Font(italic=True, size=11)
        
        # Page setup for printing
        ws.page_setup.orientation = ws.ORIENTATION_LANDSCAPE
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0  # Auto height
        ws.print_options.gridLines = True
        
        # Freeze panes at header row
        ws.freeze_panes = ws.cell(row=header_row + 1, column=1)
        
        # Save to BytesIO
        excel_buffer = BytesIO()
        wb.save(excel_buffer)
        excel_buffer.seek(0)
        
        return excel_buffer.getvalue()
    
    @staticmethod
    def generate_pdf_export(
        report_data: Dict[str, Any],
        filename: str,
        language: str = "en",
        include_charts: bool = True
    ) -> bytes:
        """
        Generate professional PDF matching Word export exactly.
        A4 landscape with RTL layout, vertical headers, signature block.
        
        Args:
            report_data: List of dictionaries OR dict with "complaints" key
            filename: Target filename
            language: Language code (en or ar)
            include_charts: Not used (kept for compatibility)
        
        Returns:
            bytes: Valid PDF file content
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "reportlab is required for PDF export. "
                "Install with: pip install reportlab"
            )
        
        def sanitize_value(value):
            """Convert value to string, handling dates and None"""
            try:
                if value is None:
                    return ""
                if isinstance(value, (datetime, date)):
                    if isinstance(value, datetime):
                        return value.strftime("%Y-%m-%d")
                    return value.isoformat()
                return str(value)
            except:
                return ""
        
        # Normalize data source
        try:
            if isinstance(report_data, dict) and "complaints" in report_data:
                rows = report_data["complaints"]
            elif isinstance(report_data, list):
                rows = report_data
            else:
                rows = []
            
            if not isinstance(rows, list):
                rows = []
        except:
            rows = []
        
        # Create PDF buffer
        pdf_buffer = BytesIO()
        
        # A4 Landscape
        from reportlab.lib.pagesizes import landscape, A4 as portrait_a4
        
        # Create PDF document
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=landscape(portrait_a4),
            rightMargin=0.4*inch,
            leftMargin=0.4*inch,
            topMargin=0.6*inch,
            bottomMargin=0.5*inch
        )
        
        # Container for PDF elements
        elements = []
        
        # Styles
        styles = getSampleStyleSheet()
        
        # Define columns with behavior (vertical vs horizontal)
        # Format: (header, field, width_ratio, is_vertical)
        columns = [
            ("تاريخ تلقي الملاحظة", "received_date", 0.4, True),          # 1 - Vertical
            ("الرقم", "id", 0.3, True),                                    # 2 - Vertical
            ("P. Full Name", "patient_name", 0.5, True),                   # 3 - Vertical
            ("قسم الصادر", "section_name", 0.5, True),           # 4 - Section (actual issuing unit)
            ("الإدارة", "administration_name", 0.4, True),               # 5 - Administration (top level)
            ("القسم المعني", "department_name", 0.5, True),          # 6 - Department (middle level)
            ("المصدر", "source_name", 0.4, True),                          # 7 - Vertical
            ("النوع", "feedback_intent_type_name", 0.4, True),             # 8 - Vertical
            ("Domain", "domain_name", 0.5, True),                          # 9 - Vertical
            ("Category", "category_name", 0.5, True),                      # 10 - Vertical
            ("Sub-Category", "subcategory_name", 0.5, True),               # 11 - Vertical
            ("Target Departments", "target_departments_display", 1.0, True),  # 12 - Vertical
            ("classification in Arabic", "classification_name", 0.6, True), # 13 - Vertical
            ("classification in English", "classification_name", 0.6, True),# 14 - Vertical
            ("محتوى الشكوى", "complaint_text", 8.0, False),                # 14 - Horizontal WIDE
            ("Immediate Action", "immediate_action", 4.0, False),          # 15 - Horizontal WIDE
            ("الإجراءات المتخذة", "taken_action", 3.0, False),             # 16 - Horizontal WIDE
            ("Severity", "severity_name", 0.4, True),                      # 17 - Vertical
            ("Stage", "stage_name", 0.4, True),                            # 18 - Vertical
            ("Harm", "harm_level", 0.3, True)                              # 19 - Vertical
        ]
        
        # Handle empty data
        if not rows:
            title_style = ParagraphStyle(
                'Title',
                parent=styles['Heading1'],
                fontSize=16,
                textColor=colors.HexColor('#366092'),
                alignment=TA_CENTER
            )
            title = Paragraph("تقرير شهري - Monthly Report", title_style)
            elements.append(title)
            elements.append(Spacer(1, 0.3*inch))
            elements.append(Paragraph("No data available", styles['Normal']))
            doc.build(elements)
            pdf_buffer.seek(0)
            return pdf_buffer.getvalue()
        
        # Extract metadata
        try:
            first_record = rows[0]
            last_record = rows[-1] if len(rows) > 1 else first_record
            
            start_date = sanitize_value(first_record.get("received_date", "—"))
            end_date = sanitize_value(last_record.get("received_date", "—"))
            idara_name = first_record.get("administration_name", "—")
            dayra_name = first_record.get("department_name", "—")
            qism_name = first_record.get("section_name", "—")
        except:
            start_date = "—"
            end_date = "—"
            idara_name = "—"
            dayra_name = "—"
            qism_name = "—"
        
        # Header block (colored background)
        header_style = ParagraphStyle(
            'Header',
            parent=styles['Normal'],
            fontSize=11,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            textColor=colors.black
        )
        
        # Date range header
        date_header = Paragraph(
            f"الشهر المعني: من {start_date} إلى {end_date}",
            header_style
        )
        elements.append(date_header)
        elements.append(Spacer(1, 0.05*inch))
        
        # Department info header
        dept_header = Paragraph(
            f"الإدارة: {idara_name}      الدائرة: {dayra_name}      القسم المعني: {qism_name}",
            header_style
        )
        elements.append(dept_header)
        elements.append(Spacer(1, 0.15*inch))
        
        # Build table data
        # Header row - for vertical columns, we'll use short text (rotation not natively supported in reportlab Table)
        # We'll use abbreviated headers or split text
        table_data = []
        
        # Create header row
        header_row = []
        for header_name, _, _, is_vertical in columns:
            if is_vertical:
                # For vertical columns, use short or split text
                header_row.append(header_name)
            else:
                # For horizontal columns, use full text
                header_row.append(header_name)
        table_data.append(header_row)
        
        # Data rows (limit to 30 for PDF)
        row_count = min(len(rows), 30)
        for row_dict in rows[:row_count]:
            row = []
            for header_name, field_name, _, is_vertical in columns:
                # Special handling for target departments: concatenate all department names
                if field_name == "target_departments_display":
                    target_depts = row_dict.get("target_departments", [])
                    if target_depts and isinstance(target_depts, list):
                        dept_names = [dept.get("department_name", "") for dept in target_depts if dept.get("department_name")]
                        value = ", ".join(dept_names) if dept_names else "—"
                    else:
                        value = "—"
                else:
                    value = sanitize_value(row_dict.get(field_name, ""))
                
                # Truncate based on column type
                if not is_vertical:  # Horizontal wide columns
                    if len(value) > 300:
                        value = value[:300] + "..."
                else:  # Vertical narrow columns
                    if len(value) > 40:
                        value = value[:40] + "..."
                
                row.append(value)
            table_data.append(row)
        
        # Note if data was truncated
        if len(rows) > 30:
            note_style = ParagraphStyle(
                'Note',
                parent=styles['Normal'],
                fontSize=8,
                alignment=TA_CENTER,
                textColor=colors.grey
            )
            note = Paragraph(
                f"Showing first 30 of {len(rows)} records. Download Excel or Word for full data.",
                note_style
            )
            elements.append(note)
            elements.append(Spacer(1, 0.05*inch))
        
        # Calculate column widths
        available_width = doc.width
        total_width_units = sum(col[2] for col in columns)
        col_widths = [(col[2] / total_width_units) * available_width for col in columns]
        
        # Create table
        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        
        # Build style commands
        style_commands = [
            # Header styling - light turquoise/green background
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#B4E7CE')),  # Light green
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 6),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
            ('TOPPADDING', (0, 0), (-1, 0), 4),
            
            # Data styling - vertical columns (narrow, centered)
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 2),
            ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            
            # Word wrap for all
            ('WORDWRAP', (0, 0), (-1, -1), True),
        ]
        
        # Apply specific alignment for columns
        for col_idx, (_, _, _, is_vertical) in enumerate(columns):
            if is_vertical:
                # Vertical columns: center align
                style_commands.append(('ALIGN', (col_idx, 1), (col_idx, -1), 'CENTER'))
            else:
                # Horizontal columns: left align, top valign
                style_commands.append(('ALIGN', (col_idx, 1), (col_idx, -1), 'LEFT'))
                style_commands.append(('VALIGN', (col_idx, 1), (col_idx, -1), 'TOP'))
        
        table.setStyle(TableStyle(style_commands))
        
        elements.append(table)
        
        # Add signature block
        elements.append(Spacer(1, 0.3*inch))
        
        # Signature table (4 rows x 4 columns)
        sig_data = [
            ["إسم مسؤول العملية:..........", "التوقيع:............................", "التاريخ:.....................", "خاص خدمات المرضى-"],
            ["إسم رئيس الدائرة:............", "التوقيع:............................", "التاريخ:.....................", "الإسم: ...................."],
            ["تاريخ الإستلام: ....................", "", "", ""],
            ["إسم مدير الإدارة:..............", "التوقيع:............................", "التاريخ:.....................", "التوقيع: ...................."]
        ]
        
        sig_table = Table(sig_data, colWidths=[doc.width * 0.25] * 4)
        sig_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            # Merge cells in row 3 (index 2)
            ('SPAN', (0, 2), (3, 2)),
        ]))
        
        elements.append(sig_table)
        
        # Add footer quote
        elements.append(Spacer(1, 0.2*inch))
        
        quote_style = ParagraphStyle(
            'Quote',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            fontName='Helvetica-Oblique',
            textColor=colors.HexColor('#555555')
        )
        
        quote = Paragraph(
            '"نؤمن أن الإبتكار لا يكون فقط في التقنيات، بل في أسلوب الخدمة والتواصل والتعاطف… فلنبتكر معًا تجربة ذات أثر طيب"',
            quote_style
        )
        elements.append(quote)
        
        # Build PDF
        try:
            doc.build(elements)
        except Exception as e:
            # Fallback: simple version if complex layout fails
            elements = []
            elements.append(Paragraph("Monthly Report - تقرير شهري", styles['Title']))
            elements.append(Paragraph(f"Error generating full report: {str(e)}", styles['Normal']))
            doc.build(elements)
        
        # Get PDF bytes
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
    
    @staticmethod
    def generate_docx_export(
        report_data: Dict[str, Any],
        filename: str,
        language: str = "en"
    ) -> bytes:
        """
        Generate official hospital audit form Word document.
        A4 Landscape with RTL layout, vertical headers, signature block.
        
        Args:
            report_data: List of dictionaries OR dict with "complaints" key
            filename: Target filename
            language: Language code (en or ar)
        
        Returns:
            bytes: Valid Word .docx file content
        """
        if not PYTHON_DOCX_AVAILABLE:
            raise ImportError(
                "python-docx is required for Word export. "
                "Install with: pip install python-docx"
            )
        
        def _safe(v):
            """Convert dimension values to int (python-docx requirement)"""
            return int(v)
        
        def sanitize_value(value):
            """Convert value to string, handling dates and None"""
            try:
                if value is None:
                    return ""
                if isinstance(value, (datetime, date)):
                    if isinstance(value, datetime):
                        return value.strftime("%Y-%m-%d")
                    return value.isoformat()
                return str(value)
            except:
                return ""
        
        def normalize_text(text: str) -> str:
            """Normalize text for Word table cells - remove manual line breaks"""
            text = str(text)
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            return " ".join(lines)
        
        # Normalize data source
        try:
            if isinstance(report_data, dict) and "complaints" in report_data:
                rows = report_data["complaints"]
            elif isinstance(report_data, list):
                rows = report_data
            else:
                rows = []
            
            if not isinstance(rows, list):
                rows = []
        except:
            rows = []
        
        # Create Word document
        doc = Document()
        
        # Set page to A4 Landscape
        section = doc.sections[0]
        section.page_height = _safe(Mm(210))  # A4 width becomes height in landscape
        section.page_width = _safe(Mm(297))   # A4 height becomes width in landscape
        section.orientation = WD_ORIENT.LANDSCAPE
        section.left_margin = _safe(Mm(15))
        section.right_margin = _safe(Mm(15))
        section.top_margin = _safe(Mm(15))
        section.bottom_margin = _safe(Mm(15))
        
        # Set document RTL
        try:
            section.start_type = 2  # New page
        except:
            pass
        
        # Extract metadata
        start_date = "—"
        end_date = "—"
        Administration = "—"
        Department = "—"
        Section = "—"
        


        if rows:
            try:
                first_record = rows[0]
                last_record = rows[-1] if len(rows) > 1 else first_record
                start_date = sanitize_value(first_record.get("received_date", "—"))
                end_date = sanitize_value(last_record.get("received_date", "—"))
                Administration = first_record.get("administration_name", "—")
                Department = first_record.get("department_name", "—")
                Section = first_record.get("section_name", "—")
            except:
                pass
        
        # ========== ADD LOGO TO WORD HEADER (TOP RIGHT) ==========

        try:
            import os
            logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'logo.png')
            if os.path.exists(logo_path):
                section = doc.sections[0]

                # Make header compact
                section.header_distance = Inches(0.1)

                header = section.header

                # Use only one paragraph and clear it
                header_para = header.paragraphs[0]
                header_para.clear()
                header_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT

                run = header_para.add_run()
                run.add_picture(logo_path, width=Inches(0.9))

        except:
            pass
        
        # ========== DOCUMENT HEADER TEXT (BODY) ==========
        
        # Title (big, bold, centered)
        # Title (big, bold, centered)
        title_para = doc.add_paragraph()
        title_run = title_para.add_run("نموذج التقرير الشهري لفرص التحسين والإجراءات التصحيحية الواردة من المرضى وذويهم")
        title_run.font.size = int(Pt(21))
        title_run.font.bold = True
        title_run.font.name = 'Traditional Arabic'
        title_run.italic = False
        title_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_para.space_after = int(Pt(3))  # Space after title
        
        # Subtitle (smaller, centered)
        subtitle_para = doc.add_paragraph()
        subtitle_run = subtitle_para.add_run("(إصدار رسمي — للاستخدام الإداري والجودة)")
        subtitle_run.font.size = int(Pt(14))
        subtitle_run.font.name = 'Traditional Arabic'
        subtitle_run.italic = False
        subtitle_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        subtitle_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle_para.space_after = int(Pt(6))  # More space after subtitle
        
        # Period line (centered, bold)
        period_para = doc.add_paragraph()
        period_run = period_para.add_run(f"الشهر المعني: من {start_date} إلى {end_date}")
        period_run.font.size = int(Pt(12))
        period_run.font.bold = True
        period_run.font.name = 'Traditional Arabic'
        period_run.italic = False
        period_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        period_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        period_para.space_after = int(Pt(3))
        
        # Department line (3-column table for proper RTL alignment)
        dept_table = doc.add_table(rows=1, cols=3)
        
        # Remove all table borders
        dept_tbl = dept_table._element
        dept_tblPr = dept_tbl.tblPr
        if dept_tblPr is None:
            dept_tblPr = OxmlElement('w:tblPr')
            dept_tbl.insert(0, dept_tblPr)
        
        # Remove borders
        dept_tblBorders = OxmlElement('w:tblBorders')
        for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
            border_elem = OxmlElement(f'w:{border_name}')
            border_elem.set(qn('w:val'), 'nil')
            dept_tblBorders.append(border_elem)
        dept_tblPr.append(dept_tblBorders)
        
        # Center the table horizontally
        dept_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        dept_tblJc = OxmlElement('w:jc')
        dept_tblJc.set(qn('w:val'), 'center')
        dept_tblPr.append(dept_tblJc)
        
        # Set table width to 70% of usable page width
        section = doc.sections[0]
        usable_width = section.page_width - section.left_margin - section.right_margin
        target_width = int(usable_width * 0.7)
        col_width = int(target_width / 3)
        
        for i in range(3):
            dept_table.columns[i].width = col_width
        
        # Fill the cells with department data (bold labels + normal values)
        dept_cells = dept_table.rows[0].cells
        dept_data = [
            ("الإدارة: ", Administration),
            ("الدائرة: ", Department), 
            ("القسم المعني: ", Section)
        ]
        
        for i, (label, value) in enumerate(dept_data):
            cell = dept_cells[i]
            
            # Clear existing content and create custom paragraph
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.right_to_left = True
            paragraph.space_after = int(Pt(6))
            
            # Add bold label run
            label_run = paragraph.add_run(label)
            label_run.font.bold = True
            label_run.font.size = int(Pt(15))
            label_run.font.name = 'Traditional Arabic'
            label_run.italic = False
            label_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
            
            # Add normal value run
            value_run = paragraph.add_run(str(value))
            value_run.font.bold = False
            value_run.font.size = int(Pt(15))
            value_run.font.name = 'Traditional Arabic'
            value_run.italic = False
            value_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        
        # Visual separator line (bottom border)
        separator_para = doc.add_paragraph()
        separator_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        # Add bottom border to paragraph
        try:
            pPr = separator_para._element.get_or_add_pPr()
            pBdr = OxmlElement('w:pBdr')
            bottom = OxmlElement('w:bottom')
            bottom.set(qn('w:val'), 'single')
            bottom.set(qn('w:sz'), '12')  # Border width
            bottom.set(qn('w:space'), '1')
            bottom.set(qn('w:color'), '4472C4')  # Hospital blue color
            pBdr.append(bottom)
            pPr.append(pBdr)
        except:
            pass
        
        # Spacer after header        
        # ========== END OF HEADER ==========
        
        # Define columns (19 columns with behavior)
        # Format: (header, field, is_vertical, width_ratio)
        # 
        # is_vertical=True: Narrow metadata columns with 90° rotated headers
        # is_vertical=False: Wide content columns with horizontal wrapped text
        # 
        # Width ratios:
        # - Narrow columns: 0.6 to 1.2 (metadata fields)
        # - Wide columns: 3.0, 4.0, 8.0 (content fields)
        columns = [
            ("تاريخ تلقي الملاحظة", "received_date", True, 0.353),     # 0.53 / 1.5
            ("الرقم", "id", True, 0.267),                               # 0.4 / 1.5
            ("P. Full Name", "patient_name", True, 0.444),             # 0.6666 / 1.5
            ("قسم الصادر", "issuing_org_unit_name", True, 0.444),
            ("الإدارة", "issuing_org_unit_name", True, 0.353),
            ("القسم المعني", "issuing_org_unit_name", True, 0.444),
            ("المصدر", "source_name", True, 0.353),
            ("النوع", "feedback_intent_type_name", True, 0.353),
            ("Domain", "domain_name", True, 0.444),
            ("Category", "category_name", True, 0.444),
            ("Sub-Category", "subcategory_name", True, 0.444),
            ("Target Departments", "target_departments_display", True, 0.8),
            ("Classification", "classification_name", True, 0.8),   # 1.2 / 1.5
            ("محتوى الشكوى (Raw Content)", "complaint_text", False, 3.555),   # 8.0 / 1.5
            ("Immediate Action (خدمات المرضى+القسم)", "immediate_action", False, 2.667), # 4.0 / 1.5
            ("الإجراءات المتخذة (القسم/الدائرة/الإدارة)", "taken_action", False, 2.0),   # 3.0 / 1.5
            ("Severity", "severity_name", True, 0.311),                       # 0.4666 / 1.5
            ("Stage", "stage_name", True, 0.353),                             # 0.53 / 1.5
            ("Harm", "harm_level", True, 0.267)                               # 0.4 / 1.5
        ]
        
        # Handle empty data
        if not rows:
            doc.add_paragraph("No data available")
            buffer = BytesIO()
            doc.save(buffer)
            buffer.seek(0)
            return buffer.getvalue()
        
        # Create main data table
        table = doc.add_table(rows=1, cols=19)
        table.style = 'Table Grid'
        # Force fixed table layout so Word respects column widths
        tbl = table._element
        tblPr = tbl.tblPr
        tblLayout = OxmlElement('w:tblLayout')
        tblLayout.set(qn('w:type'), 'fixed')
        tblPr.append(tblLayout)


        # Set table to RTL (right-to-left) direction
        tbl = table._element
        tblPr = tbl.tblPr
        if tblPr is None:
            tblPr = OxmlElement('w:tblPr')
            tbl.insert(0, tblPr)

        # Force RTL table
        bidiVisual = OxmlElement('w:bidiVisual')
        tblPr.append(bidiVisual)

        # Center the table on the page
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        
        # Also force center alignment at table level
        tblJc = OxmlElement('w:jc')
        tblJc.set(qn('w:val'), 'center')
        tblPr.append(tblJc)
        
        # CRITICAL: Set minimum row height with auto-expansion for ALL rows
        # This fixes vertical column clipping and broken stripes
        for row in table.rows:
            row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
            row.height = Inches(0.79)  # Minimum height, can grow larger
        
        # Header row
        header_cells = table.rows[0].cells
        for idx, (header_name, _, is_vertical, _) in enumerate(columns):
            cell = header_cells[idx]
            cell.text = header_name
            
            # Style header cell
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                paragraph.paragraph_format.right_to_left = True
                for run in paragraph.runs:
                    run.font.bold = True
                    run.font.size = int(Pt(8))
                    run.font.name = 'Traditional Arabic'
                    run.italic = False
                    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
            
            # Apply light turquoise/green background
            try:
                shading_elm = cell._element.get_or_add_tcPr()
                shading_elm.get_or_add_shd().fill = "B4E7CE"  # Light green
            except:
                pass
            
            # Apply vertical text rotation for narrow columns
            if is_vertical:
                try:
                    # Set text direction to vertical (bt-lr = bottom to top, left to right)
                    tc = cell._element
                    tcPr = tc.get_or_add_tcPr()
                    textDirection = OxmlElement('w:textDirection')
                    textDirection.set(qn('w:val'), 'btLr')  # Bottom to top, left to right (90° rotation)
                    tcPr.append(textDirection)
                except:
                    pass  # If rotation fails, continue without it
        
        # Set column widths (force exact page width fitting)
        # Calculate usable width from actual page dimensions
        section = doc.sections[0]
        usable_width = section.page_width - section.left_margin - section.right_margin
        
        # Disable autofit so Word respects our exact widths
        table.autofit = False
        
        # Calculate total ratio units and normalize
        total_ratio = sum(col[3] for col in columns)
        
        # Track actual widths to ensure exact sum
        assigned_widths = []
        total_assigned = 0
        
        # Calculate each column width as exact proportion of usable width
        for idx, (_, _, _, width_ratio) in enumerate(columns):
            if idx == len(columns) - 1:  # Last column gets remaining width
                col_width = int(usable_width - total_assigned)
            else:
                col_width = int((width_ratio / total_ratio) * usable_width)
                total_assigned += col_width
            
            assigned_widths.append(col_width)
            
            # Apply width to column
            table.columns[idx].width = col_width
            
            # Apply width to every cell in this column for enforcement
            for row in table.rows:
                row.cells[idx].width = col_width
        
        # Data rows (limit to 50 for Word)
        row_count = min(len(rows), 50)
        for row_dict in rows[:row_count]:
            new_row = table.add_row()
            row_cells = new_row.cells
            
            # CRITICAL: Set minimum row height with auto-expansion for this data row
            new_row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
            new_row.height = Inches(0.79)  # Minimum height, allows natural expansion
            
            # Check if this is a red flag / never event
            is_red_flag = False
            try:
                clinical_risk = row_dict.get("clinical_risk_type_name", "")
                if clinical_risk and clinical_risk != "Ordinary":
                    is_red_flag = True
            except:
                pass
            
            for idx, (header_name, field_name, is_vertical, _) in enumerate(columns):
                # Special handling for target departments: concatenate all department names
                if field_name == "target_departments_display":
                    target_depts = row_dict.get("target_departments", [])
                    if target_depts and isinstance(target_depts, list):
                        dept_names = [dept.get("department_name", "") for dept in target_depts if dept.get("department_name")]
                        raw_value = ", ".join(dept_names) if dept_names else "—"
                    else:
                        raw_value = "—"
                else:
                    raw_value = sanitize_value(row_dict.get(field_name, ""))
                
                # CRITICAL: Normalize text to remove manual line breaks from UI
                value = normalize_text(raw_value)
                
                # Truncate text appropriately
                if not is_vertical:  # Wide horizontal columns
                    if len(value) > 400:
                        value = value[:400] + "..."
                else:  # Narrow vertical columns
                    if len(value) > 60:
                        value = value[:60] + "..."
                
                cell = row_cells[idx]
                cell.text = ""  # Clear cell

                p = cell.paragraphs[0]
                run = p.add_run(value)

                # Font
                run.font.size = int(Pt(7))
                run.font.name = 'Traditional Arabic'
                run.italic = False
                run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')

                # Apply semantic coloring based on column and value (applies to ALL columns)
                cell_color = None
                    
                # Severity coloring
                if field_name == "severity_name":
                    value_lower = value.lower()
                    if "high" in value_lower:
                        cell_color = "FFB3BA"  # Light red
                    elif "medium" in value_lower:
                        cell_color = "FFDFBA"  # Light orange
                    elif "low" in value_lower:
                        cell_color = "BAFFC9"  # Light green
                    
                # Harm coloring
                elif field_name == "harm_level":
                    value_lower = value.lower()
                    if "death" in value_lower:
                        cell_color = "FF6B6B"  # Red
                    elif "severe" in value_lower:
                        cell_color = "FFA500"  # Orange
                    elif "no harm" in value_lower or "none" in value_lower:
                        cell_color = "BAFFC9"  # Light green
                    elif "minor" in value_lower or "temporary" in value_lower:
                        cell_color = "FFFFBA"  # Light yellow
                
                # Stage coloring
                elif field_name == "stage_name":
                    value_lower = value.lower()
                    if "admission" in value_lower:
                        cell_color = "BAE1FF"  # Light blue
                    elif "discharge" in value_lower or "transfer" in value_lower:
                        cell_color = "E0BBE4"  # Light purple
                    elif "examination" in value_lower or "diagnosis" in value_lower:
                        cell_color = "B4F8F8"  # Light cyan
                
                # Domain coloring
                elif field_name == "domain_name":
                    value_lower = value.lower()
                    if "clinical" in value_lower:
                        cell_color = "BAE1FF"  # Light blue
                    elif "management" in value_lower:
                        cell_color = "E0BBE4"  # Light purple
                    elif "relational" in value_lower:
                        cell_color = "FFDFBA"  # Light orange
                
                # Red flag row highlighting (very light red background for entire row)
                if is_red_flag and cell_color is None:
                    cell_color = "FFE5E5"  # Very light red
                
                # Apply cell background color if determined
                if cell_color:
                    try:
                        shading_elm = cell._element.get_or_add_tcPr()
                        shading = shading_elm.get_or_add_shd()
                        shading.fill = cell_color
                    except:
                        pass

                # Apply layout based on column type - ALL TEXT CENTERED
                if is_vertical:
                    # Vertical columns: center horizontally and vertically
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

                    # Remove spacing but allow natural wrapping
                    p.paragraph_format.space_before = Pt(0)
                    p.paragraph_format.space_after = Pt(0)
                    # Do NOT set noWrap - let Word handle text flow

                    # Set vertical text direction
                    tc = cell._element
                    tcPr = tc.get_or_add_tcPr()

                    # Remove previous textDirection if exists
                    for el in tcPr.findall(qn('w:textDirection')):
                        tcPr.remove(el)

                    textDirection = OxmlElement('w:textDirection')
                    textDirection.set(qn('w:val'), 'btLr')
                    tcPr.append(textDirection)

                    # Vertical center alignment for cell
                    vAlign = OxmlElement('w:vAlign')
                    vAlign.set(qn('w:val'), 'center')
                    tcPr.append(vAlign)

                else:
                    # Horizontal columns: center horizontally and vertically
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    
                    # Set vertical center alignment for cell
                    tc = cell._element
                    tcPr = tc.get_or_add_tcPr()
                    
                    vAlign = OxmlElement('w:vAlign')
                    vAlign.set(qn('w:val'), 'center')
                    tcPr.append(vAlign)
                    # Allow Word to wrap text naturally for proper row expansion
                
        # Note if data was truncated
        if len(rows) > 50:
            doc.add_paragraph()
            note_para = doc.add_paragraph()
            note_run = note_para.add_run(
                f"Showing first 50 of {len(rows)} records. Download Excel for full data."
            )
            note_run.font.size = int(Pt(9))
            note_run.font.name = 'Traditional Arabic'
            note_run.italic = False
            note_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
            note_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add signature block
        doc.add_paragraph()  # Spacer
        
        # Signature table (1 row x 2 columns: 70% right for approvals, 30% left for patient services)
        sig_table = doc.add_table(rows=1, cols=2)
        sig_table.style = 'Table Grid'
        
        # Set RTL direction for signature table
        sig_tbl = sig_table._element
        sig_tblPr = sig_tbl.tblPr
        if sig_tblPr is None:
            sig_tblPr = OxmlElement('w:tblPr')
            sig_tbl.insert(0, sig_tblPr)
        
        # Force RTL table
        sig_bidiVisual = OxmlElement('w:bidiVisual')
        sig_tblPr.append(sig_bidiVisual)
        
        # Set column widths - RIGHT cell 70%, LEFT cell 30%
        sig_table.columns[0].width = int(Cm(18))  # RIGHT column: Approvals (70%)
        sig_table.columns[1].width = int(Cm(8))   # LEFT column: Patient Services (30%)
        
        # RIGHT CELL (BIG - 70%): 3-row approval signatures with nested 3x3 table
        approvals_cell = sig_table.rows[0].cells[0]
        approvals_cell.text = ""  # Clear default text
        
        # Create nested 3x3 table for proper approval structure
        nested_table = approvals_cell.add_table(rows=3, cols=3)
        nested_table.style = 'Table Grid'
        
        # Remove borders from nested table
        nested_tbl = nested_table._element
        nested_tblPr = nested_tbl.tblPr
        if nested_tblPr is None:
            nested_tblPr = OxmlElement('w:tblPr')
            nested_tbl.insert(0, nested_tblPr)
        
        nested_tblBorders = OxmlElement('w:tblBorders')
        for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
            border_elem = OxmlElement(f'w:{border_name}')
            border_elem.set(qn('w:val'), 'nil')
            nested_tblBorders.append(border_elem)
        nested_tblPr.append(nested_tblBorders)
        
        # Set RTL for nested table
        nested_bidiVisual = OxmlElement('w:bidiVisual')
        nested_tblPr.append(nested_bidiVisual)
        
        # Set equal column widths for nested 3x3 table
        nested_col_width = int(Cm(18) / 3)  # Divide 70% into 3 equal parts
        for i in range(3):
            nested_table.columns[i].width = nested_col_width
        
        # Fill nested table with approval data (3 rows × 3 columns)
        approval_rows = [
            ["إسم مسؤول العملية:", "التوقيع:", "التاريخ:"],
            ["إسم رئيس الدائرة:", "التوقيع:", "التاريخ:"],
            ["إسم مدير الإدارة:", "التوقيع:", "التاريخ:"]
        ]
        
        for row_idx, row_data in enumerate(approval_rows):
            for col_idx, label_text in enumerate(row_data):
                cell = nested_table.rows[row_idx].cells[col_idx]
                para = cell.paragraphs[0]
                
                # Add bold label
                label_run = para.add_run(label_text)
                label_run.font.bold = True
                label_run.font.size = int(Pt(10))
                label_run.font.name = 'Traditional Arabic'
                label_run.italic = False
                label_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
                
                # Add dots after label
                dots_run = para.add_run(" ............")
                dots_run.font.bold = False
                dots_run.font.size = int(Pt(10))
                dots_run.font.name = 'Traditional Arabic'
                dots_run.italic = False
                dots_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
                
                # Set RTL alignment
                para.paragraph_format.right_to_left = True
                para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        
        # LEFT CELL (SMALL - 30%): Patient Services section
        patient_cell = sig_table.rows[0].cells[1]
        patient_cell.text = ""
        
        # Title: خاص خدمات المرضى (centered, bold)
        title_para = patient_cell.paragraphs[0]
        title_run = title_para.add_run("خاص خدمات المرضى")
        title_run.font.size = int(Pt(11))
        title_run.font.bold = True
        title_run.font.name = 'Traditional Arabic'
        title_run.italic = False
        title_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        title_para.paragraph_format.right_to_left = True
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Patient services fields
        patient_fields = ["الإسم:", "تاريخ الإستلام:", "التوقيع:"]
        
        for field_label in patient_fields:
            field_para = patient_cell.add_paragraph()
            
            # Bold label
            label_run = field_para.add_run(field_label)
            label_run.font.bold = True
            label_run.font.size = int(Pt(10))
            label_run.font.name = 'Traditional Arabic'
            label_run.italic = False
            label_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
            
            # Normal dots after label
            dots_run = field_para.add_run(" ............")
            dots_run.font.bold = False
            dots_run.font.size = int(Pt(10))
            dots_run.font.name = 'Traditional Arabic'
            dots_run.italic = False
            dots_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
            
            # RTL alignment
            field_para.paragraph_format.right_to_left = True
            field_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        
        # ========== ADD MOTIVATIONAL QUOTE TO REAL FOOTER ==========
        try:
            section = doc.sections[0]
            footer = section.footer
            footer_para = footer.paragraphs[0]
            footer_para.clear()
            
            # Add the Arabic quote
            run = footer_para.add_run(
                "نؤمن أن الإبتكار لا يكون فقط في التقنيات، بل في أسلوب الخدمة والتواصل والتعاطف… فلنبتكر معًا تجربة ذات أثر طيب"
            )
            run.font.size = int(Pt(10))
            run.font.name = 'Traditional Arabic'
            run.italic = False
            run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
            
            footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Add top border to footer for professional look
            pPr = footer_para._element.get_or_add_pPr()
            pBdr = OxmlElement('w:pBdr')
            top = OxmlElement('w:top')
            top.set(qn('w:val'), 'single')
            top.set(qn('w:sz'), '6')
            top.set(qn('w:color'), 'DDDDDD')
            pBdr.append(top)
            pPr.append(pBdr)
        except:
            pass  # If footer setup fails, continue without it
        
        # Save to BytesIO
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer.getvalue()


# Export service instance
reports_service = ReportsService()

