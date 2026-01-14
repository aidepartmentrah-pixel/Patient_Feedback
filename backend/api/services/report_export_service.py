"""
Report Export Service
Handles export logic for monthly and seasonal reports.
"""

from typing import Dict, Any, Literal
import traceback
from io import BytesIO
from .monthly_report_service import monthly_report_service
from .reports_service import reports_service

# Import for emergency fallback
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class ReportExportService:
    """
    Service responsible for generating report exports (PDF/CSV).
    Orchestrates data fetching and file generation.
    """

    def generate_export(
        self,
        *,
        report_type: Literal["monthly", "seasonal"],
        display_mode: Literal["detailed", "numeric", "hcat"],
        file_format: Literal["pdf", "csv", "xlsx", "docx"],
        year: int,
        month: int = None,
        trimester: int = None,
        quarter: int = None,
        filters: Dict[str, Any] = None,
        include_charts: bool = True,
        language: Literal["en", "ar"] = "en"
    ) -> Dict[str, Any]:
        """
        Generate an export file for a report.
        
        Args:
            report_type: Type of report (monthly or seasonal)
            display_mode: Display mode (detailed, numeric, hcat)
            file_format: Output format (pdf, csv, xlsx, docx)
            year: Year for the report
            month: Month for monthly reports
            trimester: Trimester for seasonal reports
            quarter: Quarter for seasonal reports
            filters: Additional filters
            include_charts: Include charts in PDF
            language: Language for export (en or ar)
        
        Returns:
            Dictionary containing:
                - filename: Generated filename
                - content: File content as bytes
                - content_type: MIME type
        
        Note:
            - xlsx uses CSV generator internally (temporary)
            - docx uses PDF generator internally (temporary)
        """
        try:
            filters = filters or {}
            
            # Map file formats to MIME types
            content_type_mapping = {
                "pdf": "application/pdf",
                "csv": "text/csv",
                "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            }
            
            content_type = content_type_mapping.get(file_format, "application/pdf")
            
            # Step 1: Fetch data based on report type and display mode
            if report_type == "monthly":
                report_data = self._fetch_monthly_data(
                    display_mode=display_mode,
                    year=year,
                    month=month,
                    filters=filters
                )
                # Normalize monthly detailed data: extract complaints list
                if display_mode == "detailed" and isinstance(report_data, dict) and "complaints" in report_data:
                    export_data = report_data["complaints"]
                else:
                    export_data = report_data
            else:  # seasonal
                report_data = self._fetch_seasonal_data(
                    display_mode=display_mode,
                    year=year,
                    trimester=trimester,
                    quarter=quarter,
                    filters=filters
                )
                export_data = report_data
            
            # Step 2: Generate file based on format
            if file_format == "pdf":
                content = reports_service.generate_pdf_export(
                    report_data=export_data,
                    filename=f"report_{year}.pdf",
                    language=language,
                    include_charts=include_charts
                )
            elif file_format == "xlsx":
                content = reports_service.generate_xlsx_export(
                    report_data=export_data,
                    filename=f"report_{year}.xlsx",
                    language=language
                )
            elif file_format == "docx":
                content = reports_service.generate_docx_export(
                    report_data=export_data,
                    filename=f"report_{year}.docx",
                    language=language
                )
            else:  # csv
                content = reports_service.generate_csv_export(
                    report_data=export_data,
                    filename=f"report_{year}.csv",
                    language=language
                )
            
            # Step 3: Build filename
            if report_type == "monthly" and month:
                filename = f"Monthly_Report_{year}_{month:02d}.{file_format}"
            elif report_type == "seasonal" and trimester:
                filename = f"Seasonal_Report_{year}_T{trimester}.{file_format}"
            elif report_type == "seasonal" and quarter:
                filename = f"Seasonal_Report_{year}_Q{quarter}.{file_format}"
            else:
                filename = f"Report_{year}.{file_format}"
            
            return {
                "filename": filename,
                "content": content,
                "content_type": content_type
            }
        
        except Exception as e:
            print("\n" + "="*80)
            print(f"[EXPORT DISPATCHER] HARD FAIL: {file_format.upper()} export")
            print(f"Report Type: {report_type}, Year: {year}, Month: {month}")
            print(f"Exception: {type(e).__name__}: {str(e)}")
            print("="*80)
            traceback.print_exc()
            print("="*80 + "\n")
            
            # Emergency fallback for Word exports
            if file_format == "docx":
                print("[EXPORT DISPATCHER] Generating emergency fallback Word document...")
                try:
                    if DOCX_AVAILABLE:
                        # Create minimal emergency Word document
                        doc = Document()
                        
                        # Title
                        title = doc.add_paragraph()
                        title_run = title.add_run("Word Export Emergency Fallback")
                        title_run.bold = True
                        title_run.font.size = 16
                        
                        # Error message
                        doc.add_paragraph()
                        doc.add_paragraph("The system failed to generate the real report.")
                        doc.add_paragraph()
                        doc.add_paragraph(f"Error Type: {type(e).__name__}")
                        doc.add_paragraph(f"Error Message: {str(e)}")
                        doc.add_paragraph()
                        doc.add_paragraph(f"Report Type: {report_type}")
                        doc.add_paragraph(f"Year: {year}")
                        if month:
                            doc.add_paragraph(f"Month: {month}")
                        
                        # Save to buffer
                        buffer = BytesIO()
                        doc.save(buffer)
                        buffer.seek(0)
                        
                        filename = f"Emergency_Fallback_{year}_{month or 0:02d}.docx"
                        
                        print("[EXPORT DISPATCHER] Emergency Word document created successfully")
                        return {
                            "filename": filename,
                            "content": buffer.getvalue(),
                            "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        }
                    else:
                        print("[EXPORT DISPATCHER] python-docx not available, cannot create fallback")
                except Exception as fallback_error:
                    print(f"[EXPORT DISPATCHER] Even fallback failed: {fallback_error}")
                    traceback.print_exc()
            
            # Re-raise if not Word or if fallback failed
            raise
    
    def _fetch_monthly_data(
        self,
        display_mode: str,
        year: int,
        month: int,
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch monthly report data based on display mode."""
        if display_mode == "detailed":
            filters_dict = {
                "report_type": "monthly",
                "year": year,
                "month": month,
                "page": 1,
                "page_size": 500,
                **filters
            }
            return monthly_report_service.get_detailed_monthly_report(filters=filters_dict)
        
        elif display_mode == "numeric":
            filters_dict = {
                "year": year,
                "month": month,
                **filters
            }
            return monthly_report_service.get_numeric_monthly_report(filters=filters_dict)
        
        else:
            raise ValueError(f"Invalid display_mode for monthly report: {display_mode}")
    
    def _fetch_seasonal_data(
        self,
        display_mode: str,
        year: int,
        trimester: int,
        quarter: int,
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch seasonal report data based on display mode."""
        if display_mode == "hcat":
            return reports_service.get_seasonal_hcat_report(
                year=year,
                trimester=trimester,
                quarter=quarter
            )
        else:
            raise ValueError(f"Invalid display_mode for seasonal report: {display_mode}")


# Singleton instance
report_export_service = ReportExportService()
