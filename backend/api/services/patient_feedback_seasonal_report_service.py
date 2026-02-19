"""
Patient Feedback Seasonal Report Word Generator

Generates comprehensive Word documents with RCA (Root Cause Analysis) 
and Satisfaction statistics for a given seasonal period.

This report aggregates:
- RCA breakdowns by cause type (Staff, Process, Equipment, Environment)
- Satisfaction status distribution (Satisfied, Not Satisfied, Not Present)
- Preventability analysis
- Feedback coverage metrics
"""

from typing import List, Dict, Any
from datetime import date, datetime
from io import BytesIO
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Mm, Cm
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml

from core.database import get_connection


def _fetch_rca_statistics(season_start: date, season_end: date) -> Dict[str, Any]:
    """
    Fetch RCA (Root Cause Analysis) statistics for the seasonal period.
    
    Returns breakdown by cause type, preventability, and department.
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Main RCA statistics query
        # Check if ANY cause column in each category has value 1
        query = """
            SELECT
                COUNT(*) as total_rca_records,
                SUM(CASE WHEN (
                    f.Cause_Staff_Training = 1 OR f.Cause_Staff_Incentives = 1 OR 
                    f.Cause_Staff_Competency = 1 OR f.Cause_Staff_Understaffed = 1 OR 
                    f.Cause_Staff_NonCompliance = 1 OR f.Cause_Staff_NoCoordination = 1 OR 
                    f.Cause_Staff_Other = 1
                ) THEN 1 ELSE 0 END) as staff_causes,
                SUM(CASE WHEN (
                    f.Cause_Process_NotComprehensive = 1 OR f.Cause_Process_Unclear = 1 OR 
                    f.Cause_Process_MissingProtocol = 1 OR f.Cause_Process_Other = 1
                ) THEN 1 ELSE 0 END) as process_causes,
                SUM(CASE WHEN (
                    f.Cause_Equipment_NotAvailable = 1 OR f.Cause_Equipment_SystemIncomplete = 1 OR 
                    f.Cause_Equipment_HardToApply = 1 OR f.Cause_Equipment_Other = 1
                ) THEN 1 ELSE 0 END) as equipment_causes,
                SUM(CASE WHEN (
                    f.Cause_Environment_PlaceNature = 1 OR f.Cause_Environment_Surroundings = 1 OR 
                    f.Cause_Environment_WorkConditions = 1 OR f.Cause_Environment_Other = 1
                ) THEN 1 ELSE 0 END) as environment_causes,
                SUM(CASE WHEN (
                    f.Preventive_MonthlyMeetings = 1 OR f.Preventive_TrainingPrograms = 1 OR 
                    f.Preventive_IncreaseStaff = 1 OR f.Preventive_MMCommitteeActions = 1 OR 
                    f.Preventive_Other = 1
                ) THEN 1 ELSE 0 END) as has_preventive_measures,
                SUM(CASE WHEN (
                    COALESCE(f.Preventive_MonthlyMeetings, 0) = 0 AND 
                    COALESCE(f.Preventive_TrainingPrograms, 0) = 0 AND 
                    COALESCE(f.Preventive_IncreaseStaff, 0) = 0 AND 
                    COALESCE(f.Preventive_MMCommitteeActions, 0) = 0 AND 
                    COALESCE(f.Preventive_Other, 0) = 0
                ) THEN 1 ELSE 0 END) as no_preventive_measures
            FROM dbo.APP_IncidentCaseFeedback f
            INNER JOIN dbo.APP_AdministrativeSubcase sub ON f.AdministrativeSubcaseID = sub.SubcaseID
            INNER JOIN dbo.APP_IncidentCase ic ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            WHERE f.AdministrativeSubcaseID IS NOT NULL
              AND CONVERT(DATE, ic.FeedbackRecievedDate) >= ?
              AND CONVERT(DATE, ic.FeedbackRecievedDate) <= ?
        """
        cursor.execute(query, (season_start.isoformat(), season_end.isoformat()))
        row = cursor.fetchone()
        
        total = row[0] or 0
        stats = {
            "total_rca_records": total,
            "by_cause_type": {
                "staff": row[1] or 0,
                "process": row[2] or 0,
                "equipment": row[3] or 0,
                "environment": row[4] or 0
            },
            "preventability": {
                "has_preventive_measures": row[5] or 0,
                "no_preventive_measures": row[6] or 0
            }
        }
        
        # RCA by department query
        dept_query = """
            SELECT
                COALESCE(ou.Name, 'غير محدد') as department,
                COUNT(*) as rca_count,
                SUM(CASE WHEN (
                    f.Cause_Staff_Training = 1 OR f.Cause_Staff_Incentives = 1 OR 
                    f.Cause_Staff_Competency = 1 OR f.Cause_Staff_Understaffed = 1 OR 
                    f.Cause_Staff_NonCompliance = 1 OR f.Cause_Staff_NoCoordination = 1 OR 
                    f.Cause_Staff_Other = 1
                ) THEN 1 ELSE 0 END) as staff,
                SUM(CASE WHEN (
                    f.Cause_Process_NotComprehensive = 1 OR f.Cause_Process_Unclear = 1 OR 
                    f.Cause_Process_MissingProtocol = 1 OR f.Cause_Process_Other = 1
                ) THEN 1 ELSE 0 END) as process,
                SUM(CASE WHEN (
                    f.Cause_Equipment_NotAvailable = 1 OR f.Cause_Equipment_SystemIncomplete = 1 OR 
                    f.Cause_Equipment_HardToApply = 1 OR f.Cause_Equipment_Other = 1
                ) THEN 1 ELSE 0 END) as equipment,
                SUM(CASE WHEN (
                    f.Cause_Environment_PlaceNature = 1 OR f.Cause_Environment_Surroundings = 1 OR 
                    f.Cause_Environment_WorkConditions = 1 OR f.Cause_Environment_Other = 1
                ) THEN 1 ELSE 0 END) as environment
            FROM dbo.APP_IncidentCaseFeedback f
            INNER JOIN dbo.APP_AdministrativeSubcase sub ON f.AdministrativeSubcaseID = sub.SubcaseID
            INNER JOIN dbo.APP_IncidentCase ic ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.AdminsrationUnit ou WITH (NOLOCK) ON sub.TargetOrgUnitID = ou.UniqueID
            WHERE f.AdministrativeSubcaseID IS NOT NULL
              AND CONVERT(DATE, ic.FeedbackRecievedDate) >= ?
              AND CONVERT(DATE, ic.FeedbackRecievedDate) <= ?
            GROUP BY COALESCE(ou.Name, 'غير محدد')
            ORDER BY COUNT(*) DESC
        """
        cursor.execute(dept_query, (season_start.isoformat(), season_end.isoformat()))
        
        stats["by_department"] = []
        for row in cursor.fetchall():
            stats["by_department"].append({
                "department": row[0],
                "rca_count": row[1],
                "staff": row[2],
                "process": row[3],
                "equipment": row[4],
                "environment": row[5]
            })
        
        return stats
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def _fetch_satisfaction_statistics(season_start: date, season_end: date) -> Dict[str, Any]:
    """
    Fetch Satisfaction statistics for the seasonal period.
    
    Returns breakdown by status and feedback metrics.
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Total cases in period
        total_query = """
            SELECT COUNT(DISTINCT ic.IncidentRequestCaseID)
            FROM dbo.APP_IncidentCase ic
            WHERE CONVERT(DATE, ic.FeedbackRecievedDate) >= ?
              AND CONVERT(DATE, ic.FeedbackRecievedDate) <= ?
        """
        cursor.execute(total_query, (season_start.isoformat(), season_end.isoformat()))
        total_cases = cursor.fetchone()[0] or 0
        
        # Satisfaction by status
        status_query = """
            SELECT
                ss.SatisfactionStatusID,
                ss.StatusNameEn,
                ss.StatusNameAr,
                COUNT(*) as count
            FROM dbo.APP_IncidentCaseSatisfaction s
            INNER JOIN dbo.APP_Lookup_SatisfactionStatus ss ON s.SatisfactionStatusID = ss.SatisfactionStatusID
            INNER JOIN dbo.APP_IncidentCase ic ON s.IncidentRequestCaseID = ic.IncidentRequestCaseID
            WHERE CONVERT(DATE, ic.FeedbackRecievedDate) >= ?
              AND CONVERT(DATE, ic.FeedbackRecievedDate) <= ?
            GROUP BY ss.SatisfactionStatusID, ss.StatusNameEn, ss.StatusNameAr
            ORDER BY ss.SatisfactionStatusID
        """
        cursor.execute(status_query, (season_start.isoformat(), season_end.isoformat()))
        
        by_status = []
        total_with_satisfaction = 0
        for row in cursor.fetchall():
            count = row[3]
            total_with_satisfaction += count
            by_status.append({
                "status_id": row[0],
                "status_en": row[1],
                "status_ar": row[2],
                "count": count
            })
        
        # Feedback needed/given stats
        feedback_query = """
            SELECT
                SUM(CASE WHEN s.FeedbackNeeded = 1 THEN 1 ELSE 0 END) as needed,
                SUM(CASE WHEN s.FeedbackGiven = 1 THEN 1 ELSE 0 END) as given
            FROM dbo.APP_IncidentCaseSatisfaction s
            INNER JOIN dbo.APP_IncidentCase ic ON s.IncidentRequestCaseID = ic.IncidentRequestCaseID
            WHERE CONVERT(DATE, ic.FeedbackRecievedDate) >= ?
              AND CONVERT(DATE, ic.FeedbackRecievedDate) <= ?
        """
        cursor.execute(feedback_query, (season_start.isoformat(), season_end.isoformat()))
        fb_row = cursor.fetchone()
        
        return {
            "total_cases": total_cases,
            "cases_with_satisfaction": total_with_satisfaction,
            "cases_without_satisfaction": total_cases - total_with_satisfaction,
            "coverage_percentage": round(total_with_satisfaction / total_cases * 100, 1) if total_cases > 0 else 0,
            "by_status": by_status,
            "feedback_stats": {
                "feedback_needed": fb_row[0] or 0 if fb_row else 0,
                "feedback_given": fb_row[1] or 0 if fb_row else 0
            }
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def _fetch_subcase_coverage(season_start: date, season_end: date) -> Dict[str, Any]:
    """
    Fetch subcase RCA coverage statistics.
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT
                COUNT(DISTINCT sub.SubcaseID) as total_subcases,
                COUNT(DISTINCT CASE WHEN f.AdministrativeSubcaseID IS NOT NULL THEN sub.SubcaseID END) as subcases_with_rca
            FROM dbo.APP_AdministrativeSubcase sub
            INNER JOIN dbo.APP_IncidentCase ic ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_IncidentCaseFeedback f ON sub.SubcaseID = f.AdministrativeSubcaseID
            WHERE CONVERT(DATE, ic.FeedbackRecievedDate) >= ?
              AND CONVERT(DATE, ic.FeedbackRecievedDate) <= ?
        """
        cursor.execute(query, (season_start.isoformat(), season_end.isoformat()))
        row = cursor.fetchone()
        
        total = row[0] or 0
        with_rca = row[1] or 0
        
        return {
            "total_subcases": total,
            "subcases_with_rca": with_rca,
            "subcases_without_rca": total - with_rca,
            "rca_coverage_percentage": round(with_rca / total * 100, 1) if total > 0 else 0
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def _set_cell_shading(cell, color_hex: str):
    """Set cell background color."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)


def _format_table_header(table, headers: List[str], color_hex: str = "667eea"):
    """Format table header row with styling."""
    header_row = table.rows[0]
    for idx, header_text in enumerate(headers):
        cell = header_row.cells[idx]
        cell.text = header_text
        _set_cell_shading(cell, color_hex)
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            for run in paragraph.runs:
                run.font.bold = True
                run.font.size = Pt(11)
                run.font.name = 'Arial'
                run.font.color.rgb = RGBColor(255, 255, 255)


def generate_patient_feedback_seasonal_word(
    season_start: date,
    season_end: date
) -> bytes:
    """
    Generate Patient Feedback Seasonal Report as Word document.
    
    Contains:
    - Executive summary with coverage metrics
    - RCA (Root Cause Analysis) breakdown
    - Satisfaction status distribution
    - Preventability analysis
    - Department-wise breakdown
    
    Args:
        season_start: Start date of reporting period
        season_end: End date of reporting period
    
    Returns:
        Word document as bytes
    """
    # Fetch all statistics
    rca_stats = _fetch_rca_statistics(season_start, season_end)
    satisfaction_stats = _fetch_satisfaction_statistics(season_start, season_end)
    subcase_coverage = _fetch_subcase_coverage(season_start, season_end)
    
    # Create document
    doc = Document()
    
    # Page setup
    section = doc.sections[0]
    section.page_height = Mm(297)
    section.page_width = Mm(210)
    section.left_margin = Cm(2)
    section.right_margin = Cm(2)
    
    # ========== TITLE PAGE ==========
    doc.add_paragraph()
    
    title_ar = doc.add_heading('تقرير ملاحظات المرضى الموسمي', 0)
    title_ar.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    for run in title_ar.runs:
        run.font.size = Pt(22)
        run.font.name = 'Arial'
        run.font.bold = True
        run.font.color.rgb = RGBColor(102, 126, 234)
    
    title_en = doc.add_heading('Patient Feedback Seasonal Report', 1)
    title_en.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    for run in title_en.runs:
        run.font.size = Pt(16)
        run.font.name = 'Arial'
        run.font.color.rgb = RGBColor(118, 75, 162)
    
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    sub_run = subtitle.add_run('تحليل الأسباب الجذرية ورضا المرضى')
    sub_run.font.size = Pt(14)
    sub_run.font.name = 'Arial'
    sub_run.font.color.rgb = RGBColor(100, 100, 100)
    
    subtitle2 = doc.add_paragraph()
    subtitle2.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    sub_run2 = subtitle2.add_run('Root Cause Analysis & Patient Satisfaction')
    sub_run2.font.size = Pt(12)
    sub_run2.font.name = 'Arial'
    sub_run2.font.italic = True
    sub_run2.font.color.rgb = RGBColor(100, 100, 100)
    
    doc.add_paragraph()
    
    # Period info
    period = doc.add_paragraph()
    period.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    period_run = period.add_run(
        f'الفترة: {season_start.strftime("%Y-%m-%d")} إلى {season_end.strftime("%Y-%m-%d")}'
    )
    period_run.font.size = Pt(12)
    period_run.font.name = 'Arial'
    period_run.font.bold = True
    
    period_en = doc.add_paragraph()
    period_en.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    period_en_run = period_en.add_run(
        f'Period: {season_start.strftime("%B %d, %Y")} to {season_end.strftime("%B %d, %Y")}'
    )
    period_en_run.font.size = Pt(11)
    period_en_run.font.name = 'Arial'
    
    # Generation timestamp
    gen_time = doc.add_paragraph()
    gen_time.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    gen_run = gen_time.add_run(f'تم الإنشاء: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    gen_run.font.size = Pt(9)
    gen_run.font.name = 'Arial'
    gen_run.font.color.rgb = RGBColor(128, 128, 128)
    
    doc.add_page_break()
    
    # ========== EXECUTIVE SUMMARY ==========
    summary_heading = doc.add_heading('الملخص التنفيذي / Executive Summary', 1)
    for run in summary_heading.runs:
        run.font.name = 'Arial'
        run.font.color.rgb = RGBColor(102, 126, 234)
    
    # Summary metrics table
    summary_table = doc.add_table(rows=6, cols=3)
    summary_table.style = 'Table Grid'
    summary_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    _format_table_header(summary_table, ['المقياس / Metric', 'القيمة / Value', 'النسبة / Percentage'])
    
    summary_data = [
        ('إجمالي الحالات / Total Cases', str(satisfaction_stats['total_cases']), '-'),
        ('إجمالي الحالات الفرعية / Total Subcases', str(subcase_coverage['total_subcases']), '-'),
        ('الحالات الفرعية مع RCA / Subcases with RCA', 
         str(subcase_coverage['subcases_with_rca']), 
         f"{subcase_coverage['rca_coverage_percentage']}%"),
        ('حالات مع رضا المريض / Cases with Satisfaction', 
         str(satisfaction_stats['cases_with_satisfaction']), 
         f"{satisfaction_stats['coverage_percentage']}%"),
        ('سجلات RCA الإجمالية / Total RCA Records', str(rca_stats['total_rca_records']), '-'),
    ]
    
    for idx, (metric, value, pct) in enumerate(summary_data):
        row = summary_table.rows[idx + 1]
        row.cells[0].text = metric
        row.cells[1].text = value
        row.cells[2].text = pct
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(10)
                    run.font.name = 'Arial'
    
    doc.add_paragraph()
    
    # ========== RCA ANALYSIS SECTION ==========
    rca_heading = doc.add_heading('تحليل الأسباب الجذرية / Root Cause Analysis', 1)
    for run in rca_heading.runs:
        run.font.name = 'Arial'
        run.font.color.rgb = RGBColor(102, 126, 234)
    
    # RCA by Cause Type
    cause_subheading = doc.add_heading('توزيع حسب نوع السبب / Distribution by Cause Type', 2)
    for run in cause_subheading.runs:
        run.font.name = 'Arial'
    
    cause_table = doc.add_table(rows=5, cols=4)
    cause_table.style = 'Table Grid'
    cause_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    _format_table_header(cause_table, ['نوع السبب / Cause Type', 'العدد / Count', 'النسبة / Percentage', 'الوصف / Description'])
    
    total_causes = (rca_stats['by_cause_type']['staff'] + 
                   rca_stats['by_cause_type']['process'] + 
                   rca_stats['by_cause_type']['equipment'] + 
                   rca_stats['by_cause_type']['environment']) or 1
    
    cause_data = [
        ('الطاقم / Staff', rca_stats['by_cause_type']['staff'], 'أسباب متعلقة بالموظفين'),
        ('العملية / Process', rca_stats['by_cause_type']['process'], 'أسباب متعلقة بالإجراءات'),
        ('المعدات / Equipment', rca_stats['by_cause_type']['equipment'], 'أسباب متعلقة بالأجهزة'),
        ('البيئة / Environment', rca_stats['by_cause_type']['environment'], 'أسباب متعلقة بالبيئة'),
    ]
    
    cause_colors = {
        'الطاقم / Staff': 'FFE5E5',
        'العملية / Process': 'FFF3E5',
        'المعدات / Equipment': 'E5F0FF',
        'البيئة / Environment': 'E5FFE5'
    }
    
    for idx, (cause_type, count, desc) in enumerate(cause_data):
        row = cause_table.rows[idx + 1]
        row.cells[0].text = cause_type
        row.cells[1].text = str(count)
        row.cells[2].text = f"{round(count / total_causes * 100, 1)}%"
        row.cells[3].text = desc
        
        # Color code the cause type cell
        _set_cell_shading(row.cells[0], cause_colors[cause_type])
        
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(10)
                    run.font.name = 'Arial'
    
    doc.add_paragraph()
    
    # Preventability Analysis
    prevent_subheading = doc.add_heading('إجراءات وقائية مقترحة / Preventive Measures Analysis', 2)
    for run in prevent_subheading.runs:
        run.font.name = 'Arial'
    
    prevent_table = doc.add_table(rows=3, cols=3)
    prevent_table.style = 'Table Grid'
    prevent_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    _format_table_header(prevent_table, ['التصنيف / Classification', 'العدد / Count', 'النسبة / Percentage'])
    
    total_prevent = (rca_stats['preventability']['has_preventive_measures'] + 
                    rca_stats['preventability']['no_preventive_measures']) or 1
    
    prevent_data = [
        ('لديه إجراءات وقائية / Has Preventive Measures', rca_stats['preventability']['has_preventive_measures']),
        ('بدون إجراءات وقائية / No Preventive Measures', rca_stats['preventability']['no_preventive_measures']),
    ]
    
    for idx, (classification, count) in enumerate(prevent_data):
        row = prevent_table.rows[idx + 1]
        row.cells[0].text = classification
        row.cells[1].text = str(count)
        row.cells[2].text = f"{round(count / total_prevent * 100, 1)}%"
        
        # Color code
        if 'Has Preventive' in classification:
            _set_cell_shading(row.cells[0], 'E5FFE5')
        else:
            _set_cell_shading(row.cells[0], 'FFE5E5')
        
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(10)
                    run.font.name = 'Arial'
    
    doc.add_paragraph()
    
    # ========== SATISFACTION SECTION ==========
    sat_heading = doc.add_heading('رضا المرضى / Patient Satisfaction', 1)
    for run in sat_heading.runs:
        run.font.name = 'Arial'
        run.font.color.rgb = RGBColor(102, 126, 234)
    
    # Satisfaction by Status
    status_subheading = doc.add_heading('توزيع حسب حالة الرضا / Distribution by Satisfaction Status', 2)
    for run in status_subheading.runs:
        run.font.name = 'Arial'
    
    status_table = doc.add_table(rows=len(satisfaction_stats['by_status']) + 2, cols=4)
    status_table.style = 'Table Grid'
    status_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    _format_table_header(status_table, ['الحالة (EN)', 'الحالة (AR)', 'العدد / Count', 'النسبة / Percentage'])
    
    total_satisfaction = satisfaction_stats['cases_with_satisfaction'] or 1
    
    status_colors = {
        1: 'F0F0F0',  # Not Present - gray
        2: 'D4EDDA',  # Satisfied - green
        3: 'F8D7DA',  # Not Satisfied - red
    }
    
    for idx, status in enumerate(satisfaction_stats['by_status']):
        row = status_table.rows[idx + 1]
        row.cells[0].text = status['status_en']
        row.cells[1].text = status['status_ar']
        row.cells[2].text = str(status['count'])
        row.cells[3].text = f"{round(status['count'] / total_satisfaction * 100, 1)}%"
        
        # Color code based on status
        color = status_colors.get(status['status_id'], 'FFFFFF')
        _set_cell_shading(row.cells[0], color)
        _set_cell_shading(row.cells[1], color)
        
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(10)
                    run.font.name = 'Arial'
    
    # Add row for cases without satisfaction
    no_sat_row = status_table.rows[-1]
    no_sat_row.cells[0].text = "No Record"
    no_sat_row.cells[1].text = "بدون سجل"
    no_sat_row.cells[2].text = str(satisfaction_stats['cases_without_satisfaction'])
    no_sat_row.cells[3].text = f"{round(satisfaction_stats['cases_without_satisfaction'] / satisfaction_stats['total_cases'] * 100, 1)}%" if satisfaction_stats['total_cases'] > 0 else "0%"
    _set_cell_shading(no_sat_row.cells[0], 'FFFACD')
    _set_cell_shading(no_sat_row.cells[1], 'FFFACD')
    
    for cell in no_sat_row.cells:
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            for run in paragraph.runs:
                run.font.size = Pt(10)
                run.font.name = 'Arial'
    
    doc.add_paragraph()
    
    # Feedback Follow-up Stats
    followup_subheading = doc.add_heading('متابعة ملاحظات المرضى / Patient Feedback Follow-up', 2)
    for run in followup_subheading.runs:
        run.font.name = 'Arial'
    
    followup_table = doc.add_table(rows=3, cols=2)
    followup_table.style = 'Table Grid'
    followup_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    _format_table_header(followup_table, ['المقياس / Metric', 'القيمة / Value'])
    
    followup_data = [
        ('حالات تحتاج متابعة / Feedback Needed', str(satisfaction_stats['feedback_stats']['feedback_needed'])),
        ('حالات تمت المتابعة / Feedback Given', str(satisfaction_stats['feedback_stats']['feedback_given'])),
    ]
    
    for idx, (metric, value) in enumerate(followup_data):
        row = followup_table.rows[idx + 1]
        row.cells[0].text = metric
        row.cells[1].text = value
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(10)
                    run.font.name = 'Arial'
    
    doc.add_paragraph()
    
    # ========== RCA BY DEPARTMENT ==========
    if rca_stats['by_department']:
        dept_heading = doc.add_heading('تحليل RCA حسب القسم / RCA Analysis by Department', 1)
        for run in dept_heading.runs:
            run.font.name = 'Arial'
            run.font.color.rgb = RGBColor(102, 126, 234)
        
        dept_table = doc.add_table(rows=len(rca_stats['by_department']) + 1, cols=6)
        dept_table.style = 'Table Grid'
        dept_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        
        _format_table_header(dept_table, ['القسم / Department', 'المجموع', 'طاقم', 'عملية', 'معدات', 'بيئة'])
        
        for idx, dept in enumerate(rca_stats['by_department']):
            row = dept_table.rows[idx + 1]
            row.cells[0].text = dept['department']
            row.cells[1].text = str(dept['rca_count'])
            row.cells[2].text = str(dept['staff'])
            row.cells[3].text = str(dept['process'])
            row.cells[4].text = str(dept['equipment'])
            row.cells[5].text = str(dept['environment'])
            
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                    for run in paragraph.runs:
                        run.font.size = Pt(9)
                        run.font.name = 'Arial'
    
    # ========== FOOTER ==========
    doc.add_paragraph()
    doc.add_paragraph()
    
    footer = doc.add_paragraph()
    footer.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    footer_run = footer.add_run('— نهاية التقرير / End of Report —')
    footer_run.font.size = Pt(10)
    footer_run.font.name = 'Arial'
    footer_run.font.italic = True
    footer_run.font.color.rgb = RGBColor(128, 128, 128)
    
    # Save to bytes
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    
    return buffer.getvalue()
