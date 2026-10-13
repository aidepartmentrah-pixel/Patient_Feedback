"""
D-B8: Seasonal Word Adapter

Formats doctor and worker seasonal report data into professional Word documents.
Uses python-docx to generate structured reports with tables, charts, and RTL support.

Architecture:
- Takes data payload from D-B6/D-B7 seasonal builders
- Generates Word document bytes (no file I/O)
- Reuses existing Word generation patterns from seasonal_report_formatter.py
"""

from typing import Dict, Any, Callable, List, Tuple
import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Mm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
import io

from .workflow_activity_word_formatter import (
    _ar_run as _ar_run_base,
    _set_rtl_table,
    _set_row_col_width as _set_row_col_width_base,
    _cell_para,
)

# Usable content width: A4 portrait 210mm - 20mm left - 20mm right margin =
# 170mm. Every incidents-table column config below sums to this so the
# table fills the page instead of being squeezed/cropped by Word's default
# autofit-to-contents behavior.
USABLE_WIDTH_MM = 170.0

# Last-resort truncation ceiling. Normal complaint text (even long ones)
# stays well under this; only pathological outliers get cut. The table
# column is wide and Word wraps text within cells, so there's no reason to
# hard-cut at 100/200 characters like the previous implementation did.
NARRATIVE_MAX_CHARS = 6000


def _ar_run(para, text: str, size: int = 10, bold: bool = False,
            italic: bool = False, color: str = None):
    """
    Local wrapper around the shared _ar_run helper that also sets the w:cs
    (complex-script) font explicitly to 'Traditional Arabic'. The shared
    helper only sets w:eastAsia, leaving Arabic glyph rendering to the
    theme's default complex-script font — some renderers substitute a font
    with incomplete Arabic coverage there, showing missing-glyph boxes for
    specific words. Forcing w:cs avoids that (same fix already applied in
    monthly_stylish_word_formatter.py; not touching the shared helper since
    other reports depend on its current behavior).
    """
    run = _ar_run_base(para, text, size=size, bold=bold, italic=italic, color=color)
    run._element.rPr.rFonts.set(qn('w:cs'), 'Traditional Arabic')
    return run


def _set_row_col_width(row, col_idx: int, mm: float):
    """
    Local wrapper that also updates the table's tblGrid column width, not
    just the cell's w:tcW. Word's fixed-layout rendering uses tblGrid to
    determine column boundaries — setting only tcW is silently ignored,
    which is why these tables were getting cropped/auto-squeezed before.
    """
    _set_row_col_width_base(row, col_idx, mm)
    row.table.columns[col_idx].width = Mm(mm)


def _truncate_for_fit(text: str, max_chars: int,
                       marker: str = '… (النص الكامل في النظام)') -> str:
    """
    Word-boundary-safe truncation used only as a last-resort safety net for
    pathological outliers — normal complaint text never reaches this ceiling.
    """
    text = (text or '').strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(' ')
    if last_space > max_chars * 0.7:
        cut = cut[:last_space]
    return cut.rstrip(' ,.;:،؛') + ' ' + marker


def _build_incidents_table(
    doc: Document,
    incidents: List[Dict[str, Any]],
    columns: List[Tuple[str, float, Callable[[Dict[str, Any]], str]]],
):
    """
    Shared incidents/complaints table builder for the patient, doctor, and
    worker Word reports, so the three stay structurally parallel and any
    future fix (RTL, column widths, fonts, truncation) only needs to happen
    once instead of being re-applied three times (which is exactly how this
    table ended up broken in only one of the three reports).

    columns: list of (arabic_header, width_mm, getter) — getter(incident)
    returns the display string for that cell. Widths should sum to
    USABLE_WIDTH_MM.
    """
    table = doc.add_table(rows=len(incidents) + 1, cols=len(columns))
    table.style = 'Light List Accent 1'
    table.autofit = False
    _set_rtl_table(table)

    header_row = table.rows[0]
    for col_idx, (header_ar, width_mm, _getter) in enumerate(columns):
        _set_row_col_width(header_row, col_idx, width_mm)
        para = _cell_para(header_row.cells[col_idx], align="center")
        _ar_run(para, header_ar, size=10, bold=True)

    for row_idx, incident in enumerate(incidents, start=1):
        row = table.rows[row_idx]
        for col_idx, (_header_ar, width_mm, getter) in enumerate(columns):
            _set_row_col_width(row, col_idx, width_mm)
            para = _cell_para(row.cells[col_idx], align="right")
            _ar_run(para, getter(incident), size=9)

    return table


class SeasonalWordAdapter:
    """
    Adapter for converting seasonal report data into Word documents.
    
    Supports both doctor and worker reports with consistent formatting.
    """
    
    @staticmethod
    def generate_doctor_seasonal_word(data: Dict[str, Any]) -> bytes:
        """
        Generate Word document for doctor seasonal report.
        
        Args:
            data: Doctor seasonal report data from D-B6
                  Structure: {
                      'doctor_identity': {...},
                      'period': {...},
                      'metrics': {...},
                      'performance': {...},
                      'category_breakdown': [...],
                      'monthly_trend': [...],
                      'incidents_summary': {...}
                  }
        
        Returns:
            bytes: Word document content
        """
        doc = Document()
        
        # Document setup - A4 Portrait
        section = doc.sections[0]
        section.page_height = Mm(297)
        section.page_width = Mm(210)
        section.orientation = WD_ORIENT.PORTRAIT
        section.left_margin = Mm(20)
        section.right_margin = Mm(20)
        section.top_margin = Mm(20)
        section.bottom_margin = Mm(20)
        
        # Set default font
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Arial'
        font.size = Pt(11)
        
        # ============================================================
        # TITLE (Changed to All Time Doctor Report)
        # ============================================================
        title = doc.add_heading('All Time Doctor Report', level=1)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # ============================================================
        # DOCTOR IDENTITY
        # ============================================================
        doc.add_heading('Doctor Information', level=2)
        identity = data.get('doctor_identity', {})
        
        identity_table = doc.add_table(rows=3, cols=2)
        identity_table.style = 'Light Grid Accent 1'
        
        identity_table.cell(0, 0).text = 'Doctor ID'
        identity_table.cell(0, 1).text = str(identity.get('id', 'N/A'))
        
        identity_table.cell(1, 0).text = 'Name (Arabic)'
        identity_table.cell(1, 1).text = identity.get('name_ar', 'N/A')
        
        identity_table.cell(2, 0).text = 'Name (English)'
        identity_table.cell(2, 1).text = identity.get('name_en', 'N/A')
        
        doc.add_paragraph()  # Spacing
        
        # ============================================================
        # REPORTING PERIOD
        # ============================================================
        period = data.get('period', {})
        doc.add_heading('Reporting Period', level=2)
        
        period_table = doc.add_table(rows=3, cols=2)
        period_table.style = 'Light Grid Accent 1'
        
        period_table.cell(0, 0).text = 'Season'
        period_table.cell(0, 1).text = period.get('season_name', 'N/A')
        
        period_table.cell(1, 0).text = 'Start Date'
        period_table.cell(1, 1).text = period.get('season_start', 'N/A')
        
        period_table.cell(2, 0).text = 'End Date'
        period_table.cell(2, 1).text = period.get('season_end', 'N/A')
        
        doc.add_paragraph()
        
        # ============================================================
        # (REMOVED) PERFORMANCE SCORE SECTION
        # ============================================================
        # This section intentionally left blank as per requirements.
        doc.add_paragraph()
        
        # ============================================================
        # METRICS SUMMARY
        # ============================================================
        metrics = data.get('metrics', {})
        doc.add_heading('Performance Metrics', level=2)
        
        metrics_table = doc.add_table(rows=13, cols=2)
        metrics_table.style = 'Light Grid Accent 1'
        
        # Incident metrics
        metrics_table.cell(0, 0).text = 'Total Incidents'
        metrics_table.cell(0, 1).text = str(metrics.get('total_incidents', 0))
        
        metrics_table.cell(1, 0).text = 'High Severity'
        metrics_table.cell(1, 1).text = str(metrics.get('high_severity', 0))
        
        metrics_table.cell(2, 0).text = 'Medium Severity'
        metrics_table.cell(2, 1).text = str(metrics.get('medium_severity', 0))
        
        metrics_table.cell(3, 0).text = 'Low Severity'
        metrics_table.cell(3, 1).text = str(metrics.get('low_severity', 0))
        
        # Intent classification
        metrics_table.cell(4, 0).text = 'Good Feedback (Notice/تنويه)'
        metrics_table.cell(4, 1).text = str(metrics.get('good_feedback_count', 0))
        
        metrics_table.cell(5, 0).text = 'Bad Feedback (Critique/نقد)'
        metrics_table.cell(5, 1).text = str(metrics.get('bad_feedback_count', 0))
        
        metrics_table.cell(6, 0).text = 'Neutral Feedback'
        metrics_table.cell(6, 1).text = str(metrics.get('neutral_feedback_count', 0))
        
        # Action metrics
        metrics_table.cell(7, 0).text = 'Total Actions'
        metrics_table.cell(7, 1).text = str(metrics.get('total_actions', 0))
        
        metrics_table.cell(8, 0).text = 'Overdue Actions'
        metrics_table.cell(8, 1).text = str(metrics.get('overdue_actions', 0))
        
        metrics_table.cell(9, 0).text = 'Completed Actions'
        metrics_table.cell(9, 1).text = str(metrics.get('completed_actions', 0))
        
        # Explanation metrics
        metrics_table.cell(10, 0).text = 'Accepted Explanations'
        metrics_table.cell(10, 1).text = str(metrics.get('accepted_explanations', 0))
        
        metrics_table.cell(11, 0).text = 'Rejected Explanations'
        metrics_table.cell(11, 1).text = str(metrics.get('rejected_explanations', 0))
        
        # Calculated metrics
        total_explanations = metrics.get('accepted_explanations', 0) + metrics.get('rejected_explanations', 0)
        acceptance_rate = (metrics.get('accepted_explanations', 0) / total_explanations * 100) if total_explanations > 0 else 0
        
        metrics_table.cell(12, 0).text = 'Explanation Acceptance Rate'
        metrics_table.cell(12, 1).text = f"{acceptance_rate:.1f}%"
        
        doc.add_paragraph()
        
        # ============================================================
        # DETAILED INCIDENTS TABLE (Arabic, RTL, fixed widths — shared
        # layout with the worker/patient reports; see _build_incidents_table)
        # ============================================================
        incidents_heading = doc.add_heading('سجل الحالات', level=2)
        for r in incidents_heading.runs:
            r.font.name = 'Traditional Arabic'
            r._element.rPr.rFonts.set(qn('w:cs'), 'Traditional Arabic')
        incidents = data.get('incidents', [])
        if incidents:
            _build_incidents_table(doc, incidents, [
                ('التاريخ', 14, lambda inc: str(inc.get('Date') or 'N/A')),
                ('رقم الحالة', 12, lambda inc: str(inc.get('RecordID') or 'N/A')),
                ('اسم المريض', 24, lambda inc: str(inc.get('PatientName') or 'N/A')[:40]),
                ('التصنيف', 18, lambda inc: str(inc.get('Category') or 'غير محدد')),
                ('الخطورة', 14, lambda inc: str(inc.get('Severity') or 'غير محدد')),
                ('الحالة', 14, lambda inc: str(inc.get('Status') or 'غير محدد')),
                ('علم أحمر', 14, lambda inc: 'نعم' if inc.get('RedFlag') == 'Yes' else 'لا'),
                ('نص الشكوى', 60, lambda inc: _truncate_for_fit(str(inc.get('Description') or 'لا يوجد نص'), NARRATIVE_MAX_CHARS)),
            ])
        else:
            # Only show the zero-complaint message if no incidents
            zero_para = doc.add_paragraph()
            zero_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _ar_run(zero_para, 'لا توجد حالات مسجلة خلال هذه الفترة', size=12, bold=True, color='008000')
            praise_para = doc.add_paragraph()
            praise_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _ar_run(praise_para, 'سجل نظيف — أداء ممتاز', size=11, color='646464')
        doc.add_paragraph()
        
        # ============================================================
        # FOOTER NOTE
        # ============================================================
        doc.add_paragraph()
        footer_note = doc.add_paragraph('Generated by Patient Feedback System - Seasonal Report Module')
        footer_note.alignment = WD_ALIGN_PARAGRAPH.CENTER
        footer_note.runs[0].font.size = Pt(9)
        footer_note.runs[0].font.color.rgb = RGBColor(128, 128, 128)
        
        # Save to bytes
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer.read()
    
    @staticmethod
    def generate_worker_seasonal_word(data: Dict[str, Any]) -> bytes:
        """
        Generate Word document for worker seasonal report.
        
        Args:
            data: Worker seasonal report data from D-B7
                  Structure: {
                      'worker_identity': {...},
                      'period': {...},
                      'metrics': {...},
                      'performance': {...}
                  }
        
        Returns:
            bytes: Word document content
        """
        doc = Document()
        
        # Document setup - A4 Portrait
        section = doc.sections[0]
        section.page_height = Mm(297)
        section.page_width = Mm(210)
        section.orientation = WD_ORIENT.PORTRAIT
        section.left_margin = Mm(20)
        section.right_margin = Mm(20)
        section.top_margin = Mm(20)
        section.bottom_margin = Mm(20)
        
        # Set default font
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Arial'
        font.size = Pt(11)
        
        # ============================================================
        # TITLE
        # ============================================================
        title = doc.add_heading('Worker Seasonal Performance Report', level=1)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # ============================================================
        # WORKER IDENTITY
        # ============================================================
        doc.add_heading('Worker Information', level=2)
        identity = data.get('worker_identity', {})
        
        identity_table = doc.add_table(rows=4, cols=2)
        identity_table.style = 'Light Grid Accent 1'
        
        identity_table.cell(0, 0).text = 'Employee ID'
        identity_table.cell(0, 1).text = str(identity.get('employee_id', 'N/A'))
        
        identity_table.cell(1, 0).text = 'Full Name'
        identity_table.cell(1, 1).text = identity.get('full_name', 'N/A')
        
        identity_table.cell(2, 0).text = 'Department'
        identity_table.cell(2, 1).text = identity.get('department', 'N/A')
        
        identity_table.cell(3, 0).text = 'Section'
        identity_table.cell(3, 1).text = identity.get('section', 'N/A')
        
        doc.add_paragraph()  # Spacing
        
        # ============================================================
        # REPORTING PERIOD
        # ============================================================
        period = data.get('period', {})
        doc.add_heading('Reporting Period', level=2)
        
        period_table = doc.add_table(rows=3, cols=2)
        period_table.style = 'Light Grid Accent 1'
        
        period_table.cell(0, 0).text = 'Season'
        period_table.cell(0, 1).text = period.get('season_name', 'N/A')
        
        period_table.cell(1, 0).text = 'Start Date'
        period_table.cell(1, 1).text = period.get('season_start', 'N/A')
        
        period_table.cell(2, 0).text = 'End Date'
        period_table.cell(2, 1).text = period.get('season_end', 'N/A')
        
        doc.add_paragraph()
        
        # ============================================================
        # PERFORMANCE SCORE
        # ============================================================
        performance = data.get('performance', {})
        doc.add_heading('Performance Score', level=2)
        
        score_para = doc.add_paragraph()
        score_run = score_para.add_run(f"Score: {performance.get('score', 0)}/100")
        score_run.bold = True
        score_run.font.size = Pt(14)
        
        # Color code based on praise level
        praise_level = performance.get('praise_level', 'average')
        if praise_level == 'excellent':
            score_run.font.color.rgb = RGBColor(0, 128, 0)  # Green
        elif praise_level == 'good':
            score_run.font.color.rgb = RGBColor(0, 100, 200)  # Blue
        elif praise_level == 'average':
            score_run.font.color.rgb = RGBColor(200, 100, 0)  # Orange
        else:
            score_run.font.color.rgb = RGBColor(200, 0, 0)  # Red
        
        doc.add_paragraph(f"Praise Level: {praise_level.upper()}")
        doc.add_paragraph(f"Risk Level: {performance.get('risk_level', 'N/A').upper()}")
        
        # Flags
        flags = performance.get('flags', [])
        if flags:
            doc.add_paragraph('Flags:')
            for flag in flags:
                doc.add_paragraph(f'  • {flag}', style='List Bullet')
        
        doc.add_paragraph()
        
        # ============================================================
        # METRICS SUMMARY
        # ============================================================
        metrics = data.get('metrics', {})
        doc.add_heading('Performance Metrics', level=2)
        
        metrics_table = doc.add_table(rows=13, cols=2)
        metrics_table.style = 'Light Grid Accent 1'
        
        # Incident metrics
        metrics_table.cell(0, 0).text = 'Total Incidents'
        metrics_table.cell(0, 1).text = str(metrics.get('total_incidents', 0))
        
        # Severity breakdown
        metrics_table.cell(1, 0).text = 'High Severity'
        metrics_table.cell(1, 1).text = str(metrics.get('high_severity', 0))
        
        metrics_table.cell(2, 0).text = 'Medium Severity'
        metrics_table.cell(2, 1).text = str(metrics.get('medium_severity', 0))
        
        metrics_table.cell(3, 0).text = 'Low Severity'
        metrics_table.cell(3, 1).text = str(metrics.get('low_severity', 0))
        
        # Intent classification
        metrics_table.cell(4, 0).text = 'Good Feedback (Notice/تنويه)'
        metrics_table.cell(4, 1).text = str(metrics.get('good_feedback_count', 0))
        
        metrics_table.cell(5, 0).text = 'Bad Feedback (Critique/نقد)'
        metrics_table.cell(5, 1).text = str(metrics.get('bad_feedback_count', 0))
        
        metrics_table.cell(6, 0).text = 'Neutral Feedback'
        metrics_table.cell(6, 1).text = str(metrics.get('neutral_feedback_count', 0))
        
        # Action metrics
        metrics_table.cell(7, 0).text = 'Total Action Items'
        metrics_table.cell(7, 1).text = str(metrics.get('total_action_items', 0))
        
        metrics_table.cell(8, 0).text = 'Overdue Actions'
        metrics_table.cell(8, 1).text = str(metrics.get('overdue_action_items', 0))
        
        metrics_table.cell(9, 0).text = 'Completed Actions'
        metrics_table.cell(9, 1).text = str(metrics.get('completed_action_items', 0))
        
        # Explanation metrics
        metrics_table.cell(10, 0).text = 'Accepted Explanations'
        metrics_table.cell(10, 1).text = str(metrics.get('explanation_accepted_count', 0))
        
        metrics_table.cell(11, 0).text = 'Rejected Explanations'
        metrics_table.cell(11, 1).text = str(metrics.get('explanation_rejected_count', 0))
        
        # Calculated metrics
        total_actions = metrics.get('total_action_items', 0)
        completion_rate = (metrics.get('completed_action_items', 0) / total_actions * 100) if total_actions > 0 else 0
        
        metrics_table.cell(12, 0).text = 'Action Completion Rate'
        metrics_table.cell(12, 1).text = f"{completion_rate:.1f}%"
        
        doc.add_paragraph()
        
        # ============================================================
        # DETAILED INCIDENTS TABLE
        # ============================================================
        incidents_heading = doc.add_heading('سجل الحالات', level=2)
        for r in incidents_heading.runs:
            r.font.name = 'Traditional Arabic'
            r._element.rPr.rFonts.set(qn('w:cs'), 'Traditional Arabic')

        incidents = data.get('incidents_details', [])
        if incidents:
            sentiment_labels = {'good': 'جيد ✓', 'bad': 'سيء ✗', 'neutral': 'محايد ―'}
            _build_incidents_table(doc, incidents, [
                ('التاريخ', 13, lambda inc: str(inc.get('date') or 'N/A')),
                ('رقم الحالة', 11, lambda inc: str(inc.get('id') or 'N/A')),
                ('اسم المريض', 20, lambda inc: str(inc.get('patient_name') or 'N/A')[:40]),
                ('التصنيف', 16, lambda inc: str(inc.get('category') or 'غير محدد')),
                ('الخطورة', 13, lambda inc: str(inc.get('severity') or 'غير محدد')),
                ('الحالة', 13, lambda inc: str(inc.get('status') or 'غير محدد')),
                ('علم أحمر', 13, lambda inc: 'نعم' if inc.get('RedFlag') == 'Yes' else 'لا'),
                ('تصنيف الشعور', 16, lambda inc: sentiment_labels.get(str(inc.get('classification') or '').lower(), 'محايد ―')),
                ('نص الشكوى', 55, lambda inc: _truncate_for_fit(str(inc.get('Description') or 'لا يوجد نص'), NARRATIVE_MAX_CHARS)),
            ])
        else:
            # Elegant zero-complaint message
            zero_para = doc.add_paragraph()
            zero_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _ar_run(zero_para, 'لا توجد حالات مسجلة خلال هذه الفترة', size=12, bold=True, color='008000')
            praise_para = doc.add_paragraph()
            praise_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _ar_run(praise_para, 'سجل نظيف — أداء ممتاز', size=11, color='646464')
        doc.add_paragraph()
        
        # ============================================================
        # FOOTER NOTE
        # ============================================================
        doc.add_paragraph()
        footer_note = doc.add_paragraph('Generated by Patient Feedback System - Seasonal Report Module')
        footer_note.alignment = WD_ALIGN_PARAGRAPH.CENTER
        footer_note.runs[0].font.size = Pt(9)
        footer_note.runs[0].font.color.rgb = RGBColor(128, 128, 128)
        
        # Save to bytes
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer.read()
    
    @staticmethod
    def generate_patient_word_report(data: Dict[str, Any]) -> bytes:
        """
        Generate Word document for patient history report.
        
        Args:
            data: Patient export data structure: {
                'patient': {patient profile dict},
                'incidents': [list of incident dicts],
                'export_date': ISO timestamp
            }
        
        Returns:
            bytes: Word document content in Arabic with hospital branding
        """
        doc = Document()

        # Document setup - A4 Portrait
        section = doc.sections[0]
        section.page_height = Mm(297)
        section.page_width = Mm(210)
        section.orientation = WD_ORIENT.PORTRAIT
        section.left_margin = Mm(20)
        section.right_margin = Mm(20)
        section.top_margin = Mm(20)
        section.bottom_margin = Mm(20)

        # Set default font
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Calibri'
        font.size = Pt(11)

        # ============================================================
        # LOGO HEADER
        # ============================================================
        try:
            logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'logo.png')
            if os.path.exists(logo_path):
                section.header_distance = Inches(0.1)
                header_section = section.header
                header_para = header_section.paragraphs[0]
                header_para.clear()
                header_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                run = header_para.add_run()
                run.add_picture(logo_path, width=Inches(0.9))
        except Exception as e:
            print(f"[PATIENT_HISTORY] Could not add logo: {e}")

        # ============================================================
        # TITLE (Arabic)
        # ============================================================
        title = doc.add_heading('تقرير تاريخ المريض', level=1)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_run = title.runs[0]
        title_run.font.size = Pt(16)
        title_run.font.name = 'Traditional Arabic'
        title_run.bold = True
        title_run._element.rPr.rFonts.set(qn('w:cs'), 'Traditional Arabic')

        # Subtitle
        subtitle = doc.add_paragraph('نظام ملاحظات المرضى')
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle.runs[0].font.size = Pt(12)
        subtitle.runs[0].font.name = 'Traditional Arabic'
        subtitle.runs[0].font.color.rgb = RGBColor(100, 100, 100)
        subtitle.runs[0]._element.rPr.rFonts.set(qn('w:cs'), 'Traditional Arabic')
        
        doc.add_paragraph()  # Spacing
        
        # ============================================================
        # PATIENT INFORMATION (Arabic)
        # ============================================================
        patient = data.get('patient', {})
        
        info_heading = doc.add_heading('معلومات المريض', level=2)
        for r in info_heading.runs:
            r.font.name = 'Traditional Arabic'
            r._element.rPr.rFonts.set(qn('w:cs'), 'Traditional Arabic')

        patient_table = doc.add_table(rows=4, cols=2)
        patient_table.style = 'Light Grid Accent 1'
        
        patient_table.cell(0, 0).text = 'رقم المريض'
        patient_table.cell(0, 1).text = str(patient.get('patient_id', 'N/A'))
        
        patient_table.cell(1, 0).text = 'رقم الملف الطبي'
        patient_table.cell(1, 1).text = str(patient.get('mrn', 'غير متوفر'))
        
        patient_table.cell(2, 0).text = 'الاسم الكامل'
        patient_table.cell(2, 1).text = str(patient.get('full_name', 'غير متوفر'))
        
        patient_table.cell(3, 0).text = 'عدد الشكاوى'
        patient_table.cell(3, 1).text = str(len(data.get('incidents', [])))
        
        doc.add_paragraph()  # Spacing
        
        # ============================================================
        # COMPLAINTS TABLE (Arabic)
        # ============================================================
        incidents = data.get('incidents', [])
        
        complaints_heading = doc.add_heading('سجل الشكاوى', level=2)
        for r in complaints_heading.runs:
            r.font.name = 'Traditional Arabic'
            r._element.rPr.rFonts.set(qn('w:cs'), 'Traditional Arabic')

        if incidents:
            _build_incidents_table(doc, incidents, [
                ('التاريخ', 16, lambda inc: str(inc.get('Date') or 'N/A')),
                ('رقم الحالة', 14, lambda inc: str(inc.get('RecordID') or 'N/A')),
                ('القسم', 20, lambda inc: str(inc.get('Department') or 'غير محدد')),
                ('التصنيف', 20, lambda inc: str(inc.get('Category') or 'غير محدد')),
                ('الخطورة', 16, lambda inc: str(inc.get('Severity') or 'غير محدد')),
                ('الحالة', 16, lambda inc: str(inc.get('Status') or 'مفتوح')),
                ('نص الشكوى', 68, lambda inc: _truncate_for_fit(str(inc.get('ComplaintText') or 'لا يوجد نص'), NARRATIVE_MAX_CHARS)),
            ])
        else:
            no_complaints = doc.add_paragraph()
            _ar_run(no_complaints, 'لا توجد شكاوى مسجلة لهذا المريض.', size=11)

        doc.add_paragraph()

        # ============================================================
        # EXPORT METADATA (Arabic)
        # ============================================================
        export_date = data.get('export_date', 'N/A')

        metadata_para = doc.add_paragraph(f'{export_date} :تاريخ التصدير')
        metadata_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        metadata_para.runs[0].font.size = Pt(9)
        metadata_para.runs[0].font.name = 'Traditional Arabic'
        metadata_para.runs[0].font.color.rgb = RGBColor(128, 128, 128)
        metadata_para.runs[0]._element.rPr.rFonts.set(qn('w:cs'), 'Traditional Arabic')

        # ============================================================
        # FOOTER NOTE
        # ============================================================
        doc.add_paragraph()
        footer_note = doc.add_paragraph('تم إنشاؤه بواسطة نظام ملاحظات المرضى')
        footer_note.alignment = WD_ALIGN_PARAGRAPH.CENTER
        footer_note.runs[0].font.size = Pt(9)
        footer_note.runs[0].font.name = 'Traditional Arabic'
        footer_note.runs[0].font.color.rgb = RGBColor(128, 128, 128)
        footer_note.runs[0]._element.rPr.rFonts.set(qn('w:cs'), 'Traditional Arabic')
        
        # Save to bytes
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer.read()
