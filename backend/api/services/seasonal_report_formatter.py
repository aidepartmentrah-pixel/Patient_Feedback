"""
Seasonal Report Formatter
Generates professional Word documents for seasonal reports with hierarchical classification tables.

Features:
- A4 Landscape orientation with Arabic header and logo
- RTL table layout (right-to-left for Arabic readers)
- Hierarchical table structure with merged cells (Domain > Category > Sub-Category > Classification)
- Bilingual classification names (Arabic + English)
- Domain-level grouping with subtotals
- Professional styling with colors, borders, and proper alignment
- Clear Severity section (Low/Medium/High)
- Clear Prevention Action section (Yes/No)
- Policy targets footer in Arabic
"""

from typing import Dict, Any, List, Tuple, Optional
import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Mm, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.table import _Cell
import io
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import patches
import numpy as np


def set_cell_shading(cell: _Cell, color: str):
    """
    Set background color for a table cell.
    
    Args:
        cell: The cell to shade
        color: Hex color code (e.g., 'D9E2F3')
    """
    shading_elm = OxmlElement('w:shd')
    shading_elm.set(qn('w:fill'), color)
    cell._tc.get_or_add_tcPr().append(shading_elm)


def center_cell_content(cell: _Cell):
    """
    Center cell content both horizontally and vertically.
    
    Args:
        cell: The cell to center
    """
    # Horizontal centering
    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Vertical centering
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def merge_cells_vertically(table, col: int, start_row: int, end_row: int):
    """
    Merge cells vertically in a table column.
    
    Args:
        table: The table object
        col: Column index
        start_row: Starting row index
        end_row: Ending row index (inclusive)
    """
    if start_row >= end_row:
        return
    
    # Get the first cell
    first_cell = table.rows[start_row].cells[col]
    
    # Merge with subsequent cells
    for row_idx in range(start_row + 1, end_row + 1):
        cell_to_merge = table.rows[row_idx].cells[col]
        first_cell.merge(cell_to_merge)


def generate_seasonal_word_report(
    seasonal_data: Dict[str, Any],
    language: str = "en"
) -> bytes:
    """
    Generate a professional Word document for a seasonal report.
    
    Creates a hierarchical table structure matching the user's specification:
    - A4 Landscape with Arabic header and logo
    - RTL table layout for Arabic readers
    - Domain level (e.g., "Clinical" n=6) with merged cells
    - Category level (e.g., "Quality of Care" n=4)
    - Sub-Category level (e.g., "Examination & Monitoring" n=2)
    - Classification level with Arabic + English names
    - Clear Severity section (Low/Medium/High)
    - Clear Prevention Action section (Yes/No)
    - Policy targets footer in Arabic
    
    Args:
        seasonal_data: Seasonal report data from orchestrator
        language: Language for the report (en or ar)
    
    Returns:
        Bytes of the generated Word document
    """
    # Debug: Log what we received
    print(f"\n[FORMATTER] Received seasonal_data type: {type(seasonal_data)}")
    print(f"[FORMATTER] seasonal_data preview: {str(seasonal_data)[:200]}")
    
    # Validate input
    if not isinstance(seasonal_data, dict):
        raise TypeError(f"Expected dict, got {type(seasonal_data)}. Data: {seasonal_data}")
    
    # Utility functions
    def _safe(v):
        """Convert dimension values to int (python-docx requirement)"""
        return int(v)
    
    doc = Document()
    
    # ============================================================
    # DOCUMENT SETUP - A4 LANDSCAPE
    # ============================================================
    section = doc.sections[0]
    section.page_height = _safe(Mm(210))  # A4 width becomes height in landscape
    section.page_width = _safe(Mm(297))   # A4 height becomes width in landscape
    section.orientation = WD_ORIENT.LANDSCAPE
    section.left_margin = _safe(Mm(15))
    section.right_margin = _safe(Mm(15))
    section.top_margin = _safe(Mm(15))
    section.bottom_margin = _safe(Mm(15))
    
    # Set default font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Traditional Arabic'
    font.size = Pt(11)
    
    # Extract data
    header = seasonal_data.get("header", {})
    classification_stats = seasonal_data.get("classification_stats", [])
    policy_snapshot = seasonal_data.get("policy_snapshot", {})
    
    # ============================================================
    # HEADER - LOGO (TOP RIGHT)
    # ============================================================
    try:
        logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'logo.png')
        if os.path.exists(logo_path):
            # Make header compact
            section.header_distance = Inches(0.1)
            header_section = section.header
            
            # Use only one paragraph and clear it
            header_para = header_section.paragraphs[0]
            header_para.clear()
            header_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            
            run = header_para.add_run()
            run.add_picture(logo_path, width=Inches(0.9))
    except Exception as e:
        print(f"[FORMATTER] Could not add logo: {e}")
        pass
    
    # ============================================================
    # TITLE SECTION (ARABIC)
    # ============================================================
    
    # Main Title (big, bold, centered)
    title_para = doc.add_paragraph()
    title_run = title_para.add_run("نموذج التقرير الموسمي لفرص التحسين والإجراءات التصحيحية")
    title_run.font.size = int(Pt(21))
    title_run.font.bold = True
    title_run.font.name = 'Traditional Arabic'
    title_run.italic = False
    title_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_para.space_after = int(Pt(3))
    
    # Subtitle (smaller, centered)
    subtitle_para = doc.add_paragraph()
    subtitle_run = subtitle_para.add_run("(إصدار رسمي — للاستخدام الإداري والجودة)")
    subtitle_run.font.size = int(Pt(14))
    subtitle_run.font.name = 'Traditional Arabic'
    subtitle_run.italic = False
    subtitle_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    subtitle_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_para.space_after = int(Pt(6))
    
    # ============================================================
    # PERIOD & ORGANIZATION INFO
    # ============================================================
    
    # Period line (centered, bold)
    period_str = header.get('period', 'N/A')
    # Convert period to Arabic
    period_arabic = period_str
    if 'Q1' in period_str:
        period_arabic = period_str.replace('Q1', 'الربع الأول')
    elif 'Q2' in period_str:
        period_arabic = period_str.replace('Q2', 'الربع الثاني')
    elif 'Q3' in period_str:
        period_arabic = period_str.replace('Q3', 'الربع الثالث')
    elif 'Q4' in period_str:
        period_arabic = period_str.replace('Q4', 'الربع الرابع')
    elif 'Trim1' in period_str:
        period_arabic = period_str.replace('Trim1', 'الفصل الأول')
    elif 'Trim2' in period_str:
        period_arabic = period_str.replace('Trim2', 'الفصل الثاني')
    elif 'Trim3' in period_str:
        period_arabic = period_str.replace('Trim3', 'الفصل الثالث')
    
    period_para = doc.add_paragraph()
    period_run = period_para.add_run(f"الموسم المعني: {period_arabic}")
    period_run.font.size = int(Pt(12))
    period_run.font.bold = True
    period_run.font.name = 'Traditional Arabic'
    period_run.italic = False
    period_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    period_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    period_para.space_after = int(Pt(3))
    
    # Organization info table (3-column, borderless)
    org_table = doc.add_table(rows=1, cols=3)
    
    # Remove all table borders
    org_tbl = org_table._element
    org_tblPr = org_tbl.tblPr
    if org_tblPr is None:
        org_tblPr = OxmlElement('w:tblPr')
        org_tbl.insert(0, org_tblPr)
    
    org_tblBorders = OxmlElement('w:tblBorders')
    for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        border_elem = OxmlElement(f'w:{border_name}')
        border_elem.set(qn('w:val'), 'nil')
        org_tblBorders.append(border_elem)
    org_tblPr.append(org_tblBorders)
    
    # Center the table
    org_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    org_tblJc = OxmlElement('w:jc')
    org_tblJc.set(qn('w:val'), 'center')
    org_tblPr.append(org_tblJc)
    
    # Set column widths
    section = doc.sections[0]
    usable_width = section.page_width - section.left_margin - section.right_margin
    target_width = int(usable_width * 0.7)
    col_width = int(target_width / 3)
    
    for i in range(3):
        org_table.columns[i].width = col_width
    
    # Determine organization type name
    orgunit_type = header.get('orgunit_type', 0)
    type_names = {
        0: "المستشفى",
        1: "الإدارة",
        2: "الدائرة",
        3: "القسم"
    }
    type_name = type_names.get(orgunit_type, "الوحدة التنظيمية")
    
    # Fill cells with data
    org_cells = org_table.rows[0].cells
    org_data = [
        ("المستشفى: ", "مستشفى الرّسول الأعظم"),
        ("الوحدة التنظيمية: ", header.get('orgunit_name', 'N/A')),
        ("النوع: ", type_name)
    ]
    
    for i, (label, value) in enumerate(org_data):
        cell = org_cells[i]
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.right_to_left = True
        paragraph.space_after = int(Pt(6))
        
        # Bold label
        label_run = paragraph.add_run(label)
        label_run.font.bold = True
        label_run.font.size = int(Pt(15))
        label_run.font.name = 'Traditional Arabic'
        label_run.italic = False
        label_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        
        # Normal value
        value_run = paragraph.add_run(str(value))
        value_run.font.bold = False
        value_run.font.size = int(Pt(15))
        value_run.font.name = 'Traditional Arabic'
        value_run.italic = False
        value_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    # Visual separator line
    separator_para = doc.add_paragraph()
    separator_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    try:
        pPr = separator_para._element.get_or_add_pPr()
        pBdr = OxmlElement('w:pBdr')
        bottom = OxmlElement('w:bottom')
        bottom.set(qn('w:val'), 'single')
        bottom.set(qn('w:sz'), '12')
        bottom.set(qn('w:space'), '1')
        bottom.set(qn('w:color'), '4472C4')
        pBdr.append(bottom)
        pPr.append(pBdr)
    except:
        pass
    
    doc.add_paragraph()  # Spacer
    
    # ============================================================
    # HIERARCHICAL CLASSIFICATION TABLES (RTL LAYOUT - ONE PER DOMAIN)
    # ============================================================
    
    # Build hierarchical structure from flat classification_stats
    hierarchy = _build_hierarchy(classification_stats)
    
    # Create separate table for each domain
    _create_hierarchical_tables_by_domain_rtl(doc, hierarchy, language)
    
    # ============================================================
    # POLICY COMPLIANCE EVALUATION (ARABIC RTL)
    # ============================================================
    
    doc.add_paragraph()  # Spacer
    _add_policy_compliance_section(doc, header, policy_snapshot)
    doc.add_paragraph()  # Spacer
    
    # ============================================================
    # SUMMARY STATISTICS (AT THE END, IN ARABIC)
    # ============================================================
    
    # Domain totals
    clinical_count = header.get('clinical_domain_count', 0)
    management_count = header.get('management_domain_count', 0)
    relational_count = header.get('relational_domain_count', 0)
    total_cases = header.get('total_cases', 0)
    
    # Severity counts
    low_count = header.get('low_severity_count', 0)
    medium_count = header.get('medium_severity_count', 0)
    high_count = header.get('high_severity_count', 0)
    
    # Add heading in Arabic
    summary_heading = doc.add_paragraph()
    summary_heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    summary_heading.paragraph_format.right_to_left = True
    
    sh_run = summary_heading.add_run("📊 إحصائيات ملخصة (Summary Statistics)")
    sh_run.font.bold = True
    sh_run.font.size = Pt(14)
    sh_run.font.name = 'Traditional Arabic'
    sh_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    doc.add_paragraph()
    
    summary_table = doc.add_table(rows=5, cols=3)
    summary_table.style = 'Table Grid'
    
    # Headers in Arabic (RTL order: النسبة المئوية, العدد, الفئة)
    summary_table.rows[0].cells[0].text = "النسبة المئوية"
    summary_table.rows[0].cells[1].text = "العدد"
    summary_table.rows[0].cells[2].text = "الفئة"
    for cell in summary_table.rows[0].cells:
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(11)
        cell.paragraphs[0].runs[0].font.name = 'Traditional Arabic'
        cell.paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        center_cell_content(cell)
        set_cell_shading(cell, '4472C4')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    # Domain breakdown in Arabic
    domain_data = [
        ("Clinical (الجوانب السريرية)", clinical_count),
        ("Management (الجوانب الإدارية)", management_count),
        ("Relational (الجوانب العلائقية)", relational_count)
    ]
    
    for idx, (label, count) in enumerate(domain_data, start=1):
        row = summary_table.rows[idx]
        
        # RTL order: percentage (col 0), count (col 1), label (col 2)
        percentage = (count / total_cases * 100) if total_cases > 0 else 0
        row.cells[0].text = f"{percentage:.1f}%"
        center_cell_content(row.cells[0])
        
        row.cells[1].text = str(count)
        center_cell_content(row.cells[1])
        
        row.cells[2].text = label
        row.cells[2].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
        row.cells[2].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        row.cells[2].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    
    # Total row in Arabic (RTL order: percentage, count, label)
    total_row = summary_table.rows[4]
    
    total_row.cells[0].text = "100.0%"
    total_row.cells[0].paragraphs[0].runs[0].bold = True
    center_cell_content(total_row.cells[0])
    
    total_row.cells[1].text = str(total_cases)
    total_row.cells[1].paragraphs[0].runs[0].bold = True
    center_cell_content(total_row.cells[1])
    
    total_row.cells[2].text = "المجموع TOTAL"
    total_row.cells[2].paragraphs[0].runs[0].bold = True
    total_row.cells[2].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
    total_row.cells[2].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    total_row.cells[2].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    
    set_cell_shading(total_row.cells[0], 'D9E2F3')
    set_cell_shading(total_row.cells[1], 'D9E2F3')
    set_cell_shading(total_row.cells[2], 'D9E2F3')
    
    doc.add_paragraph()
    
    # Severity breakdown in Arabic
    severity_para = doc.add_paragraph()
    severity_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    severity_para.paragraph_format.right_to_left = True
    
    sev_bold = severity_para.add_run("تفصيل الشدة (Severity Breakdown): ")
    sev_bold.font.bold = True
    sev_bold.font.name = 'Traditional Arabic'
    sev_bold._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    sev_text = severity_para.add_run(f"منخفضة ({low_count}) • متوسطة ({medium_count}) • عالية ({high_count})")
    sev_text.font.name = 'Traditional Arabic'
    sev_text._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    # ============================================================
    # SAVE AND RETURN
    # ============================================================
    
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def _add_policy_compliance_section(doc, header: Dict[str, Any], policy_snapshot: Dict[str, Any]):
    """
    Add policy compliance evaluation section with enabled rules only (Arabic RTL).
    
    Design C: Arabic-First RTL table with rule status
    """
    import json
    
    # Add title
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    title_para.paragraph_format.right_to_left = True
    
    title_run = title_para.add_run("📊 تقييم الامتثال للسياسة (Policy Compliance Evaluation)")
    title_run.font.bold = True
    title_run.font.size = Pt(14)
    title_run.font.name = 'Traditional Arabic'
    title_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    doc.add_paragraph()  # Spacer
    
    # If no policy, show message
    if not policy_snapshot:
        no_policy_para = doc.add_paragraph()
        no_policy_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        no_policy_para.paragraph_format.right_to_left = True
        
        np_run = no_policy_para.add_run("لا توجد سياسة محددة لهذه الوحدة التنظيمية.")
        np_run.font.size = Pt(11)
        np_run.font.name = 'Traditional Arabic'
        np_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        return
    
    # Get enable flags (convert to bool to handle 0/1 from database)
    # Note: Keys are snake_case when coming from stored policy snapshot
    enable_domain_rule = bool(policy_snapshot.get('enable_high_severity_percentage_by_domain_rule', False))
    enable_low_rule = bool(policy_snapshot.get('enable_low_severity_repetition_rule', False))
    enable_medium_rule = bool(policy_snapshot.get('enable_medium_severity_repetition_rule', False))
    enable_high_rule = bool(policy_snapshot.get('enable_high_severity_percentage_rule', False))
    
    # Debug: Log enable flags
    print(f"[FORMATTER] Enable flags: domain={enable_domain_rule}, low={enable_low_rule}, medium={enable_medium_rule}, high={enable_high_rule}")
    print(f"[FORMATTER] Policy snapshot keys: {list(policy_snapshot.keys())}")
    
    # Get data from header
    total_cases = header.get('total_cases', 0)
    clinical_count = header.get('clinical_domain_count', 0)
    management_count = header.get('management_domain_count', 0)
    relational_count = header.get('relational_domain_count', 0)
    low_count = header.get('low_severity_count', 0)
    medium_count = header.get('medium_severity_count', 0)
    high_count = header.get('high_severity_count', 0)
    
    # Build list of enabled rules to display
    rules_to_display = []
    
    # Domain rules (percentages)
    if enable_domain_rule and total_cases > 0:
        clinical_limit = policy_snapshot.get('clinical_domain_limit', 0)
        clinical_percentage = round((clinical_count / total_cases) * 100, 1)
        if clinical_limit > 0:
            rules_to_display.append({
                'name_ar': 'المجال السريري',
                'name_en': 'Clinical Domain',
                'threshold': f"{clinical_limit}%",
                'actual': f"{clinical_percentage}%",
                'passed': clinical_percentage <= clinical_limit
            })
        
        management_limit = policy_snapshot.get('management_domain_limit', 0)
        management_percentage = round((management_count / total_cases) * 100, 1)
        if management_limit > 0:
            rules_to_display.append({
                'name_ar': 'المجال الإداري',
                'name_en': 'Management Domain',
                'threshold': f"{management_limit}%",
                'actual': f"{management_percentage}%",
                'passed': management_percentage <= management_limit
            })
        
        relational_limit = policy_snapshot.get('relational_domain_limit', 0)
        relational_percentage = round((relational_count / total_cases) * 100, 1)
        if relational_limit > 0:
            rules_to_display.append({
                'name_ar': 'المجال العلائقي',
                'name_en': 'Relational Domain',
                'threshold': f"{relational_limit}%",
                'actual': f"{relational_percentage}%",
                'passed': relational_percentage <= relational_limit
            })
    
    # Severity rules (absolute counts)
    if enable_low_rule:
        low_limit = policy_snapshot.get('low_severity_limit', 0)
        rules_to_display.append({
            'name_ar': 'الحالات منخفضة الخطورة',
            'name_en': 'Low Severity Cases',
            'threshold': str(low_limit),
            'actual': str(low_count),
            'passed': low_count <= low_limit
        })
    
    if enable_medium_rule:
        medium_limit = policy_snapshot.get('medium_severity_limit', 0)
        rules_to_display.append({
            'name_ar': 'الحالات متوسطة الخطورة',
            'name_en': 'Medium Severity Cases',
            'threshold': str(medium_limit),
            'actual': str(medium_count),
            'passed': medium_count <= medium_limit
        })
    
    if enable_high_rule:
        high_limit = policy_snapshot.get('high_severity_limit', 0)
        rules_to_display.append({
            'name_ar': 'الحالات عالية الخطورة',
            'name_en': 'High Severity Cases',
            'threshold': str(high_limit),
            'actual': str(high_count),
            'passed': high_count <= high_limit
        })
    
    # If no rules enabled, show message
    if not rules_to_display:
        no_rules_para = doc.add_paragraph()
        no_rules_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        no_rules_para.paragraph_format.right_to_left = True
        
        nr_run = no_rules_para.add_run("لا توجد قواعد مفعّلة في السياسة الحالية.")
        nr_run.font.size = Pt(11)
        nr_run.font.name = 'Traditional Arabic'
        nr_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        return
    
    # Create compliance table (Design C: Arabic RTL)
    # Columns: Status | Actual | Threshold | Rule Name (AR)
    compliance_table = doc.add_table(rows=len(rules_to_display) + 1, cols=4)
    compliance_table.style = 'Table Grid'
    
    # Enable RTL for table
    tbl = compliance_table._tbl
    tblPr = tbl.tblPr
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tbl.insert(0, tblPr)
    bidiVisual = OxmlElement('w:bidiVisual')
    tblPr.append(bidiVisual)
    
    # Header row (RTL: Rule | Threshold | Actual | Status)
    headers = ['القاعدة', 'الحد الأقصى', 'الفعلي', 'الحالة']
    for idx, header_text in enumerate(headers):
        cell = compliance_table.rows[0].cells[idx]
        cell.text = header_text
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(11)
        cell.paragraphs[0].runs[0].font.name = 'Traditional Arabic'
        cell.paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        center_cell_content(cell)
        set_cell_shading(cell, '4472C4')  # Dark blue
    
    # Data rows (RTL order: Rule | Threshold | Actual | Status)
    for idx, rule in enumerate(rules_to_display, start=1):
        row = compliance_table.rows[idx]
        
        # Column 0: Rule name (Arabic)
        rule_cell = row.cells[0]
        rule_cell.text = rule['name_ar']
        rule_cell.paragraphs[0].runs[0].font.size = Pt(10)
        rule_cell.paragraphs[0].runs[0].font.name = 'Traditional Arabic'
        rule_cell.paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        rule_cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        
        # Column 1: Threshold
        threshold_cell = row.cells[1]
        threshold_cell.text = rule['threshold']
        threshold_cell.paragraphs[0].runs[0].font.size = Pt(10)
        center_cell_content(threshold_cell)
        
        # Column 2: Actual value
        actual_cell = row.cells[2]
        actual_cell.text = rule['actual']
        actual_cell.paragraphs[0].runs[0].font.size = Pt(10)
        actual_cell.paragraphs[0].runs[0].bold = not rule['passed']  # Bold if failed
        center_cell_content(actual_cell)
        if not rule['passed']:
            set_cell_shading(actual_cell, 'FFE6E6')  # Very light red for failed
        
        # Column 3: Status (✓ or ✗)
        status_cell = row.cells[3]
        status_symbol = "✓" if rule['passed'] else "✗"
        status_cell.text = status_symbol
        status_cell.paragraphs[0].runs[0].font.size = Pt(14)
        status_cell.paragraphs[0].runs[0].bold = True
        center_cell_content(status_cell)
        
        # Color coding
        if rule['passed']:
            set_cell_shading(status_cell, 'C6EFCE')  # Light green
            status_cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0, 128, 0)  # Green text
        else:
            set_cell_shading(status_cell, 'FFC7CE')  # Light red
            status_cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(192, 0, 0)  # Red text
    
    # Overall compliance status
    doc.add_paragraph()  # Spacer
    
    violations = [r for r in rules_to_display if not r['passed']]
    is_compliant = len(violations) == 0
    
    status_para = doc.add_paragraph()
    status_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    status_para.paragraph_format.right_to_left = True
    
    if is_compliant:
        status_run = status_para.add_run("✓ الحالة العامة: مطابق للسياسة (COMPLIANT)")
        status_run.font.bold = True
        status_run.font.size = Pt(12)
        status_run.font.name = 'Traditional Arabic'
        status_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        status_run.font.color.rgb = RGBColor(0, 128, 0)  # Green
    else:
        status_run = status_para.add_run(f"✗ الحالة العامة: غير مطابق ({len(violations)} مخالفات) (NON-COMPLIANT)")
        status_run.font.bold = True
        status_run.font.size = Pt(12)
        status_run.font.name = 'Traditional Arabic'
        status_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        status_run.font.color.rgb = RGBColor(192, 0, 0)  # Red


def _build_hierarchy(classification_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build hierarchical structure from flat classification statistics.
    
    Returns structure:
    {
        'Clinical': {
            'total': 6,
            'categories': {
                'Quality of Care': {
                    'total': 4,
                    'subcategories': {
                        'Examination & Monitoring': {
                            'total': 2,
                            'classifications': [...]
                        }
                    }
                }
            }
        }
    }
    """
    hierarchy = defaultdict(lambda: {
        'total': 0,
        'categories': defaultdict(lambda: {
            'total': 0,
            'subcategories': defaultdict(lambda: {
                'total': 0,
                'classifications': []
            })
        })
    })
    
    for stat in classification_stats:
        domain_name = stat.get('domain_name', 'Unknown Domain')
        category_name = stat.get('category_name', 'Unknown Category')
        subcategory_name = stat.get('subcategory_name', 'Unknown Sub-Category')
        
        # Add to hierarchy
        hierarchy[domain_name]['total'] += stat.get('total_count', 0)
        hierarchy[domain_name]['categories'][category_name]['total'] += stat.get('total_count', 0)
        hierarchy[domain_name]['categories'][category_name]['subcategories'][subcategory_name]['total'] += stat.get('total_count', 0)
        hierarchy[domain_name]['categories'][category_name]['subcategories'][subcategory_name]['classifications'].append(stat)
    
    return dict(hierarchy)
def _build_hierarchy(classification_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build hierarchical structure from flat classification statistics.
    
    Returns structure:
    {
        'Clinical': {
            'total': 6,
            'categories': {
                'Quality of Care': {
                    'total': 4,
                    'subcategories': {
                        'Examination & Monitoring': {
                            'total': 2,
                            'classifications': [...]
                        }
                    }
                }
            }
        }
    }
    """
    hierarchy = defaultdict(lambda: {
        'total': 0,
        'categories': defaultdict(lambda: {
            'total': 0,
            'subcategories': defaultdict(lambda: {
                'total': 0,
                'classifications': []
            })
        })
    })
    
    for stat in classification_stats:
        domain_name = stat.get('domain_name', 'Unknown Domain')
        category_name = stat.get('category_name', 'Unknown Category')
        subcategory_name = stat.get('subcategory_name', 'Unknown Sub-Category')
        
        # Add to hierarchy
        hierarchy[domain_name]['total'] += stat.get('total_count', 0)
        hierarchy[domain_name]['categories'][category_name]['total'] += stat.get('total_count', 0)
        hierarchy[domain_name]['categories'][category_name]['subcategories'][subcategory_name]['total'] += stat.get('total_count', 0)
        hierarchy[domain_name]['categories'][category_name]['subcategories'][subcategory_name]['classifications'].append(stat)
    
    return dict(hierarchy)


def _create_hierarchical_tables_by_domain_rtl(doc: Document, hierarchy: Dict[str, Any], language: str):
    """
    Create separate hierarchical tables for each domain with RTL layout.
    Each domain gets its own table with proper spacing.
    """
    
    if not hierarchy or len(hierarchy) == 0:
        doc.add_paragraph("لا توجد بيانات متاحة (No classification data available).")
        return
    
    # Process each domain separately
    for domain_idx, (domain_name, domain_data) in enumerate(sorted(hierarchy.items())):
        # Add domain title
        if domain_idx > 0:
            doc.add_paragraph()  # Spacing between domains
        
        domain_title = doc.add_paragraph()
        domain_title.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        domain_title.paragraph_format.right_to_left = True
        
        title_run = domain_title.add_run(f'📂 {domain_name} (n={domain_data["total"]})')
        title_run.font.bold = True
        title_run.font.size = Pt(13)
        title_run.font.name = 'Traditional Arabic'
        title_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        title_run.font.color.rgb = RGBColor(68, 114, 196)
        
        # Calculate rows for this domain
        domain_rows = 2  # Header rows
        for category_data in domain_data['categories'].values():
            for subcategory_data in category_data['subcategories'].values():
                domain_rows += len(subcategory_data['classifications'])
        
        # Create table for this domain (11 columns)
        table = doc.add_table(rows=domain_rows, cols=11)
        table.style = 'Table Grid'
        
        # Set RTL table direction
        tbl = table._element
        tblPr = tbl.tblPr
        if tblPr is None:
            tblPr = OxmlElement('w:tblPr')
            tbl.insert(0, tblPr)
        bidiVisual = OxmlElement('w:bidiVisual')
        tblPr.append(bidiVisual)
        
        # Setup headers
        header_row1 = table.rows[0]
        header_row2 = table.rows[1]
        
        headers_main = [
            "Problem Category\nفئة المشكلة",
            "Sub-Category\nالفئة الفرعية",
            "التصنيف عربي\nClassification AR",
            "التصنيف إنجليزي\nClassification EN",
            "Total\nالمجموع",
            "الشدة Severity",
            "",
            "",
            "الإجراءات الوقائية\nPrevention",
            ""
        ]
        
        # Adjust: Removed Domain column, so now 10 columns effectively used
        # But table still has 11 cols - merge first col or skip it
        # Actually, let me keep it simpler - use all 11 cols but first is Category
        
        headers_main = [
            "Problem Category\nفئة المشكلة",
            "Sub-Category\nالفئة الفرعية",
            "التصنيف عربي\nClassification AR",
            "التصنيف إنجليزي\nClassification EN",
            "Total\nالمجموع",
            "الشدة Severity",
            "",
            "",
            "",
            "الإجراءات الوقائية\nPrevention",
            ""
        ]
        
        for idx, header_text in enumerate(headers_main):
            cell = header_row1.cells[idx]
            if header_text:
                cell.text = header_text
                cell.paragraphs[0].runs[0].bold = True
                cell.paragraphs[0].runs[0].font.size = Pt(9)
                cell.paragraphs[0].runs[0].font.name = 'Traditional Arabic'
                cell.paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
                cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
                center_cell_content(cell)
            set_cell_shading(cell, '4472C4')
        
        subheaders = [
            "",
            "",
            "",
            "",
            "",
            "LOW\nمنخفض",
            "MED\nمتوسط",
            "HIGH\nعالي",
            "",
            "YES\nنعم",
            "NO\nلا"
        ]
        
        for idx, subheader_text in enumerate(subheaders):
            cell = header_row2.cells[idx]
            if subheader_text:
                cell.text = subheader_text
                cell.paragraphs[0].runs[0].bold = True
                cell.paragraphs[0].runs[0].font.size = Pt(8)
                cell.paragraphs[0].runs[0].font.name = 'Traditional Arabic'
                cell.paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
                cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
                center_cell_content(cell)
            set_cell_shading(cell, '5B9BD5')
        
        # Merge headers
        try:
            for i in range(5):
                header_row1.cells[i].merge(header_row2.cells[i])
            header_row1.cells[5].merge(header_row1.cells[7])
            header_row1.cells[9].merge(header_row1.cells[10])
        except Exception as e:
            print(f"[FORMATTER] Warning: Could not merge headers: {e}")
        
        # Fill data rows for this domain
        current_row = 2
        for category_name, category_data in sorted(domain_data['categories'].items()):
            category_start_row = current_row
            category_total = category_data['total']
            
            for subcategory_name, subcategory_data in sorted(category_data['subcategories'].items()):
                subcategory_start_row = current_row
                subcategory_total = subcategory_data['total']
                
                for classification in subcategory_data['classifications']:
                    row = table.rows[current_row]
                    
                    row.cells[0].text = ""
                    row.cells[1].text = ""
                    
                    # Classification AR
                    row.cells[2].text = classification.get('classification_name', 'N/A')
                    row.cells[2].paragraphs[0].runs[0].font.size = Pt(9)
                    row.cells[2].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
                    row.cells[2].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
                    center_cell_content(row.cells[2])
                    
                    # Classification EN
                    row.cells[3].text = classification.get('classification_name_en', 'N/A')
                    row.cells[3].paragraphs[0].runs[0].font.size = Pt(9)
                    center_cell_content(row.cells[3])
                    
                    # Total
                    row.cells[4].text = str(classification.get('total_count', 0))
                    row.cells[4].paragraphs[0].runs[0].bold = True
                    center_cell_content(row.cells[4])
                    
                    # Severity
                    low_val = classification.get('low_count', 0)
                    med_val = classification.get('medium_count', 0)
                    high_val = classification.get('high_count', 0)
                    
                    row.cells[5].text = str(low_val) if low_val > 0 else ""
                    center_cell_content(row.cells[5])
                    set_cell_shading(row.cells[5], 'C6EFCE' if low_val > 0 else 'F2F2F2')
                    
                    row.cells[6].text = str(med_val) if med_val > 0 else ""
                    center_cell_content(row.cells[6])
                    set_cell_shading(row.cells[6], 'FFEB9C' if med_val > 0 else 'F2F2F2')
                    
                    row.cells[7].text = str(high_val) if high_val > 0 else ""
                    center_cell_content(row.cells[7])
                    set_cell_shading(row.cells[7], 'FFC7CE' if high_val > 0 else 'F2F2F2')
                    
                    row.cells[8].text = ""  # Spacer column
                    
                    # Prevention
                    prev_yes = classification.get('preventive_yes_count', 0)
                    prev_no = classification.get('preventive_no_count', 0)
                    
                    row.cells[9].text = str(prev_yes)
                    center_cell_content(row.cells[9])
                    if prev_yes > 0:
                        set_cell_shading(row.cells[9], 'C6EFCE')
                    
                    row.cells[10].text = str(prev_no)
                    center_cell_content(row.cells[10])
                    if prev_no > 0:
                        set_cell_shading(row.cells[10], 'FFE6E6')
                    
                    current_row += 1
                
                # Merge subcategory
                if current_row - 1 >= subcategory_start_row:
                    table.rows[subcategory_start_row].cells[1].text = f"{subcategory_name}\n(n={subcategory_total})"
                    table.rows[subcategory_start_row].cells[1].paragraphs[0].runs[0].font.size = Pt(9)
                    table.rows[subcategory_start_row].cells[1].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
                    table.rows[subcategory_start_row].cells[1].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
                    table.rows[subcategory_start_row].cells[1].paragraphs[0].runs[0].bold = True
                    table.rows[subcategory_start_row].cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    set_cell_shading(table.rows[subcategory_start_row].cells[1], 'E7E6E6')
                    
                    if current_row - 1 > subcategory_start_row:
                        try:
                            for row_idx in range(subcategory_start_row + 1, current_row):
                                if row_idx < len(table.rows):
                                    table.rows[subcategory_start_row].cells[1].merge(table.rows[row_idx].cells[1])
                        except Exception as e:
                            print(f"[FORMATTER] Warning: subcategory merge: {e}")
            
            # Merge category
            if current_row - 1 >= category_start_row:
                table.rows[category_start_row].cells[0].text = f"{category_name}\n(n={category_total})"
                table.rows[category_start_row].cells[0].paragraphs[0].runs[0].font.size = Pt(9)
                table.rows[category_start_row].cells[0].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
                table.rows[category_start_row].cells[0].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
                table.rows[category_start_row].cells[0].paragraphs[0].runs[0].bold = True
                table.rows[category_start_row].cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                set_cell_shading(table.rows[category_start_row].cells[0], 'D9E2F3')
                
                if current_row - 1 > category_start_row:
                    try:
                        for row_idx in range(category_start_row + 1, current_row):
                            if row_idx < len(table.rows):
                                table.rows[category_start_row].cells[0].merge(table.rows[row_idx].cells[0])
                    except Exception as e:
                        print(f"[FORMATTER] Warning: category merge: {e}")


def _create_hierarchical_table_rtl(doc: Document, hierarchy: Dict[str, Any], language: str):
    """
    Create the main hierarchical classification table with RTL layout and 2-row header.
    
    Table structure (11 columns, RTL order):
    Row 1 (Main Headers):
    - Problem Domain | Problem Category | Sub-Category | Classification (عربي) | Classification (English) | Total | [Severity - 3 cols merged] | [Prevention - 2 cols merged]
    
    Row 2 (Sub-Headers):
    - [empty x6] | LOW | MEDIUM | HIGH | YES | NO
    
    RTL Order: Right-to-left for Arabic readers
    """
    
    # Calculate total rows needed (header=2 + data rows)
    total_rows = 2  # 2-row header
    for domain_name, domain_data in hierarchy.items():
        for category_name, category_data in domain_data['categories'].items():
            for subcategory_name, subcategory_data in category_data['subcategories'].items():
                total_rows += len(subcategory_data['classifications'])
    
    if total_rows == 2:
        doc.add_paragraph("لا توجد بيانات متاحة (No classification data available).")
        return
    
    # Create table with 11 columns (0-10)
    table = doc.add_table(rows=total_rows, cols=11)
    table.style = 'Table Grid'
    
    # Set RTL table direction
    tbl = table._element
    tblPr = tbl.tblPr
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tbl.insert(0, tblPr)
    
    # Enable RTL
    bidiVisual = OxmlElement('w:bidiVisual')
    tblPr.append(bidiVisual)
    
    # ============================================================
    # HEADER SETUP - FILL BOTH ROWS FIRST, THEN MERGE
    # ============================================================
    header_row1 = table.rows[0]
    header_row2 = table.rows[1]
    
    # ============================================================
    # ROW 1: Set text for all main header cells (before any merging)
    # ============================================================
    headers_main = [
        "Problem Domain\nمجال المشكلة",
        "Problem Category\nفئة المشكلة",
        "Sub-Category\nالفئة الفرعية",
        "التصنيف عربي\nClassification AR",
        "التصنيف إنجليزي\nClassification EN",
        "Total\nالمجموع",
        "الشدة Severity",  # Col 6 (will span 6-8)
        "",  # Col 7 (will merge with 6)
        "",  # Col 8 (will merge with 6)
        "الإجراءات الوقائية\nPrevention Action",  # Col 9 (will span 9-10)
        ""  # Col 10 (will merge with 9)
    ]
    
    for idx, header_text in enumerate(headers_main):
        cell = header_row1.cells[idx]
        if header_text:  # Only set text if not empty
            cell.text = header_text
            cell.paragraphs[0].runs[0].bold = True
            cell.paragraphs[0].runs[0].font.size = Pt(10 if idx == 6 else 9)
            cell.paragraphs[0].runs[0].font.name = 'Traditional Arabic'
            cell.paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
            cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_shading(cell, '4472C4')  # Dark blue for all
    
    # ============================================================
    # ROW 2: Set text for all sub-header cells (before any merging)
    # ============================================================
    subheaders = [
        "",  # Col 0 (will merge with row1)
        "",  # Col 1 (will merge with row1)
        "",  # Col 2 (will merge with row1)
        "",  # Col 3 (will merge with row1)
        "",  # Col 4 (will merge with row1)
        "",  # Col 5 (will merge with row1)
        "LOW\nمنخفضة",  # Col 6
        "MEDIUM\nمتوسطة",  # Col 7
        "HIGH\nعالية",  # Col 8
        "YES\nنعم",  # Col 9
        "NO\nلا"  # Col 10
    ]
    
    for idx, subheader_text in enumerate(subheaders):
        cell = header_row2.cells[idx]
        if subheader_text:  # Only set text if not empty
            cell.text = subheader_text
            cell.paragraphs[0].runs[0].bold = True
            cell.paragraphs[0].runs[0].font.size = Pt(8)
            cell.paragraphs[0].runs[0].font.name = 'Traditional Arabic'
            cell.paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
            cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_cell_shading(cell, '5B9BD5')  # Lighter blue
        else:
            set_cell_shading(cell, '4472C4')  # Dark blue for merged areas
    
    # ============================================================
    # NOW DO THE MERGING (after text is set)
    # ============================================================
    
    # Merge columns 0-5 vertically (row 1 with row 2)
    for col_idx in range(6):
        try:
            header_row1.cells[col_idx].merge(header_row2.cells[col_idx])
        except Exception as e:
            print(f"[FORMATTER] Warning: Could not merge col {col_idx}: {e}")
    
    # Merge Severity header horizontally in row 1 (cols 6, 7, 8)
    try:
        header_row1.cells[6].merge(header_row1.cells[7])
        header_row1.cells[6].merge(header_row1.cells[8])
    except Exception as e:
        print(f"[FORMATTER] Warning: Could not merge severity header: {e}")
    
    # Merge Prevention header horizontally in row 1 (cols 9, 10)
    try:
        header_row1.cells[9].merge(header_row1.cells[10])
    except Exception as e:
        print(f"[FORMATTER] Warning: Could not merge prevention header: {e}")
    
    # ============================================================
    # DATA ROWS WITH MERGED CELLS
    # ============================================================
    current_row = 2  # Start after 2-row header
    
    for domain_name, domain_data in sorted(hierarchy.items()):
        domain_start_row = current_row
        domain_total = domain_data['total']
        
        for category_name, category_data in sorted(domain_data['categories'].items()):
            category_start_row = current_row
            category_total = category_data['total']
            
            for subcategory_name, subcategory_data in sorted(category_data['subcategories'].items()):
                subcategory_start_row = current_row
                subcategory_total = subcategory_data['total']
                
                # Add all classifications under this subcategory
                for classification in subcategory_data['classifications']:
                    row = table.rows[current_row]
                    
                    # Column 0: Domain (will be merged later)
                    row.cells[0].text = ""
                    
                    # Column 1: Category (will be merged later)
                    row.cells[1].text = ""
                    
                    # Column 2: Sub-Category (will be merged later)
                    row.cells[2].text = ""
                    
                    # Column 3: Classification (Arabic)
                    classification_ar = classification.get('classification_name', 
                                                           classification.get('classification_name_ar', 'N/A'))
                    row.cells[3].text = classification_ar
                    row.cells[3].paragraphs[0].runs[0].font.size = Pt(9)
                    row.cells[3].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
                    row.cells[3].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
                    row.cells[3].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    
                    # Column 4: Classification (English)
                    classification_en = classification.get('classification_name_en', 
                                                          classification.get('classification_name', 'N/A'))
                    row.cells[4].text = classification_en
                    row.cells[4].paragraphs[0].runs[0].font.size = Pt(9)
                    row.cells[4].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    
                    # Column 5: Total
                    row.cells[5].text = str(classification.get('total_count', 0))
                    row.cells[5].paragraphs[0].runs[0].bold = True
                    center_cell_content(row.cells[5])
                    
                    # Columns 6-8: Severity breakdown (empty cell if 0)
                    low_val = classification.get('low_count', 0)
                    med_val = classification.get('medium_count', 0)
                    high_val = classification.get('high_count', 0)
                    
                    row.cells[6].text = str(low_val) if low_val > 0 else ""
                    center_cell_content(row.cells[6])
                    if low_val > 0:
                        set_cell_shading(row.cells[6], 'C6EFCE')  # Light green
                    else:
                        set_cell_shading(row.cells[6], 'F2F2F2')  # Light gray for empty
                    
                    row.cells[7].text = str(med_val) if med_val > 0 else ""
                    center_cell_content(row.cells[7])
                    if med_val > 0:
                        set_cell_shading(row.cells[7], 'FFEB9C')  # Light yellow
                    else:
                        set_cell_shading(row.cells[7], 'F2F2F2')  # Light gray for empty
                    
                    row.cells[8].text = str(high_val) if high_val > 0 else ""
                    center_cell_content(row.cells[8])
                    if high_val > 0:
                        set_cell_shading(row.cells[8], 'FFC7CE')  # Light red
                    else:
                        set_cell_shading(row.cells[8], 'F2F2F2')  # Light gray for empty
                    
                    # Columns 9-10: Prevention Action (now properly separated)
                    prev_yes = classification.get('preventive_yes_count', 0)
                    prev_no = classification.get('preventive_no_count', 0)
                    
                    row.cells[9].text = str(prev_yes)
                    center_cell_content(row.cells[9])
                    if prev_yes > 0:
                        set_cell_shading(row.cells[9], 'C6EFCE')  # Light green
                    
                    row.cells[10].text = str(prev_no)
                    center_cell_content(row.cells[10])
                    if prev_no > 0:
                        set_cell_shading(row.cells[10], 'FFE6E6')  # Light red
                    
                    current_row += 1
                
                # Merge Sub-Category cells
                subcategory_end_row = current_row - 1
                if subcategory_end_row >= subcategory_start_row:
                    table.rows[subcategory_start_row].cells[2].text = f"{subcategory_name}\n(n={subcategory_total})"
                    table.rows[subcategory_start_row].cells[2].paragraphs[0].runs[0].font.size = Pt(9)
                    table.rows[subcategory_start_row].cells[2].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
                    table.rows[subcategory_start_row].cells[2].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
                    table.rows[subcategory_start_row].cells[2].paragraphs[0].runs[0].bold = True
                    table.rows[subcategory_start_row].cells[2].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    set_cell_shading(table.rows[subcategory_start_row].cells[2], 'E7E6E6')
                    
                    if subcategory_end_row > subcategory_start_row:
                        try:
                            for row_idx in range(subcategory_start_row + 1, subcategory_end_row + 1):
                                if row_idx < len(table.rows):  # Safety check
                                    table.rows[subcategory_start_row].cells[2].merge(table.rows[row_idx].cells[2])
                        except Exception as e:
                            print(f"[FORMATTER] Warning: Could not merge subcategory cells: {e}")
            
            # Merge Category cells
            category_end_row = current_row - 1
            if category_end_row >= category_start_row:
                table.rows[category_start_row].cells[1].text = f"{category_name}\n(n={category_total})"
                table.rows[category_start_row].cells[1].paragraphs[0].runs[0].font.size = Pt(9)
                table.rows[category_start_row].cells[1].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
                table.rows[category_start_row].cells[1].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
                table.rows[category_start_row].cells[1].paragraphs[0].runs[0].bold = True
                table.rows[category_start_row].cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                set_cell_shading(table.rows[category_start_row].cells[1], 'D9E2F3')
                
                if category_end_row > category_start_row:
                    try:
                        for row_idx in range(category_start_row + 1, category_end_row + 1):
                            if row_idx < len(table.rows):  # Safety check
                                table.rows[category_start_row].cells[1].merge(table.rows[row_idx].cells[1])
                    except Exception as e:
                        print(f"[FORMATTER] Warning: Could not merge category cells: {e}")
        
        # Merge Domain cells
        domain_end_row = current_row - 1
        if domain_end_row >= domain_start_row:
            table.rows[domain_start_row].cells[0].text = f'"{domain_name}"\n(n={domain_total})'
            table.rows[domain_start_row].cells[0].paragraphs[0].runs[0].font.size = Pt(10)
            table.rows[domain_start_row].cells[0].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
            table.rows[domain_start_row].cells[0].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
            table.rows[domain_start_row].cells[0].paragraphs[0].runs[0].bold = True
            table.rows[domain_start_row].cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_cell_shading(table.rows[domain_start_row].cells[0], 'BDD7EE')
            
            if domain_end_row > domain_start_row:
                try:
                    for row_idx in range(domain_start_row + 1, domain_end_row + 1):
                        if row_idx < len(table.rows):  # Safety check
                            table.rows[domain_start_row].cells[0].merge(table.rows[row_idx].cells[0])
                except Exception as e:
                    print(f"[FORMATTER] Warning: Could not merge domain cells: {e}")
    
    # Set column widths for landscape orientation (11 columns total)
    try:
        table.columns[0].width = Inches(1.1)   # Domain
        table.columns[1].width = Inches(1.2)   # Category
        table.columns[2].width = Inches(1.2)   # Sub-Category
        table.columns[3].width = Inches(1.3)   # Classification (AR)
        table.columns[4].width = Inches(1.3)   # Classification (EN)
        table.columns[5].width = Inches(0.6)   # Total
        table.columns[6].width = Inches(0.6)   # Low
        table.columns[7].width = Inches(0.7)   # Medium
        table.columns[8].width = Inches(0.6)   # High
        table.columns[9].width = Inches(0.6)   # Yes
        table.columns[10].width = Inches(0.6)  # No
    except:
        pass  # Column width adjustment may fail, continue anyway


def generate_seasonal_pdf_report(
    seasonal_data: Dict[str, Any],
    language: str = "en"
) -> bytes:
    """
    Generate a PDF document for a seasonal report.
    
    Currently returns Word format - PDF conversion can be added later.
    
    Args:
        seasonal_data: Seasonal report data from orchestrator
        language: Language for the report (en or ar)
    
    Returns:
        Bytes of the generated document
    """
    # For now, return Word format
    # TODO: Add PDF conversion using reportlab or weasyprint
    return generate_seasonal_word_report(seasonal_data, language)
    
    severity_para = doc.add_paragraph()
    severity_para.add_run(f"Low Severity: ").bold = True
    severity_para.add_run(f"{header.get('low_severity_count', 0)} cases\n")
    
    severity_para.add_run(f"Medium Severity: ").bold = True
    severity_para.add_run(f"{header.get('medium_severity_count', 0)} cases\n")
    
    severity_para.add_run(f"High Severity: ").bold = True
    severity_para.add_run(f"{header.get('high_severity_count', 0)} cases\n")
    
    # Save to bytes
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    
    return buffer.getvalue()


def generate_seasonal_pdf_report(
    seasonal_data: Dict[str, Any],
    language: str = "en"
) -> bytes:
    """
    Generate a PDF for a seasonal report.
    Currently converts Word to PDF.
    
    Args:
        seasonal_data: Seasonal report data from orchestrator
        language: Language for the report (en or ar)
    
    Returns:
        Bytes of the generated PDF document
    """
    # For now, generate Word and note that PDF conversion would happen here
    # In production, use docx2pdf or similar
    word_bytes = generate_seasonal_word_report(seasonal_data, language)
    
    # TODO: Convert Word to PDF using docx2pdf or similar library
    # For now, return Word bytes with a note
    return word_bytes  # Placeholder - replace with actual PDF conversion


# ============================================================================
# VISUALIZATION FUNCTIONS FOR COMPARATIVE REPORTS
# ============================================================================

def _configure_arabic_matplotlib():
    """Configure matplotlib to support Arabic text rendering"""
    try:
        # Try to find Arabic fonts on the system
        arabic_fonts = [font for font in fm.findSystemFonts() 
                       if 'arabic' in font.lower() or 'traditional' in font.lower()]
        if arabic_fonts:
            plt.rcParams['font.family'] = 'Traditional Arabic'
    except:
        pass
    
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display


def _generate_spider_chart(labels: List[str], prev_values: List[float], 
                           curr_values: List[float], title: str,
                           prev_label: str, curr_label: str) -> io.BytesIO:
    """
    Generate a spider/radar chart comparing two datasets.
    
    Args:
        labels: List of dimension labels (e.g., domain names)
        prev_values: Previous period values
        curr_values: Current period values
        title: Chart title
        prev_label: Label for previous period
        curr_label: Label for current period
    
    Returns:
        BytesIO buffer containing PNG image
    """
    _configure_arabic_matplotlib()
    
    # Number of variables
    num_vars = len(labels)
    
    if num_vars == 0:
        # Return empty chart
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        return buf
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the circle
    prev_values = prev_values + [prev_values[0]]
    curr_values = curr_values + [curr_values[0]]
    angles += angles[:1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, prev_values, 'o-', linewidth=2, label=prev_label, color='#4472C4')
    ax.fill(angles, prev_values, alpha=0.25, color='#4472C4')
    
    ax.plot(angles, curr_values, 'o-', linewidth=2, label=curr_label, color='#ED7D31')
    ax.fill(angles, curr_values, alpha=0.25, color='#ED7D31')
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    
    # Set title and legend
    ax.set_title(title, size=14, weight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Add grid
    ax.grid(True)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf


def _generate_diverging_bar_chart(labels: List[str], changes: List[float], 
                                  title: str) -> io.BytesIO:
    """
    Generate a diverging bar chart showing positive/negative changes.
    
    Args:
        labels: List of category labels
        changes: List of change values (can be positive or negative)
        title: Chart title
    
    Returns:
        BytesIO buffer containing PNG image
    """
    _configure_arabic_matplotlib()
    
    if len(labels) == 0:
        # Return empty chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        return buf
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.4)))
    
    # Color code: green for negative (improvement), red for positive (worsening)
    colors = ['#70AD47' if c < 0 else '#C5504B' if c > 0 else '#BFBFBF' for c in changes]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, changes, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, change) in enumerate(zip(bars, changes)):
        width = bar.get_width()
        label_x = width + (1 if width > 0 else -1)
        ax.text(label_x, i, f'{change:+.0f}', 
               ha='left' if width > 0 else 'right', 
               va='center', fontsize=10, weight='bold')
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Change (← Decrease | Increase →)', fontsize=11)
    ax.set_title(title, fontsize=14, weight='bold', pad=15)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf


def _generate_heatmap(data: np.ndarray, row_labels: List[str], 
                      col_labels: List[str], title: str) -> io.BytesIO:
    """
    Generate a heatmap showing data intensity across two dimensions.
    
    Args:
        data: 2D numpy array of values
        row_labels: Labels for rows
        col_labels: Labels for columns
        title: Chart title
    
    Returns:
        BytesIO buffer containing PNG image
    """
    _configure_arabic_matplotlib()
    
    if data.size == 0:
        # Return empty chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        return buf
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(row_labels) * 0.5)))
    
    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)
    
    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, f'{data[i, j]:.0f}',
                         ha="center", va="center", color="black", fontsize=9, weight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=15)
    
    # Set title
    ax.set_title(title, fontsize=14, weight='bold', pad=15)
    
    # Tight layout
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf


def _extract_domain_data(hierarchy: Dict) -> Tuple[List[str], List[int]]:
    """Extract domain names and counts from hierarchy"""
    domains = []
    counts = []
    
    for domain_name, domain_data in hierarchy.items():
        domains.append(domain_name)
        counts.append(domain_data.get('total', 0))
    
    return domains, counts


def _extract_category_data(hierarchy: Dict) -> Tuple[List[str], List[int]]:
    """Extract category names and counts from hierarchy"""
    categories = []
    counts = []
    
    for domain_name, domain_data in hierarchy.items():
        for cat_name, cat_data in domain_data.get('categories', {}).items():
            full_name = f"{domain_name} - {cat_name}"
            categories.append(full_name)
            counts.append(cat_data.get('total', 0))
    
    return categories, counts


def _extract_subcategory_data(hierarchy: Dict) -> Tuple[List[str], List[int]]:
    """Extract subcategory names and counts from hierarchy"""
    subcategories = []
    counts = []
    
    for domain_name, domain_data in hierarchy.items():
        for cat_name, cat_data in domain_data.get('categories', {}).items():
            for subcat_name, subcat_data in cat_data.get('subcategories', {}).items():
                full_name = f"{cat_name} - {subcat_name}"
                subcategories.append(full_name)
                counts.append(subcat_data.get('total', 0))
    
    return subcategories, counts


def generate_comparative_seasonal_word_report(
    current_data: Dict[str, Any],
    previous_data: Dict[str, Any],
    language: str = "en"
) -> bytes:
    """
    Generate a comparative Word document showing current season vs previous season.
    
    The report mirrors the regular seasonal report structure but adds side-by-side
    comparison data in each table, showing both periods with delta indicators.
    
    Args:
        current_data: Current seasonal report data from orchestrator
        previous_data: Previous seasonal report data from orchestrator (may have zero data)
        language: Language for the report (en or ar)
    
    Returns:
        Bytes of the generated comparative Word document
    """
    # Utility functions
    def _safe(v):
        """Convert dimension values to int (python-docx requirement)"""
        return int(v)
    
    doc = Document()
    
    # ============================================================
    # DOCUMENT SETUP - A4 LANDSCAPE
    # ============================================================
    section = doc.sections[0]
    section.page_height = _safe(Mm(210))
    section.page_width = _safe(Mm(297))
    section.orientation = WD_ORIENT.LANDSCAPE
    section.left_margin = _safe(Mm(15))
    section.right_margin = _safe(Mm(15))
    section.top_margin = _safe(Mm(15))
    section.bottom_margin = _safe(Mm(15))
    
    # Set default font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Traditional Arabic'
    font.size = Pt(11)
    
    # Extract data from both seasons
    current_header = current_data.get("header", {})
    previous_header = previous_data.get("header", {})
    
    current_period = current_header.get('period', 'N/A')
    previous_period = previous_header.get('period', 'N/A')
    
    # ============================================================
    # HEADER - LOGO (TOP RIGHT)
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
        print(f"[FORMATTER] Could not add logo: {e}")
    
    # ============================================================
    # TITLE SECTION (ARABIC) - COMPARATIVE
    # ============================================================
    
    title_para = doc.add_paragraph()
    title_run = title_para.add_run("تقرير المقارنة الموسمية | Seasonal Comparison Report")
    title_run.font.size = int(Pt(21))
    title_run.font.bold = True
    title_run.font.name = 'Traditional Arabic'
    title_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_para.space_after = int(Pt(3))
    
    # Subtitle with both periods
    subtitle_para = doc.add_paragraph()
    subtitle_run = subtitle_para.add_run(f"{current_period} مقابل {previous_period}")
    subtitle_run.font.size = int(Pt(16))
    subtitle_run.font.bold = True
    subtitle_run.font.name = 'Traditional Arabic'
    subtitle_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    subtitle_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_para.space_after = int(Pt(6))
    
    # Organization info
    orgunit_type = current_header.get('orgunit_type', 0)
    type_names = {0: "المستشفى", 1: "الإدارة", 2: "الدائرة", 3: "القسم"}
    type_name = type_names.get(orgunit_type, "الوحدة التنظيمية")
    
    org_para = doc.add_paragraph()
    org_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    org_run = org_para.add_run(f"الوحدة التنظيمية: {current_header.get('orgunit_name', 'N/A')} • النوع: {type_name}")
    org_run.font.size = int(Pt(14))
    org_run.font.bold = True
    org_run.font.name = 'Traditional Arabic'
    org_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    doc.add_paragraph()  # Spacer
    
    # ============================================================
    # SUMMARY COMPARISON TABLE
    # ============================================================
    
    summary_heading = doc.add_paragraph()
    summary_heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    summary_heading.paragraph_format.right_to_left = True
    
    sh_run = summary_heading.add_run("📊 مقارنة الإحصائيات | Statistics Comparison")
    sh_run.font.bold = True
    sh_run.font.size = Pt(14)
    sh_run.font.name = 'Traditional Arabic'
    sh_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    doc.add_paragraph()
    
    # Create comparison table (3 columns: Category | Previous | Current) - NO DELTA
    comp_table = doc.add_table(rows=8, cols=3)
    comp_table.style = 'Table Grid'
    
    # Headers (RTL: Current, Previous, Category)
    headers_rtl = [current_period, previous_period, "الفئة\nCategory"]
    for idx, header_text in enumerate(headers_rtl):
        cell = comp_table.rows[0].cells[idx]
        cell.text = header_text
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(10)
        cell.paragraphs[0].runs[0].font.name = 'Traditional Arabic'
        cell.paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        center_cell_content(cell)
        set_cell_shading(cell, '4472C4')
    
    # Data rows
    current_total = current_header.get('total_cases', 0)
    previous_total = previous_header.get('total_cases', 0)
    
    metrics = [
        ("Total Cases | المجموع الكلي", previous_total, current_total),
        ("Clinical | السريرية", previous_header.get('clinical_domain_count', 0), current_header.get('clinical_domain_count', 0)),
        ("Management | الإدارية", previous_header.get('management_domain_count', 0), current_header.get('management_domain_count', 0)),
        ("Relational | العلائقية", previous_header.get('relational_domain_count', 0), current_header.get('relational_domain_count', 0)),
        ("Low Severity | منخفضة", previous_header.get('low_severity_count', 0), current_header.get('low_severity_count', 0)),
        ("Medium Severity | متوسطة", previous_header.get('medium_severity_count', 0), current_header.get('medium_severity_count', 0)),
        ("High Severity | عالية", previous_header.get('high_severity_count', 0), current_header.get('high_severity_count', 0))
    ]
    
    for idx, (label, prev_val, curr_val) in enumerate(metrics, start=1):
        row = comp_table.rows[idx]
        
        # Category (rightmost - col 2)
        row.cells[2].text = label
        row.cells[2].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
        row.cells[2].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        row.cells[2].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        
        # Previous value (col 1)
        row.cells[1].text = str(prev_val)
        center_cell_content(row.cells[1])
        
        # Current value (col 0 - leftmost)
        row.cells[0].text = str(curr_val)
        center_cell_content(row.cells[0])
    
    doc.add_paragraph()
    
    # ============================================================
    # HIERARCHICAL CLASSIFICATION COMPARISON BY DOMAIN
    # ============================================================
    
    # Build hierarchies for table generation (moved before graphs)
    current_hierarchy = _build_hierarchy(current_data.get("classification_stats", []))
    previous_hierarchy = _build_hierarchy(previous_data.get("classification_stats", []))
    
    # Merge all domains from both seasons
    all_domains = set(list(current_hierarchy.keys()) + list(previous_hierarchy.keys()))
    
    # Create comparative tables for each domain
    _create_comparative_hierarchical_tables_by_domain(
        doc, 
        current_hierarchy, 
        previous_hierarchy, 
        current_period,
        previous_period,
        all_domains,
        language
    )
    
    doc.add_paragraph()
    
    # ============================================================
    # POLICY COMPLIANCE COMPARISON
    # ============================================================
    
    policy_heading = doc.add_paragraph()
    policy_heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    policy_heading.paragraph_format.right_to_left = True
    
    ph_run = policy_heading.add_run("📊 مقارنة الامتثال للسياسة | Policy Compliance Comparison")
    ph_run.font.bold = True
    ph_run.font.size = Pt(14)
    ph_run.font.name = 'Traditional Arabic'
    ph_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    doc.add_paragraph()
    
    current_compliant = current_header.get('is_compliant', False)
    previous_compliant = previous_header.get('is_compliant', False)
    
    compliance_para = doc.add_paragraph()
    compliance_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    compliance_para.paragraph_format.right_to_left = True
    
    cp_text = f"{previous_period}: {'✓ مطابق' if previous_compliant else '✗ غير مطابق'}  |  {current_period}: {'✓ مطابق' if current_compliant else '✗ غير مطابق'}"
    cp_run = compliance_para.add_run(cp_text)
    cp_run.font.size = Pt(12)
    cp_run.font.bold = True
    cp_run.font.name = 'Traditional Arabic'
    cp_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    doc.add_paragraph()
    
    # ============================================================
    # PAGE BREAK BEFORE VISUALIZATION SECTION
    # ============================================================
    
    doc.add_page_break()
    
    # ============================================================
    # VISUALIZATION SECTION - GRAPHS AT THE END
    # ============================================================
    
    try:
        # Add visual analysis heading
        visual_heading = doc.add_paragraph()
        visual_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        vh_run = visual_heading.add_run("📊 التحليل المرئي | Visual Analysis")
        vh_run.font.bold = True
        vh_run.font.size = Pt(16)
        vh_run.font.name = 'Traditional Arabic'
        vh_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        
        doc.add_paragraph()
        
        # =================== DOMAIN LEVEL CHARTS ===================
        
        # Extract domain data
        prev_domains, prev_domain_counts = _extract_domain_data(previous_hierarchy)
        curr_domains, curr_domain_counts = _extract_domain_data(current_hierarchy)
        
        # Merge domain lists
        all_domain_names = sorted(set(prev_domains + curr_domains))
        
        # Align data
        prev_domain_values = []
        curr_domain_values = []
        domain_changes = []
        
        for domain in all_domain_names:
            prev_val = prev_domain_counts[prev_domains.index(domain)] if domain in prev_domains else 0
            curr_val = curr_domain_counts[curr_domains.index(domain)] if domain in curr_domains else 0
            prev_domain_values.append(prev_val)
            curr_domain_values.append(curr_val)
            domain_changes.append(curr_val - prev_val)
        
        # Generate Domain Spider Chart
        if all_domain_names:
            spider_buf = _generate_spider_chart(
                labels=all_domain_names,
                prev_values=prev_domain_values,
                curr_values=curr_domain_values,
                title="Domain Comparison | مقارنة المجالات",
                prev_label=previous_period,
                curr_label=current_period
            )
            
            # Add image centered
            spider_para = doc.add_paragraph()
            spider_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            spider_run = spider_para.add_run()
            spider_run.add_picture(spider_buf, width=Inches(6))
            
            # Add bilingual caption centered
            caption_para = doc.add_paragraph()
            caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption_run = caption_para.add_run("🕸️ مخطط العنكبوت - المجالات | Domain Spider Chart")
            caption_run.font.bold = True
            caption_run.font.size = Pt(11)
            caption_run.font.name = 'Traditional Arabic'
            caption_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
            
            doc.add_paragraph()  # Spacer
        
        # Generate Domain Bar Subtraction Chart
        if all_domain_names:
            bar_buf = _generate_diverging_bar_chart(
                labels=all_domain_names,
                changes=domain_changes,
                title="Domain Change Analysis | تحليل تغيرات المجالات"
            )
            
            # Add image centered
            bar_para = doc.add_paragraph()
            bar_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            bar_run = bar_para.add_run()
            bar_run.add_picture(bar_buf, width=Inches(6))
            
            # Add bilingual caption centered
            bar_caption_para = doc.add_paragraph()
            bar_caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            bar_caption_run = bar_caption_para.add_run("📊 مخطط الأعمدة - الفروقات (المجالات) | Domain Bar Subtraction Chart")
            bar_caption_run.font.bold = True
            bar_caption_run.font.size = Pt(11)
            bar_caption_run.font.name = 'Traditional Arabic'
            bar_caption_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
            
            doc.add_paragraph()  # Spacer
        
        # =================== CATEGORY LEVEL CHARTS ===================
        
        # Extract category data
        prev_categories, prev_category_counts = _extract_category_data(previous_hierarchy)
        curr_categories, curr_category_counts = _extract_category_data(current_hierarchy)
        
        # Merge category lists
        all_category_names = sorted(set(prev_categories + curr_categories))
        
        # Align data
        prev_category_values = []
        curr_category_values = []
        category_changes = []
        
        for category in all_category_names:
            prev_val = prev_category_counts[prev_categories.index(category)] if category in prev_categories else 0
            curr_val = curr_category_counts[curr_categories.index(category)] if category in curr_categories else 0
            prev_category_values.append(prev_val)
            curr_category_values.append(curr_val)
            category_changes.append(curr_val - prev_val)
        
        # Limit to top 10 for readability
        if len(all_category_names) > 10:
            # Sort by absolute change
            sorted_indices = sorted(range(len(category_changes)), 
                                   key=lambda i: abs(category_changes[i]), reverse=True)[:10]
            all_category_names = [all_category_names[i] for i in sorted_indices]
            prev_category_values = [prev_category_values[i] for i in sorted_indices]
            curr_category_values = [curr_category_values[i] for i in sorted_indices]
            category_changes = [category_changes[i] for i in sorted_indices]
        
        # Generate Category Spider Chart
        if all_category_names:
            cat_spider_buf = _generate_spider_chart(
                labels=all_category_names,
                prev_values=prev_category_values,
                curr_values=curr_category_values,
                title="Category Comparison (Top 10) | مقارنة الفئات (أعلى 10)",
                prev_label=previous_period,
                curr_label=current_period
            )
            
            # Add image centered
            cat_spider_para = doc.add_paragraph()
            cat_spider_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cat_spider_run = cat_spider_para.add_run()
            cat_spider_run.add_picture(cat_spider_buf, width=Inches(6))
            
            # Add bilingual caption centered
            cat_caption_para = doc.add_paragraph()
            cat_caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cat_caption_run = cat_caption_para.add_run("🕸️ مخطط العنكبوت - الفئات | Category Spider Chart")
            cat_caption_run.font.bold = True
            cat_caption_run.font.size = Pt(11)
            cat_caption_run.font.name = 'Traditional Arabic'
            cat_caption_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
            
            doc.add_paragraph()  # Spacer
        
        # Generate Category Bar Subtraction Chart
        if all_category_names:
            cat_bar_buf = _generate_diverging_bar_chart(
                labels=all_category_names,
                changes=category_changes,
                title="Category Change Analysis (Top 10) | تحليل تغيرات الفئات (أعلى 10)"
            )
            
            # Add image centered
            cat_bar_para = doc.add_paragraph()
            cat_bar_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cat_bar_run = cat_bar_para.add_run()
            cat_bar_run.add_picture(cat_bar_buf, width=Inches(6))
            
            # Add bilingual caption centered
            cat_bar_caption_para = doc.add_paragraph()
            cat_bar_caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cat_bar_caption_run = cat_bar_caption_para.add_run("📊 مخطط الأعمدة - الفروقات (الفئات) | Category Bar Subtraction Chart")
            cat_bar_caption_run.font.bold = True
            cat_bar_caption_run.font.size = Pt(11)
            cat_bar_caption_run.font.name = 'Traditional Arabic'
            cat_bar_caption_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
            
            doc.add_paragraph()  # Spacer
        
        # =================== SUBCATEGORY LEVEL CHART ===================
        
        # Extract subcategory data
        prev_subcategories, prev_subcategory_counts = _extract_subcategory_data(previous_hierarchy)
        curr_subcategories, curr_subcategory_counts = _extract_subcategory_data(current_hierarchy)
        
        # Merge subcategory lists
        all_subcategory_names = sorted(set(prev_subcategories + curr_subcategories))
        
        # Align data
        prev_subcategory_values = []
        curr_subcategory_values = []
        subcategory_changes = []
        
        for subcategory in all_subcategory_names:
            prev_val = prev_subcategory_counts[prev_subcategories.index(subcategory)] if subcategory in prev_subcategories else 0
            curr_val = curr_subcategory_counts[curr_subcategories.index(subcategory)] if subcategory in curr_subcategories else 0
            prev_subcategory_values.append(prev_val)
            curr_subcategory_values.append(curr_val)
            subcategory_changes.append(curr_val - prev_val)
        
        # Limit to top 10 for readability
        if len(all_subcategory_names) > 10:
            # Sort by absolute change
            sorted_indices = sorted(range(len(subcategory_changes)), 
                                   key=lambda i: abs(subcategory_changes[i]), reverse=True)[:10]
            all_subcategory_names = [all_subcategory_names[i] for i in sorted_indices]
            prev_subcategory_values = [prev_subcategory_values[i] for i in sorted_indices]
            curr_subcategory_values = [curr_subcategory_values[i] for i in sorted_indices]
            subcategory_changes = [subcategory_changes[i] for i in sorted_indices]
        
        # Generate Subcategory Spider Chart ONLY (no bar or heatmap)
        if all_subcategory_names:
            subcat_spider_buf = _generate_spider_chart(
                labels=all_subcategory_names,
                prev_values=prev_subcategory_values,
                curr_values=curr_subcategory_values,
                title="Subcategory Comparison (Top 10) | مقارنة الفئات الفرعية (أعلى 10)",
                prev_label=previous_period,
                curr_label=current_period
            )
            
            # Add image centered
            subcat_spider_para = doc.add_paragraph()
            subcat_spider_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            subcat_spider_run = subcat_spider_para.add_run()
            subcat_spider_run.add_picture(subcat_spider_buf, width=Inches(6))
            
            # Add bilingual caption centered
            subcat_caption_para = doc.add_paragraph()
            subcat_caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            subcat_caption_run = subcat_caption_para.add_run("🕸️ مخطط العنكبوت - الفئات الفرعية | SubCategory Spider Chart")
            subcat_caption_run.font.bold = True
            subcat_caption_run.font.size = Pt(11)
            subcat_caption_run.font.name = 'Traditional Arabic'
            subcat_caption_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
            
            doc.add_paragraph()  # Spacer
        
    except Exception as e:
        # If chart generation fails, add error message but continue with report
        error_para = doc.add_paragraph()
        error_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        error_run = error_para.add_run(f"⚠️ Chart generation encountered an issue: {str(e)}")
        error_run.font.size = Pt(11)
        error_run.font.color.rgb = RGBColor(192, 0, 0)
        doc.add_paragraph()
    
    # ============================================================
    # HIERARCHICAL CLASSIFICATION COMPARISON BY DOMAIN
    # ============================================================
    
    # Merge all domains from both seasons
    all_domains = set(list(current_hierarchy.keys()) + list(previous_hierarchy.keys()))
    
    # Create comparative tables for each domain
    _create_comparative_hierarchical_tables_by_domain(
        doc, 
        current_hierarchy, 
        previous_hierarchy, 
        current_period,
        previous_period,
        all_domains,
        language
    )
    
    doc.add_paragraph()
    
    # ============================================================
    # POLICY COMPLIANCE COMPARISON
    # ============================================================
    
    policy_heading = doc.add_paragraph()
    policy_heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    policy_heading.paragraph_format.right_to_left = True
    
    ph_run = policy_heading.add_run("📊 مقارنة الامتثال للسياسة | Policy Compliance Comparison")
    ph_run.font.bold = True
    ph_run.font.size = Pt(14)
    ph_run.font.name = 'Traditional Arabic'
    ph_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    doc.add_paragraph()
    
    current_compliant = current_header.get('is_compliant', False)
    previous_compliant = previous_header.get('is_compliant', False)
    
    compliance_para = doc.add_paragraph()
    compliance_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    compliance_para.paragraph_format.right_to_left = True
    
    cp_text = f"{previous_period}: {'✓ مطابق' if previous_compliant else '✗ غير مطابق'}  |  {current_period}: {'✓ مطابق' if current_compliant else '✗ غير مطابق'}"
    cp_run = compliance_para.add_run(cp_text)
    cp_run.font.size = Pt(12)
    cp_run.font.bold = True
    cp_run.font.name = 'Traditional Arabic'
    cp_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    # ============================================================
    # SAVE AND RETURN
    # ============================================================
    
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def _create_comparative_hierarchical_tables_by_domain(
    doc: Document,
    current_hierarchy: Dict[str, Any],
    previous_hierarchy: Dict[str, Any],
    current_period: str,
    previous_period: str,
    all_domains: set,
    language: str
):
    """
    Create comparative hierarchical tables for each domain showing current vs previous data.
    
    Table structure (RTL, 9 columns):
    - Domain
    - Classification (AR + EN)
    - Previous Period: Total + Severity (L/M/H)
    - Current Period: Total + Severity (L/M/H)
    """
    if not all_domains:
        doc.add_paragraph("لا توجد بيانات متاحة (No classification data available).")
        return
    
    for domain_idx, domain_name in enumerate(sorted(all_domains)):
        if domain_idx > 0:
            doc.add_paragraph()  # Spacing between domains
        
        current_domain_data = current_hierarchy.get(domain_name, {'total': 0, 'categories': {}})
        previous_domain_data = previous_hierarchy.get(domain_name, {'total': 0, 'categories': {}})
        
        # Domain title with comparison
        domain_title = doc.add_paragraph()
        domain_title.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        domain_title.paragraph_format.right_to_left = True
        
        title_text = f'📂 {domain_name} | {previous_period}: n={previous_domain_data["total"]} → {current_period}: n={current_domain_data["total"]}'
        title_run = domain_title.add_run(title_text)
        title_run.font.bold = True
        title_run.font.size = Pt(13)
        title_run.font.name = 'Traditional Arabic'
        title_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        title_run.font.color.rgb = RGBColor(68, 114, 196)
        
        # Collect all classifications from both periods
        classification_map = {}  # classification_id -> {current_data, previous_data}
        
        # Process current period
        for category_name, category_data in current_domain_data['categories'].items():
            for subcat_name, subcat_data in category_data['subcategories'].items():
                for classification in subcat_data['classifications']:
                    class_id = classification.get('classification_id')
                    if class_id not in classification_map:
                        classification_map[class_id] = {'current': None, 'previous': None}
                    classification_map[class_id]['current'] = classification
        
        # Process previous period
        for category_name, category_data in previous_domain_data['categories'].items():
            for subcat_name, subcat_data in category_data['subcategories'].items():
                for classification in subcat_data['classifications']:
                    class_id = classification.get('classification_id')
                    if class_id not in classification_map:
                        classification_map[class_id] = {'current': None, 'previous': None}
                    classification_map[class_id]['previous'] = classification
        
        if not classification_map:
            # No classifications in this domain
            no_data = doc.add_paragraph()
            no_data.alignment = WD_ALIGN_PARAGRAPH.CENTER
            no_data_run = no_data.add_run("لا توجد بيانات لهذا المجال (No data for this domain)")
            no_data_run.font.size = Pt(10)
            no_data_run.font.italic = True
            continue
        
        # Create table: 11 columns (Domain | Class AR | Class EN | Prev[Total+L+M+H] | Curr[Total+L+M+H])
        num_rows = len(classification_map) + 2  # data rows + 2 header rows
        table = doc.add_table(rows=num_rows, cols=11)
        table.style = 'Table Grid'
        
        # Set RTL
        tbl = table._element
        tblPr = tbl.tblPr
        if tblPr is None:
            tblPr = OxmlElement('w:tblPr')
            tbl.insert(0, tblPr)
        bidiVisual = OxmlElement('w:bidiVisual')
        tblPr.append(bidiVisual)
        
        header_row1 = table.rows[0]
        header_row2 = table.rows[1]
        
        # Row 1: Domain | Class AR | Class EN | Previous (4 cols merged) | Current (4 cols merged)
        headers_main = [
            f"{domain_name}",
            "التصنيف عربي",
            "EN",
            previous_period,
            "", "", "",
            current_period,
            "", "", ""
        ]
        
        for idx, header_text in enumerate(headers_main):
            cell = header_row1.cells[idx]
            if header_text:
                cell.text = header_text
                cell.paragraphs[0].runs[0].bold = True
                cell.paragraphs[0].runs[0].font.size = Pt(9)
                cell.paragraphs[0].runs[0].font.name = 'Traditional Arabic'
                cell.paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
                cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
                center_cell_content(cell)
            set_cell_shading(cell, '4472C4')
        
        # Row 2: Empty | Empty | Empty | Total L M H | Total L M H
        subheaders = ["", "", "", "المجموع", "L", "M", "H", "المجموع", "L", "M", "H"]
        
        for idx, subheader_text in enumerate(subheaders):
            cell = header_row2.cells[idx]
            if subheader_text:
                cell.text = subheader_text
                cell.paragraphs[0].runs[0].bold = True
                cell.paragraphs[0].runs[0].font.size = Pt(8)
                cell.paragraphs[0].runs[0].font.name = 'Traditional Arabic'
                cell.paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
                cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
                center_cell_content(cell)
            set_cell_shading(cell, '5B9BD5')
        
        # Merge headers
        try:
            # Merge domain, class AR, class EN vertically
            for i in range(3):
                header_row1.cells[i].merge(header_row2.cells[i])
            # Merge previous period header (cols 3-6)
            header_row1.cells[3].merge(header_row1.cells[6])
            # Merge current period header (cols 7-10)
            header_row1.cells[7].merge(header_row1.cells[10])
        except Exception as e:
            print(f"[FORMATTER] Warning: Could not merge headers: {e}")
        
        # Fill data rows
        current_row = 2
        domain_start_row = current_row
        
        for class_id, data in sorted(classification_map.items()):
            row = table.rows[current_row]
            
            current_class = data['current']
            previous_class = data['previous']
            
            # Get classification names (prefer current, fallback to previous)
            if current_class:
                class_name_ar = current_class.get('classification_name', 'N/A')
                class_name_en = current_class.get('classification_name_en', 'N/A')
            else:
                class_name_ar = previous_class.get('classification_name', 'N/A')
                class_name_en = previous_class.get('classification_name_en', 'N/A')
            
            # Column 0: Domain (will be merged later)
            row.cells[0].text = ""
            
            # Column 1: Classification AR
            row.cells[1].text = class_name_ar
            row.cells[1].paragraphs[0].runs[0].font.size = Pt(9)
            row.cells[1].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
            row.cells[1].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
            center_cell_content(row.cells[1])
            
            # Column 2: Classification EN
            row.cells[2].text = class_name_en
            row.cells[2].paragraphs[0].runs[0].font.size = Pt(8)
            center_cell_content(row.cells[2])
            
            # Previous period data (cols 3-6)
            if previous_class:
                prev_total = previous_class.get('total_count', 0)
                prev_low = previous_class.get('low_count', 0)
                prev_med = previous_class.get('medium_count', 0)
                prev_high = previous_class.get('high_count', 0)
            else:
                prev_total = prev_low = prev_med = prev_high = 0
            
            row.cells[3].text = str(prev_total)
            row.cells[3].paragraphs[0].runs[0].bold = True
            center_cell_content(row.cells[3])
            
            row.cells[4].text = str(prev_low) if prev_low > 0 else ""
            center_cell_content(row.cells[4])
            set_cell_shading(row.cells[4], 'C6EFCE' if prev_low > 0 else 'F2F2F2')
            
            row.cells[5].text = str(prev_med) if prev_med > 0 else ""
            center_cell_content(row.cells[5])
            set_cell_shading(row.cells[5], 'FFEB9C' if prev_med > 0 else 'F2F2F2')
            
            row.cells[6].text = str(prev_high) if prev_high > 0 else ""
            center_cell_content(row.cells[6])
            set_cell_shading(row.cells[6], 'FFC7CE' if prev_high > 0 else 'F2F2F2')
            
            # Current period data (cols 7-10)
            if current_class:
                curr_total = current_class.get('total_count', 0)
                curr_low = current_class.get('low_count', 0)
                curr_med = current_class.get('medium_count', 0)
                curr_high = current_class.get('high_count', 0)
            else:
                curr_total = curr_low = curr_med = curr_high = 0
            
            row.cells[7].text = str(curr_total)
            row.cells[7].paragraphs[0].runs[0].bold = True
            center_cell_content(row.cells[7])
            
            row.cells[8].text = str(curr_low) if curr_low > 0 else ""
            center_cell_content(row.cells[8])
            set_cell_shading(row.cells[8], 'C6EFCE' if curr_low > 0 else 'F2F2F2')
            
            row.cells[9].text = str(curr_med) if curr_med > 0 else ""
            center_cell_content(row.cells[9])
            set_cell_shading(row.cells[9], 'FFEB9C' if curr_med > 0 else 'F2F2F2')
            
            row.cells[10].text = str(curr_high) if curr_high > 0 else ""
            center_cell_content(row.cells[10])
            set_cell_shading(row.cells[10], 'FFC7CE' if curr_high > 0 else 'F2F2F2')
            
            current_row += 1
        
        # Merge Domain column for all rows
        if current_row - 1 > domain_start_row:
            table.rows[domain_start_row].cells[0].text = domain_name
            table.rows[domain_start_row].cells[0].paragraphs[0].runs[0].font.size = Pt(11)
            table.rows[domain_start_row].cells[0].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
            table.rows[domain_start_row].cells[0].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
            table.rows[domain_start_row].cells[0].paragraphs[0].runs[0].font.bold = True
            center_cell_content(table.rows[domain_start_row].cells[0])
            set_cell_shading(table.rows[domain_start_row].cells[0], 'BDD7EE')
            
            try:
                for row_idx in range(domain_start_row + 1, current_row):
                    if row_idx < len(table.rows):
                        table.rows[domain_start_row].cells[0].merge(table.rows[row_idx].cells[0])
            except Exception as e:
                print(f"[FORMATTER] Warning: Could not merge domain cells: {e}")
    
    doc.add_paragraph()


def generate_3_quarter_comparison_report(comparison_data: Dict[str, Any], language: str = 'ar') -> Document:
    """
    Generate a professional Word document comparing 3 seasonal quarters.
    
    Features:
    - Summary table with 3 quarters + trend indicators
    - Hierarchical tables showing domain/category/subcategory comparisons
    - Spider charts only (3 graphs: Domain, Category, SubCategory)
    - Trend indicators (↑↑, ↑, →, ↓, ↓↓)
    - A4 Landscape orientation with Arabic support
    
    Args:
        comparison_data: Dictionary containing comparison data from seasonal_comparison_service
        language: Language code ('ar' or 'en')
        
    Returns:
        Document object ready for saving
    """
    doc = Document()
    
    # Configure page layout (A4 Landscape)
    section = doc.sections[0]
    section.page_width = Mm(297)
    section.page_height = Mm(210)
    section.left_margin = Cm(1.5)
    section.right_margin = Cm(1.5)
    section.top_margin = Cm(1.5)
    section.bottom_margin = Cm(1.5)
    
    reports = comparison_data['reports']
    periods = comparison_data['periods']
    trends = comparison_data['trends']
    orgunit_name = comparison_data.get('orgunit_name', 'N/A')
    
    # =============================
    # 1. HEADER WITH LOGO
    # =============================
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
        print(f"[FORMATTER] Could not add logo: {e}")
        pass
    
    # Title
    title_text = "تقرير المقارنة الموسمية - 3 أرباع" if language == 'ar' else "Seasonal Comparison Report - 3 Quarters"
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run(title_text)
    title_run.font.size = Pt(18)
    title_run.font.bold = True
    title_run.font.name = 'Traditional Arabic'
    title_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    doc.add_paragraph()
    
    # =============================
    # 2. SUMMARY TABLE (3 QUARTERS + TRENDS)
    # =============================
    summary_heading = doc.add_paragraph()
    summary_heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    summary_run = summary_heading.add_run("📊 ملخص المقارنة | Comparison Summary")
    summary_run.font.size = Pt(14)
    summary_run.font.bold = True
    summary_run.font.name = 'Traditional Arabic'
    summary_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    # Create summary table: 5 columns (Metric | Q1 | Q2 | Q3 | Trend)
    summary_table = doc.add_table(rows=11, cols=5)
    summary_table.alignment = WD_TABLE_ALIGNMENT.RIGHT
    summary_table.style = 'Table Grid'
    
    # Header row
    headers = ['الاتجاه | Trend', periods[2], periods[1], periods[0], 'المؤشر | Metric']
    for i, header_text in enumerate(headers):
        cell = summary_table.rows[0].cells[i]
        cell.text = header_text
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(11)
        cell.paragraphs[0].runs[0].font.name = 'Traditional Arabic'
        cell.paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        center_cell_content(cell)
        set_cell_shading(cell, '4472C4')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    # Data rows
    metrics = [
        ('إجمالي الحالات | Total Cases', 'total_cases', 'total_cases'),
        ('السريرية | Clinical', 'clinical_domain_count', 'clinical'),
        ('الإدارية | Management', 'management_domain_count', 'management'),
        ('العلاقاتية | Relational', 'relational_domain_count', 'relational'),
        ('منخفضة الخطورة | Low Severity', 'low_severity_count', 'low_severity'),
        ('متوسطة الخطورة | Medium Severity', 'medium_severity_count', 'medium_severity'),
        ('عالية الخطورة | High Severity', 'high_severity_count', 'high_severity'),
        ('إجراءات وقائية | Prevention Actions', 'prevention_action_count', None),
        ('تفسيرات مقدمة | Explanations Submitted', 'explanation_count', None),
        ('حالات مفتوحة | Open Cases', 'open_cases_count', None)
    ]
    
    for row_idx, (metric_label, metric_key, trend_key) in enumerate(metrics, start=1):
        row = summary_table.rows[row_idx]
        
        # Metric name
        row.cells[4].text = metric_label
        row.cells[4].paragraphs[0].runs[0].font.size = Pt(10)
        row.cells[4].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
        row.cells[4].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        row.cells[4].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        set_cell_shading(row.cells[4], 'D9E2F3')
        
        # Q1, Q2, Q3 values
        for i in range(3):
            value = reports[i]['header'].get(metric_key, 0)
            row.cells[3-i].text = str(value)
            row.cells[3-i].paragraphs[0].runs[0].font.size = Pt(10)
            center_cell_content(row.cells[3-i])
        
        # Trend indicator
        if trend_key and trend_key in trends:
            row.cells[0].text = trends[trend_key]
            row.cells[0].paragraphs[0].runs[0].font.size = Pt(14)
            center_cell_content(row.cells[0])
            
            # Color code trends
            trend_value = trends[trend_key]
            if trend_value in ['↑', '↑↑']:
                set_cell_shading(row.cells[0], 'C6EFCE')  # Green
            elif trend_value in ['↓', '↓↓']:
                set_cell_shading(row.cells[0], 'FFC7CE')  # Red
            else:
                set_cell_shading(row.cells[0], 'FFEB9C')  # Yellow
        else:
            row.cells[0].text = '—'
            center_cell_content(row.cells[0])
    
    doc.add_paragraph()
    
    # =============================
    # 3. HIERARCHICAL COMPARISON TABLES
    # =============================
    
    # 3A. Domain-Level Comparison
    domain_heading = doc.add_paragraph()
    domain_heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    domain_run = domain_heading.add_run("🔷 مقارنة المجالات | Domain Comparison")
    domain_run.font.size = Pt(14)
    domain_run.font.bold = True
    domain_run.font.name = 'Traditional Arabic'
    domain_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    domain_data = comparison_data['domain_comparison']
    _create_3quarter_hierarchical_table(doc, domain_data, periods, level='domain')
    
    doc.add_paragraph()
    
    # 3B. Category-Level Comparison
    category_heading = doc.add_paragraph()
    category_heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    category_run = category_heading.add_run("🔶 مقارنة الفئات | Category Comparison")
    category_run.font.size = Pt(14)
    category_run.font.bold = True
    category_run.font.name = 'Traditional Arabic'
    category_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    category_data = comparison_data['category_comparison']
    _create_3quarter_hierarchical_table(doc, category_data, periods, level='category')
    
    doc.add_paragraph()
    
    # 3C. SubCategory-Level Comparison
    subcategory_heading = doc.add_paragraph()
    subcategory_heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    subcategory_run = subcategory_heading.add_run("🔸 مقارنة الفئات الفرعية | SubCategory Comparison")
    subcategory_run.font.size = Pt(14)
    subcategory_run.font.bold = True
    subcategory_run.font.name = 'Traditional Arabic'
    subcategory_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    subcategory_data = comparison_data['subcategory_comparison']
    _create_3quarter_hierarchical_table(doc, subcategory_data, periods, level='subcategory')
    
    # =============================
    # 4. PAGE BREAK BEFORE GRAPHS
    # =============================
    doc.add_page_break()
    
    # =============================
    # 5. VISUAL ANALYSIS - SPIDER CHARTS ONLY
    # =============================
    visual_heading = doc.add_paragraph()
    visual_heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    visual_run = visual_heading.add_run("📊 التحليل البصري | Visual Analysis")
    visual_run.font.size = Pt(16)
    visual_run.font.bold = True
    visual_run.font.name = 'Traditional Arabic'
    visual_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    doc.add_paragraph()
    
    # 5A. Domain Spider Chart
    domain_spider = _generate_3quarter_spider_chart(
        domain_data,
        periods,
        title_ar="مخطط العنكبوت - المجالات",
        title_en="Domain Spider Chart"
    )
    
    spider_para = doc.add_paragraph()
    spider_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    spider_run = spider_para.add_run()
    spider_run.add_picture(domain_spider, width=Inches(7))
    
    caption_para = doc.add_paragraph()
    caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_run = caption_para.add_run("🕸️ مخطط العنكبوت - المجالات | Domain Spider Chart")
    caption_run.font.bold = True
    caption_run.font.size = Pt(11)
    caption_run.font.name = 'Traditional Arabic'
    caption_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    doc.add_paragraph()
    
    # 5B. Category Spider Chart
    category_spider = _generate_3quarter_spider_chart(
        category_data,
        periods,
        title_ar="مخطط العنكبوت - الفئات",
        title_en="Category Spider Chart",
        max_items=10
    )
    
    spider_para = doc.add_paragraph()
    spider_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    spider_run = spider_para.add_run()
    spider_run.add_picture(category_spider, width=Inches(7))
    
    caption_para = doc.add_paragraph()
    caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_run = caption_para.add_run("🕸️ مخطط العنكبوت - الفئات | Category Spider Chart")
    caption_run.font.bold = True
    caption_run.font.size = Pt(11)
    caption_run.font.name = 'Traditional Arabic'
    caption_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    doc.add_paragraph()
    
    # 5C. SubCategory Spider Chart
    subcategory_spider = _generate_3quarter_spider_chart(
        subcategory_data,
        periods,
        title_ar="مخطط العنكبوت - الفئات الفرعية",
        title_en="SubCategory Spider Chart",
        max_items=12
    )
    
    spider_para = doc.add_paragraph()
    spider_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    spider_run = spider_para.add_run()
    spider_run.add_picture(subcategory_spider, width=Inches(7))
    
    caption_para = doc.add_paragraph()
    caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_run = caption_para.add_run("🕸️ مخطط العنكبوت - الفئات الفرعية | SubCategory Spider Chart")
    caption_run.font.bold = True
    caption_run.font.size = Pt(11)
    caption_run.font.name = 'Traditional Arabic'
    caption_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    return doc


def _create_3quarter_hierarchical_table(doc: Document, data: Dict[str, List[int]], periods: List[str], level: str):
    """
    Create a hierarchical table comparing 3 quarters for a specific level (domain/category/subcategory).
    
    Args:
        doc: Document object
        data: Dictionary mapping item names to list of 3 values
        periods: List of 3 period labels
        level: 'domain', 'category', or 'subcategory'
    """
    # Create table: 5 columns (Item | Q1 | Q2 | Q3 | Trend)
    num_rows = len(data) + 2  # +1 for header, +1 for totals
    table = doc.add_table(rows=num_rows, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.RIGHT
    table.style = 'Table Grid'
    
    # Header row
    headers = ['الاتجاه | Trend', periods[2], periods[1], periods[0], f'{"المجال" if level == "domain" else "الفئة" if level == "category" else "الفئة الفرعية"} | {level.title()}']
    for i, header_text in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header_text
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(11)
        cell.paragraphs[0].runs[0].font.name = 'Traditional Arabic'
        cell.paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        center_cell_content(cell)
        set_cell_shading(cell, '4472C4')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    # Data rows
    totals = [0, 0, 0]
    for row_idx, (item_name, values) in enumerate(sorted(data.items()), start=1):
        row = table.rows[row_idx]
        
        # Item name
        row.cells[4].text = item_name
        row.cells[4].paragraphs[0].runs[0].font.size = Pt(10)
        row.cells[4].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
        row.cells[4].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        row.cells[4].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        
        # Quarter values
        for i in range(3):
            row.cells[3-i].text = str(values[i])
            row.cells[3-i].paragraphs[0].runs[0].font.size = Pt(10)
            center_cell_content(row.cells[3-i])
            totals[i] += values[i]
        
        # Trend indicator
        trend = _calculate_trend_indicator(values[0], values[-1])
        row.cells[0].text = trend
        row.cells[0].paragraphs[0].runs[0].font.size = Pt(14)
        center_cell_content(row.cells[0])
        
        # Color code trends
        if trend in ['↑', '↑↑']:
            set_cell_shading(row.cells[0], 'C6EFCE')  # Green
        elif trend in ['↓', '↓↓']:
            set_cell_shading(row.cells[0], 'FFC7CE')  # Red
        else:
            set_cell_shading(row.cells[0], 'FFEB9C')  # Yellow
    
    # Totals row
    totals_row = table.rows[-1]
    totals_row.cells[4].text = 'الإجمالي | Total'
    totals_row.cells[4].paragraphs[0].runs[0].font.bold = True
    totals_row.cells[4].paragraphs[0].runs[0].font.size = Pt(11)
    totals_row.cells[4].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
    totals_row.cells[4].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    totals_row.cells[4].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_cell_shading(totals_row.cells[4], 'D9E2F3')
    
    for i in range(3):
        totals_row.cells[3-i].text = str(totals[i])
        totals_row.cells[3-i].paragraphs[0].runs[0].font.bold = True
        totals_row.cells[3-i].paragraphs[0].runs[0].font.size = Pt(11)
        center_cell_content(totals_row.cells[3-i])
        set_cell_shading(totals_row.cells[3-i], 'D9E2F3')
    
    # Overall trend
    overall_trend = _calculate_trend_indicator(totals[0], totals[-1])
    totals_row.cells[0].text = overall_trend
    totals_row.cells[0].paragraphs[0].runs[0].font.size = Pt(14)
    totals_row.cells[0].paragraphs[0].runs[0].font.bold = True
    center_cell_content(totals_row.cells[0])
    set_cell_shading(totals_row.cells[0], 'BDD7EE')


def _calculate_trend_indicator(first_value: int, last_value: int) -> str:
    """
    Calculate trend indicator for 3-quarter comparison.
    
    Returns:
        Trend string: '↑↑', '↑', '→', '↓', '↓↓'
    """
    if first_value == 0:
        if last_value == 0:
            return '→'
        else:
            return '↑↑'
    
    change_percent = ((last_value - first_value) / first_value) * 100
    
    if change_percent > 20:
        return '↑↑'
    elif change_percent > 5:
        return '↑'
    elif change_percent < -20:
        return '↓↓'
    elif change_percent < -5:
        return '↓'
    else:
        return '→'


def _generate_3quarter_spider_chart(
    data: Dict[str, List[int]],
    periods: List[str],
    title_ar: str,
    title_en: str,
    max_items: int = 8
) -> io.BytesIO:
    """
    Generate a spider chart comparing 3 quarters for multiple items.
    
    Args:
        data: Dictionary mapping item names to list of 3 values
        periods: List of 3 period labels
        title_ar: Arabic title
        title_en: English title
        max_items: Maximum number of items to display
        
    Returns:
        BytesIO buffer containing the chart image
    """
    # Setup Arabic font
    font_path = 'C:/Windows/Fonts/trado.ttf'
    if os.path.exists(font_path):
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
    
    # Select top items by total count
    sorted_items = sorted(data.items(), key=lambda x: sum(x[1]), reverse=True)[:max_items]
    
    if not sorted_items:
        # Return empty chart
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        ax.text(0.5, 0.5, 'لا توجد بيانات\nNo Data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf
    
    categories = [item[0] for item in sorted_items]
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # Colors for 3 quarters
    colors = ['#4472C4', '#ED7D31', '#A5A5A5']
    
    # Plot data for each quarter
    for i in range(3):
        values = [item[1][i] for item in sorted_items]
        values += values[:1]  # Close the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=periods[i], color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    
    # Add title
    title_text = f"{title_ar}\n{title_en}"
    plt.title(title_text, size=14, weight='bold', pad=20)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf


def generate_4_quarter_comparison_report(comparison_data: Dict[str, Any], language: str = 'ar') -> Document:
    """
    Generate a professional Word document comparing 4 seasonal quarters (full year).
    
    Features:
    - Summary table with 4 quarters + yearly total + trend indicators
    - Hierarchical tables showing domain/category/subcategory comparisons
    - Spider charts only (3 graphs: Domain, Category, SubCategory)
    - Trend indicators (↑↑, ↑, →, ↓, ↓↓)
    - A4 Landscape orientation with Arabic support
    
    Args:
        comparison_data: Dictionary containing comparison data from seasonal_comparison_service
        language: Language code ('ar' or 'en')
        
    Returns:
        Document object ready for saving
    """
    doc = Document()
    
    # Configure page layout (A4 Landscape)
    section = doc.sections[0]
    section.page_width = Mm(297)
    section.page_height = Mm(210)
    section.left_margin = Cm(1.5)
    section.right_margin = Cm(1.5)
    section.top_margin = Cm(1.5)
    section.bottom_margin = Cm(1.5)
    
    reports = comparison_data['reports']
    periods = comparison_data['periods']
    trends = comparison_data['trends']
    yearly_totals = comparison_data['yearly_totals']
    orgunit_name = comparison_data.get('orgunit_name', 'N/A')
    
    # =============================
    # 1. HEADER WITH LOGO
    # =============================
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
        print(f"[FORMATTER] Could not add logo: {e}")
        pass
    
    # Title
    title_text = "تقرير المقارنة الموسمية - التقرير السنوي (4 أرباع)" if language == 'ar' else "Seasonal Comparison Report - Annual (4 Quarters)"
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run(title_text)
    title_run.font.size = Pt(18)
    title_run.font.bold = True
    title_run.font.name = 'Traditional Arabic'
    title_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    doc.add_paragraph()
    
    # =============================
    # 2. SUMMARY TABLE (4 QUARTERS + YEARLY TOTAL + TRENDS)
    # =============================
    summary_heading = doc.add_paragraph()
    summary_heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    summary_run = summary_heading.add_run("📊 ملخص المقارنة السنوية | Annual Comparison Summary")
    summary_run.font.size = Pt(14)
    summary_run.font.bold = True
    summary_run.font.name = 'Traditional Arabic'
    summary_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    # Create summary table: 7 columns (Metric | Q1 | Q2 | Q3 | Q4 | Yearly Total | Trend)
    summary_table = doc.add_table(rows=11, cols=7)
    summary_table.alignment = WD_TABLE_ALIGNMENT.RIGHT
    summary_table.style = 'Table Grid'
    
    # Header row
    headers = ['الاتجاه | Trend', 'الإجمالي السنوي | Yearly Total', periods[3], periods[2], periods[1], periods[0], 'المؤشر | Metric']
    for i, header_text in enumerate(headers):
        cell = summary_table.rows[0].cells[i]
        cell.text = header_text
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(10)
        cell.paragraphs[0].runs[0].font.name = 'Traditional Arabic'
        cell.paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        center_cell_content(cell)
        set_cell_shading(cell, '4472C4')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    # Data rows
    metrics = [
        ('إجمالي الحالات | Total Cases', 'total_cases', 'total_cases'),
        ('السريرية | Clinical', 'clinical_domain_count', 'clinical'),
        ('الإدارية | Management', 'management_domain_count', 'management'),
        ('العلاقاتية | Relational', 'relational_domain_count', 'relational'),
        ('منخفضة الخطورة | Low Severity', 'low_severity_count', 'low_severity'),
        ('متوسطة الخطورة | Medium Severity', 'medium_severity_count', 'medium_severity'),
        ('عالية الخطورة | High Severity', 'high_severity_count', 'high_severity'),
        ('إجراءات وقائية | Prevention Actions', 'prevention_action_count', None),
        ('تفسيرات مقدمة | Explanations Submitted', 'explanation_count', None),
        ('حالات مفتوحة | Open Cases', 'open_cases_count', None)
    ]
    
    for row_idx, (metric_label, metric_key, trend_key) in enumerate(metrics, start=1):
        row = summary_table.rows[row_idx]
        
        # Metric name
        row.cells[6].text = metric_label
        row.cells[6].paragraphs[0].runs[0].font.size = Pt(9)
        row.cells[6].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
        row.cells[6].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        row.cells[6].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        set_cell_shading(row.cells[6], 'D9E2F3')
        
        # Q1, Q2, Q3, Q4 values
        for i in range(4):
            value = reports[i]['header'].get(metric_key, 0)
            row.cells[5-i].text = str(value)
            row.cells[5-i].paragraphs[0].runs[0].font.size = Pt(9)
            center_cell_content(row.cells[5-i])
        
        # Yearly total
        if trend_key and trend_key in yearly_totals:
            yearly_value = yearly_totals[trend_key]
        else:
            yearly_value = sum(reports[i]['header'].get(metric_key, 0) for i in range(4))
        
        row.cells[1].text = str(yearly_value)
        row.cells[1].paragraphs[0].runs[0].font.size = Pt(9)
        row.cells[1].paragraphs[0].runs[0].font.bold = True
        center_cell_content(row.cells[1])
        set_cell_shading(row.cells[1], 'E7E6E6')
        
        # Trend indicator
        if trend_key and trend_key in trends:
            row.cells[0].text = trends[trend_key]
            row.cells[0].paragraphs[0].runs[0].font.size = Pt(14)
            center_cell_content(row.cells[0])
            
            # Color code trends
            trend_value = trends[trend_key]
            if trend_value in ['↑', '↑↑']:
                set_cell_shading(row.cells[0], 'C6EFCE')  # Green
            elif trend_value in ['↓', '↓↓']:
                set_cell_shading(row.cells[0], 'FFC7CE')  # Red
            else:
                set_cell_shading(row.cells[0], 'FFEB9C')  # Yellow
        else:
            row.cells[0].text = '—'
            center_cell_content(row.cells[0])
    
    doc.add_paragraph()
    
    # =============================
    # 3. HIERARCHICAL COMPARISON TABLES
    # =============================
    
    # 3A. Domain-Level Comparison
    domain_heading = doc.add_paragraph()
    domain_heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    domain_run = domain_heading.add_run("🔷 مقارنة المجالات | Domain Comparison")
    domain_run.font.size = Pt(14)
    domain_run.font.bold = True
    domain_run.font.name = 'Traditional Arabic'
    domain_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    domain_data = comparison_data['domain_comparison']
    _create_4quarter_hierarchical_table(doc, domain_data, periods, yearly_totals=None, level='domain')
    
    doc.add_paragraph()
    
    # 3B. Category-Level Comparison
    category_heading = doc.add_paragraph()
    category_heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    category_run = category_heading.add_run("🔶 مقارنة الفئات | Category Comparison")
    category_run.font.size = Pt(14)
    category_run.font.bold = True
    category_run.font.name = 'Traditional Arabic'
    category_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    category_data = comparison_data['category_comparison']
    _create_4quarter_hierarchical_table(doc, category_data, periods, yearly_totals=None, level='category')
    
    doc.add_paragraph()
    
    # 3C. SubCategory-Level Comparison
    subcategory_heading = doc.add_paragraph()
    subcategory_heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    subcategory_run = subcategory_heading.add_run("🔸 مقارنة الفئات الفرعية | SubCategory Comparison")
    subcategory_run.font.size = Pt(14)
    subcategory_run.font.bold = True
    subcategory_run.font.name = 'Traditional Arabic'
    subcategory_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    subcategory_data = comparison_data['subcategory_comparison']
    _create_4quarter_hierarchical_table(doc, subcategory_data, periods, yearly_totals=None, level='subcategory')
    
    # =============================
    # 4. PAGE BREAK BEFORE GRAPHS
    # =============================
    doc.add_page_break()
    
    # =============================
    # 5. VISUAL ANALYSIS - SPIDER CHARTS ONLY
    # =============================
    visual_heading = doc.add_paragraph()
    visual_heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    visual_run = visual_heading.add_run("📊 التحليل البصري | Visual Analysis")
    visual_run.font.size = Pt(16)
    visual_run.font.bold = True
    visual_run.font.name = 'Traditional Arabic'
    visual_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    doc.add_paragraph()
    
    # 5A. Domain Spider Chart
    domain_spider = _generate_4quarter_spider_chart(
        domain_data,
        periods,
        title_ar="مخطط العنكبوت - المجالات",
        title_en="Domain Spider Chart"
    )
    
    spider_para = doc.add_paragraph()
    spider_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    spider_run = spider_para.add_run()
    spider_run.add_picture(domain_spider, width=Inches(7))
    
    caption_para = doc.add_paragraph()
    caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_run = caption_para.add_run("🕸️ مخطط العنكبوت - المجالات | Domain Spider Chart")
    caption_run.font.bold = True
    caption_run.font.size = Pt(11)
    caption_run.font.name = 'Traditional Arabic'
    caption_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    doc.add_paragraph()
    
    # 5B. Category Spider Chart
    category_spider = _generate_4quarter_spider_chart(
        category_data,
        periods,
        title_ar="مخطط العنكبوت - الفئات",
        title_en="Category Spider Chart",
        max_items=10
    )
    
    spider_para = doc.add_paragraph()
    spider_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    spider_run = spider_para.add_run()
    spider_run.add_picture(category_spider, width=Inches(7))
    
    caption_para = doc.add_paragraph()
    caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_run = caption_para.add_run("🕸️ مخطط العنكبوت - الفئات | Category Spider Chart")
    caption_run.font.bold = True
    caption_run.font.size = Pt(11)
    caption_run.font.name = 'Traditional Arabic'
    caption_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    doc.add_paragraph()
    
    # 5C. SubCategory Spider Chart
    subcategory_spider = _generate_4quarter_spider_chart(
        subcategory_data,
        periods,
        title_ar="مخطط العنكبوت - الفئات الفرعية",
        title_en="SubCategory Spider Chart",
        max_items=12
    )
    
    spider_para = doc.add_paragraph()
    spider_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    spider_run = spider_para.add_run()
    spider_run.add_picture(subcategory_spider, width=Inches(7))
    
    caption_para = doc.add_paragraph()
    caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_run = caption_para.add_run("🕸️ مخطط العنكبوت - الفئات الفرعية | SubCategory Spider Chart")
    caption_run.font.bold = True
    caption_run.font.size = Pt(11)
    caption_run.font.name = 'Traditional Arabic'
    caption_run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    
    return doc


def _create_4quarter_hierarchical_table(doc: Document, data: Dict[str, List[int]], periods: List[str], yearly_totals: Optional[Dict] = None, level: str = 'domain'):
    """
    Create a hierarchical table comparing 4 quarters for a specific level (domain/category/subcategory).
    
    Args:
        doc: Document object
        data: Dictionary mapping item names to list of 4 values
        periods: List of 4 period labels
        yearly_totals: Optional dictionary of yearly totals
        level: 'domain', 'category', or 'subcategory'
    """
    # Create table: 7 columns (Item | Q1 | Q2 | Q3 | Q4 | Yearly Total | Trend)
    num_rows = len(data) + 2  # +1 for header, +1 for totals
    table = doc.add_table(rows=num_rows, cols=7)
    table.alignment = WD_TABLE_ALIGNMENT.RIGHT
    table.style = 'Table Grid'
    
    # Header row
    level_label = "المجال" if level == "domain" else "الفئة" if level == "category" else "الفئة الفرعية"
    headers = ['الاتجاه | Trend', 'الإجمالي السنوي | Yearly', periods[3], periods[2], periods[1], periods[0], f'{level_label} | {level.title()}']
    for i, header_text in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header_text
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(10)
        cell.paragraphs[0].runs[0].font.name = 'Traditional Arabic'
        cell.paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        center_cell_content(cell)
        set_cell_shading(cell, '4472C4')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    # Data rows
    totals = [0, 0, 0, 0]
    for row_idx, (item_name, values) in enumerate(sorted(data.items()), start=1):
        row = table.rows[row_idx]
        
        # Item name
        row.cells[6].text = item_name
        row.cells[6].paragraphs[0].runs[0].font.size = Pt(9)
        row.cells[6].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
        row.cells[6].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
        row.cells[6].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        
        # Quarter values
        for i in range(4):
            row.cells[5-i].text = str(values[i])
            row.cells[5-i].paragraphs[0].runs[0].font.size = Pt(9)
            center_cell_content(row.cells[5-i])
            totals[i] += values[i]
        
        # Yearly total for this item
        yearly = sum(values)
        row.cells[1].text = str(yearly)
        row.cells[1].paragraphs[0].runs[0].font.size = Pt(9)
        row.cells[1].paragraphs[0].runs[0].font.bold = True
        center_cell_content(row.cells[1])
        set_cell_shading(row.cells[1], 'F2F2F2')
        
        # Trend indicator
        trend = _calculate_trend_indicator(values[0], values[-1])
        row.cells[0].text = trend
        row.cells[0].paragraphs[0].runs[0].font.size = Pt(14)
        center_cell_content(row.cells[0])
        
        # Color code trends
        if trend in ['↑', '↑↑']:
            set_cell_shading(row.cells[0], 'C6EFCE')  # Green
        elif trend in ['↓', '↓↓']:
            set_cell_shading(row.cells[0], 'FFC7CE')  # Red
        else:
            set_cell_shading(row.cells[0], 'FFEB9C')  # Yellow
    
    # Totals row
    totals_row = table.rows[-1]
    totals_row.cells[6].text = 'الإجمالي | Total'
    totals_row.cells[6].paragraphs[0].runs[0].font.bold = True
    totals_row.cells[6].paragraphs[0].runs[0].font.size = Pt(10)
    totals_row.cells[6].paragraphs[0].runs[0].font.name = 'Traditional Arabic'
    totals_row.cells[6].paragraphs[0].runs[0]._element.rPr.rFonts.set(qn('w:eastAsia'), 'Traditional Arabic')
    totals_row.cells[6].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_cell_shading(totals_row.cells[6], 'D9E2F3')
    
    for i in range(4):
        totals_row.cells[5-i].text = str(totals[i])
        totals_row.cells[5-i].paragraphs[0].runs[0].font.bold = True
        totals_row.cells[5-i].paragraphs[0].runs[0].font.size = Pt(10)
        center_cell_content(totals_row.cells[5-i])
        set_cell_shading(totals_row.cells[5-i], 'D9E2F3')
    
    # Yearly grand total
    grand_yearly = sum(totals)
    totals_row.cells[1].text = str(grand_yearly)
    totals_row.cells[1].paragraphs[0].runs[0].font.bold = True
    totals_row.cells[1].paragraphs[0].runs[0].font.size = Pt(10)
    center_cell_content(totals_row.cells[1])
    set_cell_shading(totals_row.cells[1], 'BDD7EE')
    
    # Overall trend
    overall_trend = _calculate_trend_indicator(totals[0], totals[-1])
    totals_row.cells[0].text = overall_trend
    totals_row.cells[0].paragraphs[0].runs[0].font.size = Pt(14)
    totals_row.cells[0].paragraphs[0].runs[0].font.bold = True
    center_cell_content(totals_row.cells[0])
    set_cell_shading(totals_row.cells[0], 'BDD7EE')


def _generate_4quarter_spider_chart(
    data: Dict[str, List[int]],
    periods: List[str],
    title_ar: str,
    title_en: str,
    max_items: int = 8
) -> io.BytesIO:
    """
    Generate a spider chart comparing 4 quarters for multiple items.
    
    Args:
        data: Dictionary mapping item names to list of 4 values
        periods: List of 4 period labels
        title_ar: Arabic title
        title_en: English title
        max_items: Maximum number of items to display
        
    Returns:
        BytesIO buffer containing the chart image
    """
    # Setup Arabic font
    font_path = 'C:/Windows/Fonts/trado.ttf'
    if os.path.exists(font_path):
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
    
    # Select top items by total count
    sorted_items = sorted(data.items(), key=lambda x: sum(x[1]), reverse=True)[:max_items]
    
    if not sorted_items:
        # Return empty chart
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        ax.text(0.5, 0.5, 'لا توجد بيانات\nNo Data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf
    
    categories = [item[0] for item in sorted_items]
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # Colors for 4 quarters
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']
    
    # Plot data for each quarter
    for i in range(4):
        values = [item[1][i] for item in sorted_items]
        values += values[:1]  # Close the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=periods[i], color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    
    # Add title
    title_text = f"{title_ar}\n{title_en}"
    plt.title(title_text, size=14, weight='bold', pad=20)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf



