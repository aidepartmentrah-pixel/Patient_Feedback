"""
Monthly Report – Stylish Word Formatter  (HCAT Reporting Refactor – Session 6)

Philosophy
----------
Investigation-assistant document, not just an archive.
Each complaint occupies one full landscape page structured as a visual card.
Notices are grouped compactly (multiple per page).

Layout per complaint card (5 zones):
  Zone 1 – Identity    : 7 labeled boxes  (who / where / when)
  Zone 2 – Class       : 7 labeled boxes  (domain / category / classification)
                         + Severity badge  + Harm Stage dot-scale
  Zone 3 – Stage flow  : 5-step visual timeline + Status badge
  Zone 4 – Content     : Complaint narrative (left 60%) | Actions (right 40%)
  Zone 5 – Approvals   : 4-role signature block + RCA instruction note

Pure renderer: zero DB queries, zero calculations.
All data arrives pre-computed in report_data.

Imports low-level helpers from workflow_activity_word_formatter — no parallel utilities.
"""

import os
from io import BytesIO
from typing import Any, Dict, List, Optional

from docx import Document
from docx.shared import Pt, Mm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from .workflow_activity_word_formatter import (
    _ar_run as _ar_run_base,
    _new_para,
    _set_rtl_table,
    _apply_minimal_table_borders,
    _set_cell_shading,
    _cell_para,
    _set_row_col_width as _set_row_col_width_base,
    _set_para_shading,
    _set_para_bottom_border,
    _fmt_date,
    _mm_to_dxa,
)


def _ar_run(para, text: str, size: int = 10, bold: bool = False,
            italic: bool = False, color: str = None):
    """
    Local wrapper around the shared _ar_run helper that additionally sets
    the w:cs (complex-script) font. The shared helper only sets w:eastAsia,
    which leaves RTL Arabic glyph rendering to the theme's default complex-
    script font — can show as missing-glyph boxes in some renderers. Not
    touching the shared helper itself since other reports depend on it.
    """
    run = _ar_run_base(para, text, size=size, bold=bold, italic=italic, color=color)
    run._element.rPr.rFonts.set(qn('w:cs'), 'Traditional Arabic')
    return run


def _set_row_col_width(row, col_idx: int, mm: float):
    """
    Local wrapper around the shared _set_row_col_width helper that ALSO
    updates the table's tblGrid column width via the column object.

    The shared helper only sets w:tcW on the specific cell — it never
    touches w:tblGrid, which is what Word's tblLayout="fixed" rendering
    actually uses to determine column boundaries. Every prior "width fix"
    in this file changed cell tcW values that Word was silently ignoring,
    because the table's real grid stayed frozen at its auto-generated
    default. Not touching the shared helper since other reports use it.
    """
    _set_row_col_width_base(row, col_idx, mm)
    row.table.columns[col_idx].width = Mm(mm)


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

NAVY       = '1C3A7A'
NAVY_LIGHT = 'EBF3FB'   # identity zone bg
MINT_LIGHT = 'EAFAF1'   # classification zone bg
STAGE_BG   = 'F4F6F9'   # stage zone bg
WHITE      = 'FFFFFF'
GREY_LINE  = 'D0D7E4'
GREY_TEXT  = '6B7A99'
DARK_TEXT  = '1A1A2E'

# Severity
_SEV = {
    'low':    ('D5F5E3', '1E8449', '▼', 'منخفض', 'LOW'),
    'medium': ('FEF9E7', 'D4AC0D', '►', 'متوسط', 'MEDIUM'),
    'high':   ('FADBD8', 'C0392B', '▲', 'مرتفع', 'HIGH'),
}

# Harm stage — 5 levels, escalating palette
_HARM = [
    ('بلا ضرر',   'No Harm',        'D5E8D4', '82B366'),
    ('طفيف',      'Minor',          'DAE8FC', '6C8EBF'),
    ('متوسطة',    'Moderate',       'FFF2CC', 'D6B656'),
    ('ضرر شديد',  'Severe Harm',    'F8CECC', 'B85450'),
    ('وفاة',      'Death',          'E1D5E7', '9673A6'),
]

# Stage of care — 5 steps
_STAGES = [
    ('القبول',             'Admission'),
    ('الفحص والتشخيص',    'Examination & Diagnosis'),
    ('الرعاية في القسم',  'Care on the Ward'),
    ('الإجراء / العملية', 'Operation / Procedure'),
    ('الخروج / التحويل',  'Discharge / Transfer'),
]

# Arabic month names for summary page
_MONTHS_AR = {
    1: 'يناير', 2: 'فبراير', 3: 'مارس', 4: 'أبريل',
    5: 'مايو', 6: 'يونيو', 7: 'يوليو', 8: 'أغسطس',
    9: 'سبتمبر', 10: 'أكتوبر', 11: 'نوفمبر', 12: 'ديسمبر',
}


# ---------------------------------------------------------------------------
# LOW-LEVEL DOCX UTILITIES (supplement to imported helpers)
# ---------------------------------------------------------------------------

def _apply_borders_no_vertical(table, outer: str = 'AAAAAA', outer_sz: int = 4,
                               inner_h: str = 'DDDDDD', inner_h_sz: int = 4):
    """
    Like the shared _apply_minimal_table_borders, but drops the vertical
    inner gridlines entirely (insideV=none). Softens dense grids (e.g. the
    signature block) that otherwise read as a spreadsheet, while keeping
    horizontal row separators and the outer border for structure.
    """
    tbl = table._tbl
    tblPr = tbl.find(qn('w:tblPr'))
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tbl.insert(0, tblPr)
    for old in tblPr.findall(qn('w:tblBorders')):
        tblPr.remove(old)
    bdr = OxmlElement('w:tblBorders')
    for name, color, sz in [
        ('top', outer, outer_sz), ('left', outer, outer_sz),
        ('bottom', outer, outer_sz), ('right', outer, outer_sz),
        ('insideH', inner_h, inner_h_sz),
    ]:
        b = OxmlElement(f'w:{name}')
        b.set(qn('w:val'), 'single')
        b.set(qn('w:sz'), str(sz))
        b.set(qn('w:space'), '0')
        b.set(qn('w:color'), color)
        bdr.append(b)
    v = OxmlElement('w:insideV')
    v.set(qn('w:val'), 'none')
    v.set(qn('w:sz'), '0')
    v.set(qn('w:space'), '0')
    v.set(qn('w:color'), 'auto')
    bdr.append(v)
    tblPr.append(bdr)


def _truncate_for_fit(text: str, max_chars: int,
                       marker: str = '… (النص الكامل في النظام)') -> str:
    """
    Hard single-page guarantee: Content Zone is capped at a fixed height
    (not just a floor), so any field that could grow past its character
    budget gets cut at the last word boundary and tagged with a marker
    pointing back to the system for the full record. Without this, a
    sufficiently long complaint_text/immediate_action/taken_action makes
    the per-complaint page structurally unbounded — this is what makes
    "one page per case, no exceptions" actually true regardless of data.
    """
    text = (text or '').strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(' ')
    if last_space > max_chars * 0.7:
        cut = cut[:last_space]
    return cut.rstrip(' ,.;:،؛') + ' ' + marker


def _set_cell_borders(cell, color: str = GREY_LINE, sz: int = 4):
    """Apply uniform single-line borders to a cell."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    for old in tcPr.findall(qn('w:tcBorders')):
        tcPr.remove(old)
    bdr = OxmlElement('w:tcBorders')
    for side in ('top', 'left', 'bottom', 'right'):
        b = OxmlElement(f'w:{side}')
        b.set(qn('w:val'), 'single')
        b.set(qn('w:sz'), str(sz))
        b.set(qn('w:space'), '0')
        b.set(qn('w:color'), color)
        bdr.append(b)
    tcPr.append(bdr)


def _remove_cell_borders(cell):
    """Remove all borders from a cell (none style)."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    for old in tcPr.findall(qn('w:tcBorders')):
        tcPr.remove(old)
    bdr = OxmlElement('w:tcBorders')
    for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        b = OxmlElement(f'w:{side}')
        b.set(qn('w:val'), 'none')
        b.set(qn('w:sz'), '0')
        b.set(qn('w:space'), '0')
        b.set(qn('w:color'), 'auto')
        bdr.append(b)
    tcPr.append(bdr)


def _set_table_outer_border(table, color: str = NAVY, sz: int = 8):
    """Navy outer border, no inner borders."""
    tbl = table._tbl
    tblPr = tbl.find(qn('w:tblPr'))
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tbl.insert(0, tblPr)
    for old in tblPr.findall(qn('w:tblBorders')):
        tblPr.remove(old)
    bdr = OxmlElement('w:tblBorders')
    for name in ('top', 'left', 'bottom', 'right'):
        b = OxmlElement(f'w:{name}')
        b.set(qn('w:val'), 'single')
        b.set(qn('w:sz'), str(sz))
        b.set(qn('w:space'), '0')
        b.set(qn('w:color'), color)
        bdr.append(b)
    for name in ('insideH', 'insideV'):
        b = OxmlElement(f'w:{name}')
        b.set(qn('w:val'), 'none')
        b.set(qn('w:sz'), '0')
        b.set(qn('w:space'), '0')
        b.set(qn('w:color'), 'auto')
        bdr.append(b)
    tblPr.append(bdr)


def _cell_v_center(cell):
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def _labeled_cell(cell, label: str, value: str,
                  label_color: str = GREY_TEXT,
                  value_color: str = DARK_TEXT,
                  value_size: int = 10,
                  value_bold: bool = True,
                  bg: str = None,
                  align: str = 'right'):
    """
    Two-paragraph cell: small bilingual label on top (always centered —
    label text is "Arabic / English"), larger value below (right-aligned —
    pure Arabic/data content, ragged-left like classic RTL print layouts).
    """
    if bg:
        _set_cell_shading(cell, bg)
    _cell_v_center(cell)
    cell.text = ''

    # label — bilingual "X / Y" → always centered
    lp = cell.paragraphs[0]
    lp.clear()
    lp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    lp.paragraph_format.space_before = Pt(3)
    lp.paragraph_format.space_after  = Pt(1)
    lp._p.get_or_add_pPr().append(OxmlElement('w:bidi'))
    _ar_run(lp, label, size=7, color=label_color)

    # value — pure Arabic/data → right-aligned (RTL start)
    vp = cell.add_paragraph()
    vp.alignment = WD_ALIGN_PARAGRAPH.RIGHT if align == 'right' else WD_ALIGN_PARAGRAPH.CENTER
    vp.paragraph_format.space_before = Pt(2)
    vp.paragraph_format.space_after  = Pt(3)
    vp._p.get_or_add_pPr().append(OxmlElement('w:bidi'))
    _ar_run(vp, value or '—', size=value_size, bold=value_bold, color=value_color)


def _spacer_para(doc: Document, pt: float = 2):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(pt)


def _gap(doc: Document, mm: float):
    """
    Precise vertical gap between zones, in mm. Uses a near-zero-size run
    so the empty paragraph's own line-height doesn't add unaccounted
    height — the gap is controlled almost entirely by space_after.
    """
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(mm * 2.6)
    r = p.add_run('')
    r.font.size = Pt(1)


def _page_break(doc: Document):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(0)
    run = p.add_run()
    run.add_break(WD_BREAK.PAGE)


def _zone_label_row(table, col_span: int, label_ar: str, label_en: str,
                    bg: str = NAVY, text_color: str = WHITE):
    """Single merged header row across all columns — zone title strip."""
    row = table.add_row()
    row.height = _mm_to_dxa(6)
    from docx.enum.table import WD_ROW_HEIGHT_RULE
    row.height_rule = WD_ROW_HEIGHT_RULE.EXACTLY

    cell = row.cells[0]
    # Merge across all columns
    if col_span > 1:
        cell = row.cells[0].merge(row.cells[col_span - 1])

    _set_cell_shading(cell, bg)
    p = cell.paragraphs[0]
    p.clear()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(0)
    p._p.get_or_add_pPr().append(OxmlElement('w:bidi'))
    _ar_run(p, label_ar, size=8, bold=True, color=text_color)
    if label_en:
        _ar_run(p, f'  /  {label_en}', size=7, color=text_color)


# ---------------------------------------------------------------------------
# TARGET DEPARTMENT HELPERS
# ---------------------------------------------------------------------------

def _primary_target(complaint: Dict) -> Dict:
    tds = complaint.get('target_departments') or []
    primary = next((d for d in tds if d.get('is_primary')), tds[0] if tds else {})
    return primary


def _target_names(complaint: Dict):
    p = _primary_target(complaint)
    return (
        p.get('section_name') or '—',
        p.get('department_name') or '—',
        p.get('administration_name') or '—',
    )


def _target_display(complaint: Dict) -> str:
    """Most specific named target unit as a single string."""
    p = _primary_target(complaint)
    return (p.get('section_name') or p.get('department_name') or
            p.get('administration_name') or '—')


# ---------------------------------------------------------------------------
# SEVERITY BADGE  (3-cell mini-table)
# ---------------------------------------------------------------------------

def _severity_key(raw: str) -> str:
    r = (raw or '').lower()
    if 'high' in r or 'مرتفع' in r:   return 'high'
    if 'medium' in r or 'متوسط' in r: return 'medium'
    return 'low'


def _build_severity_cell(parent_cell, severity_raw: str):
    """Render a 3-cell Low/Medium/High badge inside the given cell."""
    _cell_v_center(parent_cell)
    parent_cell.text = ''

    lp = parent_cell.paragraphs[0]
    lp.clear()
    lp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    lp.paragraph_format.space_before = Pt(1)
    lp.paragraph_format.space_after  = Pt(1)
    _ar_run(lp, 'مستوى الخطورة / Severity', size=7, color=GREY_TEXT)

    # Inner mini-table — width must fit within parent cell (~36mm)
    inner = parent_cell.add_table(1, 3)
    inner.autofit = False
    inner.alignment = WD_TABLE_ALIGNMENT.CENTER
    _apply_minimal_table_borders(inner, outer='CCCCCC', outer_sz=4, inner='CCCCCC', inner_sz=4)

    key = _severity_key(severity_raw)
    levels = [('low', 'Low', 'منخفض'), ('medium', 'Medium', 'متوسط'), ('high', 'High', 'مرتفع')]
    for ci, (lk, en, ar) in enumerate(levels):
        cfg = _SEV[lk]
        c = inner.rows[0].cells[ci]
        is_selected = (lk == key)
        _set_cell_shading(c, cfg[0] if is_selected else 'F8F9FA')
        _cell_v_center(c)
        cp = c.paragraphs[0]
        cp.clear()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_before = Pt(1)
        cp.paragraph_format.space_after  = Pt(1)
        _ar_run(cp, cfg[2] + ' ' if is_selected else '   ',
                size=8, bold=is_selected, color=cfg[1] if is_selected else 'AAAAAA')
        _ar_run(cp, ar, size=8, bold=is_selected,
                color=cfg[1] if is_selected else 'CCCCCC')

    # Column widths — 13mm each = 39mm total (parent cell 43mm, 4mm padding).
    # "Medium" (6 chars) at 8pt needs ~9.3mm minimum; old 10mm cell had almost
    # no buffer, letting Word silently expand the column past tblLayout=fixed.
    for ci in range(3):
        _set_row_col_width(inner.rows[0], ci, 13)


# ---------------------------------------------------------------------------
# HARM STAGE DOT-SCALE  (5-cell mini-table)
# ---------------------------------------------------------------------------

def _harm_key(raw: str) -> int:
    """Map harm_level string to 0-based index into _HARM."""
    r = (raw or '').lower()
    if 'death' in r or 'وفاة' in r:           return 4
    if 'severe' in r or 'ضرر شديد' in r:      return 3
    if 'moderate' in r or 'متوسطة' in r:      return 2
    if 'minor' in r or 'طفيف' in r:           return 1
    return 0  # No harm / unknown


def _build_harm_cell(parent_cell, harm_raw: str):
    """5-dot scale rendered as a mini-table inside the given cell."""
    _cell_v_center(parent_cell)
    parent_cell.text = ''

    lp = parent_cell.paragraphs[0]
    lp.clear()
    lp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    lp.paragraph_format.space_before = Pt(1)
    lp.paragraph_format.space_after  = Pt(1)
    _ar_run(lp, 'مرحلة الضرر / Harm Stage', size=7, color=GREY_TEXT)

    selected_idx = _harm_key(harm_raw)

    # Inner mini-table — width must fit within parent cell (~54mm), 5 cols × 10mm = 50mm
    inner = parent_cell.add_table(2, 5)
    inner.autofit = False
    inner.alignment = WD_TABLE_ALIGNMENT.CENTER
    _apply_minimal_table_borders(inner, outer='DDDDDD', outer_sz=4, inner='DDDDDD', inner_sz=2)

    for ci, (ar, en, bg_sel, fg_sel) in enumerate(_HARM):
        is_sel = (ci == selected_idx)
        is_before = (ci < selected_idx)

        # Row 0: dot
        dot_cell = inner.rows[0].cells[ci]
        _set_cell_shading(dot_cell, bg_sel if is_sel else ('F0F0F0' if is_before else 'FAFAFA'))
        _cell_v_center(dot_cell)
        dp = dot_cell.paragraphs[0]
        dp.clear()
        dp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        dp.paragraph_format.space_before = Pt(1)
        dp.paragraph_format.space_after  = Pt(1)
        sym = '■' if is_sel else ('▪' if is_before else '□')
        _ar_run(dp, sym, size=10, bold=is_sel, color=fg_sel if is_sel else ('AAAAAA' if is_before else 'CCCCCC'))

        # Row 1: label
        lbl_cell = inner.rows[1].cells[ci]
        _set_cell_shading(lbl_cell, 'FFFFFF')
        _cell_v_center(lbl_cell)
        lp2 = lbl_cell.paragraphs[0]
        lp2.clear()
        lp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        lp2.paragraph_format.space_before = Pt(0)
        lp2.paragraph_format.space_after  = Pt(1)
        _ar_run(lp2, ar, size=6, bold=is_sel, color=fg_sel if is_sel else GREY_TEXT)

    # Column widths — 12mm each = 60mm total (parent cell 64mm, 4mm padding).
    # "متوسطة" (Moderate, single unbreakable Arabic word, 6 chars) needs ~9mm
    # at 6pt; old 9mm cell had ~0mm buffer — Word silently expanded it past
    # tblLayout=fixed rather than clip the word.
    for ci in range(5):
        for ri in range(2):
            _set_row_col_width(inner.rows[ri], ci, 12)


# ---------------------------------------------------------------------------
# STAGE OF CARE FLOW  (5-step visual bar)
# ---------------------------------------------------------------------------

def _stage_key(raw: str) -> int:
    """Map stage_name to 0-based index in _STAGES."""
    r = (raw or '').lower()
    if 'discharge' in r or 'transfer' in r or 'خروج' in r: return 4
    if 'operation' in r or 'procedure' in r or 'إجراء' in r or 'عملية' in r: return 3
    if 'ward' in r or 'رعاية' in r: return 2
    if 'examination' in r or 'diagnosis' in r or 'فحص' in r: return 1
    if 'admission' in r or 'قبول' in r: return 0
    return -1  # unknown


def _build_stage_row(doc: Document, stage_raw: str, status_raw: str,
                     risk_type: str, page_width_mm: float):
    """
    Full-width stage-flow table.
    Cols: [5 stage cells + 1 status cell + 1 risk-type cell].
    """
    sel_idx = _stage_key(stage_raw)

    # 9 cols: stage0 | arrow | stage1 | arrow | stage2 | arrow | stage3 | arrow | stage4 + status + risk
    # Simpler: 7 cols (5 stages + status + risk), no explicit arrow cells
    n_cols = 7
    tbl = doc.add_table(rows=2, cols=n_cols)
    tbl.autofit = False
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _apply_minimal_table_borders(tbl, outer=NAVY, outer_sz=6, inner='E4E8F0', inner_sz=2)
    _set_rtl_table(tbl)

    # Header strip
    hdr_row = tbl.rows[0]
    hdr_cell = hdr_row.cells[0].merge(hdr_row.cells[n_cols - 1])
    _set_cell_shading(hdr_cell, NAVY)
    hp = hdr_cell.paragraphs[0]
    hp.clear()
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hp.paragraph_format.space_before = Pt(0)
    hp.paragraph_format.space_after  = Pt(0)
    _ar_run(hp, 'مرحلة تقديم الخدمة / Stage of Care', size=8, bold=True, color=WHITE)

    # Data row — target ~250mm (17mm slack vs 267mm usable). Stage cells already
    # have generous room for their content (multi-word, wraps fine) so trimmed
    # slightly; risk-type badge widened ("Never Event" = 11 unbreakable chars).
    data_row = tbl.rows[1]
    stage_widths = [36.0, 42.0, 39.0, 42.0, 42.0]   # 5 stages = 201mm
    status_width  = 26.0
    risk_width    = 23.0                             # 201+26+23 = 250mm

    for ci, (ar, en) in enumerate(_STAGES):
        cell = data_row.cells[ci]
        is_sel = (ci == sel_idx)
        is_prev = (ci < sel_idx) and sel_idx >= 0
        bg = NAVY if is_sel else (NAVY_LIGHT if is_prev else STAGE_BG)
        txt_color = WHITE if is_sel else (NAVY if is_prev else GREY_TEXT)

        _set_cell_shading(cell, bg)
        _cell_v_center(cell)

        cp = cell.paragraphs[0]
        cp.clear()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_before = Pt(3)
        cp.paragraph_format.space_after  = Pt(1)
        cp._p.get_or_add_pPr().append(OxmlElement('w:bidi'))
        _ar_run(cp, ar, size=8, bold=is_sel, color=txt_color)

        ep = cell.add_paragraph()
        ep.alignment = WD_ALIGN_PARAGRAPH.CENTER
        ep.paragraph_format.space_before = Pt(0)
        ep.paragraph_format.space_after  = Pt(3)
        _ar_run(ep, en, size=6, color=txt_color)

        _set_row_col_width(data_row, ci, stage_widths[ci])

    # Visual divider between the 5-stage progression (cols 0-4) and the
    # Status/Risk-type metadata badges (cols 5-6) — those two are NOT part
    # of the stage sequence, so a distinct double-line border separates
    # them from the flow, instead of all 7 cells reading as one sequence.
    for cell_idx, side in ((4, 'right'), (5, 'left')):
        tc = data_row.cells[cell_idx]._tc
        tcPr = tc.get_or_add_tcPr()
        tcBorders = tcPr.find(qn('w:tcBorders'))
        if tcBorders is None:
            tcBorders = OxmlElement('w:tcBorders')
            tcPr.append(tcBorders)
        b = OxmlElement(f'w:{side}')
        b.set(qn('w:val'), 'double')
        b.set(qn('w:sz'), '18')
        b.set(qn('w:space'), '0')
        b.set(qn('w:color'), NAVY)
        tcBorders.append(b)

    # Status cell (col 5)
    st_cell = data_row.cells[5]
    status_lower = (status_raw or '').lower()
    if 'close' in status_lower or 'مغلق' in status_lower:
        st_bg, st_fg = 'D5F5E3', '1E8449'
        st_label = 'مغلق / Closed'
    else:
        st_bg, st_fg = 'FEF9E7', 'D4AC0D'
        st_label = 'مفتوح / Open'

    _set_cell_shading(st_cell, st_bg)
    _cell_v_center(st_cell)
    sp = st_cell.paragraphs[0]
    sp.clear()
    sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sp.paragraph_format.space_before = Pt(2)
    sp.paragraph_format.space_after  = Pt(1)
    sp._p.get_or_add_pPr().append(OxmlElement('w:bidi'))
    _ar_run(sp, 'الحالة / Status', size=7, color=GREY_TEXT)
    vp = st_cell.add_paragraph()
    vp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    vp.paragraph_format.space_before = Pt(1)
    vp.paragraph_format.space_after  = Pt(2)
    _ar_run(vp, st_label, size=9, bold=True, color=st_fg)
    _set_row_col_width(data_row, 5, status_width)

    # Risk type cell (col 6)
    rk_cell = data_row.cells[6]
    rk_lower = (risk_type or '').lower()
    if 'never' in rk_lower:
        rk_bg, rk_fg = 'FADBD8', 'C0392B'
    elif 'red' in rk_lower or 'flag' in rk_lower:
        rk_bg, rk_fg = 'FEF9E7', 'D4AC0D'
    else:
        rk_bg, rk_fg = 'F4F6F9', GREY_TEXT

    _set_cell_shading(rk_cell, rk_bg)
    _cell_v_center(rk_cell)
    rp = rk_cell.paragraphs[0]
    rp.clear()
    rp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rp.paragraph_format.space_before = Pt(2)
    rp.paragraph_format.space_after  = Pt(1)
    _ar_run(rp, 'نوع السجل', size=7, color=GREY_TEXT)
    rvp = rk_cell.add_paragraph()
    rvp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rvp.paragraph_format.space_before = Pt(1)
    rvp.paragraph_format.space_after  = Pt(2)
    _ar_run(rvp, risk_type or 'Ordinary', size=8, bold=True, color=rk_fg)
    _set_row_col_width(data_row, 6, risk_width)

    # Header row height — AT_LEAST (not EXACTLY): a too-tight exact height on
    # a merged cell can collapse to invisible in some renderers.
    from docx.enum.table import WD_ROW_HEIGHT_RULE
    hdr_row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    hdr_row.height = _mm_to_dxa(7)

    # 17mm (slightly taller than Identity's 15mm to give the divider/group
    # styling a bit more breathing room).
    data_row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    data_row.height = _mm_to_dxa(17)


# ---------------------------------------------------------------------------
# COMPLAINT CARD ZONES
# ---------------------------------------------------------------------------

def _build_identity_zone(doc: Document, complaint: Dict, period: Dict,
                         report_entity_name: str, report_entity_type: str):
    """Zone 1 — 7-box identity row."""
    sec_name, dept_name, admin_name = _target_names(complaint)

    # Derive month display from period
    period_label = period.get('label_ar') or period.get('label') or '—'

    cols = 7
    tbl = doc.add_table(rows=2, cols=cols)
    tbl.autofit = False
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _apply_minimal_table_borders(tbl, outer=NAVY, outer_sz=8, inner='E4E8F0', inner_sz=2)
    _set_rtl_table(tbl)

    # Header strip
    hdr = tbl.rows[0]
    hdr_cell = hdr.cells[0].merge(hdr.cells[cols - 1])
    _set_cell_shading(hdr_cell, NAVY)
    hp = hdr_cell.paragraphs[0]
    hp.clear()
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hp.paragraph_format.space_before = Pt(0)
    hp.paragraph_format.space_after  = Pt(0)
    _ar_run(hp, 'بيانات الحالة  /  Case Identity', size=9, bold=True, color=WHITE)

    from docx.enum.table import WD_ROW_HEIGHT_RULE
    hdr.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    hdr.height = _mm_to_dxa(7)

    # Data row
    data = tbl.rows[1]
    inc_id = complaint.get('incident_id')
    id_str = f"INC-{int(inc_id):06d}" if inc_id is not None else str(complaint.get('id', '—'))

    boxes = [
        ('القسم / Section',         sec_name),
        ('الدائرة / Department',     dept_name),
        ('الإدارة / Administration', admin_name),
        ('الشهر / Month',            period_label),
        ('تاريخ التلقي / Received',  _fmt_date(complaint.get('received_date'))),
        ('رقم الشكوى / Case No.',    id_str),
        ('نوع السجل / Type',         complaint.get('feedback_intent_type_name_ar') or 'شكوى'),
    ]

    # sum = 244mm (target ~250mm budget, 23mm slack vs 267mm usable).
    # Case No./Received widened (unbreakable digit-strings); Section/Dept/Admin
    # trimmed (multi-word Arabic wraps fine across 2 lines, low overflow risk).
    widths = [37, 40, 40, 27, 36, 38, 26]
    for ci, ((label, value), w) in enumerate(zip(boxes, widths)):
        cell = data.cells[ci]
        _labeled_cell(cell, label, value, bg=NAVY_LIGHT, value_size=9, align='center')
        _set_row_col_width(data, ci, w)

    # Standardized to 15mm to match Stage row's rhythm (Classification stays
    # taller at 18mm — it genuinely needs the room for its nested mini-tables).
    data.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    data.height = _mm_to_dxa(15)


def _build_classification_zone(doc: Document, complaint: Dict):
    """Zone 2 — 8-box classification row:
    Source | Patient | Domain | Category | Sub-Category | Classification | Severity | Harm."""
    cols = 8
    tbl = doc.add_table(rows=2, cols=cols)
    tbl.autofit = False
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _apply_minimal_table_borders(tbl, outer=NAVY, outer_sz=6, inner='E4E8F0', inner_sz=2)
    _set_rtl_table(tbl)

    from docx.enum.table import WD_ROW_HEIGHT_RULE

    # Header strip
    hdr = tbl.rows[0]
    hdr_cell = hdr.cells[0].merge(hdr.cells[cols - 1])
    _set_cell_shading(hdr_cell, '2E6DA4')
    hp = hdr_cell.paragraphs[0]
    hp.clear()
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hp.paragraph_format.space_before = Pt(0)
    hp.paragraph_format.space_after  = Pt(0)
    _ar_run(hp, 'التصنيف والتحليل  /  Classification & Analysis', size=9, bold=True, color=WHITE)
    hdr.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    hdr.height = _mm_to_dxa(7)

    # Data row
    data = tbl.rows[1]

    class_ar = complaint.get('classification_name') or '—'
    subcat    = complaint.get('subcategory_name') or '—'

    # Cols 0-5: labeled boxes (Source | Patient | Domain | Category | Sub-Category | Classification)
    # Severity (43mm) fits its widened 39mm inner table; Harm (64mm) fits its widened 60mm
    # inner table — these two were the overflow source: short unbreakable English words
    # ("Moderate", "Medium") in 9-10mm cells at 6-8pt don't actually fit, so Word silently
    # expands them past tblLayout=fixed. Other cols trimmed to compensate (multi-word
    # Arabic/English content wraps fine, low overflow risk). Total = 250mm (17mm slack).
    simple_boxes = [
        ('المصدر / Source',            complaint.get('source_name') or '—'),
        ('المريض / Patient',           complaint.get('patient_name') or '—'),
        ('المجال / Domain',            complaint.get('domain_name') or '—'),
        ('فئة المشكلة / Category',     complaint.get('category_name') or '—'),
        ('الفئة الفرعية / Sub-Cat.',   subcat),
        ('التصنيف / Classification',   class_ar),
    ]
    widths = [22, 26, 22, 25, 24, 24, 43, 64]
    for ci, (label, value) in enumerate(simple_boxes):
        _labeled_cell(data.cells[ci], label, value, bg=MINT_LIGHT, value_size=8, align='center')
        _set_row_col_width(data, ci, widths[ci])

    # Col 6: Severity badge (mini-table)
    _build_severity_cell(data.cells[6], complaint.get('severity_name') or '')
    _set_cell_shading(data.cells[6], MINT_LIGHT)
    _set_row_col_width(data, 6, widths[6])

    # Col 7: Harm stage scale (mini-table)
    _build_harm_cell(data.cells[7], complaint.get('harm_level') or '')
    _set_cell_shading(data.cells[7], MINT_LIGHT)
    _set_row_col_width(data, 7, widths[7])

    data.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    data.height = _mm_to_dxa(18)


def _build_content_zone(doc: Document, complaint: Dict, page_width_mm: float):
    """Zone 4 — Complaint text (left 60%) | Immediate Action + Follow-up (right 40%)."""
    from docx.enum.table import WD_ROW_HEIGHT_RULE

    tbl = doc.add_table(rows=1, cols=2)
    tbl.autofit = False
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _apply_minimal_table_borders(tbl, outer=NAVY, outer_sz=6, inner=GREY_LINE, inner_sz=6)

    # page_width_mm is already the usable width (margins already subtracted by caller)
    left_w  = page_width_mm * 0.60
    right_w = page_width_mm * 0.40

    left_cell  = tbl.rows[0].cells[0]
    right_cell = tbl.rows[0].cells[1]

    _set_row_col_width(tbl.rows[0], 0, left_w)
    _set_row_col_width(tbl.rows[0], 1, right_w)

    # LEFT — complaint text
    _set_cell_shading(left_cell, WHITE)
    _cell_v_center(left_cell)
    left_cell.text = ''

    lp_hdr = left_cell.paragraphs[0]
    lp_hdr.clear()
    _set_para_shading(lp_hdr, NAVY_LIGHT)
    lp_hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER   # bilingual label → centered
    lp_hdr.paragraph_format.space_before = Pt(6)
    lp_hdr.paragraph_format.space_after  = Pt(6)
    lp_hdr._p.get_or_add_pPr().append(OxmlElement('w:bidi'))
    _ar_run(lp_hdr, 'محتوى الشكوى  /  Complaint Details', size=9, bold=True, color=NAVY)

    # Hard cap at 850 chars — Content Zone has a fixed height ceiling (see
    # tbl.rows[0].height below); this is what makes the cap actually true.
    complaint_text = _truncate_for_fit(complaint.get('complaint_text'), 950)
    lp_body = left_cell.add_paragraph()
    lp_body.alignment = WD_ALIGN_PARAGRAPH.CENTER
    lp_body.paragraph_format.space_before = Pt(6)
    lp_body.paragraph_format.space_after  = Pt(6)
    lp_body.paragraph_format.left_indent  = Mm(3)
    lp_body.paragraph_format.right_indent = Mm(3)
    lp_body._p.get_or_add_pPr().append(OxmlElement('w:bidi'))

    # Line spacing 1.15 (was 1.3) — saves vertical space on long complaint
    # text, the main driver of per-card page overflow.
    pPr = lp_body._p.get_or_add_pPr()
    sp_el = OxmlElement('w:spacing')
    sp_el.set(qn('w:line'), '276')
    sp_el.set(qn('w:lineRule'), 'auto')
    pPr.append(sp_el)

    _ar_run(lp_body, complaint_text or 'لا يوجد نص للشكوى', size=9, color=DARK_TEXT)

    # RIGHT — two stacked action boxes
    _set_cell_shading(right_cell, WHITE)
    right_cell.text = ''

    # Immediate Action
    ra_hdr = right_cell.paragraphs[0]
    ra_hdr.clear()
    _set_para_shading(ra_hdr, 'FEF9E7')
    ra_hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER   # bilingual label → centered
    ra_hdr.paragraph_format.space_before = Pt(6)
    ra_hdr.paragraph_format.space_after  = Pt(6)
    ra_hdr._p.get_or_add_pPr().append(OxmlElement('w:bidi'))
    _ar_run(ra_hdr, 'الإجراء الفوري  /  Immediate Action', size=9, bold=True, color='8B4000')

    imm = _truncate_for_fit(complaint.get('immediate_action'), 240)
    ra_body = right_cell.add_paragraph()
    ra_body.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ra_body.paragraph_format.space_before = Pt(4)
    ra_body.paragraph_format.space_after  = Pt(4)
    ra_body.paragraph_format.left_indent  = Mm(2)
    ra_body.paragraph_format.right_indent = Mm(2)
    ra_body._p.get_or_add_pPr().append(OxmlElement('w:bidi'))
    _ar_run(ra_body, imm or '—', size=8, color=DARK_TEXT)

    # Divider
    div = right_cell.add_paragraph()
    div.paragraph_format.space_before = Pt(4)
    div.paragraph_format.space_after  = Pt(0)
    _set_para_bottom_border(div, color=GREY_LINE, sz=4)

    # Follow-up / Taken Action
    fu_hdr = right_cell.add_paragraph()
    _set_para_shading(fu_hdr, 'EBF3FB')
    fu_hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER   # bilingual label → centered
    fu_hdr.paragraph_format.space_before = Pt(6)
    fu_hdr.paragraph_format.space_after  = Pt(6)
    fu_hdr._p.get_or_add_pPr().append(OxmlElement('w:bidi'))
    _ar_run(fu_hdr, 'المتابعة والرد  /  Follow-up Response', size=9, bold=True, color=NAVY)

    taken = _truncate_for_fit(complaint.get('taken_action'), 240)
    fu_body = right_cell.add_paragraph()
    fu_body.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fu_body.paragraph_format.space_before = Pt(4)
    fu_body.paragraph_format.space_after  = Pt(4)
    fu_body.paragraph_format.left_indent  = Mm(2)
    fu_body.paragraph_format.right_indent = Mm(2)
    fu_body._p.get_or_add_pPr().append(OxmlElement('w:bidi'))
    _ar_run(fu_body, taken or '—', size=8, color=DARK_TEXT)

    # V8 hard cap: 55mm — Content Zone is now a CEILING, not just a floor.
    # Every text field above is truncated to a character budget calculated
    # to fit within this height, so this row can never grow past 58mm
    # regardless of source data length — this is what makes "one page per
    # case, no exceptions" structurally true rather than just typical-case.
    tbl.rows[0].height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    tbl.rows[0].height = _mm_to_dxa(58)


def _build_approvals_zone(doc: Document, complaint: Dict):
    """Zone 5 — 4-role signature block (left) + RCA instruction note (right)."""
    from docx.enum.table import WD_ROW_HEIGHT_RULE

    outer = doc.add_table(rows=1, cols=2)
    outer.autofit = False
    outer.alignment = WD_TABLE_ALIGNMENT.CENTER
    _apply_minimal_table_borders(outer, outer=NAVY, outer_sz=6, inner=GREY_LINE, inner_sz=4)

    sig_cell  = outer.rows[0].cells[0]
    note_cell = outer.rows[0].cells[1]
    # Target ~250mm total (17mm slack vs 267mm usable)
    _set_row_col_width(outer.rows[0], 0, 135)
    _set_row_col_width(outer.rows[0], 1, 115)

    # — Signature block inside sig_cell (width 135mm) —
    _set_cell_shading(sig_cell, WHITE)
    sig_cell.text = ''
    inner = sig_cell.add_table(4, 5)
    inner.autofit = False
    _apply_borders_no_vertical(inner, outer='AAAAAA', outer_sz=4, inner_h='E8E8E8', inner_h_sz=3)
    _set_rtl_table(inner)

    roles = ['مسؤول العملية\nProcess Owner',
             'رئيس الدائرة\nDept. Head',
             'مدير الإدارة\nAdmin. Manager',
             'خاص خدمات المرضى\nPatient Services']
    col_widths = [25, 27, 27, 27, 29]   # sum = 135mm

    # Header row — explicit height (6mm). Was never set before; Word's
    # default row sizing for 4 blank-ish rows was the actual cause of the
    # repeated page-2 spillover (~35-45mm unbounded vs the 27mm this gives).
    from docx.enum.table import WD_ROW_HEIGHT_RULE as _SIG_HR
    hdr = inner.rows[0]
    hdr.height_rule = _SIG_HR.EXACTLY
    hdr.height = _mm_to_dxa(7)
    for ci, label in enumerate([''] + roles):
        c = hdr.cells[ci]
        _set_cell_shading(c, NAVY)
        cp = _cell_para(c, 'center')
        _ar_run(cp, label, size=7, bold=True, color=WHITE)
        _set_row_col_width(hdr, ci, col_widths[ci])

    field_labels = ['الاسم / Name', 'التاريخ / Date', 'التوقيع / Signature']
    for ri, field_lbl in enumerate(field_labels):
        row = inner.rows[ri + 1]
        row.height_rule = _SIG_HR.EXACTLY
        row.height = _mm_to_dxa(7)
        c0 = row.cells[0]
        _set_cell_shading(c0, 'F4F6F9')
        p0 = _cell_para(c0, 'right')
        _ar_run(p0, field_lbl, size=8, bold=True, color=NAVY)
        _set_row_col_width(row, 0, col_widths[0])
        for ci in range(1, 5):
            _set_cell_shading(row.cells[ci], WHITE)
            _set_row_col_width(row, ci, col_widths[ci])

    # — RCA note in note_cell —
    _set_cell_shading(note_cell, 'FFFDE7')
    _cell_v_center(note_cell)
    note_cell.text = ''

    nh = note_cell.paragraphs[0]
    nh.clear()
    nh.alignment = WD_ALIGN_PARAGRAPH.CENTER
    nh.paragraph_format.space_before = Pt(4)
    nh.paragraph_format.space_after  = Pt(3)
    nh._p.get_or_add_pPr().append(OxmlElement('w:bidi'))
    _ar_run(nh, 'ملاحظات مهمة', size=9, bold=True, color='8B4000')

    intro = note_cell.add_paragraph()
    intro.alignment = WD_ALIGN_PARAGRAPH.CENTER
    intro.paragraph_format.space_before = Pt(0)
    intro.paragraph_format.space_after  = Pt(3)
    intro.paragraph_format.left_indent  = Mm(2)
    intro.paragraph_format.right_indent = Mm(2)
    intro._p.get_or_add_pPr().append(OxmlElement('w:bidi'))
    _ar_run(intro, 'يلزم ملء استمارة RCA (Root Cause Analysis) عند تحديد مستوى الشكوى كالتالي:',
            size=8, color='5D4037')

    # Each line: English level word, bullet, Arabic clause — bullet sits
    # BETWEEN two strong-direction runs (not leading the Arabic text), so
    # Word's bidi engine places it deterministically instead of pulling it
    # to the wrong visual side of an RTL paragraph.
    level_lines = [
        ('High',         'يلزم ملؤها باستمارة RCA خلال المتابعة'),
        ('Medium / Low', 'يكون ملؤها تبعاً للحاجة بقرار مسؤول العملية'),
    ]
    for level, desc in level_lines:
        lp = note_cell.add_paragraph()
        lp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        lp.paragraph_format.space_before = Pt(1)
        lp.paragraph_format.space_after  = Pt(1)
        lp.paragraph_format.left_indent  = Mm(2)
        lp.paragraph_format.right_indent = Mm(2)
        lp._p.get_or_add_pPr().append(OxmlElement('w:bidi'))
        _ar_run(lp, f'{level}  •  {desc}', size=8, color='5D4037')

    closing = note_cell.add_paragraph()
    closing.alignment = WD_ALIGN_PARAGRAPH.CENTER
    closing.paragraph_format.space_before = Pt(2)
    closing.paragraph_format.space_after  = Pt(1)
    closing.paragraph_format.left_indent  = Mm(2)
    closing.paragraph_format.right_indent = Mm(2)
    closing._p.get_or_add_pPr().append(OxmlElement('w:bidi'))
    _ar_run(closing, 'التقرير الفصلي  —  ترفع استمارة تحسين تلقائياً تبعاً للشكاوى',
            size=8, color='5D4037')

    # Signature table is now explicitly bounded at 28mm (4 rows x 7mm,
    # EXACTLY) — this floor matches it so the outer row settles at exactly
    # that height rather than Word picking its own.
    outer.rows[0].height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    outer.rows[0].height = _mm_to_dxa(28)


# ---------------------------------------------------------------------------
# FULL COMPLAINT PAGE
# ---------------------------------------------------------------------------

def _render_complaint_page(doc: Document, complaint: Dict, index: int, total: int,
                            period: Dict, page_width_mm: float,
                            report_entity_name: str, report_entity_type: str):
    # Zone 1: Identity
    _build_identity_zone(doc, complaint, period, report_entity_name, report_entity_type)
    _gap(doc, 1.8)   # 60% of 3mm

    # Zone 2: Classification
    _build_classification_zone(doc, complaint)
    _gap(doc, 1.8)   # 60% of 3mm

    # Zone 3: Stage of care
    _build_stage_row(doc,
                     complaint.get('stage_name') or '',
                     complaint.get('status_name') or '',
                     complaint.get('clinical_risk_type_name') or 'Ordinary',
                     page_width_mm)
    _gap(doc, 2.4)   # 60% of 4mm

    # Zone 4: Content
    _build_content_zone(doc, complaint, page_width_mm)
    _gap(doc, 2.4)   # 60% of 4mm

    # Zone 5: Approvals
    _build_approvals_zone(doc, complaint)

    # Page counter footer line
    pg_para = _new_para(doc, align='center', space_before=2, space_after=0)
    _ar_run(pg_para, f'شكوى {index} من {total}  •  {_fmt_date(complaint.get("received_date"))}',
            size=7, color=GREY_TEXT)

    _page_break(doc)


# ---------------------------------------------------------------------------
# NOTICE COMPACT CARD  (multiple per page)
# ---------------------------------------------------------------------------

def _render_notice_card(doc: Document, notice: Dict):
    """Compact 3-row notice card."""
    from docx.enum.table import WD_ROW_HEIGHT_RULE

    target_unit = _target_display(notice)
    inc_id = notice.get('incident_id')
    id_str = f"RTG-{int(inc_id):06d}" if inc_id is not None else str(notice.get('id', '—'))

    tbl = doc.add_table(rows=3, cols=6)
    tbl.autofit = False
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _apply_minimal_table_borders(tbl, outer=NAVY, outer_sz=6, inner='DCEEE2', inner_sz=2)
    _set_rtl_table(tbl)

    # Row 0: Identity strip (light teal header)
    id_row = tbl.rows[0]
    id_header = id_row.cells[0].merge(id_row.cells[5])
    _set_cell_shading(id_header, '1B7A5E')
    hp = id_header.paragraphs[0]
    hp.clear()
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hp.paragraph_format.space_before = Pt(1)
    hp.paragraph_format.space_after  = Pt(1)
    hp._p.get_or_add_pPr().append(OxmlElement('w:bidi'))
    _ar_run(hp, f'تنويه / Notice   —   {id_str}   |   {_fmt_date(notice.get("received_date"))}   |   المصدر: {notice.get("source_name") or "—"}   |   {notice.get("patient_name") or "—"}',
            size=8, bold=True, color=WHITE)
    id_row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    id_row.height = _mm_to_dxa(7)

    # Row 1: Notice text (large, readable)
    txt_row = tbl.rows[1]
    txt_cell = txt_row.cells[0].merge(txt_row.cells[3])
    _set_cell_shading(txt_cell, 'F0FFF8')
    txt_cell.text = ''
    tp = txt_cell.paragraphs[0]
    tp.clear()
    tp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    tp.paragraph_format.space_before = Pt(5)
    tp.paragraph_format.space_after  = Pt(5)
    tp.paragraph_format.right_indent = Mm(3)
    tp._p.get_or_add_pPr().append(OxmlElement('w:bidi'))
    pPr = tp._p.get_or_add_pPr()
    sp_el = OxmlElement('w:spacing')
    sp_el.set(qn('w:line'), '300')
    sp_el.set(qn('w:lineRule'), 'auto')
    pPr.append(sp_el)
    notice_text = (notice.get('notice_text') or '').strip()
    _ar_run(tp, f'" {notice_text} "' if notice_text else '—', size=10, italic=True, color='1B5E20')

    # Target unit cell (col 4-5 merged)
    target_cell = txt_row.cells[4].merge(txt_row.cells[5])
    _set_cell_shading(target_cell, 'E8F5E9')
    _cell_v_center(target_cell)
    _labeled_cell(target_cell, 'الوحدة المُنوَّه بها / Praised Unit', target_unit,
                  value_size=9, value_color='1B5E20', bg='E8F5E9')

    txt_row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    txt_row.height = _mm_to_dxa(18)

    # Row 2: Section info — was 6 cols with a wasted empty 88mm trailing cell
    # (true total 280mm, exceeding even the old 267mm usable). Merged the
    # unused cell into Type; target ~250mm total (17mm safety margin).
    info_row = tbl.rows[2]
    info_meta = [
        ('قسم الصادر / Issuing', notice.get('section_name') or '—'),
        ('الدائرة / Department',  notice.get('department_name') or '—'),
        ('الإدارة / Admin',       notice.get('administration_name') or '—'),
        ('الحالة / Status',       notice.get('status_name') or '—'),
    ]
    widths_n = [48, 48, 48, 30]
    for ci, (lbl, val) in enumerate(info_meta):
        _labeled_cell(info_row.cells[ci], lbl, val, bg='F4FBF7', value_size=8)
        _set_row_col_width(info_row, ci, widths_n[ci])

    type_cell = info_row.cells[4].merge(info_row.cells[5])
    _labeled_cell(type_cell, 'نوع السجل / Type',
                  notice.get('feedback_intent_type_name_ar') or 'تنويه',
                  bg='F4FBF7', value_size=8)
    # Merged cell spans 2 underlying grid columns — both must be set explicitly,
    # otherwise the unset one keeps its stale auto-generated grid width and
    # still counts toward the table's real total even though visually merged away.
    _set_row_col_width(info_row, 4, 38)
    _set_row_col_width(info_row, 5, 38)   # 48+48+48+30+38+38 = 250mm

    info_row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    info_row.height = _mm_to_dxa(10)

    _spacer_para(doc, 3)


# ---------------------------------------------------------------------------
# SUMMARY PAGE
# ---------------------------------------------------------------------------

def _render_summary_page(doc: Document, report_data: Dict,
                          report_entity_name: str, report_entity_type: str):
    """Page 1: Period info + Intent counts table."""
    from docx.enum.table import WD_ROW_HEIGHT_RULE

    period   = report_data.get('period', {})
    counts   = report_data.get('intent_counts', {})
    complaints = report_data.get('complaints', [])
    notices    = report_data.get('notices', [])

    # Title block
    title_p = _new_para(doc, align='center', space_before=0, space_after=4)
    _ar_run(title_p, 'ملخص التقرير الشهري  /  Monthly Report Summary',
            size=16, bold=True, color=NAVY)

    # Period + scope
    scope_p = _new_para(doc, align='center', space_before=0, space_after=6)
    scope_name = report_entity_name or 'المستشفى (Hospital Level)'
    _ar_run(scope_p, f'الفترة: {period.get("label_ar") or period.get("label", "—")}   |   النطاق: {scope_name}',
            size=11, color=GREY_TEXT)

    # Stats bar (3 boxes)
    stats_tbl = doc.add_table(rows=2, cols=3)
    stats_tbl.autofit = False
    stats_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _apply_minimal_table_borders(stats_tbl, outer=NAVY, outer_sz=8, inner=GREY_LINE, inner_sz=6)
    _set_rtl_table(stats_tbl)

    hdr_r = stats_tbl.rows[0]
    hdr_merged = hdr_r.cells[0].merge(hdr_r.cells[2])
    _set_cell_shading(hdr_merged, NAVY)
    shp = hdr_merged.paragraphs[0]
    shp.clear()
    shp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    shp.paragraph_format.space_before = Pt(2)
    shp.paragraph_format.space_after  = Pt(2)
    _ar_run(shp, 'إجمالي السجلات للفترة  /  Total Records for Period', size=10, bold=True, color=WHITE)
    hdr_r.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    hdr_r.height = _mm_to_dxa(8)

    total_complaints = len(complaints)
    total_notices    = len(notices)
    data_r = stats_tbl.rows[1]
    stat_boxes = [
        ('إجمالي الشكاوى\nComplaints', str(total_complaints), 'FADBD8', 'C0392B'),
        ('إجمالي التنويهات\nNotices',  str(total_notices),    'D5F5E3', '1E8449'),
        ('المجموع الكلي\nCombined Total', str(total_complaints + total_notices), NAVY_LIGHT, NAVY),
    ]
    stat_widths = [88, 88, 90]
    for ci, (lbl, val, bg, fg) in enumerate(stat_boxes):
        c = data_r.cells[ci]
        _set_cell_shading(c, bg)
        _cell_v_center(c)
        c.text = ''
        vp = c.paragraphs[0]
        vp.clear()
        vp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        vp.paragraph_format.space_before = Pt(4)
        vp.paragraph_format.space_after  = Pt(2)
        _ar_run(vp, val, size=22, bold=True, color=fg)
        lp = c.add_paragraph()
        lp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        lp.paragraph_format.space_before = Pt(0)
        lp.paragraph_format.space_after  = Pt(4)
        _ar_run(lp, lbl, size=8, color=GREY_TEXT)
        _set_row_col_width(data_r, ci, stat_widths[ci])

    data_r.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    data_r.height = _mm_to_dxa(24)

    _spacer_para(doc, 6)

    # Intent counts detail table
    sections  = counts.get('sections', [])
    depts     = counts.get('departments', [])
    admins    = counts.get('administrations', [])

    rows_data = []
    for u in sections:      rows_data.append((u, 'قسم (Section)'))
    for u in depts:         rows_data.append((u, 'دائرة (Department)'))
    for u in admins:        rows_data.append((u, 'إدارة (Administration)'))

    if rows_data:
        title2 = _new_para(doc, align='center', space_before=4, space_after=3)
        _set_para_bottom_border(title2, color=NAVY, sz=6)
        _ar_run(title2, 'توزيع الشكاوى والتنويهات بحسب الوحدة  /  Complaint & Notice Count by Unit',
                size=11, bold=True, color=NAVY)

        ct = doc.add_table(rows=1, cols=5)
        ct.autofit = False
        ct.alignment = WD_TABLE_ALIGNMENT.CENTER
        _apply_minimal_table_borders(ct, outer=NAVY, outer_sz=6, inner=GREY_LINE, inner_sz=4)
        _set_rtl_table(ct)

        headers = [
            ('اسم الوحدة / Unit Name', 80),
            ('نوع الوحدة / Type', 45),
            ('الشكاوى / Complaints', 38),
            ('التنويهات / Notices', 38),
            ('المجموع / Total', 35),
        ]
        for ci, (hdr_txt, w) in enumerate(headers):
            c = ct.rows[0].cells[ci]
            _set_cell_shading(c, NAVY)
            cp = _cell_para(c, 'center')
            _ar_run(cp, hdr_txt, size=8, bold=True, color=WHITE)
            _set_row_col_width(ct.rows[0], ci, w)

        for ri, (unit, type_lbl) in enumerate(rows_data):
            row = ct.add_row()
            bg = 'F4F6F9' if ri % 2 == 0 else WHITE
            vals = [
                unit.get('unit_name', '—'),
                type_lbl,
                str(unit.get('complaint_count', 0)),
                str(unit.get('notice_count', 0)),
                str(unit.get('total_count', 0)),
            ]
            widths_c = [80, 45, 38, 38, 35]
            for ci, (val, w) in enumerate(zip(vals, widths_c)):
                c = row.cells[ci]
                _set_cell_shading(c, bg)
                _cell_v_center(c)
                cp = _cell_para(c, 'center' if ci > 0 else 'right')
                _ar_run(cp, val, size=9, color=DARK_TEXT)
                _set_row_col_width(row, ci, w)

    _page_break(doc)


# ---------------------------------------------------------------------------
# DOCUMENT SETUP + REPEATING HEADER
# ---------------------------------------------------------------------------

def _setup_document(report_data: Dict, report_entity_name: str) -> Document:
    """Create document, set landscape A4, configure repeating header."""
    doc = Document()
    doc.styles['Normal'].font.name = 'Traditional Arabic'
    doc.styles['Normal'].font.size = Pt(10)

    sec = doc.sections[0]
    sec.page_width    = int(Mm(297))
    sec.page_height   = int(Mm(210))
    sec.orientation   = WD_ORIENT.LANDSCAPE
    sec.left_margin   = int(Mm(12))   # recovered from 15mm — more canvas for content
    sec.right_margin  = int(Mm(12))
    sec.top_margin    = int(Mm(13))   # only safe because header is now a single ~9mm row, not 4 stacked paragraphs
    sec.bottom_margin = int(Mm(3))    # footer trimmed to 6pt to fit
    sec.header_distance = int(Mm(4))
    sec.footer_distance = int(Mm(3))

    logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'logo.png')

    # Load config — same keys as the classical formatter and Settings page
    try:
        from ..db_layer.report_config_db import get_report_config
        cfg = get_report_config()
    except Exception:
        cfg = {}
    header_title    = cfg.get('header_title',
                              'التقرير الشهري لفرص التحسين والإجراءات التصحيحية الواردة من المرضى وذويهم')
    header_subtitle = cfg.get('header_subtitle', 'HCAT Monthly Patient Feedback Report')
    footer_text     = cfg.get('footer_text',
                              'نؤمن أن الابتكار لا يكون فقط في التقنيات، بل في أسلوب الخدمة والتواصل والتعاطف')
    report_code     = cfg.get('report_code', '')

    period     = report_data.get('period', {})
    period_str = (f"{period.get('start_date', '—')}  —  {period.get('end_date', '—')}")
    scope_str  = report_entity_name or 'مستوى المستشفى'

    # ── Repeating header — compact single-row table (logo beside the title
    # block, not stacked above it) so total header height fits a 13mm top
    # margin. Plain 2-column table, no borders, no merged cells, AT_LEAST
    # height — avoids both Round-1 collapse causes (merged-cell + EXACTLY,
    # and _Cell.add_table() rejecting a width kwarg) since neither applies
    # to a borderless, unmerged, header-level add_table(rows, cols, width).
    from docx.enum.table import WD_ROW_HEIGHT_RULE as _HDR_HR
    hdr = sec.header
    hdr.paragraphs[0].clear()

    hdr_tbl = hdr.add_table(1, 2, int(Mm(273)))
    hdr_tbl.autofit = False
    hdr_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_rtl_table(hdr_tbl)

    logo_cell  = hdr_tbl.rows[0].cells[0]
    title_cell = hdr_tbl.rows[0].cells[1]
    _set_row_col_width(hdr_tbl.rows[0], 0, 32)
    _set_row_col_width(hdr_tbl.rows[0], 1, 241)

    # Logo — small, square-ish (178x179px), 0.35in keeps row height ~9mm
    _cell_v_center(logo_cell)
    logo_cell.text = ''
    lp = logo_cell.paragraphs[0]
    lp.clear()
    lp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    try:
        if os.path.exists(logo_path):
            lp.add_run().add_picture(logo_path, width=int(Inches(0.35)))
    except Exception:
        pass

    # Title + info line, stacked inside the (wider) second column
    _cell_v_center(title_cell)
    title_cell.text = ''
    tp = title_cell.paragraphs[0]
    tp.clear()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tp.paragraph_format.space_before = int(Pt(0))
    tp.paragraph_format.space_after  = int(Pt(1))
    _ar_run(tp, header_title, size=10, bold=True, color=NAVY)

    # One combined info line: subtitle (config value is Arabic by default,
    # e.g. "(إصدار رسمي — للاستخدام الإداري والجودة)" — same key the classical
    # formatter uses, rendered with Traditional Arabic there too) | period+scope | report code
    hdr_info_para = title_cell.add_paragraph()
    hdr_info_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hdr_info_para.paragraph_format.space_before = int(Pt(0))
    hdr_info_para.paragraph_format.space_after = int(Pt(1))
    _ar_run(hdr_info_para, header_subtitle, size=7, italic=True, color=GREY_TEXT)
    _ar_run(hdr_info_para, '   |   ', size=7, color=GREY_TEXT)
    _ar_run(hdr_info_para, f'الفترة: {period_str}   |   النطاق: {scope_str}', size=7, bold=True, color=GREY_TEXT)
    if report_code:
        _ar_run(hdr_info_para, '   |   ', size=7, color=GREY_TEXT)
        _ar_run(hdr_info_para, f'رمز التقرير: {report_code}', size=7, color=GREY_TEXT)

    hdr_tbl.rows[0].height_rule = _HDR_HR.AT_LEAST
    hdr_tbl.rows[0].height = _mm_to_dxa(9)

    # Blue separator line (matches classical formatter) — minimal height
    hdr_sep = hdr.add_paragraph()
    hdr_sep.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hdr_sep.paragraph_format.space_before = int(Pt(0))
    hdr_sep.paragraph_format.space_after  = int(Pt(0))
    hdr_sep_run = hdr_sep.add_run('')
    hdr_sep_run.font.size = int(Pt(2))
    _pPr = hdr_sep._element.get_or_add_pPr()
    _pBdr = OxmlElement('w:pBdr')
    _bot = OxmlElement('w:bottom')
    _bot.set(qn('w:val'), 'single')
    _bot.set(qn('w:sz'), '12')
    _bot.set(qn('w:space'), '1')
    _bot.set(qn('w:color'), '4472C4')
    _pBdr.append(_bot)
    _pPr.append(_pBdr)

    # ── Footer (compact — single tiny line, fits a 3mm bottom margin) ──
    ftr = sec.footer
    ftr.is_linked_to_previous = False
    fp = ftr.paragraphs[0]
    fp.clear()
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.paragraph_format.space_before = Pt(0)
    fp.paragraph_format.space_after  = Pt(0)
    _set_para_bottom_border(fp, color=GREY_LINE, sz=4)
    _ar_run(fp, footer_text, size=6, italic=True, color=GREY_TEXT)

    return doc


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def generate_monthly_stylish_docx(
    report_data: Dict[str, Any],
    filename: str = '',
    language: str = 'ar',
    report_entity_name: str = None,
    report_entity_type: str = None,
    report_administration: str = None,
    report_department: str = None,
    report_section: str = None,
) -> bytes:
    """
    Generate the Stylish Monthly Report DOCX.

    Args:
        report_data: Prepared report model from get_detailed_monthly_report()
                     Keys used: complaints, notices, period, intent_counts
        report_entity_name: Org unit name for scope display
        report_entity_type: 'section' | 'department' | 'administration' | 'hospital'

    Returns:
        bytes: Valid DOCX file content.
    """
    # Normalise inputs
    complaints = []
    notices    = []
    try:
        if isinstance(report_data, dict):
            raw = report_data.get('complaints', [])
            complaints = raw if isinstance(raw, list) else []
            notices    = report_data.get('notices', []) or []
        elif isinstance(report_data, list):
            complaints = report_data
    except Exception:
        complaints = []

    # Derive scope label
    if not report_entity_name:
        if report_administration:    report_entity_name = report_administration
        elif report_department:      report_entity_name = report_department
        elif report_section:         report_entity_name = report_section
        else:                        report_entity_name = 'مستوى المستشفى'

    period = {}
    if isinstance(report_data, dict):
        period = report_data.get('period', {})

    doc = _setup_document(report_data if isinstance(report_data, dict) else {}, report_entity_name)
    # Usable width is 273mm (297mm page - 12mm margins each side), but every zone
    # table targets ~250mm — 23mm safety margin against Word's known behavior of
    # silently widening a "fixed layout" column when it holds an unbreakable
    # token too narrow to fit (see zone-builder functions for the per-cell fix).
    page_width_mm = 250.0

    # Page 1: Summary
    _render_summary_page(doc, report_data if isinstance(report_data, dict) else {},
                         report_entity_name, report_entity_type or 'hospital')

    # Complaint cards (one per page, unlimited)
    total_c = len(complaints)
    for idx, complaint in enumerate(complaints, start=1):
        try:
            _render_complaint_page(doc, complaint, idx, total_c,
                                   period, page_width_mm,
                                   report_entity_name, report_entity_type or '')
        except Exception as e:
            print(f'[STYLISH] Warning: failed to render complaint #{idx}: {e}')
            _page_break(doc)

    # Notice page(s) — compact cards grouped
    if notices:
        # Notice section header page intro paragraph
        notice_hdr = _new_para(doc, align='center', space_before=0, space_after=6)
        _set_para_shading(notice_hdr, NAVY)
        _ar_run(notice_hdr,
                f'التنويهات  /  Notices   —   إجمالي: {len(notices)}',
                size=14, bold=True, color=WHITE)
        _spacer_para(doc, 4)

        for notice in notices:
            try:
                _render_notice_card(doc, notice)
            except Exception as e:
                print(f'[STYLISH] Warning: failed to render notice: {e}')

    # Empty state
    if not complaints and not notices:
        ep = _new_para(doc, align='center', space_before=20, space_after=0)
        _ar_run(ep, 'لا توجد سجلات لهذه الفترة — No records for this period.',
                size=13, italic=True, color=GREY_TEXT)

    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()
