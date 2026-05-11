"""
Patient Feedback Seasonal Report Word Generator

Generates comprehensive Word documents with RCA (Root Cause Analysis) 
and Satisfaction statistics for a given seasonal period.

This report aggregates:
- RCA breakdowns by cause type (Staff, Process, Equipment, Environment)
- Detailed sub-cause analysis with individual cause factors
- Satisfaction status distribution (Satisfied, Not Satisfied, Not Present)
- Preventability analysis with specific measures breakdown
- Feedback coverage metrics
- Visual charts and graphs for data visualization
"""

from typing import List, Dict, Any
from datetime import date, datetime
from io import BytesIO
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Mm, Cm
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import nsdecls, qn
from docx.oxml import parse_xml

# For chart generation
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import numpy as np

from core.database import get_connection

# Configure matplotlib for Arabic text support
rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']


def _fetch_rca_statistics(season_start: date, season_end: date) -> Dict[str, Any]:
    """
    Fetch RCA (Root Cause Analysis) statistics for the seasonal period.
    
    Returns breakdown by cause type, individual sub-causes, preventability, and department.
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Enhanced RCA statistics query with detailed sub-cause breakdown
        query = """
            SELECT
                COUNT(*) as total_rca_records,
                -- Staff causes aggregate
                SUM(CASE WHEN (
                    f.Cause_Staff_Training = 1 OR f.Cause_Staff_Incentives = 1 OR 
                    f.Cause_Staff_Competency = 1 OR f.Cause_Staff_Understaffed = 1 OR 
                    f.Cause_Staff_NonCompliance = 1 OR f.Cause_Staff_NoCoordination = 1 OR 
                    f.Cause_Staff_Other = 1
                ) THEN 1 ELSE 0 END) as staff_causes,
                -- Process causes aggregate
                SUM(CASE WHEN (
                    f.Cause_Process_NotComprehensive = 1 OR f.Cause_Process_Unclear = 1 OR 
                    f.Cause_Process_MissingProtocol = 1 OR f.Cause_Process_Other = 1
                ) THEN 1 ELSE 0 END) as process_causes,
                -- Equipment causes aggregate
                SUM(CASE WHEN (
                    f.Cause_Equipment_NotAvailable = 1 OR f.Cause_Equipment_SystemIncomplete = 1 OR 
                    f.Cause_Equipment_HardToApply = 1 OR f.Cause_Equipment_Other = 1
                ) THEN 1 ELSE 0 END) as equipment_causes,
                -- Environment causes aggregate
                SUM(CASE WHEN (
                    f.Cause_Environment_PlaceNature = 1 OR f.Cause_Environment_Surroundings = 1 OR 
                    f.Cause_Environment_WorkConditions = 1 OR f.Cause_Environment_Other = 1
                ) THEN 1 ELSE 0 END) as environment_causes,
                -- Preventive measures aggregate
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
                ) THEN 1 ELSE 0 END) as no_preventive_measures,
                -- Individual Staff sub-causes
                SUM(CAST(COALESCE(f.Cause_Staff_Training, 0) AS INT)) as staff_training,
                SUM(CAST(COALESCE(f.Cause_Staff_Incentives, 0) AS INT)) as staff_incentives,
                SUM(CAST(COALESCE(f.Cause_Staff_Competency, 0) AS INT)) as staff_competency,
                SUM(CAST(COALESCE(f.Cause_Staff_Understaffed, 0) AS INT)) as staff_understaffed,
                SUM(CAST(COALESCE(f.Cause_Staff_NonCompliance, 0) AS INT)) as staff_noncompliance,
                SUM(CAST(COALESCE(f.Cause_Staff_NoCoordination, 0) AS INT)) as staff_nocoordination,
                SUM(CAST(COALESCE(f.Cause_Staff_Other, 0) AS INT)) as staff_other,
                -- Individual Process sub-causes
                SUM(CAST(COALESCE(f.Cause_Process_NotComprehensive, 0) AS INT)) as process_notcomprehensive,
                SUM(CAST(COALESCE(f.Cause_Process_Unclear, 0) AS INT)) as process_unclear,
                SUM(CAST(COALESCE(f.Cause_Process_MissingProtocol, 0) AS INT)) as process_missingprotocol,
                SUM(CAST(COALESCE(f.Cause_Process_Other, 0) AS INT)) as process_other,
                -- Individual Equipment sub-causes
                SUM(CAST(COALESCE(f.Cause_Equipment_NotAvailable, 0) AS INT)) as equipment_notavailable,
                SUM(CAST(COALESCE(f.Cause_Equipment_SystemIncomplete, 0) AS INT)) as equipment_systemincomplete,
                SUM(CAST(COALESCE(f.Cause_Equipment_HardToApply, 0) AS INT)) as equipment_hardtoapply,
                SUM(CAST(COALESCE(f.Cause_Equipment_Other, 0) AS INT)) as equipment_other,
                -- Individual Environment sub-causes
                SUM(CAST(COALESCE(f.Cause_Environment_PlaceNature, 0) AS INT)) as environment_placenature,
                SUM(CAST(COALESCE(f.Cause_Environment_Surroundings, 0) AS INT)) as environment_surroundings,
                SUM(CAST(COALESCE(f.Cause_Environment_WorkConditions, 0) AS INT)) as environment_workconditions,
                SUM(CAST(COALESCE(f.Cause_Environment_Other, 0) AS INT)) as environment_other,
                -- Individual Preventive measures
                SUM(CAST(COALESCE(f.Preventive_MonthlyMeetings, 0) AS INT)) as preventive_monthlymeetings,
                SUM(CAST(COALESCE(f.Preventive_TrainingPrograms, 0) AS INT)) as preventive_trainingprograms,
                SUM(CAST(COALESCE(f.Preventive_IncreaseStaff, 0) AS INT)) as preventive_increasestaff,
                SUM(CAST(COALESCE(f.Preventive_MMCommitteeActions, 0) AS INT)) as preventive_mmcommitteeactions,
                SUM(CAST(COALESCE(f.Preventive_Other, 0) AS INT)) as preventive_other
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
            },
            # Detailed sub-cause breakdown
            "staff_subcauses": {
                "التدريب (Training)": row[7] or 0,
                "الحوافز (Incentives)": row[8] or 0,
                "الكفاءة (Competency)": row[9] or 0,
                "نقص الموظفين (Understaffed)": row[10] or 0,
                "عدم الالتزام (Non-Compliance)": row[11] or 0,
                "عدم التنسيق (No Coordination)": row[12] or 0,
                "أخرى (Other)": row[13] or 0
            },
            "process_subcauses": {
                "غير شامل (Not Comprehensive)": row[14] or 0,
                "غير واضح (Unclear)": row[15] or 0,
                "بروتوكول مفقود (Missing Protocol)": row[16] or 0,
                "أخرى (Other)": row[17] or 0
            },
            "equipment_subcauses": {
                "غير متوفر (Not Available)": row[18] or 0,
                "نظام غير مكتمل (System Incomplete)": row[19] or 0,
                "صعب التطبيق (Hard to Apply)": row[20] or 0,
                "أخرى (Other)": row[21] or 0
            },
            "environment_subcauses": {
                "طبيعة المكان (Place Nature)": row[22] or 0,
                "المحيط (Surroundings)": row[23] or 0,
                "ظروف العمل (Work Conditions)": row[24] or 0,
                "أخرى (Other)": row[25] or 0
            },
            "preventive_measures_detail": {
                "اجتماعات شهرية (Monthly Meetings)": row[26] or 0,
                "برامج تدريبية (Training Programs)": row[27] or 0,
                "زيادة الموظفين (Increase Staff)": row[28] or 0,
                "إجراءات لجنة MM (MM Committee Actions)": row[29] or 0,
                "أخرى (Other)": row[30] or 0
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
                run.font.name = 'Calibri'
                run.font.color.rgb = RGBColor(255, 255, 255)
                run._element.rPr.rFonts.set(qn('w:cs'), 'Calibri')


def _generate_rca_pie_chart(rca_stats: Dict[str, Any]) -> BytesIO:
    """
    Generate a pie chart showing RCA cause type distribution.
    
    Returns the chart as a BytesIO object for embedding in Word document.
    """
    cause_types = rca_stats.get("by_cause_type", {})
    
    labels = [
        'الكوادر البشرية\n(Staff)',
        'العمليات\n(Process)', 
        'المعدات\n(Equipment)',
        'البيئة\n(Environment)'
    ]
    sizes = [
        cause_types.get("staff", 0),
        cause_types.get("process", 0),
        cause_types.get("equipment", 0),
        cause_types.get("environment", 0)
    ]
    
    # Filter out zero values
    non_zero_data = [(l, s) for l, s in zip(labels, sizes) if s > 0]
    if not non_zero_data:
        return None
    
    labels, sizes = zip(*non_zero_data)
    
    # Professional color palette
    colors = ['#667eea', '#4CAF50', '#FF9800', '#E91E63'][:len(sizes)]
    explode = [0.02] * len(sizes)  # Slight separation
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(sizes))})',
        explode=explode,
        colors=colors,
        shadow=False,
        startangle=90,
        textprops={'fontsize': 10}
    )
    
    # Style the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    ax.set_title('RCA Cause Type Distribution\nتوزيع أنواع الأسباب الجذرية', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.axis('equal')
    
    plt.tight_layout()
    
    # Save to BytesIO
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    img_buffer.seek(0)
    
    return img_buffer


def _generate_subcause_bar_chart(subcauses: Dict[str, int], title: str, color: str) -> BytesIO:
    """
    Generate a horizontal bar chart for sub-cause analysis.
    
    Returns the chart as a BytesIO object for embedding in Word document.
    """
    # Filter out zero values
    filtered = {k: v for k, v in subcauses.items() if v > 0}
    if not filtered:
        return None
    
    # Sort by value descending
    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_items)
    
    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.5)), facecolor='white')
    
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=color, edgecolor='white', height=0.6)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # Highest value at top
    
    ax.set_xlabel('عدد الحالات (Count)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(int(val)), va='center', fontsize=9, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    img_buffer.seek(0)
    
    return img_buffer


def _generate_preventive_measures_chart(measures: Dict[str, int]) -> BytesIO:
    """
    Generate a bar chart for preventive measures breakdown.
    """
    # Filter out zero values
    filtered = {k: v for k, v in measures.items() if v > 0}
    if not filtered:
        return None
    
    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_items)
    
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, values, color='#4CAF50', edgecolor='white', width=0.6)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
    
    ax.set_ylabel('عدد الحالات (Count)', fontsize=10)
    ax.set_title('Preventive Measures Distribution\nتوزيع الإجراءات الوقائية', 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(int(val)), ha='center', fontsize=9, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    img_buffer.seek(0)
    
    return img_buffer


def _get_rca_insights(rca_stats: Dict[str, Any]) -> List[str]:
    """
    Generate analytical insights based on RCA data.
    
    Returns a list of insight strings for the report.
    """
    insights = []
    
    total = rca_stats.get("total_rca_records", 0)
    if total == 0:
        return ["لا توجد بيانات كافية لتحليل الأسباب الجذرية في هذه الفترة."]
    
    cause_types = rca_stats.get("by_cause_type", {})
    
    # Find dominant cause type
    max_cause = max(cause_types.items(), key=lambda x: x[1])
    cause_names = {
        "staff": "الكوادر البشرية (Staff)",
        "process": "العمليات (Process)",
        "equipment": "المعدات (Equipment)",
        "environment": "البيئة (Environment)"
    }
    
    if max_cause[1] > 0:
        pct = round(max_cause[1] / total * 100, 1)
        insights.append(f"• السبب الأكثر شيوعاً هو {cause_names[max_cause[0]]} بنسبة {pct}% من الحالات المحللة.")
    
    # Staff analysis
    staff_subcauses = rca_stats.get("staff_subcauses", {})
    if staff_subcauses:
        top_staff = max(staff_subcauses.items(), key=lambda x: x[1])
        if top_staff[1] > 0:
            insights.append(f"• أبرز مشكلة في الكوادر البشرية: {top_staff[0]} ({top_staff[1]} حالة).")
    
    # Process analysis
    process_subcauses = rca_stats.get("process_subcauses", {})
    if process_subcauses:
        top_process = max(process_subcauses.items(), key=lambda x: x[1])
        if top_process[1] > 0:
            insights.append(f"• أبرز مشكلة في العمليات: {top_process[0]} ({top_process[1]} حالة).")
    
    # Preventive measures analysis
    preventability = rca_stats.get("preventability", {})
    has_prev = preventability.get("has_preventive_measures", 0)
    no_prev = preventability.get("no_preventive_measures", 0)
    
    if has_prev + no_prev > 0:
        prev_pct = round(has_prev / (has_prev + no_prev) * 100, 1)
        if prev_pct >= 70:
            insights.append(f"• نسبة عالية ({prev_pct}%) من الحالات لديها إجراءات وقائية مقترحة - مؤشر إيجابي.")
        elif prev_pct < 50:
            insights.append(f"• تحذير: نسبة منخفضة ({prev_pct}%) من الحالات لديها إجراءات وقائية - يُنصح بتعزيز التحليل.")
    
    # Recommendations based on data
    insights.append("")
    insights.append("التوصيات (Recommendations):")
    
    if cause_types.get("staff", 0) > total * 0.3:
        insights.append("• التركيز على برامج التدريب وتطوير الكفاءات للموظفين.")
    if cause_types.get("process", 0) > total * 0.2:
        insights.append("• مراجعة وتحديث البروتوكولات والإجراءات التشغيلية.")
    if cause_types.get("equipment", 0) > total * 0.15:
        insights.append("• تقييم احتياجات المعدات والأنظمة التقنية.")
    if cause_types.get("environment", 0) > total * 0.1:
        insights.append("• تحسين بيئة العمل وظروفها.")
    
    return insights


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

    # Set default font to Traditional Arabic
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)

    # Page setup
    section = doc.sections[0]
    section.page_height = Mm(297)
    section.page_width = Mm(210)
    section.left_margin = Cm(2)
    section.right_margin = Cm(2)

    # ========== LOGO HEADER ==========
    try:
        logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'logo.png')
        if os.path.exists(logo_path):
            section.header_distance = Inches(0.1)
            header_section = section.header
            header_para = header_section.paragraphs[0]
            header_para.clear()
            header_para.alignment = WD_PARAGRAPH_ALIGNMENT.RIGHT
            run = header_para.add_run()
            run.add_picture(logo_path, width=Inches(0.9))
    except Exception as e:
        print(f"[PATIENT_FEEDBACK_SEASONAL] Could not add logo: {e}")

    # ========== TITLE PAGE ==========
    doc.add_paragraph()

    title_ar = doc.add_heading('تقرير ملاحظات المرضى الموسمي', 0)
    title_ar.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    for run in title_ar.runs:
        run.font.size = Pt(22)
        run.font.name = 'Calibri'
        run.font.bold = True
        run.font.color.rgb = RGBColor(102, 126, 234)
        run._element.rPr.rFonts.set(qn('w:cs'), 'Calibri')

    title_en = doc.add_heading('Patient Feedback Seasonal Report', 1)
    title_en.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    for run in title_en.runs:
        run.font.size = Pt(16)
        run.font.name = 'Calibri'
        run.font.color.rgb = RGBColor(118, 75, 162)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    sub_run = subtitle.add_run('تحليل الأسباب الجذرية ورضا المرضى')
    sub_run.font.size = Pt(14)
    sub_run.font.name = 'Calibri'
    sub_run.font.color.rgb = RGBColor(100, 100, 100)
    sub_run._element.rPr.rFonts.set(qn('w:cs'), 'Calibri')

    subtitle2 = doc.add_paragraph()
    subtitle2.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    sub_run2 = subtitle2.add_run('Root Cause Analysis & Patient Satisfaction')
    sub_run2.font.size = Pt(12)
    sub_run2.font.name = 'Calibri'
    sub_run2.font.color.rgb = RGBColor(100, 100, 100)

    doc.add_paragraph()

    # Period info — keep as a single Arabic line to avoid LTR/RTL number collision
    period = doc.add_paragraph()
    period.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    period_run = period.add_run(
        f'{season_start.strftime("%Y-%m-%d")} — {season_end.strftime("%Y-%m-%d")} :الفترة'
    )
    period_run.font.size = Pt(12)
    period_run.font.name = 'Calibri'
    period_run.font.bold = True
    period_run._element.rPr.rFonts.set(qn('w:cs'), 'Calibri')

    # Generation timestamp
    gen_time = doc.add_paragraph()
    gen_time.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    gen_run = gen_time.add_run(f'{datetime.now().strftime("%Y-%m-%d %H:%M")} :تم الإنشاء')
    gen_run.font.size = Pt(9)
    gen_run.font.name = 'Calibri'
    gen_run.font.color.rgb = RGBColor(128, 128, 128)
    gen_run._element.rPr.rFonts.set(qn('w:cs'), 'Calibri')
    
    doc.add_page_break()
    
    # ========== EXECUTIVE SUMMARY ==========
    summary_heading = doc.add_heading('الملخص التنفيذي / Executive Summary', 1)
    for run in summary_heading.runs:
        run.font.name = 'Calibri'
        run.font.color.rgb = RGBColor(102, 126, 234)
        run._element.rPr.rFonts.set(qn('w:cs'), 'Calibri')
    
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
                    run.font.name = 'Calibri'
    
    doc.add_paragraph()
    
    # ========== RCA ANALYSIS SECTION ==========
    rca_heading = doc.add_heading('تحليل الأسباب الجذرية / Root Cause Analysis', 1)
    for run in rca_heading.runs:
        run.font.name = 'Calibri'
        run.font.color.rgb = RGBColor(102, 126, 234)
    
    # RCA Introduction paragraph
    rca_intro = doc.add_paragraph()
    rca_intro_run = rca_intro.add_run(
        f"يوضح هذا القسم التحليل الشامل للأسباب الجذرية (RCA) لعدد {rca_stats['total_rca_records']} "
        f"حالة تم تحليلها خلال الفترة من {season_start.strftime('%Y-%m-%d')} إلى {season_end.strftime('%Y-%m-%d')}. "
        "يشمل التحليل أربع فئات رئيسية: الكوادر البشرية، العمليات، المعدات، والبيئة."
    )
    rca_intro_run.font.size = Pt(11)
    rca_intro_run.font.name = 'Calibri'
    rca_intro.paragraph_format.line_spacing = 1.5
    
    doc.add_paragraph()
    
    # ========== RCA PIE CHART ==========
    pie_chart = _generate_rca_pie_chart(rca_stats)
    if pie_chart:
        chart_heading = doc.add_heading('الرسم البياني لتوزيع الأسباب / Cause Distribution Chart', 2)
        for run in chart_heading.runs:
            run.font.name = 'Calibri'
        
        # Center the chart
        chart_para = doc.add_paragraph()
        chart_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        chart_run = chart_para.add_run()
        chart_run.add_picture(pie_chart, width=Inches(5.5))
        
        doc.add_paragraph()
    
    # ========== RCA CAUSE TYPE OVERVIEW TABLE ==========
    cause_subheading = doc.add_heading('ملخص أنواع الأسباب / Cause Type Summary', 2)
    for run in cause_subheading.runs:
        run.font.name = 'Calibri'
    
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
        
        _set_cell_shading(row.cells[0], cause_colors[cause_type])
        
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(10)
                    run.font.name = 'Calibri'
    
    doc.add_paragraph()
    
    # ========== DETAILED STAFF SUB-CAUSES ==========
    staff_subheading = doc.add_heading('تفصيل أسباب الكوادر البشرية / Staff Causes Breakdown', 2)
    for run in staff_subheading.runs:
        run.font.name = 'Calibri'
    
    staff_subcauses = rca_stats.get('staff_subcauses', {})
    staff_chart = _generate_subcause_bar_chart(
        staff_subcauses, 
        'Staff Sub-Causes Analysis\nتحليل أسباب الكوادر البشرية', 
        '#667eea'
    )
    if staff_chart:
        chart_para = doc.add_paragraph()
        chart_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        chart_run = chart_para.add_run()
        chart_run.add_picture(staff_chart, width=Inches(5.5))
    
    # Staff sub-causes table
    staff_table = doc.add_table(rows=len(staff_subcauses) + 1, cols=3)
    staff_table.style = 'Table Grid'
    staff_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _format_table_header(staff_table, ['السبب الفرعي / Sub-Cause', 'العدد / Count', 'النسبة / Percentage'], "667eea")
    
    staff_total = sum(staff_subcauses.values()) or 1
    for idx, (subcause, count) in enumerate(staff_subcauses.items()):
        row = staff_table.rows[idx + 1]
        row.cells[0].text = subcause
        row.cells[1].text = str(count)
        row.cells[2].text = f"{round(count / staff_total * 100, 1)}%"
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    run.font.name = 'Calibri'
    
    doc.add_paragraph()
    
    # ========== DETAILED PROCESS SUB-CAUSES ==========
    process_subheading = doc.add_heading('تفصيل أسباب العمليات / Process Causes Breakdown', 2)
    for run in process_subheading.runs:
        run.font.name = 'Calibri'
    
    process_subcauses = rca_stats.get('process_subcauses', {})
    process_chart = _generate_subcause_bar_chart(
        process_subcauses,
        'Process Sub-Causes Analysis\nتحليل أسباب العمليات',
        '#FF9800'
    )
    if process_chart:
        chart_para = doc.add_paragraph()
        chart_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        chart_run = chart_para.add_run()
        chart_run.add_picture(process_chart, width=Inches(5.5))
    
    # Process sub-causes table
    process_table = doc.add_table(rows=len(process_subcauses) + 1, cols=3)
    process_table.style = 'Table Grid'
    process_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _format_table_header(process_table, ['السبب الفرعي / Sub-Cause', 'العدد / Count', 'النسبة / Percentage'], "FF9800")
    
    process_total = sum(process_subcauses.values()) or 1
    for idx, (subcause, count) in enumerate(process_subcauses.items()):
        row = process_table.rows[idx + 1]
        row.cells[0].text = subcause
        row.cells[1].text = str(count)
        row.cells[2].text = f"{round(count / process_total * 100, 1)}%"
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    run.font.name = 'Calibri'
    
    doc.add_paragraph()
    
    # ========== DETAILED EQUIPMENT SUB-CAUSES ==========
    equip_subheading = doc.add_heading('تفصيل أسباب المعدات / Equipment Causes Breakdown', 2)
    for run in equip_subheading.runs:
        run.font.name = 'Calibri'
    
    equipment_subcauses = rca_stats.get('equipment_subcauses', {})
    equip_chart = _generate_subcause_bar_chart(
        equipment_subcauses,
        'Equipment Sub-Causes Analysis\nتحليل أسباب المعدات',
        '#E91E63'
    )
    if equip_chart:
        chart_para = doc.add_paragraph()
        chart_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        chart_run = chart_para.add_run()
        chart_run.add_picture(equip_chart, width=Inches(5.5))
    
    # Equipment sub-causes table
    equip_table = doc.add_table(rows=len(equipment_subcauses) + 1, cols=3)
    equip_table.style = 'Table Grid'
    equip_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _format_table_header(equip_table, ['السبب الفرعي / Sub-Cause', 'العدد / Count', 'النسبة / Percentage'], "E91E63")
    
    equip_total = sum(equipment_subcauses.values()) or 1
    for idx, (subcause, count) in enumerate(equipment_subcauses.items()):
        row = equip_table.rows[idx + 1]
        row.cells[0].text = subcause
        row.cells[1].text = str(count)
        row.cells[2].text = f"{round(count / equip_total * 100, 1)}%"
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    run.font.name = 'Calibri'
    
    doc.add_paragraph()
    
    # ========== DETAILED ENVIRONMENT SUB-CAUSES ==========
    env_subheading = doc.add_heading('تفصيل أسباب البيئة / Environment Causes Breakdown', 2)
    for run in env_subheading.runs:
        run.font.name = 'Calibri'
    
    environment_subcauses = rca_stats.get('environment_subcauses', {})
    env_chart = _generate_subcause_bar_chart(
        environment_subcauses,
        'Environment Sub-Causes Analysis\nتحليل أسباب البيئة',
        '#4CAF50'
    )
    if env_chart:
        chart_para = doc.add_paragraph()
        chart_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        chart_run = chart_para.add_run()
        chart_run.add_picture(env_chart, width=Inches(5.5))
    
    # Environment sub-causes table
    env_table = doc.add_table(rows=len(environment_subcauses) + 1, cols=3)
    env_table.style = 'Table Grid'
    env_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _format_table_header(env_table, ['السبب الفرعي / Sub-Cause', 'العدد / Count', 'النسبة / Percentage'], "4CAF50")
    
    env_total = sum(environment_subcauses.values()) or 1
    for idx, (subcause, count) in enumerate(environment_subcauses.items()):
        row = env_table.rows[idx + 1]
        row.cells[0].text = subcause
        row.cells[1].text = str(count)
        row.cells[2].text = f"{round(count / env_total * 100, 1)}%"
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    run.font.name = 'Calibri'
    
    doc.add_paragraph()
    
    # ========== PREVENTIVE MEASURES ANALYSIS ==========
    prevent_subheading = doc.add_heading('تحليل الإجراءات الوقائية / Preventive Measures Analysis', 2)
    for run in prevent_subheading.runs:
        run.font.name = 'Calibri'
    
    # Preventive measures chart
    preventive_measures = rca_stats.get('preventive_measures_detail', {})
    prev_chart = _generate_preventive_measures_chart(preventive_measures)
    if prev_chart:
        chart_para = doc.add_paragraph()
        chart_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        chart_run = chart_para.add_run()
        chart_run.add_picture(prev_chart, width=Inches(5.5))
    
    # Preventive measures summary table
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
        
        if 'Has Preventive' in classification:
            _set_cell_shading(row.cells[0], 'E5FFE5')
        else:
            _set_cell_shading(row.cells[0], 'FFE5E5')
        
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(10)
                    run.font.name = 'Calibri'
    
    doc.add_paragraph()
    
    # Detailed preventive measures table
    prev_detail_subheading = doc.add_heading('تفصيل الإجراءات الوقائية المقترحة / Detailed Preventive Measures', 3)
    for run in prev_detail_subheading.runs:
        run.font.name = 'Calibri'
    
    prev_detail_table = doc.add_table(rows=len(preventive_measures) + 1, cols=3)
    prev_detail_table.style = 'Table Grid'
    prev_detail_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _format_table_header(prev_detail_table, ['الإجراء الوقائي / Preventive Measure', 'العدد / Count', 'النسبة / Percentage'], "4CAF50")
    
    prev_total = sum(preventive_measures.values()) or 1
    for idx, (measure, count) in enumerate(preventive_measures.items()):
        row = prev_detail_table.rows[idx + 1]
        row.cells[0].text = measure
        row.cells[1].text = str(count)
        row.cells[2].text = f"{round(count / prev_total * 100, 1)}%"
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    run.font.name = 'Calibri'
    
    doc.add_paragraph()
    
    # ========== RCA INSIGHTS AND RECOMMENDATIONS ==========
    insights_heading = doc.add_heading('التحليل والتوصيات / Analysis & Recommendations', 2)
    for run in insights_heading.runs:
        run.font.name = 'Calibri'
        run.font.color.rgb = RGBColor(102, 126, 234)
    
    # Generate insights
    insights = _get_rca_insights(rca_stats)
    
    for insight in insights:
        if insight.startswith("التوصيات"):
            # Make recommendations header bold
            rec_para = doc.add_paragraph()
            rec_run = rec_para.add_run(insight)
            rec_run.font.bold = True
            rec_run.font.size = Pt(11)
            rec_run.font.name = 'Calibri'
            rec_run.font.color.rgb = RGBColor(76, 175, 80)
        elif insight:
            insight_para = doc.add_paragraph()
            insight_run = insight_para.add_run(insight)
            insight_run.font.size = Pt(10)
            insight_run.font.name = 'Calibri'
            insight_para.paragraph_format.line_spacing = 1.3
    
    doc.add_paragraph()
    doc.add_page_break()
    
    # ========== SATISFACTION SECTION ==========
    sat_heading = doc.add_heading('رضا المرضى / Patient Satisfaction', 1)
    for run in sat_heading.runs:
        run.font.name = 'Calibri'
        run.font.color.rgb = RGBColor(102, 126, 234)
    
    # Satisfaction by Status
    status_subheading = doc.add_heading('توزيع حسب حالة الرضا / Distribution by Satisfaction Status', 2)
    for run in status_subheading.runs:
        run.font.name = 'Calibri'
    
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
                    run.font.name = 'Calibri'
    
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
                run.font.name = 'Calibri'
    
    doc.add_paragraph()
    
    # Feedback Follow-up Stats
    followup_subheading = doc.add_heading('متابعة ملاحظات المرضى / Patient Feedback Follow-up', 2)
    for run in followup_subheading.runs:
        run.font.name = 'Calibri'
    
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
                    run.font.name = 'Calibri'
    
    doc.add_paragraph()
    
    # ========== RCA BY DEPARTMENT ==========
    if rca_stats['by_department']:
        dept_heading = doc.add_heading('تحليل RCA حسب القسم / RCA Analysis by Department', 1)
        for run in dept_heading.runs:
            run.font.name = 'Calibri'
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
                        run.font.name = 'Calibri'
    
    # ========== FOOTER ==========
    doc.add_paragraph()
    doc.add_paragraph()
    
    footer = doc.add_paragraph()
    footer.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    footer_run = footer.add_run('— نهاية التقرير / End of Report —')
    footer_run.font.size = Pt(10)
    footer_run.font.name = 'Calibri'
    footer_run.font.color.rgb = RGBColor(128, 128, 128)
    footer_run._element.rPr.rFonts.set(qn('w:cs'), 'Calibri')
    
    # Save to bytes
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    
    return buffer.getvalue()
