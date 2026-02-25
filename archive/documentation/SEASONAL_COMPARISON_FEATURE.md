# Seasonal Comparison Reporting Feature

## Overview
Automatic comparative seasonal reports that show current season vs previous season data side-by-side.

## Feature Details

### ✅ What's Implemented

1. **Automatic Previous Season Detection**
   - `get_previous_season()` function in `seasonal_report.py`
   - Logic: Q1→Q4 (previous year), Q2→Q1, Q3→Q2, Q4→Q3
   - Trimester support: Trim1→Trim3 (previous year), Trim2→Trim1, Trim3→Trim2

2. **Dual Report Orchestration**
   - `get_or_generate_comparative_seasonal_reports()` in `seasonal_report_orchestrator.py`
   - Fetches/generates both current AND previous season reports
   - Handles cases where previous season doesn't exist (returns zero-data structure)

3. **Comparative Word Document Generator**
   - `generate_comparative_seasonal_word_report()` in `seasonal_report_formatter.py`
   - Side-by-side comparison tables
   - Delta indicators (↑↓) with color coding
   - Percentage change calculations
   - Full Arabic RTL support

4. **Download Integration**
   - Modified `report_export_service.py` to always generate comparative reports
   - Works for both PDF and DOCX formats
   - Filename reflects comparison: `Seasonal_Comparison_2025_Q2.docx`

### 📊 Report Structure

The comparative report includes:

#### 1. Header Section
- Title: "تقرير المقارنة الموسمية | Seasonal Comparison Report"
- Periods: "Q2-2025 مقابل Q1-2025"
- Organization info

#### 2. Summary Comparison Table
```
┌─────────────────────────────────────────────────────────┐
│ Category      │ Previous │ Current │ Delta │ % Change │
├───────────────┼──────────┼─────────┼───────┼──────────┤
│ Total Cases   │    45    │   52    │ +7 ↑  │  +15.6%  │
│ Clinical      │    20    │   18    │ -2 ↓  │  -10.0%  │
│ Management    │    15    │   22    │ +7 ↑  │  +46.7%  │
│ ...           │   ...    │  ...    │  ...  │   ...    │
└─────────────────────────────────────────────────────────┘
```

#### 3. Domain-by-Domain Comparison
- Separate tables for each domain (Clinical, Management, Relational)
- Classification-level comparison
- Severity breakdown (Low/Medium/High) for both periods
- Prevention action counts (Yes/No) for both periods

#### 4. Policy Compliance Comparison
- Shows compliance status for both periods
- Visual indicators (✓ compliant / ✗ non-compliant)

### 🔧 Technical Details

#### Files Modified

1. **`backend/api/db_layer/seasonal_report.py`**
   - Added: `get_previous_season(season_id: int) -> Optional[int]`
   - Determines previous season based on naming convention

2. **`backend/api/services/seasonal_report_orchestrator.py`**
   - Added: `get_or_generate_comparative_seasonal_reports(...)`
   - Orchestrates generation of both current and previous reports
   - Returns dict with `current_report`, `previous_report`, `has_previous`

3. **`backend/api/services/seasonal_report_formatter.py`**
   - Added: `generate_comparative_seasonal_word_report(...)`
   - Generates Word document with side-by-side comparison
   - Added: `_create_comparative_hierarchical_tables_by_domain(...)`
   - Helper function for building comparative tables

4. **`backend/api/services/report_export_service.py`**
   - Modified: `generate_export()` method
   - Changed orchestrator call from single to comparative
   - Updated DOCX and PDF generation to use comparative formatter
   - Updated filename generation to reflect comparison

### 🎯 User Experience

#### Before (Old Behavior)
- Download button → Single season report
- User must manually compare multiple downloaded files
- No trend indicators

#### After (New Behavior - AUTOMATIC)
- Download button → Comparative report (current vs previous)
- Automatic side-by-side comparison
- Delta indicators (↑↓) with color coding
- Percentage change calculations
- Even works for first season (shows zero data for "previous")

### 💡 Design Decisions

1. **Always Generated**: No checkbox needed - comparison is automatic
2. **Zero Data Handling**: If no previous season exists, shows zero-data structure (not an error)
3. **Format**: Mirrors regular seasonal report but adds comparison columns
4. **RTL Support**: Fully respects Arabic right-to-left reading direction
5. **Filename**: Clearly indicates it's a comparison report

### 🔄 Flow Diagram

```
User clicks "Download Q2-2025 Report"
         ↓
get_or_generate_comparative_seasonal_reports()
         ↓
    ┌────────┴────────┐
    │                 │
Generate Q2-2025   Detect Previous (Q1-2025)
    │                 │
    │            Generate Q1-2025
    │                 │
    └────────┬────────┘
         ↓
generate_comparative_seasonal_word_report(current, previous)
         ↓
Word document: "Seasonal_Comparison_2025_Q2.docx"
         ↓
Downloaded with side-by-side comparison
```

### 🚀 Next Steps (Optional Enhancements)

1. **Add visual charts** showing trend lines across multiple seasons
2. **Add "Top 3 Changes"** executive summary section
3. **Color-code entire rows** based on improvement/degradation
4. **Add comparison insights** - auto-generated text explaining key changes
5. **Support comparison across multiple periods** (not just previous, but Q1 vs Q3, etc.)

### 📝 Testing

To test the feature:

```python
from backend.api.services.seasonal_report_orchestrator import get_or_generate_comparative_seasonal_reports

# Example: Generate comparative report for Q2-2025
result = get_or_generate_comparative_seasonal_reports(
    season_id=10,  # Q2-2025
    orgunit_id=1,
    orgunit_type=0,
    user_id=1
)

# Result contains:
# - result['current_report']  # Q2-2025 data
# - result['previous_report'] # Q1-2025 data
# - result['has_previous']    # True
```

### ✅ Completed Tasks

- [x] Add `get_previous_season()` utility function
- [x] Create comparative report formatter function
- [x] Modify orchestrator to fetch both seasons
- [x] Update download endpoint to return comparison report
- [x] Ensure numbers are centered horizontally and vertically
- [x] Fix RTL table ordering for Arabic

---

**Status**: ✅ **FULLY IMPLEMENTED AND READY FOR TESTING**

**Author**: GitHub Copilot  
**Date**: January 16, 2026  
**Version**: Seasonal_Reporting_3
