# 📊 Seasonal Comparison Feature - Multi-Quarter Implementation Plan

## 🎯 Executive Summary

This document outlines the implementation plan for a comprehensive **Seasonal Comparison System** that allows comparing 2, 3, or 4 quarters of data at various organizational levels (Hospital, Administration, Department, Section).

---

## 📋 Current State Analysis

### ✅ What Currently Exists
1. **2-Quarter Comparison (Automatic)** - Already implemented
   - Compares current season vs. previous season
   - Generates comparative Word reports with tables and 9 charts
   - Available at all organizational levels
   - Files: `seasonal_report_formatter.py` (function: `generate_comparative_seasonal_word_report`)

2. **Single Seasonal Report Generation**
   - Hospital, Administration, Department, Section levels
   - Word (DOCX) and PDF export formats
   - Hierarchical tables with Domain → Category → Sub-Category → Classification

3. **Multi-Unit Export**
   - Generates ZIP files with multiple reports
   - Service: `MultiSeasonalExportService`

### 🔴 What Needs to Be Built
1. **3-Quarter Comparison Service & Endpoint**
2. **4-Quarter (One Year) Comparison Service & Endpoint**
3. **Refactored 2-Quarter Comparison Endpoint** (dedicated, not automatic)
4. **Modified Word Report Structure** (tables first, then graphs)
5. **Frontend Integration** (UI to select comparison type)
6. **Removal of Forced Automatic Comparison** in standard seasonal reports

---

## 🏗️ Implementation Phases

## **PHASE 1: Planning & Architecture** (1-2 days)

### Objectives
- Finalize technical specifications
- Design database schema (if needed)
- Design API contracts
- Create test data

### Tasks
1. **Review Current Implementation**
   - [ ] Analyze `seasonal_report_formatter.py` (line 1625: `generate_comparative_seasonal_word_report`)
   - [ ] Analyze `seasonal_report_orchestrator.py` (line 93: `get_or_generate_comparative_seasonal_reports`)
   - [ ] Review `multi_seasonal_export_service.py` for multi-unit logic
   - [ ] Document current 2-quarter comparison logic

2. **Design New Services**
   - [ ] Design `SeasonalComparisonService` class
   - [ ] Design data structure for 3-quarter comparison
   - [ ] Design data structure for 4-quarter comparison
   - [ ] Design graph selection logic (Domain: Spider + Bar Subtraction, Category: Spider + Bar Subtraction)

3. **Define API Contracts**
   ```python
   POST /api/reports/seasonal/comparison/2-quarters
   POST /api/reports/seasonal/comparison/3-quarters
   POST /api/reports/seasonal/comparison/4-quarters
   
   Request Body:
   {
     "season_ids": [1, 2] or [1, 2, 3] or [1, 2, 3, 4],
     "orgunit_id": 1,
     "orgunit_type": 0,
     "format": "docx",
     "language": "en"
   }
   ```

4. **Database Review**
   - [ ] Check if `APP_LOOKUP_SEASON` table supports quarter sequence queries
   - [ ] Verify season ID retrieval for consecutive quarters

### Deliverables
- ✅ Technical specification document
- ✅ API contract definitions
- ✅ Database schema validation

---

## **PHASE 2: Backend Foundation - Comparison Services** (3-4 days)

### Objectives
- Build reusable comparison service layer
- Create data aggregation functions
- Implement 3-quarter and 4-quarter comparison logic

### Tasks

#### 2.1 Create Base Comparison Service
**File**: `backend/api/services/seasonal_comparison_service.py` (NEW)

```python
class SeasonalComparisonService:
    """
    Service for generating multi-quarter seasonal comparisons.
    Supports 2, 3, and 4 quarter comparisons.
    """
    
    def fetch_seasons_data(
        self,
        season_ids: List[int],
        orgunit_id: int,
        orgunit_type: int,
        user_id: int
    ) -> List[Dict[str, Any]]:
        """Fetch multiple seasonal reports."""
        pass
    
    def generate_2_quarter_comparison(
        self,
        season_ids: List[int],
        orgunit_id: int,
        orgunit_type: int,
        user_id: int
    ) -> Dict[str, Any]:
        """Generate 2-quarter comparison (refactored from existing)."""
        pass
    
    def generate_3_quarter_comparison(
        self,
        season_ids: List[int],
        orgunit_id: int,
        orgunit_type: int,
        user_id: int
    ) -> Dict[str, Any]:
        """Generate 3-quarter comparison with trend analysis."""
        pass
    
    def generate_4_quarter_comparison(
        self,
        season_ids: List[int],
        orgunit_id: int,
        orgunit_type: int,
        user_id: int
    ) -> Dict[str, Any]:
        """Generate 4-quarter (yearly) comparison."""
        pass
```

#### 2.2 Implement Data Aggregation Helpers
- [ ] Create `_aggregate_domain_data(reports: List[Dict]) -> Dict`
- [ ] Create `_aggregate_category_data(reports: List[Dict]) -> Dict`
- [ ] Create `_calculate_trends(reports: List[Dict]) -> Dict` (for 3/4 quarters)
- [ ] Create `_calculate_percentage_changes(reports: List[Dict]) -> Dict`

#### 2.3 Database Layer Enhancements
**File**: `backend/api/db_layer/seasonal_report.py`

- [ ] Add `get_consecutive_quarters(start_season_id: int, count: int) -> List[int]`
- [ ] Add `validate_quarter_sequence(season_ids: List[int]) -> bool`
- [ ] Add `get_season_metadata(season_id: int) -> Dict` (year, quarter, period name)

### Deliverables
- ✅ `seasonal_comparison_service.py` with all comparison methods
- ✅ Database helper functions
- ✅ Unit tests for data aggregation

---

## **PHASE 3: Word Report Restructuring** (3-4 days)

### Objectives
- Modify Word report generation to place **tables first**, **graphs last**
- Implement graph selection logic (Domain: Spider + Bar Subtraction, Category: Spider + Bar Subtraction)

### Tasks

#### 3.1 Refactor 2-Quarter Comparison Report Generator
**File**: `backend/api/services/seasonal_report_formatter.py`

Current function: `generate_comparative_seasonal_word_report(current_data, previous_data, language)`

Modifications:
```python
def generate_comparative_seasonal_word_report_v2(
    seasons_data: List[Dict[str, Any]],  # Changed: Accept list of 2+ seasons
    comparison_type: Literal["2q", "3q", "4q"],
    language: str = "en",
    graph_config: Dict = None  # NEW: Control which graphs to include
) -> bytes:
    """
    Generate comparison report with TABLES FIRST, then GRAPHS.
    
    Structure:
    1. Header (Title, Period, Organization Info)
    2. TABLES SECTION
       - Summary comparison table
       - Domain-by-domain comparison tables
       - Category-by-category comparison tables (if applicable)
    3. GRAPHS SECTION (at the end)
       - Domain Level: Spider Chart + Bar Subtraction Chart
       - Category Level: Spider Chart + Bar Subtraction Chart
    """
    pass
```

- [ ] **Extract table generation** into separate function: `_generate_all_comparison_tables()`
- [ ] **Extract graph generation** into separate function: `_generate_all_comparison_graphs()`
- [ ] **Reorder document structure**: Tables → Page Break → Graphs
- [ ] **Implement graph filtering logic**:
  - Domain graphs: Spider + Bar Subtraction only
  - Category graphs: Spider + Bar Subtraction only

#### 3.2 Create 3-Quarter Report Generator
**File**: `backend/api/services/seasonal_report_formatter.py`

```python
def generate_3_quarter_comparison_report(
    seasons_data: List[Dict[str, Any]],  # 3 quarters
    language: str = "en"
) -> bytes:
    """
    Generate 3-quarter comparison report.
    
    Special features:
    - Trend indicators (↑ trending up, → stable, ↓ trending down)
    - 3-column comparison tables
    - Line graphs showing trends across 3 quarters
    """
    pass
```

- [ ] Design 3-column table layout (Q1 | Q2 | Q3 | Trend)
- [ ] Implement trend calculation logic
- [ ] Add line graphs for trend visualization

#### 3.3 Create 4-Quarter (Yearly) Report Generator
**File**: `backend/api/services/seasonal_report_formatter.py`

```python
def generate_4_quarter_comparison_report(
    seasons_data: List[Dict[str, Any]],  # 4 quarters (1 year)
    language: str = "en"
) -> bytes:
    """
    Generate 4-quarter (yearly) comparison report.
    
    Special features:
    - 4-column comparison tables
    - Year-over-year summary
    - Seasonal patterns analysis
    - Annual totals and averages
    """
    pass
```

- [ ] Design 4-column table layout
- [ ] Calculate annual totals and quarterly averages
- [ ] Add yearly summary section

### Deliverables
- ✅ Refactored 2-quarter report generator with tables-first structure
- ✅ New 3-quarter report generator
- ✅ New 4-quarter report generator
- ✅ Graph filtering logic implemented

---

## **PHASE 4: Backend API Endpoints** (2-3 days)

### Objectives
- Create dedicated comparison endpoints
- Implement multi-level support (Hospital, Administration, Department, Section)
- Add selectable number of units feature

### Tasks

#### 4.1 Create Comparison Router
**File**: `backend/api/routers/seasonal_comparison_router.py` (NEW)

```python
from fastapi import APIRouter, Query, HTTPException, Response
from pydantic import BaseModel
from typing import List, Literal, Optional

router = APIRouter(prefix="/api/reports/seasonal/comparison", tags=["Seasonal Comparison"])


class ComparisonRequest(BaseModel):
    season_ids: List[int]  # 2, 3, or 4 season IDs
    orgunit_id: int
    orgunit_type: int  # 0=Hospital, 1=Admin, 2=Dept, 3=Section
    format: Literal["docx", "pdf"] = "docx"
    language: Literal["en", "ar"] = "en"


@router.post("/2-quarters")
async def compare_2_quarters(request: ComparisonRequest):
    """Generate 2-quarter comparison report."""
    pass


@router.post("/3-quarters")
async def compare_3_quarters(request: ComparisonRequest):
    """Generate 3-quarter comparison report."""
    pass


@router.post("/4-quarters")
async def compare_4_quarters(request: ComparisonRequest):
    """Generate 4-quarter (yearly) comparison report."""
    pass


@router.post("/multi-unit")
async def compare_multi_units(
    season_ids: List[int] = Query(...),
    comparison_type: Literal["2q", "3q", "4q"] = Query(...),
    report_level: Literal["administration", "department", "section"] = Query(...),
    selected_unit_ids: Optional[List[int]] = Query(None),
    format: Literal["docx", "pdf"] = Query("docx"),
    language: Literal["en", "ar"] = Query("en")
):
    """
    Generate comparison reports for multiple units.
    
    Examples:
    - All Administrations: report_level="administration", selected_unit_ids=None
    - Specific Departments: report_level="department", selected_unit_ids=[1, 2, 3]
    """
    pass
```

#### 4.2 Implement Endpoint Logic
- [ ] **2-quarters endpoint**: Call `SeasonalComparisonService.generate_2_quarter_comparison()`
- [ ] **3-quarters endpoint**: Call `SeasonalComparisonService.generate_3_quarter_comparison()`
- [ ] **4-quarters endpoint**: Call `SeasonalComparisonService.generate_4_quarter_comparison()`
- [ ] **Multi-unit endpoint**: Create ZIP with multiple comparison reports

#### 4.3 Register Router in Main Application
**File**: `backend/main.py`

```python
from backend.api.routers.seasonal_comparison_router import router as comparison_router

app.include_router(comparison_router)
```

### Deliverables
- ✅ `seasonal_comparison_router.py` with all endpoints
- ✅ Integration with `seasonal_comparison_service.py`
- ✅ Router registered in FastAPI app

---

## **PHASE 5: Remove Forced Automatic Comparison** (1 day)

### Objectives
- Refactor existing seasonal export to NOT automatically include comparison
- Make comparison optional/explicit via new endpoints

### Tasks

#### 5.1 Modify Report Export Service
**File**: `backend/api/services/report_export_service.py`

Current behavior (lines 160-230):
- Automatically calls `get_or_generate_comparative_seasonal_reports()`
- Always generates comparison report in ZIP

**New behavior**:
- Call `get_or_generate_seasonal_report()` (single report only)
- Remove automatic ZIP packaging with comparison
- Comparison only available via dedicated endpoints

Changes:
```python
# OLD (lines 160-165)
report_data = get_or_generate_comparative_seasonal_reports(
    season_id=season_id,
    orgunit_id=orgunit_id,
    orgunit_type=orgunit_type,
    user_id=1
)

# NEW
report_data = get_or_generate_seasonal_report(
    season_id=season_id,
    orgunit_id=orgunit_id,
    orgunit_type=orgunit_type,
    user_id=1
)
```

- [ ] Remove automatic comparison generation from `report_export_service.py`
- [ ] Update `multi_seasonal_export_service.py` to NOT include comparisons by default
- [ ] Add deprecation notice for old behavior

#### 5.2 Update Seasonal Export Endpoint
**File**: `backend/api/routers/reports_router.py`

- [ ] Ensure `/api/reports/seasonal/export` only returns single-season reports
- [ ] Remove ZIP packaging logic for comparisons
- [ ] Update documentation

### Deliverables
- ✅ Seasonal export returns single reports only
- ✅ Comparison removed from automatic generation
- ✅ Updated endpoint documentation

---

## **PHASE 6: Frontend Integration** (3-4 days)

### Objectives
- Add UI to select between "Seasonal Report" or "Seasonal Comparison"
- Add comparison type selector (2, 3, or 4 quarters)
- Add quarter/season selector (multi-select for comparison)
- Integrate with new backend endpoints

### Tasks

#### 6.1 Update Seasonal Report Page UI
**Location**: Frontend seasonal reporting component

**New UI Elements**:
1. **Report Type Selector** (Radio buttons)
   - [ ] Option 1: "Single Seasonal Report"
   - [ ] Option 2: "Seasonal Comparison"

2. **Comparison Type Selector** (Conditional - only if "Seasonal Comparison" selected)
   - [ ] Option 1: "2 Quarters"
   - [ ] Option 2: "3 Quarters"
   - [ ] Option 3: "4 Quarters"

3. **Season/Quarter Selector** (Multi-select dropdown)
   - [ ] Fetch available seasons from API
   - [ ] Allow selecting 2, 3, or 4 quarters based on comparison type
   - [ ] Validate consecutive quarters (optional)

4. **Organizational Level Selector** (Dropdown)
   - [ ] Hospital
   - [ ] Administration (single or multiple)
   - [ ] Department (single or multiple)
   - [ ] Section (single or multiple)

5. **Export Format Selector**
   - [ ] Word (DOCX)
   - [ ] PDF (future)

#### 6.2 Implement Frontend API Calls

```typescript
// New API service functions
export const generateSeasonalComparison = async (
  comparisonType: '2q' | '3q' | '4q',
  seasonIds: number[],
  orgunitId: number,
  orgunitType: number,
  format: 'docx' | 'pdf',
  language: 'en' | 'ar'
) => {
  const endpoint = `/api/reports/seasonal/comparison/${comparisonType}-quarters`;
  return await axios.post(endpoint, {
    season_ids: seasonIds,
    orgunit_id: orgunitId,
    orgunit_type: orgunitType,
    format,
    language
  });
};

export const generateMultiUnitComparison = async (
  comparisonType: '2q' | '3q' | '4q',
  seasonIds: number[],
  reportLevel: 'administration' | 'department' | 'section',
  selectedUnitIds: number[] | null,
  format: 'docx' | 'pdf',
  language: 'en' | 'ar'
) => {
  return await axios.post('/api/reports/seasonal/comparison/multi-unit', null, {
    params: {
      season_ids: seasonIds.join(','),
      comparison_type: comparisonType,
      report_level: reportLevel,
      selected_unit_ids: selectedUnitIds?.join(',') || 'all',
      format,
      language
    }
  });
};
```

#### 6.3 Update UI Logic
- [ ] Add state management for report type selection
- [ ] Add state management for comparison type
- [ ] Add state management for selected seasons
- [ ] Implement conditional rendering based on selections
- [ ] Add loading states during report generation
- [ ] Add success/error notifications

#### 6.4 Update Download Handler
```typescript
const handleDownload = async () => {
  if (reportType === 'single') {
    // Existing single report logic
    await downloadSingleSeasonalReport();
  } else {
    // New comparison logic
    if (multiUnit) {
      await downloadMultiUnitComparison();
    } else {
      await downloadSingleComparison();
    }
  }
};
```

### Deliverables
- ✅ Updated UI with report type selector
- ✅ Comparison type selector (2, 3, 4 quarters)
- ✅ Season multi-select component
- ✅ Integration with new API endpoints
- ✅ Updated download handlers

---

## **PHASE 7: Testing & Validation** (2-3 days)

### Objectives
- Test all comparison types at all organizational levels
- Validate report structure (tables first, graphs last)
- Test multi-unit export functionality
- Verify graph selection logic

### Tasks

#### 7.1 Create Test Files
**File**: `test_seasonal_comparison.py` (NEW)

```python
"""
Test Seasonal Comparison Feature
Tests 2, 3, and 4 quarter comparisons at all organizational levels.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import requests

BASE_URL = "http://localhost:8000"

def test_2_quarter_comparison_hospital():
    """Test 2-quarter comparison at hospital level."""
    pass

def test_3_quarter_comparison_department():
    """Test 3-quarter comparison at department level."""
    pass

def test_4_quarter_comparison_administration():
    """Test 4-quarter comparison at administration level."""
    pass

def test_multi_unit_comparison_all_departments():
    """Test multi-unit comparison for all departments."""
    pass

def test_report_structure():
    """Verify tables appear before graphs in generated report."""
    pass

def test_graph_selection():
    """Verify only Spider and Bar Subtraction graphs are included."""
    pass

if __name__ == "__main__":
    main()
```

#### 7.2 Test Scenarios

**Backend Testing:**
- [ ] Test 2-quarter comparison (Hospital, Admin, Dept, Section)
- [ ] Test 3-quarter comparison (Hospital, Admin, Dept, Section)
- [ ] Test 4-quarter comparison (Hospital, Admin, Dept, Section)
- [ ] Test multi-unit comparison (All Admins, All Depts, All Sections)
- [ ] Test with invalid season IDs (error handling)
- [ ] Test with non-consecutive quarters (if validation is enabled)

**Report Structure Testing:**
- [ ] Verify tables appear first in Word document
- [ ] Verify graphs appear after all tables
- [ ] Verify only Spider and Bar Subtraction graphs are included
- [ ] Verify trend indicators in 3-quarter reports
- [ ] Verify annual summaries in 4-quarter reports

**Frontend Testing:**
- [ ] Test UI state transitions
- [ ] Test season multi-select validation
- [ ] Test download functionality
- [ ] Test error handling and user feedback
- [ ] Test with different languages (en, ar)

**Performance Testing:**
- [ ] Measure generation time for 4-quarter comparison at hospital level
- [ ] Test ZIP generation for multi-unit exports
- [ ] Test with large datasets (100+ cases per quarter)

#### 7.3 Documentation Updates
- [ ] Update API documentation with new endpoints
- [ ] Create user guide for seasonal comparison feature
- [ ] Update SEASONAL_QUICK_REFERENCE.md
- [ ] Add examples and screenshots

### Deliverables
- ✅ Comprehensive test suite
- ✅ Test results documentation
- ✅ Bug fixes from testing
- ✅ Updated documentation

---

## **PHASE 8: Deployment & Monitoring** (1-2 days)

### Objectives
- Deploy to production
- Monitor performance
- Gather user feedback

### Tasks

#### 8.1 Pre-Deployment Checklist
- [ ] All tests passing
- [ ] Code review completed
- [ ] Documentation updated
- [ ] Database migrations (if needed)
- [ ] Performance benchmarks met

#### 8.2 Deployment
- [ ] Deploy backend services
- [ ] Deploy frontend changes
- [ ] Update API documentation
- [ ] Notify users of new feature

#### 8.3 Post-Deployment Monitoring
- [ ] Monitor API response times
- [ ] Monitor error rates
- [ ] Monitor user adoption
- [ ] Collect user feedback

#### 8.4 Training & Support
- [ ] Create training materials
- [ ] Conduct user training sessions
- [ ] Set up support channels
- [ ] Create FAQ documentation

### Deliverables
- ✅ Production deployment
- ✅ Monitoring dashboards
- ✅ User training materials
- ✅ Support documentation

---

## 📊 Implementation Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| **Phase 1: Planning** | 1-2 days | None |
| **Phase 2: Backend Services** | 3-4 days | Phase 1 |
| **Phase 3: Word Report Restructuring** | 3-4 days | Phase 2 |
| **Phase 4: API Endpoints** | 2-3 days | Phase 2, 3 |
| **Phase 5: Remove Forced Comparison** | 1 day | Phase 4 |
| **Phase 6: Frontend Integration** | 3-4 days | Phase 4 |
| **Phase 7: Testing** | 2-3 days | Phase 2-6 |
| **Phase 8: Deployment** | 1-2 days | Phase 7 |
| **TOTAL** | **16-23 days** | (~3-4 weeks) |

---

## 🎯 Success Criteria

### Functional Requirements
✅ Users can select between single report or comparison
✅ Users can compare 2, 3, or 4 consecutive quarters
✅ Comparisons available at all organizational levels
✅ Reports show tables first, then graphs
✅ Only Spider and Bar Subtraction graphs included
✅ Multi-unit exports generate ZIP files

### Performance Requirements
✅ 2-quarter comparison generates in <10 seconds
✅ 4-quarter comparison generates in <20 seconds
✅ Multi-unit export (10 units) generates in <60 seconds

### Quality Requirements
✅ All tests passing (unit, integration, E2E)
✅ Zero data inconsistencies in reports
✅ Proper error handling and user feedback
✅ Arabic (RTL) and English support

---

## 🚨 Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Report generation too slow | High | Medium | Implement caching, optimize queries |
| Memory issues with 4-quarter reports | High | Low | Stream data, use pagination |
| Frontend complexity | Medium | Medium | Incremental development, code reviews |
| User adoption | Medium | Low | User training, clear documentation |
| Data inconsistencies | High | Low | Comprehensive testing, validation |

---

## 📝 Key Technical Decisions

### 1. Report Structure Change
**Decision**: Tables first, graphs last
**Rationale**: Better readability, easier to print tables separately

### 2. Graph Selection
**Decision**: Only Spider + Bar Subtraction for Domain and Category
**Rationale**: User request, reduces report size, focuses on key comparisons

### 3. Endpoint Design
**Decision**: Separate endpoints for 2q, 3q, 4q comparisons
**Rationale**: Clear intent, easier to maintain, better API design

### 4. Multi-Unit Support
**Decision**: Single endpoint with parameters for "all" or specific IDs
**Rationale**: Flexible, avoids endpoint proliferation

### 5. Backward Compatibility
**Decision**: Remove automatic comparison from existing endpoint
**Rationale**: Cleaner separation, comparison should be explicit

---

## 📚 Related Documents

- [SEASONAL_COMPARISON_FEATURE.md](SEASONAL_COMPARISON_FEATURE.md) - Current 2-quarter comparison
- [SEASONAL_QUICK_REFERENCE.md](SEASONAL_QUICK_REFERENCE.md) - Seasonal reporting overview
- [REPORTING_SERVICES_ANALYSIS.md](REPORTING_SERVICES_ANALYSIS.md) - Service architecture
- [PHASE8_FRONTEND_REQUIREMENTS.md](PHASE8_FRONTEND_REQUIREMENTS.md) - Frontend requirements

---

## ✅ Next Steps

1. **Review this plan** with stakeholders
2. **Approve technical approach**
3. **Create Jira/GitHub issues** for each phase
4. **Assign developers** to phases
5. **Begin Phase 1: Planning**

---

## 📧 Contact & Support

For questions or clarifications about this implementation plan:
- Technical Lead: [Name]
- Project Manager: [Name]
- Business Owner: [Name]

---

**Document Version**: 1.0
**Last Updated**: January 19, 2026
**Status**: ✅ Ready for Review
