# Analytics Endpoints - Testing Guide

## Overview

Four new analytics endpoints have been added to power dashboard cards for Red Flags and Never Events.

**Base URL:** `http://127.0.0.1:8000`

---

## Red Flags Analytics

### 1. Category Breakdown

**Endpoint:**
```
GET /api/red-flags/category-breakdown
```

**Purpose:** Distribution of red flags across categories with severity breakdown

**Query Parameters:**
- `from_date` (optional): Filter from date (YYYY-MM-DD)
- `to_date` (optional): Filter to date (YYYY-MM-DD)

**Test URLs:**
```
http://127.0.0.1:8000/api/red-flags/category-breakdown
http://127.0.0.1:8000/api/red-flags/category-breakdown?from_date=2024-01-01
http://127.0.0.1:8000/api/red-flags/category-breakdown?from_date=2024-01-01&to_date=2024-12-31
```

**Response Format:**
```json
{
  "total": 245,
  "period": "2024-01-01 to 2024-12-31",
  "categories": [
    {
      "category_name": "Patient Safety",
      "category_name_ar": "سلامة المريض",
      "count": 98,
      "percentage": 40.0,
      "severity_breakdown": {
        "CRITICAL": 45,
        "HIGH": 53
      }
    },
    {
      "category_name": "Medical Errors",
      "category_name_ar": "الأخطاء الطبية",
      "count": 67,
      "percentage": 27.3,
      "severity_breakdown": {
        "CRITICAL": 22,
        "HIGH": 45
      }
    }
  ]
}
```

**Use Cases:**
- Pie chart showing category distribution
- KPI card showing top category
- Severity analysis by category

---

### 2. Department Breakdown

**Endpoint:**
```
GET /api/red-flags/department-breakdown
```

**Purpose:** Distribution of red flags across departments with status breakdown

**Query Parameters:**
- `from_date` (optional): Filter from date (YYYY-MM-DD)
- `to_date` (optional): Filter to date (YYYY-MM-DD)
- `limit` (optional): Max departments to return (default: 10, max: 50)

**Test URLs:**
```
http://127.0.0.1:8000/api/red-flags/department-breakdown
http://127.0.0.1:8000/api/red-flags/department-breakdown?limit=5
http://127.0.0.1:8000/api/red-flags/department-breakdown?from_date=2024-01-01&limit=10
```

**Response Format:**
```json
{
  "total": 245,
  "period": "2024-01-01 to 2024-12-31",
  "departments": [
    {
      "department": "الطوارئ",
      "department_en": "Emergency",
      "count": 45,
      "percentage": 18.4,
      "status_breakdown": {
        "OPEN": 12,
        "UNDER_REVIEW": 18,
        "FINISHED": 15
      }
    },
    {
      "department": "الجراحة",
      "department_en": "Surgery",
      "count": 38,
      "percentage": 15.5,
      "status_breakdown": {
        "OPEN": 8,
        "UNDER_REVIEW": 15,
        "FINISHED": 15
      }
    }
  ]
}
```

**Use Cases:**
- Bar chart showing top departments
- KPI card showing most affected department
- Status analysis by department

---

## Never Events Analytics

### 3. Category Breakdown

**Endpoint:**
```
GET /api/never-events/category-breakdown
```

**Purpose:** Distribution of never events across categories with specific event types

**Query Parameters:**
- `from_date` (optional): Filter from date (YYYY-MM-DD)
- `to_date` (optional): Filter to date (YYYY-MM-DD)

**Test URLs:**
```
http://127.0.0.1:8000/api/never-events/category-breakdown
http://127.0.0.1:8000/api/never-events/category-breakdown?from_date=2024-01-01
http://127.0.0.1:8000/api/never-events/category-breakdown?from_date=2024-01-01&to_date=2024-12-31
```

**Response Format:**
```json
{
  "total": 12,
  "goal": 0,
  "period": "2024-01-01 to 2024-12-31",
  "categories": [
    {
      "category_name": "Surgical Events",
      "category_name_ar": "أحداث جراحية",
      "count": 5,
      "percentage": 41.7,
      "types": [
        {
          "type": "Wrong Site Surgery",
          "type_ar": "جراحة في موقع خاطئ",
          "count": 3
        },
        {
          "type": "Wrong Patient Surgery",
          "type_ar": "جراحة لمريض خاطئ",
          "count": 2
        }
      ]
    },
    {
      "category_name": "Medication Events",
      "category_name_ar": "أحداث دوائية",
      "count": 4,
      "percentage": 33.3,
      "types": [
        {
          "type": "Wrong Patient Medication",
          "type_ar": "دواء لمريض خاطئ",
          "count": 4
        }
      ]
    }
  ]
}
```

**Use Cases:**
- Pie chart showing category distribution
- Drill-down view showing specific event types
- Zero-tolerance monitoring card
- Goal comparison (actual vs target of 0)

---

### 4. Timeline Comparison

**Endpoint:**
```
GET /api/never-events/timeline-comparison
```

**Purpose:** Compare current vs previous period to track progress toward zero

**Query Parameters:**
- `period` (optional): Time period (month, quarter, or year) - default: month

**Test URLs:**
```
http://127.0.0.1:8000/api/never-events/timeline-comparison
http://127.0.0.1:8000/api/never-events/timeline-comparison?period=month
http://127.0.0.1:8000/api/never-events/timeline-comparison?period=quarter
http://127.0.0.1:8000/api/never-events/timeline-comparison?period=year
```

**Response Format:**
```json
{
  "goal": 0,
  "current": {
    "period": "December 2024",
    "period_ar": "ديسمبر 2024",
    "count": 2,
    "start_date": "2024-12-01",
    "end_date": "2024-12-31"
  },
  "previous": {
    "period": "November 2024",
    "period_ar": "نوفمبر 2024",
    "count": 3
  },
  "change": {
    "absolute": -1,
    "percentage": -33.3,
    "trend": "improving"
  },
  "year_to_date": {
    "count": 12,
    "average_per_month": 1.0
  }
}
```

**Use Cases:**
- Comparison card (current vs previous)
- Trend indicator (improving/worsening)
- YTD summary card
- Progress toward zero goal
- Month-over-month/quarter-over-quarter tracking

---

## Key Features

### Common Across All Endpoints

✅ **Always return HTTP 200** (even if no data)
✅ **Empty arrays** when no data matches filters
✅ **Sorted by count DESC** (most to least)
✅ **Arabic and English names** included
✅ **Optional date filtering** on all endpoints
✅ **Percentage calculations** (rounded to 1 decimal)
✅ **Consistent JSON structure**

### Red Flags Specific

- **Severity breakdown**: CRITICAL vs HIGH
- **Status breakdown**: OPEN, UNDER_REVIEW, FINISHED
- **Department limit**: Top N departments

### Never Events Specific

- **Goal: 0** (zero tolerance)
- **Event types**: Drill-down to specific never event types
- **Trend analysis**: Improving vs worsening
- **YTD tracking**: Year-to-date summaries

---

## Testing Checklist

### Red Flags - Category Breakdown
- [ ] Test without filters (all time)
- [ ] Test with from_date only
- [ ] Test with both from_date and to_date
- [ ] Verify categories sorted by count DESC
- [ ] Verify percentages sum to ~100%
- [ ] Verify severity_breakdown has CRITICAL and HIGH
- [ ] Verify Arabic names present

### Red Flags - Department Breakdown
- [ ] Test without filters (all time)
- [ ] Test with limit=5
- [ ] Test with limit=10
- [ ] Test with date range
- [ ] Verify departments sorted by count DESC
- [ ] Verify status_breakdown has OPEN, UNDER_REVIEW, FINISHED
- [ ] Verify Arabic department names present

### Never Events - Category Breakdown
- [ ] Test without filters (all time)
- [ ] Test with date range
- [ ] Verify goal=0 always present
- [ ] Verify categories sorted by count DESC
- [ ] Verify types array includes specific event types
- [ ] Verify Arabic names present

### Never Events - Timeline Comparison
- [ ] Test period=month
- [ ] Test period=quarter
- [ ] Test period=year
- [ ] Verify goal=0 always present
- [ ] Verify trend is "improving" when count decreases
- [ ] Verify trend is "worsening" when count increases
- [ ] Verify YTD calculations are correct
- [ ] Verify Arabic period names present

---

## Integration Notes

### Frontend Usage

**Category Breakdown Cards:**
```typescript
// Pie Chart
const categoryData = response.categories.map(cat => ({
  name: cat.category_name_ar,
  value: cat.count
}));

// KPI Card
const topCategory = response.categories[0];
```

**Department Breakdown Cards:**
```typescript
// Bar Chart
const deptData = response.departments.map(dept => ({
  name: dept.department,
  value: dept.count,
  percentage: dept.percentage
}));
```

**Never Events Comparison Card:**
```typescript
// Trend Indicator
const trendIcon = response.change.trend === 'improving' ? '↓' : '↑';
const trendColor = response.change.trend === 'improving' ? 'green' : 'red';

// Progress to Goal
const progressPercent = (response.current.count / response.goal) * 100; // Always > 100% (bad)
```

---

## API Documentation

Full interactive API documentation available at:
```
http://127.0.0.1:8000/docs
```

Navigate to:
- **Red Flags** section → See category-breakdown and department-breakdown
- **Never Events** section → See category-breakdown and timeline-comparison

---

## Priority Testing Order

1. ✅ **Red Flags - Category Breakdown** (highest priority)
2. ✅ **Never Events - Category Breakdown**
3. ✅ **Never Events - Timeline Comparison**
4. ✅ **Red Flags - Department Breakdown**

---

**Last Updated:** 2024-12-26  
**Status:** ✅ All 4 endpoints implemented and ready for testing
