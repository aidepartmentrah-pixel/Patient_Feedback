# Quick Reference: Seasonal Reporting Periods

## ✅ What Works Now

### Frontend Sends:
```javascript
// Option 1: Quarters (Recommended)
{ "year": 2025, "trimester": "Q1" }  // Jan-Mar
{ "year": 2025, "trimester": "Q2" }  // Apr-Jun
{ "year": 2025, "trimester": "Q3" }  // Jul-Sep
{ "year": 2025, "trimester": "Q4" }  // Oct-Dec

// Option 2: Trimesters (Legacy)
{ "year": 2025, "trimester": "Trim1" }  // Jan-Apr
{ "year": 2025, "trimester": "Trim2" }  // May-Aug
{ "year": 2025, "trimester": "Trim3" }  // Sep-Dec
```

### Backend Accepts:
- ✅ `Q1`, `Q2`, `Q3`, `Q4` (3-month quarters)
- ✅ `Trim1`, `Trim2`, `Trim3` (4-month trimesters)
- ❌ Anything else → 400 Bad Request

### Database Contains:
```
Q1-2025: Jan 01 - Mar 31 (Season ID: 1)
Q2-2025: Apr 01 - Jun 30 (Season ID: 2)
Q3-2025: Jul 01 - Sep 30 (Season ID: 3)
Q4-2025: Oct 01 - Dec 31 (Season ID: 4)
```

## 📝 API Examples

### Generate Seasonal Report
```http
POST /api/reports/seasonal/view
Content-Type: application/json

{
  "year": 2025,
  "trimester": "Q1",
  "orgunit_id": 12,
  "orgunit_type": 1,
  "user_id": 1
}
```

### Export Seasonal Report
```http
POST /api/reports/export?format=pdf
Content-Type: application/json

{
  "report_type": "seasonal",
  "display_mode": "detailed",
  "year": 2025,
  "period": "Q2",
  "filters": {},
  "language": "en"
}
```

## 🎯 Recommendations

1. **Use Q1-Q4** - Matches database exactly, no ambiguity
2. **Avoid Trim1** - May find multiple seasons (overlaps Q1 and Q2)
3. **Frontend works as-is** - No changes needed if already sending Q1-Q4
4. **Parameter name** - Still called "trimester" for backward compatibility

## ⚠️ Error Handling

### Invalid Format:
```javascript
{ "year": 2025, "trimester": "Q5" }
→ 400 Bad Request
→ "Invalid period: Q5. Must be one of ['Q1', 'Q2', 'Q3', 'Q4', 'Trim1', 'Trim2', 'Trim3']"
```

### Season Not Found:
```javascript
{ "year": 2024, "trimester": "Q1" }
→ 404 Not Found
→ "Season not found for year=2024, period=Q1"
```

### Ambiguous Season:
```javascript
{ "year": 2025, "trimester": "Trim1" }
→ 409 Conflict (if multiple matches)
→ "Ambiguous season: Multiple seasons found for year=2025, trimester=Trim1"
```
