# Quick Reference: Three-Type Explanation System

## 🎯 When to Use Which Endpoint

```
┌─────────────────────────────────────────────────────────────────┐
│  Is it a Red Flag (ID=2) or Never Event (ID=3)?                 │
│  ├─ YES → POST /api/explanations/red-flag/{case_id}             │
│  │         Complex form with root cause analysis                │
│  │         Creates new record in APP_IncidentCaseFeedback       │
│  │                                                               │
│  └─ NO → Is it Ordinary (ID=1) with RequiresExplanation=1?     │
│           ├─ YES → POST /api/explanations/ordinary/{case_id}    │
│           │         Simple text form                            │
│           │         Updates TakenAction field                   │
│           │                                                      │
│           └─ NO → Is it a Seasonal Report?                      │
│                    └─ YES → POST /api/explanations/seasonal/    │
│                              {report_id}                        │
│                              Report explanation form            │
│                              Updates ExplanationText            │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Request Body Examples

### Red Flag / Never Event
```json
POST /api/explanations/red-flag/123
{
  "explanation_text": "Detailed explanation (min 50 chars)...",
  "causes_staff": {
    "training": true,
    "competency": false,
    "understaffed": true,
    "non_compliance": false,
    "no_coordination": false,
    "other": false,
    "other_text": null
  },
  "causes_process": {
    "not_comprehensive": false,
    "unclear": true,
    "missing_protocol": false,
    "other": false,
    "other_text": null
  },
  "causes_equipment": {
    "not_available": true,
    "system_incomplete": false,
    "hard_to_apply": false,
    "other": false,
    "other_text": null
  },
  "causes_environment": {
    "place_nature": false,
    "surroundings": false,
    "work_conditions": true,
    "other": false,
    "other_text": null
  },
  "preventive_actions": {
    "monthly_meetings": true,
    "training_programs": true,
    "increase_staff": false,
    "mm_committee_actions": true,
    "other": false,
    "other_text": null
  },
  "user_id": 123
}
```

### Ordinary
```json
POST /api/explanations/ordinary/124
{
  "explanation_text": "Simple explanation (min 20 chars)...",
  "user_id": 123
}
```

### Seasonal
```json
POST /api/explanations/seasonal/456
{
  "explanation_text": "Report explanation (min 50 chars)...",
  "user_id": 123
}
```

## 🔍 Get Pending Items

### Cases
```javascript
GET /api/explanations/pending/cases?dept_id=10&include_red_flags_only=false

Response:
{
  "success": true,
  "data": [
    {
      "incident_request_case_id": 123,
      "clinical_risk_type_id": 2,
      "explanation_type": "red_flag",           // ← Use this!
      "explanation_endpoint": "/api/explanations/red-flag/123",  // ← Or this!
      "complaint_text": "...",
      "patient_name": "...",
      ...
    }
  ],
  "statistics": { "total_count": 10, "red_flag_count": 3 }
}
```

### Seasonal Reports
```javascript
GET /api/explanations/pending/seasonal?org_unit_id=10&non_compliant_only=true

Response:
{
  "success": true,
  "data": [
    {
      "seasonal_report_id": 456,
      "season_name": "Q1 2026",
      "org_unit_id": 10,
      "is_compliant": false,
      "violated_rules": "Rule A, Rule B",
      ...
    }
  ],
  "statistics": { "total_count": 5, "non_compliant_count": 2 }
}
```

## 💾 Database Impact

| Endpoint | Table | Operation | Fields |
|----------|-------|-----------|--------|
| `/red-flag/{id}` | `APP_IncidentCaseFeedback` | **INSERT** new record | 30+ fields (causes + preventive actions) |
| `/ordinary/{id}` | `APP_IncidentCase` | **UPDATE** existing | `TakenAction` (append) |
| `/seasonal/{id}` | `APP_SeasonalOrgUnitReport` | **UPDATE** existing | `ExplanationText`, `ExplanationStatusID`, `ExplanationSubmittedAt` |

## ⚡ Quick Start

### Backend
```bash
cd backend
uvicorn main:app --reload --port 8000
# Visit http://localhost:8000/docs
```

### Frontend (React/JS)
```javascript
// 1. Fetch pending cases
const { data } = await axios.get('/api/explanations/pending/cases');

// 2. Check type and render form
data.data.forEach(case => {
  if (case.explanation_type === 'red_flag' || case.explanation_type === 'never_event') {
    renderComplexForm(case);
  } else if (case.explanation_type === 'ordinary') {
    renderSimpleForm(case);
  }
});

// 3. Submit to correct endpoint
await axios.post(case.explanation_endpoint, requestBody);
```

## 🚨 Common Errors

### ❌ "INVALID_CASE_TYPE"
- You used `/red-flag/` endpoint for an Ordinary case
- **Fix:** Use `/ordinary/` endpoint instead

### ❌ "EXPLANATION_NOT_REQUIRED"
- Ordinary case has `RequiresExplanation = 0`
- **Fix:** This case doesn't need explanation

### ❌ "INVALID_STATE"
- Case is not in S0 (Open + Waiting) state
- **Fix:** Case already has explanation or is closed

### ❌ "FEEDBACK_EXISTS"
- Red Flag case already has feedback record
- **Fix:** Cannot submit twice (use GET endpoint to view existing)

## 📚 Documentation Files

- **CRISIS_SOLVED.md** ← This file (quick start)
- **THREE_TYPE_EXPLANATION_SYSTEM.md** ← Full documentation
- **test_three_type_explanation_system.py** ← Verification script

## 🎉 You're All Set!

Your three-type explanation system is ready. Each type now has:
- ✅ Separate endpoint
- ✅ Appropriate database operation
- ✅ Type-specific validation
- ✅ Clear error messages
- ✅ FSM state management

**Questions?** Check the full docs in THREE_TYPE_EXPLANATION_SYSTEM.md
