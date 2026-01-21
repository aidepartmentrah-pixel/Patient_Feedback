# Three-Type Explanation System - Complete Documentation

## 🚨 Problem Summary

The original explanation system treated all cases identically, but they actually require **three completely different database operations**:

| Explanation Type | Database Operation | Target Table | FSM |
|-----------------|-------------------|--------------|-----|
| **Red Flag / Never Event** | Create NEW record | `APP_IncidentCaseFeedback` | S0 → S1 |
| **Ordinary** | Update existing field | `APP_IncidentCase.TakenAction` | S0 → S1 |
| **Seasonal** | Update report field | `APP_SeasonalOrgUnitReport.ExplanationText` | N/A |

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (React)                          │
│  Three Different Forms Based on explanation_type             │
└─────────────┬───────────────┬────────────────┬──────────────┘
              │               │                │
              │               │                │
    ┌─────────▼──────┐  ┌────▼─────┐  ┌───────▼────────┐
    │ Red Flag Form  │  │ Ordinary │  │ Seasonal Form  │
    │ (Complex)      │  │  Form    │  │ (Report-level) │
    │ - Root Causes  │  │ (Simple) │  │ - Explanation  │
    │ - Preventive   │  │ - Text   │  │   Text         │
    └────────┬───────┘  └─────┬────┘  └────────┬───────┘
             │                │                 │
    ┌────────▼────────────────▼─────────────────▼───────────┐
    │         BACKEND API - Separate Endpoints               │
    │  /red-flag/{id}  /ordinary/{id}  /seasonal/{id}       │
    └─────────┬────────────────┬───────────────┬────────────┘
              │                │               │
    ┌─────────▼──────┐  ┌──────▼─────┐  ┌─────▼──────────┐
    │ Feedback DB    │  │ Case DB    │  │ Seasonal DB    │
    │ Layer          │  │ Layer      │  │ Layer          │
    └────────┬───────┘  └──────┬─────┘  └─────┬──────────┘
             │                 │                │
    ┌────────▼─────────────────▼────────────────▼──────────┐
    │                  SQL Server Database                  │
    │  APP_IncidentCaseFeedback | APP_IncidentCase |       │
    │  APP_SeasonalOrgUnitReport                            │
    └───────────────────────────────────────────────────────┘
```

---

## 🔧 Backend Structure

### New Files Created

```
backend/
├── api/
│   ├── db_layer/
│   │   ├── explanation_red_flag_db.py       ✅ NEW - Red Flag/Never Event
│   │   ├── explanation_ordinary_db.py       ✅ NEW - Ordinary cases
│   │   └── explanation_seasonal_db.py       ✅ NEW - Seasonal reports
│   ├── services/
│   │   └── explanation_service_refactored.py ✅ NEW - Unified service layer
│   └── routers/
│       └── explanation_routes_refactored.py  ✅ NEW - Separate endpoints
└── main.py                                   ✅ UPDATED - Uses refactored routes
```

### Database Schema Reference

#### 1. APP_IncidentCaseFeedback (Red Flag/Never Event)
```sql
CREATE TABLE APP_IncidentCaseFeedback (
    IncidentRequestCaseID INT PRIMARY KEY,
    
    -- Root Cause Analysis: Staff
    Cause_Staff_Training BIT,
    Cause_Staff_Incentives BIT,
    Cause_Staff_Competency BIT,
    Cause_Staff_Understaffed BIT,
    Cause_Staff_NonCompliance BIT,
    Cause_Staff_NoCoordination BIT,
    Cause_Staff_Other BIT,
    Cause_Staff_OtherText NVARCHAR(MAX),
    
    -- Root Cause Analysis: Process
    Cause_Process_NotComprehensive BIT,
    Cause_Process_Unclear BIT,
    Cause_Process_MissingProtocol BIT,
    Cause_Process_Other BIT,
    Cause_Process_OtherText NVARCHAR(MAX),
    
    -- Root Cause Analysis: Equipment
    Cause_Equipment_NotAvailable BIT,
    Cause_Equipment_SystemIncomplete BIT,
    Cause_Equipment_HardToApply BIT,
    Cause_Equipment_Other BIT,
    Cause_Equipment_OtherText NVARCHAR(MAX),
    
    -- Root Cause Analysis: Environment
    Cause_Environment_PlaceNature BIT,
    Cause_Environment_Surroundings BIT,
    Cause_Environment_WorkConditions BIT,
    Cause_Environment_Other BIT,
    Cause_Environment_OtherText NVARCHAR(MAX),
    
    -- Preventive Actions
    Preventive_MonthlyMeetings BIT,
    Preventive_TrainingPrograms BIT,
    Preventive_IncreaseStaff BIT,
    Preventive_MMCommitteeActions BIT,
    Preventive_Other BIT,
    Preventive_OtherText NVARCHAR(MAX),
    
    -- Explanation
    DepartmentExplanationText NVARCHAR(MAX),
    DepartmentExplanationStatusID INT,
    DepartmentExplanationReceivalDate DATE,
    
    CreatedAt DATETIME DEFAULT GETDATE(),
    CreatedByUserID INT
);
```

#### 2. APP_IncidentCase (Ordinary)
```sql
-- Only updates this field:
TakenAction NVARCHAR(MAX)
-- Appends explanation with timestamp
```

#### 3. APP_SeasonalOrgUnitReport (Seasonal)
```sql
-- Updates these fields:
ExplanationText NVARCHAR(MAX)
ExplanationStatusID INT
ExplanationSubmittedAt DATETIME
```

---

## 📡 API Endpoints

### 1. Red Flag / Never Event Explanations

#### POST `/api/explanations/red-flag/{case_id}`
Creates new comprehensive feedback record.

**Request:**
```json
{
  "explanation_text": "Detailed explanation of what happened...",
  "causes_staff": {
    "training": true,
    "competency": false,
    "understaffed": true,
    "other": false,
    "other_text": null
  },
  "causes_process": {
    "not_comprehensive": false,
    "unclear": true,
    "missing_protocol": false,
    "other": false
  },
  "causes_equipment": {
    "not_available": true,
    "system_incomplete": false,
    "hard_to_apply": false
  },
  "causes_environment": {
    "place_nature": false,
    "surroundings": false,
    "work_conditions": true
  },
  "preventive_actions": {
    "monthly_meetings": true,
    "training_programs": true,
    "increase_staff": false,
    "mm_committee_actions": true,
    "other": false
  },
  "user_id": 123
}
```

**Response:**
```json
{
  "success": true,
  "message": "Red Flag/Never Event feedback submitted successfully",
  "feedback_created": true,
  "fsm_transition": "S0 → S1 (Open + Waiting → In Progress + Responded)"
}
```

#### GET `/api/explanations/red-flag/{case_id}`
Retrieves existing feedback.

---

### 2. Ordinary Case Explanations

#### POST `/api/explanations/ordinary/{case_id}`
Updates TakenAction field with explanation.

**Request:**
```json
{
  "explanation_text": "Brief explanation of actions taken",
  "user_id": 123
}
```

**Response:**
```json
{
  "success": true,
  "message": "Ordinary case explanation submitted successfully",
  "updated_field": "TakenAction",
  "fsm_transition": "S0 → S1 (Open + Waiting → In Progress + Responded)"
}
```

#### GET `/api/explanations/ordinary/{case_id}`
Retrieves TakenAction content.

---

### 3. Seasonal Report Explanations

#### POST `/api/explanations/seasonal/{report_id}`
Updates ExplanationText field in seasonal report.

**Request:**
```json
{
  "explanation_text": "Explanation for seasonal report performance...",
  "user_id": 123
}
```

**Response:**
```json
{
  "success": true,
  "message": "Seasonal report explanation submitted successfully",
  "seasonal_report_id": 456,
  "org_unit_id": 10,
  "season_id": 5,
  "updated_field": "ExplanationText"
}
```

#### GET `/api/explanations/seasonal/{report_id}`
Retrieves seasonal report explanation.

---

### 4. Unified Listing Endpoints

#### GET `/api/explanations/pending/cases`
Returns all cases (Red Flag, Never Event, Ordinary) needing explanation.

**Query Parameters:**
- `dept_id` (optional): Filter by department
- `start_date` (optional): YYYY-MM-DD
- `end_date` (optional): YYYY-MM-DD
- `include_red_flags_only` (optional): Boolean

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "incident_request_case_id": 123,
      "clinical_risk_type_id": 2,
      "explanation_type": "red_flag",
      "explanation_endpoint": "/api/explanations/red-flag/123",
      "complaint_text": "...",
      "patient_name": "...",
      "feedback_received_date": "2026-01-15"
    },
    {
      "incident_request_case_id": 124,
      "clinical_risk_type_id": 1,
      "explanation_type": "ordinary",
      "explanation_endpoint": "/api/explanations/ordinary/124",
      "requires_explanation": true
    }
  ],
  "statistics": {
    "total_count": 10,
    "red_flag_count": 3,
    "ordinary_count": 7
  }
}
```

#### GET `/api/explanations/pending/seasonal`
Returns all seasonal reports needing explanation.

**Query Parameters:**
- `org_unit_id` (optional): Filter by organization unit
- `season_id` (optional): Filter by season
- `non_compliant_only` (optional): Boolean

---

## 🎨 Frontend Implementation Guide

### Step 1: Fetch Pending Cases

```javascript
// Fetch all cases needing explanation
const response = await axios.get('/api/explanations/pending/cases');
const cases = response.data.data;

// Each case has explanation_type field
cases.forEach(case => {
  switch (case.explanation_type) {
    case 'red_flag':
    case 'never_event':
      // Show complex form with root cause analysis
      renderRedFlagForm(case);
      break;
    
    case 'ordinary':
      // Show simple text input form
      renderOrdinaryForm(case);
      break;
  }
});
```

### Step 2: Red Flag Form Component (React Example)

```jsx
function RedFlagExplanationForm({ caseId }) {
  const [formData, setFormData] = useState({
    explanation_text: '',
    causes_staff: {
      training: false,
      competency: false,
      understaffed: false,
      non_compliance: false,
      no_coordination: false,
      other: false,
      other_text: ''
    },
    causes_process: {
      not_comprehensive: false,
      unclear: false,
      missing_protocol: false,
      other: false,
      other_text: ''
    },
    causes_equipment: {
      not_available: false,
      system_incomplete: false,
      hard_to_apply: false,
      other: false,
      other_text: ''
    },
    causes_environment: {
      place_nature: false,
      surroundings: false,
      work_conditions: false,
      other: false,
      other_text: ''
    },
    preventive_actions: {
      monthly_meetings: false,
      training_programs: false,
      increase_staff: false,
      mm_committee_actions: false,
      other: false,
      other_text: ''
    },
    user_id: getCurrentUserId()
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      const response = await axios.post(
        `/api/explanations/red-flag/${caseId}`,
        formData
      );
      
      if (response.data.success) {
        alert('Red Flag feedback submitted successfully!');
        // Refresh list
      }
    } catch (error) {
      console.error('Submission failed:', error.response.data);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <h2>Red Flag / Never Event Explanation</h2>
      
      {/* Explanation Text */}
      <div>
        <label>Detailed Explanation (min 50 chars)</label>
        <textarea
          value={formData.explanation_text}
          onChange={e => setFormData({
            ...formData,
            explanation_text: e.target.value
          })}
          minLength={50}
          required
        />
      </div>

      {/* Staff Causes */}
      <fieldset>
        <legend>Staff-Related Causes</legend>
        <label>
          <input
            type="checkbox"
            checked={formData.causes_staff.training}
            onChange={e => setFormData({
              ...formData,
              causes_staff: {
                ...formData.causes_staff,
                training: e.target.checked
              }
            })}
          />
          Lack of Training
        </label>
        
        <label>
          <input
            type="checkbox"
            checked={formData.causes_staff.competency}
            onChange={e => setFormData({
              ...formData,
              causes_staff: {
                ...formData.causes_staff,
                competency: e.target.checked
              }
            })}
          />
          Competency Issues
        </label>
        
        {/* ... more checkboxes ... */}
      </fieldset>

      {/* Process Causes */}
      <fieldset>
        <legend>Process-Related Causes</legend>
        {/* ... checkboxes ... */}
      </fieldset>

      {/* Equipment Causes */}
      <fieldset>
        <legend>Equipment-Related Causes</legend>
        {/* ... checkboxes ... */}
      </fieldset>

      {/* Environment Causes */}
      <fieldset>
        <legend>Environment-Related Causes</legend>
        {/* ... checkboxes ... */}
      </fieldset>

      {/* Preventive Actions */}
      <fieldset>
        <legend>Preventive Actions</legend>
        {/* ... checkboxes ... */}
      </fieldset>

      <button type="submit">Submit Red Flag Feedback</button>
    </form>
  );
}
```

### Step 3: Ordinary Form Component (React Example)

```jsx
function OrdinaryExplanationForm({ caseId }) {
  const [explanationText, setExplanationText] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      const response = await axios.post(
        `/api/explanations/ordinary/${caseId}`,
        {
          explanation_text: explanationText,
          user_id: getCurrentUserId()
        }
      );
      
      if (response.data.success) {
        alert('Explanation submitted successfully!');
        // Refresh list
      }
    } catch (error) {
      console.error('Submission failed:', error.response.data);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <h2>Ordinary Case Explanation</h2>
      
      <div>
        <label>Explanation (min 20 chars)</label>
        <textarea
          value={explanationText}
          onChange={e => setExplanationText(e.target.value)}
          minLength={20}
          required
        />
      </div>

      <button type="submit">Submit Explanation</button>
    </form>
  );
}
```

### Step 4: Seasonal Form Component (React Example)

```jsx
function SeasonalExplanationForm({ reportId }) {
  const [explanationText, setExplanationText] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      const response = await axios.post(
        `/api/explanations/seasonal/${reportId}`,
        {
          explanation_text: explanationText,
          user_id: getCurrentUserId()
        }
      );
      
      if (response.data.success) {
        alert('Seasonal explanation submitted successfully!');
        // Refresh list
      }
    } catch (error) {
      console.error('Submission failed:', error.response.data);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <h2>Seasonal Report Explanation</h2>
      
      <div>
        <label>Report Explanation (min 50 chars)</label>
        <textarea
          value={explanationText}
          onChange={e => setExplanationText(e.target.value)}
          minLength={50}
          required
        />
      </div>

      <button type="submit">Submit Explanation</button>
    </form>
  );
}
```

---

## 🔄 FSM State Transitions

### Case Explanations (Red Flag, Never Event, Ordinary)

```
S0: Open + Waiting
     ↓ (submit_explanation)
S1: In Progress + Responded
     ↓ (complete_action_items)
S3: Closed + Responded

Alternative:
S0: Open + Waiting
     ↓ (force_close)
S2: Closed + Forcibly Closed
```

### Seasonal Reports
- No FSM - just updates ExplanationStatusID
- 1 (Waiting) → 2 (Responded)

---

## ✅ Validation Rules

### Red Flag / Never Event
- ✅ Must be ClinicalRiskTypeID = 2 or 3
- ✅ Must be in S0 state (Open + Waiting)
- ✅ Explanation text min 50 characters
- ✅ Cannot submit if feedback already exists
- ✅ FSM transition: S0 → S1

### Ordinary
- ✅ Must be ClinicalRiskTypeID = 1
- ✅ Must have RequiresExplanation = 1
- ✅ Must be in S0 state (Open + Waiting)
- ✅ Explanation text min 20 characters
- ✅ Appends to TakenAction (doesn't overwrite)
- ✅ FSM transition: S0 → S1

### Seasonal
- ✅ Must be valid SeasonalReportID
- ✅ Explanation text min 50 characters
- ✅ Updates ExplanationText (overwrites previous)
- ✅ No FSM validation

---

## 🧪 Testing

### Test Red Flag Endpoint
```bash
curl -X POST http://localhost:8000/api/explanations/red-flag/123 \
  -H "Content-Type: application/json" \
  -d '{
    "explanation_text": "Comprehensive explanation of the red flag incident with detailed analysis...",
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
    "user_id": 1
  }'
```

### Test Ordinary Endpoint
```bash
curl -X POST http://localhost:8000/api/explanations/ordinary/124 \
  -H "Content-Type: application/json" \
  -d '{
    "explanation_text": "Simple explanation of actions taken for this ordinary case",
    "user_id": 1
  }'
```

### Test Seasonal Endpoint
```bash
curl -X POST http://localhost:8000/api/explanations/seasonal/456 \
  -H "Content-Type: application/json" \
  -d '{
    "explanation_text": "Explanation for seasonal report performance and policy violations",
    "user_id": 1
  }'
```

---

## 🚀 Deployment Steps

1. **Stop the backend server** (Ctrl+C in terminal)

2. **The new routes are already active** in `main.py`

3. **Restart the backend:**
   ```bash
   cd backend
   uvicorn main:app --reload --port 8000
   ```

4. **Test the API:**
   - Visit http://localhost:8000/docs
   - You'll see new endpoints:
     - `/api/explanations/red-flag/{case_id}`
     - `/api/explanations/ordinary/{case_id}`
     - `/api/explanations/seasonal/{report_id}`
     - `/api/explanations/pending/cases`
     - `/api/explanations/pending/seasonal`

5. **Update Frontend:**
   - Fetch pending cases from `/api/explanations/pending/cases`
   - Check `explanation_type` field
   - Render appropriate form based on type
   - Submit to correct endpoint

---

## 📝 Summary

✅ **Created 6 new files:**
- `explanation_red_flag_db.py` - Red Flag/Never Event database layer
- `explanation_ordinary_db.py` - Ordinary case database layer
- `explanation_seasonal_db.py` - Seasonal report database layer
- `explanation_service_refactored.py` - Unified service layer
- `explanation_routes_refactored.py` - Separate API endpoints
- Updated `main.py` to use refactored routes

✅ **Three separate workflows:**
- Red Flag/Never Event → Complex form → Create feedback record
- Ordinary → Simple form → Update TakenAction field
- Seasonal → Report form → Update ExplanationText field

✅ **Unified dashboard:**
- Single endpoint returns all pending cases
- `explanation_type` field indicates which form to show
- `explanation_endpoint` field provides submission URL

Your crisis is solved! 🎉
