# PHASE 8: Frontend Integration Requirements

## Overview
This document outlines the data requirements and API modifications needed for the Explanation Workflow frontend integration.

---

## 🎯 Three Types of Explanations

### 1. **Seasonal Reports** (UNCLEAR - Need Clarification)
- **Data Source:** APP_SeasonalReport table (?)
- **Status:** Not yet implemented in explanation workflow
- **Questions:**
  - What triggers a seasonal report to need explanation?
  - Is this a separate system from incident cases?
  - What data structure does it use?

### 2. **Red Flag / Never Event Cases**
- **Data Source:** APP_IncidentCase WHERE ClinicalRiskTypeID IN (2, 3)
- **Status:** ✅ Implemented in backend
- **Current Implementation:**
  - FSM: Open + Waiting → In Progress + Responded → Closed + Responded
  - Always requires explanation (RequiresExplanation flag ignored)
  - Can have action items

### 3. **Ordinary Complaints**
- **Data Source:** APP_IncidentCase WHERE ClinicalRiskTypeID = 1
- **Status:** ✅ Implemented in backend
- **Current Implementation:**
  - Two paths based on RequiresExplanation flag:
    - RequiresExplanation=1: Open + Waiting (needs explanation)
    - RequiresExplanation=0: Closed + No Explanation Needed (auto-closed)
  - Can have action items if explanation submitted

---

## 📋 Missing Data for Frontend

### **CRITICAL: Need to Add**

#### 1. Due Date Calculation
**Current:** Not implemented
**Needed:**
```javascript
{
  "due_date": "2026-01-26",  // Calculated based on case type
  "days_until_due": 7,
  "is_overdue": false,
  "days_overdue": 0
}
```

**Business Rules (NEED CONFIRMATION):**
- Red Flag: 7 days from FeedbackRecievedDate?
- Never Event: 3 days from FeedbackRecievedDate?
- Ordinary: 14 days from FeedbackRecievedDate?

#### 2. Department Information
**Current:** Only IssuingDepartmentID (integer)
**Needed:**
```javascript
{
  "issuing_department_id": 43,
  "issuing_department_name": "Emergency Department",
  "issuing_department_name_ar": "قسم الطوارئ",
  "administration_name": "Medical Services",
  "section_name": "Acute Care"
}
```

#### 3. Visual Priority Indicators
**Current:** Only ClinicalRiskTypeID
**Needed:**
```javascript
{
  "priority_level": "HIGH",  // HIGH/MEDIUM/LOW
  "priority_color": "#FF0000",  // For UI color coding
  "priority_icon": "alert-triangle",
  "risk_type_name": "Red Flag",
  "risk_type_name_ar": "العلم الأحمر"
}
```

#### 4. Case Summary Info
**Current:** Only ComplaintText (can be very long)
**Needed:**
```javascript
{
  "case_reference": "RF-2026-001",  // Formatted reference number
  "case_summary": "Patient fell from bed...",  // First 100 chars
  "patient_name": "Patient Name",  // Anonymized if needed
  "feedback_date": "2026-01-19",
  "created_at": "2026-01-19T10:30:00"
}
```

#### 5. Action Items Progress
**Current:** Available via separate endpoint
**Needed:** Include in main response
```javascript
{
  "action_items_summary": {
    "total": 3,
    "completed": 1,
    "pending": 2,
    "completion_percentage": 33.3,
    "can_close_case": false
  }
}
```

---

## 🔧 Required API Modifications

### **Option A: Enhance Existing Endpoints** (Recommended)

Modify `GET /api/explanations/pending` to return enhanced data:

```python
# ADD to response
{
  "cases": [
    {
      # Existing fields...
      "incident_case_id": 123,
      "clinical_risk_type_id": 2,
      "explanation_status": "Waiting",
      
      # NEW FIELDS:
      "due_date": "2026-01-26",
      "days_until_due": 7,
      "is_overdue": false,
      "priority": {
        "level": "HIGH",
        "color": "#FF0000",
        "label": "Red Flag",
        "label_ar": "العلم الأحمر"
      },
      "department": {
        "id": 43,
        "name": "Emergency Department",
        "name_ar": "قسم الطوارئ",
        "administration": "Medical Services",
        "section": "Acute Care"
      },
      "case_info": {
        "reference": "RF-2026-001",
        "summary": "Patient fell from bed causing...",
        "feedback_date": "2026-01-19",
        "patient_name": "Anonymous"
      },
      "action_items": {
        "total": 3,
        "completed": 1,
        "pending": 2,
        "completion_percentage": 33.3
      }
    }
  ],
  
  # Enhanced statistics
  "statistics": {
    "overdue_count": 5,
    "due_today_count": 3,
    "due_this_week_count": 12,
    "by_priority": {
      "HIGH": 8,
      "MEDIUM": 10,
      "LOW": 4
    }
  }
}
```

### **Option B: Create New Enhanced Endpoints**

```python
# New endpoints:
GET /api/explanations/enhanced-pending  # With all new fields
GET /api/explanations/dashboard-data    # For dashboard widgets
GET /api/explanations/overdue           # Overdue cases only
```

---

## 📝 Frontend Components Needed

### **1. Explanation List Page**
**Purpose:** Display all pending explanations with filtering/sorting

**Required Data:**
- ✅ Case ID & Reference
- ✅ Case Type (Red Flag/Never Event/Ordinary)
- ⚠️ Department Name (missing)
- ⚠️ Due Date (missing)
- ⚠️ Days Overdue (missing)
- ✅ Explanation Status
- ⚠️ Priority Indicator (missing)
- ✅ Brief Summary

**Filters Needed:**
- By Case Type (All/Red Flag/Never Event/Ordinary)
- By Department
- By Status (All/Waiting/Overdue)
- By Date Range

**Actions:**
- Click row → Open explanation form
- Bulk operations (if needed)

### **2. Explanation Form/Modal**
**Purpose:** Submit explanation for a specific case

**Required Data Display:**
- Case details (read-only)
- Department info
- Complaint text (full)
- Current status
- Due date with countdown

**Required Inputs:**
- Explanation text (multiline, min 10 chars)
- Action items (dynamic list)
  - Title
  - Description
  - Due date
- Submit button

**Validation:**
- Explanation text: min 10 characters
- Action items: optional but if added, must have title
- Due dates must be in future

### **3. Insert Page - Add RequiresExplanation Checkbox**
**Purpose:** Allow marking ordinary complaints for explanation during creation

**Modifications:**
```jsx
// Add after ClinicalRiskType selection
{clinicalRiskType === 1 && ( // Only show for Ordinary complaints
  <Checkbox
    label="Requires Detailed Explanation"
    labelAr="يتطلب شرحًا مفصلاً"
    checked={requiresExplanation}
    onChange={setRequiresExplanation}
    helpText="Check if this complaint requires management to provide explanation"
  />
)}
```

---

## 🎨 UI/UX Design Considerations

### **Visual Differentiation**

```javascript
const RISK_TYPE_STYLES = {
  RED_FLAG: {
    badge: 'bg-red-500 text-white',
    border: 'border-l-4 border-red-500',
    icon: '🚩',
    priority: 'HIGH'
  },
  NEVER_EVENT: {
    badge: 'bg-purple-600 text-white',
    border: 'border-l-4 border-purple-600',
    icon: '⚠️',
    priority: 'CRITICAL'
  },
  ORDINARY: {
    badge: 'bg-blue-500 text-white',
    border: 'border-l-4 border-blue-500',
    icon: '📋',
    priority: 'MEDIUM'
  }
}
```

### **Overdue Indicators**

```javascript
const getOverdueStatus = (daysOverdue) => {
  if (daysOverdue === 0) return { color: 'green', label: 'On Time' }
  if (daysOverdue <= 2) return { color: 'yellow', label: 'Due Soon' }
  if (daysOverdue <= 7) return { color: 'orange', label: 'Overdue' }
  return { color: 'red', label: 'Critical Overdue' }
}
```

---

## ❓ Questions for User

### **1. Seasonal Reports** (URGENT - Clarification Needed)
- [ ] What is a seasonal report in your system?
- [ ] Where is seasonal data stored?
- [ ] What triggers a seasonal report to need explanation?
- [ ] Is this the same as the seasonal comparison exports?
- [ ] Should seasonal explanations be in the same UI as case explanations?

### **2. Due Date Policy**
- [ ] Red Flag: How many days to respond?
- [ ] Never Event: How many days to respond?
- [ ] Ordinary: How many days to respond?
- [ ] What happens when overdue? Auto force-close after X days?
- [ ] Should there be escalation notifications?

### **3. Department Information**
- [ ] Which department name to show? (Issuing? Target? Both?)
- [ ] Should administration/section be displayed?
- [ ] Arabic names needed?

### **4. UI Framework**
- [ ] What framework is the frontend using? (React, Vue, Angular, Blazor?)
- [ ] Do you have existing component library?
- [ ] Are there existing table/form components to reuse?

### **5. Permissions**
- [ ] Who can submit explanations? (Anyone? Department managers only?)
- [ ] Who can force-close? (Admins only?)
- [ ] Who can toggle RequiresExplanation flag?

---

## 📦 Deliverables Needed from You

### **For me to implement Phase 8, please provide:**

1. **Seasonal Reports Clarification**
   - Explain what seasonal reports are
   - Show me the table structure
   - How they relate to explanations

2. **Business Rules Document**
   - Due date policies
   - Escalation rules
   - Force-close policies

3. **Current Frontend Code**
   - Insert page component (if exists)
   - Any existing explanation page
   - API service files

4. **UI Mockup or Requirements**
   - Screenshot of desired layout
   - List of all fields to display
   - Filter/sort requirements

5. **Department Structure**
   - Confirm which department info to display
   - Arabic translations needed?

---

## 🚀 Proposed Implementation Order

1. **Backend: Enhance API endpoints** (1-2 hours)
   - Add due date calculation
   - Add department name JOIN
   - Add priority indicators
   - Add action items summary

2. **Frontend: Explanation List Page** (2-3 hours)
   - Display pending explanations
   - Filters and sorting
   - Visual indicators

3. **Frontend: Explanation Form** (2-3 hours)
   - Modal or page
   - Submit explanation
   - Add action items

4. **Frontend: Insert Page Modification** (30 mins)
   - Add RequiresExplanation checkbox
   - Conditional display logic

5. **Testing & Polish** (1-2 hours)
   - End-to-end testing
   - UI polish
   - Arabic support

---

## 📞 Next Steps

**Please respond with:**
1. Answers to the questions above
2. Clarification on seasonal reports
3. Current frontend code files
4. UI mockup or detailed requirements

Once I have this information, I can:
- ✅ Enhance the API endpoints with missing data
- ✅ Create frontend components
- ✅ Implement the complete explanation workflow UI
