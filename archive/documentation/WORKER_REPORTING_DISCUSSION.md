# 📋 WORKER REPORTING DISCUSSION & PLANNING

## **CURRENT STATE ANALYSIS**

### ✅ **What Already Exists:**

#### **API V1** (Legacy)
1. **Worker Profile** - `GET /api/workers/{employee_id}/profile`
   - ✅ Returns worker identity from `APP_VIEWTABLE_HR_EMPLOYEES`
   - ✅ Returns metrics (incidents, action items, explanations)
   - ✅ Optional date filtering (`date_from`, `date_to`)
   - ⚠️ **BUT**: Doesn't join with `APP_IncidentCaseEmployee` to show actual complaints

#### **API V2**
2. **Search Workers** - `GET /api/v2/workers/search?q={query}&limit={limit}`
   - ✅ Already implemented with HR system integration
   - ✅ Searches `APP_VIEWTABLE_HR_EMPLOYEES`
   - ✅ Returns normalized V2 response format

3. **Get Actions** - `GET /api/v2/workers/{id}/actions?limit={num}&offset={num}`
   - ✅ Already implemented
   - ✅ Returns paginated action items

4. **Get Profile V2** - `GET /api/v2/workers/{id}/profile`
   - ✅ Already implemented
   - ✅ Reuses V1 service layer

#### **Seasonal Reports**
5. **Worker Seasonal Word** - `GET /api/person-reports/worker/{id}/seasonal-word?season_start={date}&season_end={date}`
   - ✅ Already implemented
   - ✅ Returns .docx blob
   - ✅ Has hospital template with icons
   - ⚠️ **BUT**: Needs to be updated to use `APP_IncidentCaseEmployee` linkage

---

## **🔄 WHAT NEEDS TO BE MODIFIED**

### **1. Single Worker Full Report** - `GET /api/workers/{id}/profile`

| Feature | Current State | Required State |
|---------|--------------|----------------|
| **Template** | ❓ Unknown | ✅ Use hospital template with icons + Arabic |
| **Personal Info** | ✅ Shows from HR | ✅ Keep as is |
| **Complaints Table** | ❌ NOT showing | ✅ **JOIN `APP_IncidentCaseEmployee` with `APP_IncidentCase`** |
| **Query Logic** | ❓ Need to verify | ✅ Show complaints FOR or AGAINST employee |

**Required Changes:**
```sql
-- Add this query to show complaints
SELECT 
    inc.IncidentRequestCaseID,
    inc.ComplaintText,
    inc.FeedbackRecievedDate,
    inc.SeverityID,
    inc.CaseStatusID,
    s.SeverityName,
    emp.IsPrimary as IsEmployeePrimary
FROM APP_IncidentCaseEmployee emp
INNER JOIN APP_IncidentCase inc ON emp.IncidentRequestCaseID = inc.IncidentRequestCaseID
LEFT JOIN APP_LOOKUP_SEVERITY s ON inc.SeverityID = s.SeverityID
WHERE emp.EmployeeID = {employee_id}
  AND (date_from IS NULL OR inc.FeedbackRecievedDate >= date_from)
  AND (date_to IS NULL OR inc.FeedbackRecievedDate <= date_to)
ORDER BY inc.FeedbackRecievedDate DESC
```

---

### **2. Worker Seasonal Report** - `GET /api/person-reports/worker/{id}/seasonal-word`

| Feature | Current State | Required State |
|---------|--------------|----------------|
| **Template** | ✅ Has template | ✅ Use hospital template with icons + Arabic |
| **Count Complaints** | ❓ Unknown | ✅ Count workers WITH complaints |
| **High Severity** | ❌ NOT separated | ✅ Specify HIGH severity separately |
| **Zero Complaints** | ❓ Unknown | ✅ Do NOT show if zero complaints |

**Required Changes:**
```sql
-- Count complaints by severity for this employee
SELECT 
    COUNT(*) as TotalComplaints,
    SUM(CASE WHEN inc.SeverityID = 3 THEN 1 ELSE 0 END) as HighSeverityCount,
    SUM(CASE WHEN inc.SeverityID < 3 THEN 1 ELSE 0 END) as OtherSeverityCount
FROM APP_IncidentCaseEmployee emp
INNER JOIN APP_IncidentCase inc ON emp.IncidentRequestCaseID = inc.IncidentRequestCaseID
WHERE emp.EmployeeID = {employee_id}
  AND inc.FeedbackRecievedDate >= season_start
  AND inc.FeedbackRecievedDate <= season_end

-- Only generate report if TotalComplaints > 0
```

---

## **🎯 PROPOSED IMPLEMENTATION PLAN**

### **Phase 1: Data Layer** (Foundation)

**File**: `backend/api/db_layer/worker_reporting_db.py`

**Tasks:**
1. ✅ Add function `get_worker_incidents(employee_id, date_from, date_to)` 
   - Joins `APP_IncidentCaseEmployee` with `APP_IncidentCase`
   - Returns list of complaints with severity, status, date
   
2. ✅ Modify `get_employee_profile(employee_id)` to include incidents
   - Add `incidents` array to existing response
   
3. ✅ Add function `count_seasonal_incidents_by_severity(employee_id, season_start, season_end)`
   - Returns total count and high-severity count
   - Returns None if zero complaints (so we can skip report generation)

---

### **Phase 2: Service Layer** (Business Logic)

**File**: `backend/api/services/worker_reporting_service.py`

**Tasks:**
1. ✅ Update `get_worker_profile()` to include incidents list
   - Call new DB function to get incidents
   - Transform into clean response format
   
2. ✅ Update `build_worker_seasonal_report_data()` 
   - Add severity breakdown (total vs high)
   - Return None if zero complaints
   - Include template metadata (hospital icons, Arabic)

---

### **Phase 3: Router Layer** (API Endpoints)

**File**: `backend/api/routers/worker_reporting_router.py`

**Tasks:**
1. ✅ Update profile endpoint response schema
   - Add `incidents` array to `WorkerProfileResponse`
   
**File**: `backend/api/routers/person_seasonal_report_router.py`

**Tasks:**
1. ✅ Update seasonal word endpoint
   - Check if employee has complaints before generating
   - Return 404 or message if zero complaints
   - Include severity breakdown in report

---

### **Phase 4: Template Updates** (Word Document)

**File**: `backend/api/services/word_generator.py` (or similar)

**Tasks:**
1. ✅ Add complaints table to worker profile template
2. ✅ Add severity breakdown to seasonal report template
3. ✅ Ensure icons and Arabic text are properly formatted

---

### **Phase 5: Testing**

**Test Files to Create:**
1. `test_worker_incidents.py` - Test incident linkage
2. `test_worker_seasonal_updated.py` - Test seasonal report with severity

**Test Cases:**
- ✅ Worker with zero complaints
- ✅ Worker with only low severity complaints
- ✅ Worker with high severity complaints
- ✅ Worker with mix of severities
- ✅ Date range filtering works correctly
- ✅ Primary employee designation is respected

---

## **❓ QUESTIONS FOR DISCUSSION**

### **1. Severity Levels** 
**Question**: What severity ID = "High"? Is it:
- SeverityID = 3 (assuming 1=Low, 2=Medium, 3=High)?
- Or different mapping?

**Action**: Need to check `APP_LOOKUP_SEVERITY` table structure

---

### **2. "For" vs "Against"**
**Question**: You mentioned "complaints FOR or AGAINST this worker". How do we distinguish?
- Is it based on `IsPrimary` flag?
- Or based on relationship type?
- Or are ALL records in `APP_IncidentCaseEmployee` considered?

**Action**: Clarify business logic

---

### **3. Template Location**
**Question**: Where is the hospital template with icons located?
- Is it in `backend/api/services/word_templates/`?
- What's the filename?

**Action**: Verify template path

---

### **4. Zero Complaints Behavior**
**Question**: For seasonal report, if worker has zero complaints:
- Return 404 error?
- Return 200 with message "No complaints for this period"?
- Return empty report?

**Action**: Decide on user experience

---

### **5. Existing Data Migration**
**Question**: We linked employees to incidents starting at incident #490. Should we:
- Generate reports only for incidents >= 490?
- Backfill old incidents with employee linkage?
- Accept that old incidents won't appear in reports?

**Action**: Decide on data completeness strategy

---

## **📊 ESTIMATED EFFORT**

| Phase | Estimated Time | Complexity |
|-------|---------------|------------|
| **Phase 1** - Data Layer | 2-3 hours | Medium |
| **Phase 2** - Service Layer | 1-2 hours | Low |
| **Phase 3** - Router Updates | 1 hour | Low |
| **Phase 4** - Template Updates | 2-3 hours | Medium |
| **Phase 5** - Testing | 2-3 hours | Medium |
| **TOTAL** | **8-12 hours** | **Medium** |

---

## **🚦 DECISION TIME**

### **Ready to Start?**

- [ ] Answer the 5 questions above
- [ ] Confirm severity mapping
- [ ] Confirm "for/against" logic
- [ ] Approve implementation plan
- [ ] Decide on zero-complaints behavior
- [ ] Decide on old incidents handling

### **Once Decided:**

1. I'll implement Phase 1 (Data Layer)
2. Test it works correctly
3. Move to Phase 2 (Service Layer)
4. Test again
5. Continue through each phase with testing

---

## **💬 YOUR FEEDBACK NEEDED**

Please answer:
1. **Severity**: Which SeverityID = "High"?
2. **Logic**: Should we show ALL incidents linked to employee, or filter by `IsPrimary`?
3. **Template**: Where is the hospital template file?
4. **Zero Complaints**: What should the API return?
5. **Old Data**: Should we care about incidents < 490?

Once you answer these, we can start implementation! 🚀
