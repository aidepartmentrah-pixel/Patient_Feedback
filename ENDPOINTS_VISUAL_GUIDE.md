# 🎯 Your 4 Endpoints - Where to Use Them

## ✅ All Endpoints Are Live and Working!

---

## 📌 **1. LEAF NODES** (Smallest Units)
### **216 leaf units available**

**Endpoint**: 
```
GET http://localhost:8000/api/org-units/leaves
```

**Use in UI**:
- ✅ Add Patient Form → "Issuing Department" dropdown
- ✅ Add Incident Form → "Target Department" dropdown
- ✅ Complaint Form → "Department" dropdown
- ✅ Any form where user reports about a **specific operational unit**

**Why**: People complain about what they **actually experience** at the operational level.

**Example Response**:
```json
{
  "leaves": [
    {
      "id": 14,
      "name": "دائرة الهندسة",
      "name_ar": "دائرة الهندسة",
      "parent_id": 10,
      "parent_name": "الإدارة الإدارية",
      "type": 324,
      "type_name": "SECTION"
    }
  ],
  "count": 216
}
```

---

## 📌 **2. ALL ADMINISTRATIONS** (Top Level)
### **9 administrations available**

**Endpoint**: 
```
GET http://localhost:8000/api/org-units/administrations
```

**Use in UI**:
- ✅ Report Configuration → "Report Scope" dropdown
- ✅ Monthly Report → "Organization Level" dropdown
- ✅ Seasonal Report → "Administration" dropdown
- ✅ Dashboard → "Administration Filter" (high-level)
- ✅ Any place you need **"All Administrations"** option

**Why**: For **aggregate analysis** and comparison across major divisions.

**Example Response**:
```json
{
  "administrations": [
    {
      "id": 1,
      "name": "الادارة العامة",
      "name_ar": "الادارة العامة"
    },
    {
      "id": 2,
      "name": "الادارة المالية",
      "name_ar": "الادارة المالية"
    }
  ],
  "count": 9
}
```

---

## 📌 **3. SELECT ADMINISTRATIONS** (Choose One or More)
### **Same as #2** ← Use same endpoint!

**Endpoint**: 
```
GET http://localhost:8000/api/org-units/administrations
```

**Use in UI**:
- ✅ User Management → "Assign to Administration" dropdown
- ✅ Settings → "Default Administration" dropdown
- ✅ Filter Panel → "Administration" filter
- ✅ Investigation Page → "Administration" selector (when not using cascade)
- ✅ Any place you need to **select one or more administrations**

**Why**: Same list of administrations, just different UI usage (single-select vs multi-select).

---

## 📌 **4. SELECT DEPARTMENTS** (Only Departments)
### **134 departments available**

**Endpoint**: 
```
GET http://localhost:8000/api/org-units/departments
```

**Use in UI**:
- ✅ User Assignment → "Assign to Department" dropdown
- ✅ Department Report → "Select Department" dropdown
- ✅ Filter Panel → "Department" filter (multi-select)
- ✅ Performance Dashboard → "Compare Departments" selector
- ✅ Any place you need to **select departments specifically**

**Why**: When you need **department-level** selection, not sections or administrations.

**Example Response**:
```json
{
  "departments": [
    {
      "id": 5,
      "name": "دائرة المواد",
      "name_ar": "دائرة المواد",
      "administration_id": 1
    },
    {
      "id": 10,
      "name": "Emergency Department",
      "name_ar": "قسم الطوارئ",
      "administration_id": 4
    }
  ],
  "count": 134
}
```

---

## 🎨 **Visual UI Map**

```
┌─────────────────────────────────────────────────────────────────┐
│                      YOUR APPLICATION UI                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┐  ┌─────────────────────────────┐
│     ADD PATIENT FORM            │  │    REPORT CONFIGURATION     │
│                                 │  │                             │
│  [Patient Name]                 │  │  Report Scope:              │
│  [Phone Number]                 │  │  ┌───────────────────────┐  │
│                                 │  │  │ ⭕ All Administrations │  │
│  Issuing Department:            │  │  │ ⚪ Medical Admin       │  │
│  ┌───────────────────────────┐  │  │  │ ⚪ Surgical Admin      │  │
│  │ Emergency Section       ▼ │  │  │  └───────────────────────┘  │
│  │ ICU Section               │  │  │                             │
│  │ Pharmacy Department       │  │  │  [Start Date] [End Date]    │
│  │ Lab Section               │  │  │                             │
│  │ ...216 more leaf units    │  │  │  [Generate Report]          │
│  └───────────────────────────┘  │  │                             │
│                                 │  │  Uses:                      │
│  [Submit]                       │  │  /api/org-units/            │
│                                 │  │  administrations            │
│  Uses:                          │  │                             │
│  /api/org-units/leaves          │  └─────────────────────────────┘
│                                 │
└─────────────────────────────────┘

┌─────────────────────────────────┐  ┌─────────────────────────────┐
│     USER MANAGEMENT             │  │    DEPARTMENT FILTER        │
│                                 │  │                             │
│  Assign User To:                │  │  Filter by Department:      │
│                                 │  │                             │
│  ⚪ Administration               │  │  ☑ Emergency Dept           │
│     ┌───────────────────────┐   │  │  ☐ Surgery Dept             │
│     │ Medical Admin       ▼ │   │  │  ☑ Pharmacy Dept            │
│     │ Surgical Admin        │   │  │  ☐ Lab Dept                 │
│     │ Finance Admin         │   │  │  ☐ Imaging Dept             │
│     │ ...9 administrations  │   │  │  ...134 departments         │
│     └───────────────────────┘   │  │                             │
│                                 │  │  [Apply Filter]             │
│  ⚪ Department                   │  │                             │
│     ┌───────────────────────┐   │  │  Uses:                      │
│     │ Emergency Dept      ▼ │   │  │  /api/org-units/            │
│     │ Surgery Dept          │   │  │  departments                │
│     │ Pharmacy Dept         │   │  │                             │
│     │ ...134 departments    │   │  └─────────────────────────────┘
│     └───────────────────────┘   │
│                                 │
│  [Assign]                       │
│                                 │
│  Uses:                          │
│  /api/org-units/administrations │
│  /api/org-units/departments     │
│                                 │
└─────────────────────────────────┘
```

---

## 🔥 **Quick Copy-Paste Examples**

### **Example 1: Load Leaf Nodes (INSERT Form)**
```javascript
// Use this in: AddPatient.vue, AddIncident.vue, ComplaintForm.vue
const response = await fetch('http://localhost:8000/api/org-units/leaves');
const data = await response.json();

// Populate dropdown
issuingDeptOptions.value = data.leaves.map(leaf => ({
  value: leaf.id,
  label: leaf.name
}));
```

### **Example 2: Load Administrations (REPORT Form)**
```javascript
// Use this in: ReportConfig.vue, MonthlyReport.vue, SeasonalReport.vue
const response = await fetch('http://localhost:8000/api/org-units/administrations');
const data = await response.json();

// Add "All" option
reportScopeOptions.value = [
  { value: 'all', label: 'All Administrations' },
  ...data.administrations.map(admin => ({
    value: admin.id,
    label: admin.name
  }))
];
```

### **Example 3: Load Administrations (USER Assignment)**
```javascript
// Use this in: UserManagement.vue, Settings.vue
const response = await fetch('http://localhost:8000/api/org-units/administrations');
const data = await response.json();

// Single-select dropdown
adminOptions.value = data.administrations.map(admin => ({
  value: admin.id,
  text: admin.name
}));
```

### **Example 4: Load Departments (FILTER Panel)**
```javascript
// Use this in: FilterPanel.vue, DepartmentReport.vue
const response = await fetch('http://localhost:8000/api/org-units/departments');
const data = await response.json();

// Multi-select checkboxes
departmentFilters.value = data.departments.map(dept => ({
  id: dept.id,
  name: dept.name,
  checked: false
}));
```

---

## 📊 **Summary Table**

| Your Need | Endpoint | Count | Example Use |
|-----------|----------|-------|-------------|
| **1. Section/Leaf** | `/api/org-units/leaves` | 216 | Add Patient Form |
| **2. All Administrations** | `/api/org-units/administrations` | 9 | Report Config |
| **3. Select Administrations** | `/api/org-units/administrations` | 9 | User Assignment |
| **4. Select Departments** | `/api/org-units/departments` | 134 | Department Filter |

---

## ✅ **Test Right Now**

Open these URLs in your browser:

1. **Leaf Nodes**: http://localhost:8000/api/org-units/leaves
2. **Administrations**: http://localhost:8000/api/org-units/administrations
3. **Departments**: http://localhost:8000/api/org-units/departments
4. **Summary**: http://localhost:8000/api/org-units/summary

Or visit: **http://localhost:8000/docs** → Look for "Organizational Units" section

---

## 🎯 **Action Plan**

### **Do This Now**:

1. ✅ Open your **Add Patient Form**
   - Find "Issuing Department" dropdown
   - Replace with: `GET /api/org-units/leaves`

2. ✅ Open your **Report Configuration**
   - Find "Report Scope" dropdown
   - Replace with: `GET /api/org-units/administrations`

3. ✅ Find any **Department filters**
   - Replace with: `GET /api/org-units/departments`

4. ✅ Test each form to verify correct units appear

### **Keep Unchanged**:

- ❌ Investigation Page (keep using `/api/investigation/hierarchy`)
- ❌ Dashboard cascade filters (keep using hierarchy)

---

**All endpoints tested and working!** ✅  
**Start updating your UI now!** 🚀
