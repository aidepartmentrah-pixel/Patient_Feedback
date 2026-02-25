# Organization Selection Quick Reference

## 🎯 Problem Solved

You had **different organization selection needs** in different parts of your application:

1. **INSERT Forms** → Need **LEAF NODES ONLY** (smallest units)
2. **REPORTS** → Need **ALL ADMINISTRATIONS** (top-level units)

## ✅ Solution Implemented

New specialized endpoints have been created to solve this problem:

---

## 📍 Endpoints Overview

| Endpoint | Purpose | Use Case | Returns |
|----------|---------|----------|---------|
| **`/api/org-units/leaves`** | Leaf nodes only | INSERT/ADD forms | Sections with no children |
| **`/api/org-units/administrations`** | Top-level units | REPORTS | All administrations |
| **`/api/org-units/departments`** | Mid-level units | Filtering | All departments |
| **`/api/org-units/sections`** | Bottom-level units | Filtering | All sections |
| **`/api/org-units/unit/{id}`** | Single unit + ancestry | Breadcrumbs | Unit with full path |
| **`/api/org-units/summary`** | Overview counts | Dashboard | Count by type |

---

## 🔥 When to Use Each

### 1. INSERT/ADD PATIENT FORMS ← **LEAF NODES**

**Endpoint**: `GET /api/org-units/leaves`

**Why**: Users need to select the **ACTUAL department** where an incident occurred.

**Example Response**:
```json
{
  "leaves": [
    {
      "id": 45,
      "name": "Emergency Section",
      "name_ar": "قسم الطوارئ",
      "parent_id": 10,
      "parent_name": "Emergency Department",
      "type": 324,
      "type_name": "SECTION"
    }
  ],
  "count": 216
}
```

**Frontend Usage**:
```javascript
// Populate issuing department dropdown in INSERT form
const response = await fetch('/api/org-units/leaves');
const data = await response.json();

const issuingDeptOptions = data.leaves.map(leaf => ({
  value: leaf.id,
  label: leaf.name,
  labelAr: leaf.name_ar
}));
```

---

### 2. REPORTS (All Administrations) ← **TOP LEVEL ONLY**

**Endpoint**: `GET /api/org-units/administrations`

**Why**: Reports need to compare **major hospital divisions**.

**Example Response**:
```json
{
  "administrations": [
    {
      "id": 1,
      "name": "Medical Administration",
      "name_ar": "الإدارة الطبية"
    },
    {
      "id": 2,
      "name": "Surgical Administration",
      "name_ar": "الإدارة الجراحية"
    }
  ],
  "count": 9
}
```

**Frontend Usage**:
```javascript
// Populate report scope dropdown
const response = await fetch('/api/org-units/administrations');
const data = await response.json();

const reportScopeOptions = [
  { value: 'all', label: 'All Administrations' },
  ...data.administrations.map(admin => ({
    value: admin.id,
    label: admin.name
  }))
];
```

---

### 3. CASCADING FILTERS ← **FULL HIERARCHY**

**Endpoint**: `GET /api/investigation/hierarchy` (existing)

**Why**: Users drill down: Admin → Dept → Section

**Use this endpoint** for:
- Investigation page
- Dashboard filters
- Anywhere you need cascading selectors

---

### 4. BREADCRUMB NAVIGATION ← **UNIT WITH ANCESTORS**

**Endpoint**: `GET /api/org-units/unit/{id}`

**Example Response**:
```json
{
  "id": 45,
  "name": "Emergency Section",
  "type": 324,
  "type_name": "SECTION",
  "ancestors": [
    {
      "id": 1,
      "name": "Medical Administration",
      "type": 323,
      "type_name": "ADMINISTRATION"
    },
    {
      "id": 10,
      "name": "Emergency Department",
      "type": 325,
      "type_name": "DEPARTMENT"
    }
  ],
  "breadcrumb": "Medical Administration > Emergency Department > Emergency Section"
}
```

**Frontend Usage**:
```javascript
// Display breadcrumb for a unit
const response = await fetch(`/api/org-units/unit/${unitId}`);
const data = await response.json();
setBreadcrumb(data.breadcrumb);
// Output: "Medical Administration > Emergency Department > Emergency Section"
```

---

## 🧪 Testing

### Test with curl:

```bash
# 1. Get leaf nodes (for insert forms)
curl http://localhost:8000/api/org-units/leaves

# 2. Get all administrations (for reports)
curl http://localhost:8000/api/org-units/administrations

# 3. Get all departments
curl http://localhost:8000/api/org-units/departments

# 4. Get all sections  
curl http://localhost:8000/api/org-units/sections

# 5. Get unit with ancestors
curl http://localhost:8000/api/org-units/unit/45

# 6. Get summary
curl http://localhost:8000/api/org-units/summary
```

### Test with Python:

```bash
# Run comprehensive test
python test_org_unit_endpoints.py

# Run API test (requires server running)
python test_org_unit_api.py
```

### Test in Browser:

Visit: http://localhost:8000/docs

Look for **"Organizational Units"** section

---

## 📊 Current Database Stats

**Your database has:**
- **9 Administrations** (top-level)
- **134 Departments** (mid-level)
- **85 Sections** (bottom-level)
- **216 Leaf Nodes** (units with no children)
- **228 Total Units**

---

## 🚀 Implementation Status

✅ **Backend Service** - Created `org_unit_service.py`  
✅ **Backend Router** - Created `org_unit_router.py`  
✅ **Main Registration** - Added to `main.py`  
✅ **Testing** - Both service and API tests passing  
✅ **Documentation** - Complete with examples  

---

## 📝 Frontend Action Items

### 1. UPDATE INSERT/ADD PATIENT FORMS

**File**: Your patient/incident creation form  
**Change**: Use `/api/org-units/leaves` instead of full hierarchy

```javascript
// OLD (Don't use this anymore for INSERT)
const hierarchy = await fetch('/api/investigation/hierarchy');
// Then filter for leaf nodes client-side... ❌

// NEW (Use this for INSERT)
const response = await fetch('/api/org-units/leaves');
const data = await response.json();
const options = data.leaves.map(leaf => ({
  value: leaf.id,
  label: leaf.name
}));
// ✅ Server already filtered to leaf nodes only
```

### 2. UPDATE REPORT CONFIGURATION

**File**: Your report scope selector  
**Change**: Use `/api/org-units/administrations`

```javascript
// For "All Administrations" view in reports
const response = await fetch('/api/org-units/administrations');
const data = await response.json();

const options = [
  { value: 'all', label: 'All Administrations' },
  ...data.administrations.map(admin => ({
    value: admin.id,
    label: admin.name
  }))
];
```

### 3. KEEP EXISTING HIERARCHY ENDPOINT

**Don't change**: Investigation page, Dashboard filters  
**Keep using**: `/api/investigation/hierarchy`

These pages need the full cascading hierarchy.

---

## 🔗 Related Files

**Service Layer**:
- `backend/api/services/org_unit_service.py` (NEW)

**Router Layer**:
- `backend/api/routers/org_unit_router.py` (NEW)

**Database Layer**:
- `backend/api/db_layer/admin_units.py` (existing)

**Main App**:
- `backend/main.py` (updated to register router)

**Tests**:
- `test_org_unit_endpoints.py` (service tests)
- `test_org_unit_api.py` (API tests)

**Documentation**:
- `ORGANIZATION_SELECTOR_SOLUTION.md` (complete guide)

---

## 💡 Summary

| Need | Old Problem | New Solution |
|------|-------------|--------------|
| **INSERT forms** | Had to use full hierarchy and filter client-side | Use `/api/org-units/leaves` - returns leaf nodes only |
| **REPORTS** | Had to manually build "All Administrations" option | Use `/api/org-units/administrations` - returns top level only |
| **Investigation** | Needed cascading selectors | Keep using `/api/investigation/hierarchy` - unchanged |

---

## ✅ Next Steps

1. **Backend** - Already done! ✅
2. **Frontend** - Update INSERT forms to use `/api/org-units/leaves`
3. **Frontend** - Update REPORT selectors to use `/api/org-units/administrations`
4. **Testing** - Verify in your UI that the correct units appear in each context

---

## 🎯 The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR APPLICATION                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  INSERT FORMS    │         │     REPORTS      │         │
│  │                  │         │                  │         │
│  │  "Where did the  │         │  "Compare all    │         │
│  │   incident       │         │   major hospital │         │
│  │   happen?"       │         │   divisions"     │         │
│  │                  │         │                  │         │
│  │  Uses:           │         │  Uses:           │         │
│  │  /org-units/     │         │  /org-units/     │         │
│  │  leaves          │         │  administrations │         │
│  │                  │         │                  │         │
│  │  Returns:        │         │  Returns:        │         │
│  │  ✓ Sections only │         │  ✓ Top level     │         │
│  │  ✓ 216 units     │         │  ✓ 9 units       │         │
│  └──────────────────┘         └──────────────────┘         │
│                                                             │
│  ┌──────────────────────────────────────────────┐          │
│  │        INVESTIGATION / DASHBOARD             │          │
│  │                                              │          │
│  │  "Drill down through hierarchy"             │          │
│  │                                              │          │
│  │  Uses: /investigation/hierarchy (existing)  │          │
│  │                                              │          │
│  │  Returns: Full 3-level hierarchy            │          │
│  │  Admin → Dept → Section                     │          │
│  └──────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

**Questions? Check `ORGANIZATION_SELECTOR_SOLUTION.md` for complete details!**
