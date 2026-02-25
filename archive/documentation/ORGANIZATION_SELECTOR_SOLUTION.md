# Organization Selector Solution

## 🎯 Problem Statement

The application needs **different types of organization selection** for different purposes:

1. **INSERT Page** (Patient/Incident Creation): Needs **LEAF NODES ONLY** (Sections)
   - Why: Users complain about what they actually experience at the smallest operational unit
   - Example: A patient has an issue in "Emergency Section" not "Medical Administration"

2. **REPORTS Page** (Aggregate Reports): Needs **ALL ADMINISTRATIONS** (Top level only)
   - Why: For hospital-wide reporting and comparison across major divisions
   - Example: Compare "Medical Administration" vs "Surgical Administration" performance

## ✅ Implemented Solution

### 1. New Endpoint: Get Leaf Nodes for Insert Forms
**Endpoint**: `GET /api/org-units/leaves`

**Purpose**: Returns ONLY the smallest organizational units (sections with no children)

**Response Structure**:
```json
{
  "leaves": [
    {
      "id": 45,
      "name": "Emergency Section",
      "name_ar": "قسم الطوارئ",
      "parent_id": 10,
      "parent_name": "Emergency Department"
    },
    {
      "id": 46,
      "name": "ICU Section",
      "name_ar": "قسم العناية المركزة",
      "parent_id": 11,
      "parent_name": "Critical Care Department"
    }
  ],
  "count": 2
}
```

**When to Use**: Insert/Add Patient forms, Incident creation forms

---

### 2. New Endpoint: Get All Administrations for Reports
**Endpoint**: `GET /api/org-units/administrations`

**Purpose**: Returns ONLY top-level administration units

**Response Structure**:
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
  "count": 2
}
```

**When to Use**: Report configuration, aggregate analysis, high-level filtering

---

### 3. Existing Endpoint: Full Hierarchy (Keep for Backwards Compatibility)
**Endpoint**: `GET /api/investigation/hierarchy`

**Purpose**: Returns complete 3-level hierarchy (Administrations → Departments → Sections)

**When to Use**: Investigation trees, detailed filtering, cascading selectors

---

## 🔧 Technical Implementation

### Files Modified:
1. **`backend/api/routers/org_unit_router.py`** ← NEW ROUTER
   - `/api/org-units/leaves` endpoint
   - `/api/org-units/administrations` endpoint
   - `/api/org-units/departments` endpoint (bonus: get all departments)
   - `/api/org-units/sections` endpoint (bonus: get all sections)

2. **`backend/api/services/org_unit_service.py`** ← NEW SERVICE
   - `get_leaf_units()` - Returns leaf nodes only
   - `get_administrations()` - Returns top-level units only
   - `get_units_by_level()` - Generic function for any level

3. **`backend/main.py`** ← UPDATED
   - Register new org_unit_router

---

## 📋 Usage Guide

### For Frontend Developers:

#### 1. INSERT/ADD FORMS (Patient, Incident Creation)
```javascript
// Use: /api/org-units/leaves
const loadIssuingDepartments = async () => {
  const response = await fetch('/api/org-units/leaves');
  const data = await response.json();
  
  // Populate dropdown with ONLY leaf nodes
  const dropdown = data.leaves.map(leaf => ({
    value: leaf.id,
    label: leaf.name,
    labelAr: leaf.name_ar
  }));
};
```

**Why**: Users select the ACTUAL department where the incident occurred, not abstract administrative divisions.

---

#### 2. REPORTS (All Administrations View)
```javascript
// Use: /api/org-units/administrations
const loadReportScope = async () => {
  const response = await fetch('/api/org-units/administrations');
  const data = await response.json();
  
  // Show "All Administrations" option + individual admins
  const options = [
    { value: 'all', label: 'All Administrations' },
    ...data.administrations.map(admin => ({
      value: admin.id,
      label: admin.name
    }))
  ];
};
```

**Why**: Reports need high-level grouping for hospital-wide analysis.

---

#### 3. CASCADING FILTERS (Investigation, Dashboard)
```javascript
// Use: /api/investigation/hierarchy (existing)
const loadHierarchy = async () => {
  const response = await fetch('/api/investigation/hierarchy');
  const data = await response.json();
  
  // Build cascading dropdowns: Admin → Dept → Section
  buildCascadingSelectors(data);
};
```

**Why**: Users drill down from top to bottom in exploratory analysis.

---

## 🧪 Testing

### Test the New Endpoints:

```bash
# 1. Get leaf nodes (for insert forms)
curl http://localhost:8000/api/org-units/leaves

# 2. Get all administrations (for reports)
curl http://localhost:8000/api/org-units/administrations

# 3. Get all departments
curl http://localhost:8000/api/org-units/departments

# 4. Get all sections
curl http://localhost:8000/api/org-units/sections
```

---

## 📊 Database Schema Reference

```
AdminsrationUnit Table:
- UniqueID (PK)
- ParentID (Self-referencing FK)
- Name
- Type (323=Admin, 324=Section, 325=Department)
- Frozen (Boolean)

Hierarchy Rules:
- Administration: ParentID == UniqueID (self-referencing root)
- Department: ParentID points to Administration
- Section: ParentID points to Department

Leaf Node: Any unit with NO children (no other unit has ParentID = this unit's UniqueID)
```

---

## ✅ Benefits

1. **Clearer API semantics**: Each endpoint has a single, clear purpose
2. **Performance**: Leaf nodes endpoint returns ONLY what's needed (smaller payload)
3. **Frontend simplicity**: No need to filter hierarchy client-side
4. **Flexibility**: Can add more specialized endpoints as needed (e.g., sections by department)

---

## 🚀 Next Steps

### For Backend:
- ✅ Implemented specialized org unit router
- ✅ Created org unit service layer
- ✅ Registered router in main.py

### For Frontend:
1. **Update INSERT forms** to use `/api/org-units/leaves`
   - Patient creation form
   - Incident creation form
   - Any "issuing department" selector

2. **Update REPORT forms** to use `/api/org-units/administrations`
   - Monthly report scope selector
   - Seasonal report scope selector
   - Aggregate analysis filters

3. **Keep existing hierarchy endpoint** for:
   - Investigation page
   - Dashboard filters
   - Any cascading selectors (Admin → Dept → Section)

---

## 📝 Summary

| Feature | Endpoint | Returns | Use Case |
|---------|----------|---------|----------|
| **Leaf Nodes** | `/api/org-units/leaves` | Sections only (no children) | INSERT forms |
| **Administrations** | `/api/org-units/administrations` | Top-level units only | REPORTS |
| **Full Hierarchy** | `/api/investigation/hierarchy` | All 3 levels nested | Investigation, Dashboard |
| **All Departments** | `/api/org-units/departments` | All departments | Optional filtering |
| **All Sections** | `/api/org-units/sections` | All sections | Optional filtering |

---

## 🔗 Related Files

- Router: `backend/api/routers/org_unit_router.py`
- Service: `backend/api/services/org_unit_service.py`
- DB Layer: `backend/api/db_layer/admin_units.py` (existing)
- Constants: `backend/api/constants/org_unit_types.py` (existing)
