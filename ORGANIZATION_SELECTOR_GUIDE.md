# 🎯 Organization Selector - Complete Guide

## 📊 Quick Summary

You have **4 different organization selector endpoints** ready to use:

| # | Endpoint | Count | Use For |
|---|----------|-------|---------|
| **1** | `/api/org-units/leaves` | 216 units | **INSERT forms** - where incidents happen |
| **2** | `/api/org-units/administrations` | 9 units | **REPORTS** - high-level aggregate |
| **3** | `/api/org-units/administrations` | 9 units | **USER ASSIGNMENT** - select admin |
| **4** | `/api/org-units/departments` | 134 units | **FILTERS** - department selection |

---

## 🔍 Detailed Breakdown

### 1️⃣ LEAF NODES - For INSERT Forms

**Endpoint**: `GET /api/org-units/leaves`

**Returns**: 216 leaf units (smallest organizational units)

**Use Case**: 
- ✅ Add Patient Form → "Issuing Department" field
- ✅ Create Incident Form → "Where did this happen?" field  
- ✅ Any form where user needs to select **actual location** of event

**Why**: Users experience issues at the **smallest operational unit** (sections/departments with no children), not at abstract administrative levels.

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

**Frontend Code**:
```javascript
// Vue/React example
async function loadIssuingDepartments() {
  const response = await fetch('http://localhost:8000/api/org-units/leaves');
  const data = await response.json();
  
  issuingDeptOptions.value = data.leaves.map(leaf => ({
    value: leaf.id,
    label: leaf.name,
    labelAr: leaf.name_ar
  }));
}
```

---

### 2️⃣ ALL ADMINISTRATIONS - For Reports

**Endpoint**: `GET /api/org-units/administrations`

**Returns**: 9 top-level administration units

**Use Case**:
- ✅ Report Configuration → "Report Scope" dropdown
- ✅ Monthly Reports → "Organization Level" selector
- ✅ Seasonal Reports → "Select Administration" field
- ✅ Dashboard → "All Administrations" filter option

**Why**: Reports need to show **aggregate data** across major hospital divisions for comparison.

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

**Frontend Code**:
```javascript
// Report configuration page
async function loadReportScope() {
  const response = await fetch('http://localhost:8000/api/org-units/administrations');
  const data = await response.json();
  
  // Add "All" option + individual administrations
  reportScopeOptions.value = [
    { value: 'all', label: 'All Administrations', labelAr: 'جميع الإدارات' },
    ...data.administrations.map(admin => ({
      value: admin.id,
      label: admin.name,
      labelAr: admin.name_ar
    }))
  ];
}
```

---

### 3️⃣ SELECT ADMINISTRATIONS - For User Assignment

**Endpoint**: `GET /api/org-units/administrations` (same as #2)

**Returns**: 9 administrations

**Use Case**:
- ✅ User Management → "Assign to Administration" dropdown
- ✅ Settings → "Default Administration" selector
- ✅ Investigation Page → "Administration" filter

**Why**: Same endpoint as #2, just different UI usage (single-select vs view-all).

**Frontend Code**:
```javascript
// User assignment form
async function loadAdministrationOptions() {
  const response = await fetch('http://localhost:8000/api/org-units/administrations');
  const data = await response.json();
  
  adminOptions.value = data.administrations.map(admin => ({
    value: admin.id,
    text: admin.name,
    textAr: admin.name_ar
  }));
}
```

---

### 4️⃣ DEPARTMENTS ONLY - For Filtering

**Endpoint**: `GET /api/org-units/departments`

**Returns**: 134 department units

**Use Case**:
- ✅ Filter Panel → "Department" multi-select filter
- ✅ User Assignment → "Assign to Department" dropdown
- ✅ Department Reports → "Select Department" field
- ✅ Performance Dashboard → "Compare Departments" selector

**Why**: When you need **department-level** selection specifically, not sections or administrations.

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
      "id": 6,
      "name": "دائرة شؤون المرضى",
      "name_ar": "دائرة شؤون المرضى",
      "administration_id": 2
    }
  ],
  "count": 134
}
```

**Frontend Code**:
```javascript
// Department filter (multi-select)
async function loadDepartmentFilters() {
  const response = await fetch('http://localhost:8000/api/org-units/departments');
  const data = await response.json();
  
  departmentFilters.value = data.departments.map(dept => ({
    id: dept.id,
    name: dept.name,
    nameAr: dept.name_ar,
    administrationId: dept.administration_id,
    checked: false
  }));
}
```

---

## 📋 Usage Matrix

| UI Component | Endpoint to Use | Field Name |
|--------------|----------------|------------|
| **Add Patient Form** | `/api/org-units/leaves` | Issuing Department |
| **Create Incident Form** | `/api/org-units/leaves` | Department Where Incident Occurred |
| **Report Configuration** | `/api/org-units/administrations` | Report Scope |
| **Monthly Report** | `/api/org-units/administrations` | Organization Level |
| **Seasonal Report** | `/api/org-units/administrations` | Select Administration |
| **User Management** | `/api/org-units/administrations` | Assign to Administration |
| **User Management** | `/api/org-units/departments` | Assign to Department |
| **Filter Panel** | `/api/org-units/departments` | Department Filter |
| **Performance Dashboard** | `/api/org-units/departments` | Compare Departments |
| **Investigation Page** | Keep existing hierarchy endpoint | Cascading Filters |

---

## 🧪 Test URLs

Open these in your browser to see live data:

1. **Leaf Nodes**: http://localhost:8000/api/org-units/leaves
2. **Administrations**: http://localhost:8000/api/org-units/administrations
3. **Departments**: http://localhost:8000/api/org-units/departments
4. **Summary Stats**: http://localhost:8000/api/org-units/summary
5. **API Documentation**: http://localhost:8000/docs

---

## 🎨 UI Implementation Examples

### Example 1: Add Patient Form (Dropdown)

```vue
<template>
  <div class="form-group">
    <label>Issuing Department</label>
    <select v-model="selectedDepartment">
      <option value="">-- Select Department --</option>
      <option 
        v-for="leaf in leafOptions" 
        :key="leaf.value" 
        :value="leaf.value"
      >
        {{ leaf.label }}
      </option>
    </select>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const selectedDepartment = ref('');
const leafOptions = ref([]);

onMounted(async () => {
  const response = await fetch('http://localhost:8000/api/org-units/leaves');
  const data = await response.json();
  
  leafOptions.value = data.leaves.map(leaf => ({
    value: leaf.id,
    label: leaf.name
  }));
});
</script>
```

### Example 2: Report Configuration (Radio + Dropdown)

```vue
<template>
  <div class="report-config">
    <h3>Report Scope</h3>
    
    <div class="radio-group">
      <label>
        <input type="radio" v-model="scopeType" value="all" />
        All Administrations
      </label>
      
      <label>
        <input type="radio" v-model="scopeType" value="specific" />
        Specific Administration
      </label>
    </div>
    
    <select v-if="scopeType === 'specific'" v-model="selectedAdmin">
      <option value="">-- Select Administration --</option>
      <option 
        v-for="admin in adminOptions" 
        :key="admin.value" 
        :value="admin.value"
      >
        {{ admin.label }}
      </option>
    </select>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const scopeType = ref('all');
const selectedAdmin = ref('');
const adminOptions = ref([]);

onMounted(async () => {
  const response = await fetch('http://localhost:8000/api/org-units/administrations');
  const data = await response.json();
  
  adminOptions.value = data.administrations.map(admin => ({
    value: admin.id,
    label: admin.name
  }));
});
</script>
```

### Example 3: Department Filter (Multi-select Checkboxes)

```vue
<template>
  <div class="filter-panel">
    <h3>Filter by Department</h3>
    
    <div 
      v-for="dept in departments" 
      :key="dept.id" 
      class="checkbox-item"
    >
      <label>
        <input 
          type="checkbox" 
          v-model="dept.checked" 
          @change="onFilterChange"
        />
        {{ dept.name }}
      </label>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const departments = ref([]);

onMounted(async () => {
  const response = await fetch('http://localhost:8000/api/org-units/departments');
  const data = await response.json();
  
  departments.value = data.departments.map(dept => ({
    id: dept.id,
    name: dept.name,
    administrationId: dept.administration_id,
    checked: false
  }));
});

function onFilterChange() {
  const selectedDepartments = departments.value
    .filter(d => d.checked)
    .map(d => d.id);
  
  // Apply filter to your data
  console.log('Selected departments:', selectedDepartments);
}
</script>
```

---

## ✅ Testing Checklist

Run this test to verify all endpoints:
```bash
cd "c:\Users\IT\Documents\GitHub Repository\Patient_Feedback"
python test_4_endpoints.py
```

Expected Results:
- ✅ 216 leaf nodes
- ✅ 9 administrations
- ✅ 134 departments
- ✅ All endpoints return `200 OK`

---

## 🎯 Summary

| Need | Solution |
|------|----------|
| **"Where did incident happen?"** | Use `/api/org-units/leaves` (216 leaf units) |
| **"Report on all administrations"** | Use `/api/org-units/administrations` (9 admins) |
| **"Assign user to administration"** | Use `/api/org-units/administrations` (9 admins) |
| **"Filter by department"** | Use `/api/org-units/departments` (134 depts) |

All 4 endpoints are **working**, **tested**, and ready for frontend integration! 🎉
