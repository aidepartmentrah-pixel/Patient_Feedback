# UI Component to Endpoint Mapping Guide

## 🎯 **Exact Endpoints for Your UI Needs**

This guide shows **exactly which endpoint** to call for each UI component in your application.

---

## 📍 **1. SECTIONS/LEAF NODES** (Smallest Units)

### **Use Case**: Insert/Add Patient Forms, Incident Creation

**Endpoint**: `GET /api/org-units/leaves`

**Returns**: All leaf nodes (216 units) - sections and departments with no children

### **Where to Use in UI**:

#### ✅ **Patient Creation Form** (`AddPatient.vue` or similar)
- **Field**: "Issuing Department" dropdown
- **Why**: Patient filed complaint about a specific unit

```javascript
// Example: Populate issuing department dropdown
async function loadIssuingDepartments() {
  const response = await fetch('http://localhost:8000/api/org-units/leaves');
  const data = await response.json();
  
  // Populate dropdown
  issuingDepartmentOptions.value = data.leaves.map(leaf => ({
    value: leaf.id,
    label: leaf.name,
    labelAr: leaf.name_ar,
    parentInfo: leaf.parent_name // Optional: show parent context
  }));
}
```

#### ✅ **Incident Creation Form** (`AddIncident.vue` or similar)
- **Field**: "Target Department" dropdown
- **Why**: Incident happened at a specific operational unit

```javascript
// Example: Populate target department dropdown
async function loadTargetDepartments() {
  const response = await fetch('http://localhost:8000/api/org-units/leaves');
  const data = await response.json();
  
  targetDepartmentOptions.value = data.leaves.map(leaf => ({
    value: leaf.id,
    label: `${leaf.name} (${leaf.type_name})`,
    type: leaf.type_name
  }));
}
```

#### ✅ **Complaint Form** (Any form where users report issues)
- **Field**: "Department" or "Section" dropdown
- **Why**: Users report about actual operational units

**API Response Structure**:
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

---

## 📍 **2. ALL ADMINISTRATIONS** (Top-Level Units)

### **Use Case**: Reports, Aggregate Analysis, High-Level Selection

**Endpoint**: `GET /api/org-units/administrations`

**Returns**: Only top-level administrations (9 units)

### **Where to Use in UI**:

#### ✅ **Report Configuration Page** (`Reports.vue` or `ReportConfig.vue`)
- **Field**: "Report Scope" dropdown
- **Why**: Compare performance across major hospital divisions

```javascript
// Example: Populate report scope dropdown
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

#### ✅ **Monthly Report Export** (`MonthlyReport.vue`)
- **Field**: "Organization Level" dropdown
- **Why**: User wants to see "All Administrations" aggregate

```javascript
// Example: Report scope selector
async function loadAdministrationOptions() {
  const response = await fetch('http://localhost:8000/api/org-units/administrations');
  const data = await response.json();
  
  administrationOptions.value = data.administrations.map(admin => ({
    value: admin.id,
    label: admin.name
  }));
}
```

#### ✅ **Seasonal Report Configuration** (`SeasonalReport.vue`)
- **Field**: "Select Administration" dropdown
- **Why**: Generate reports for specific major divisions

#### ✅ **Dashboard High-Level Filter** (Optional)
- **Field**: "Administration Filter" at the top
- **Why**: Quick filter by major division

**API Response Structure**:
```json
{
  "administrations": [
    {
      "id": 1,
      "name": "الادارة العامة",
      "name_ar": "الادارة العامة"
    },
    {
      "id": 4,
      "name": "الادارة الطبية",
      "name_ar": "الادارة الطبية"
    }
  ],
  "count": 9
}
```

---

## 📍 **3. SELECT ADMINISTRATIONS** (Choose One or More Administrations)

### **Use Case**: Filter by Administration, Multi-Select Administrations

**Same Endpoint**: `GET /api/org-units/administrations`

**Returns**: Only administrations (9 units)

### **Where to Use in UI**:

#### ✅ **User Management** (`UserManagement.vue`)
- **Field**: "Assign to Administration" dropdown
- **Why**: Assign user to a specific administration

```javascript
// Example: Administration assignment
async function loadAdministrationsList() {
  const response = await fetch('http://localhost:8000/api/org-units/administrations');
  const data = await response.json();
  
  administrationList.value = data.administrations.map(admin => ({
    value: admin.id,
    text: admin.name
  }));
}
```

#### ✅ **Settings Page** (`Settings.vue`)
- **Field**: "Default Administration" dropdown
- **Why**: Set default administration for reports/views

#### ✅ **Filter Panel** (Any page with filters)
- **Field**: "Administration" filter dropdown
- **Why**: Filter data by specific administration

```javascript
// Example: Multi-select for filtering
async function loadAdministrationFilters() {
  const response = await fetch('http://localhost:8000/api/org-units/administrations');
  const data = await response.json();
  
  // For multi-select component
  administrationFilters.value = data.administrations.map(admin => ({
    id: admin.id,
    name: admin.name,
    selected: false
  }));
}
```

#### ✅ **Investigation Page** - Administration Level
- **Field**: First dropdown in cascade (when NOT using full hierarchy)
- **Why**: Start filtering at administration level

---

## 📍 **4. SELECT DEPARTMENTS** (Only Departments)

### **Use Case**: Filter by Department, Department Selection

**Endpoint**: `GET /api/org-units/departments`

**Returns**: Only departments (134 units)

### **Where to Use in UI**:

#### ✅ **User Assignment** (`UserRoles.vue`)
- **Field**: "Assign to Department" dropdown
- **Why**: Assign user to a specific department

```javascript
// Example: Department assignment
async function loadDepartmentsList() {
  const response = await fetch('http://localhost:8000/api/org-units/departments');
  const data = await response.json();
  
  departmentList.value = data.departments.map(dept => ({
    value: dept.id,
    label: dept.name,
    administrationId: dept.administration_id // For grouping
  }));
  
  // Optional: Group by administration
  departmentListGrouped.value = groupBy(data.departments, 'administration_id');
}
```

#### ✅ **Department-Level Reports** (`DepartmentReport.vue`)
- **Field**: "Select Department" dropdown
- **Why**: Generate report for specific department

```javascript
// Example: Department selector for reports
async function loadDepartmentOptions() {
  const response = await fetch('http://localhost:8000/api/org-units/departments');
  const data = await response.json();
  
  departmentOptions.value = data.departments.map(dept => ({
    value: dept.id,
    text: dept.name,
    group: dept.administration_id // For optgroup
  }));
}
```

#### ✅ **Filter Panel** - Department Level
- **Field**: "Department" filter dropdown
- **Why**: Filter data by specific departments

```javascript
// Example: Multi-select departments filter
async function loadDepartmentFilters() {
  const response = await fetch('http://localhost:8000/api/org-units/departments');
  const data = await response.json();
  
  departmentFilters.value = data.departments.map(dept => ({
    id: dept.id,
    name: dept.name,
    administrationId: dept.administration_id,
    checked: false
  }));
}
```

#### ✅ **Department Performance Dashboard** (`PerformanceDashboard.vue`)
- **Field**: "Compare Departments" multi-select
- **Why**: Compare multiple departments side-by-side

**API Response Structure**:
```json
{
  "departments": [
    {
      "id": 10,
      "name": "Emergency Department",
      "name_ar": "قسم الطوارئ",
      "administration_id": 4
    },
    {
      "id": 15,
      "name": "Imaging Department",
      "name_ar": "دائرة التصوير الطبي",
      "administration_id": 4
    }
  ],
  "count": 134
}
```

---

## 📍 **5. BONUS: SECTIONS ONLY** (All Sections, Not Leaves)

### **Use Case**: When you need ALL sections, not just leaf sections

**Endpoint**: `GET /api/org-units/sections`

**Returns**: Only sections (85 units)

### **Where to Use in UI**:

#### ✅ **Section Management** (`SectionManagement.vue`)
- **Field**: "Select Section" dropdown
- **Why**: Manage sections specifically

```javascript
// Example: Section selector
async function loadSections() {
  const response = await fetch('http://localhost:8000/api/org-units/sections');
  const data = await response.json();
  
  sectionList.value = data.sections.map(section => ({
    value: section.id,
    label: section.name,
    departmentId: section.department_id
  }));
}
```

**API Response Structure**:
```json
{
  "sections": [
    {
      "id": 45,
      "name": "Emergency Section",
      "name_ar": "قسم الطوارئ",
      "department_id": 10
    }
  ],
  "count": 85
}
```

---

## 📍 **6. CASCADING HIERARCHY** (Keep Existing)

### **Use Case**: Investigation Page, Dashboard with Drill-Down

**Endpoint**: `GET /api/investigation/hierarchy` (existing - don't change)

**Returns**: Full 3-level nested hierarchy

### **Where to Use in UI**:

#### ✅ **Investigation Page** (`Investigation.vue`)
- **Component**: Cascading dropdowns (Admin → Dept → Section)
- **Why**: Users drill down through hierarchy

#### ✅ **Dashboard** (`Dashboard.vue`)
- **Component**: Hierarchical filters
- **Why**: Navigate through organization structure

**Keep using this for cascading selectors!**

---

## 🎯 **Quick Reference Table**

| UI Component | Field Name | Endpoint | What It Returns |
|--------------|------------|----------|-----------------|
| **Add Patient Form** | Issuing Department | `/api/org-units/leaves` | 216 leaf units |
| **Add Incident Form** | Target Department | `/api/org-units/leaves` | 216 leaf units |
| **Report Configuration** | Report Scope | `/api/org-units/administrations` | 9 administrations |
| **Monthly Report** | Organization Level | `/api/org-units/administrations` | 9 administrations |
| **User Assignment** | Assign to Administration | `/api/org-units/administrations` | 9 administrations |
| **User Assignment** | Assign to Department | `/api/org-units/departments` | 134 departments |
| **Department Report** | Select Department | `/api/org-units/departments` | 134 departments |
| **Filter Panel** | Department Filter | `/api/org-units/departments` | 134 departments |
| **Section Management** | Select Section | `/api/org-units/sections` | 85 sections |
| **Investigation Page** | Cascading Filters | `/api/investigation/hierarchy` | Full nested tree |
| **Dashboard** | Hierarchy Navigation | `/api/investigation/hierarchy` | Full nested tree |

---

## 🔥 **Complete Implementation Examples**

### **Example 1: Add Patient Form (Leaf Nodes)**

```vue
<template>
  <div class="add-patient-form">
    <h2>Add Patient</h2>
    
    <!-- Patient Name -->
    <input v-model="patientName" placeholder="Patient Name" />
    
    <!-- Issuing Department (USE LEAF NODES) -->
    <select v-model="issuingDepartmentId">
      <option value="">Select Department</option>
      <option 
        v-for="leaf in leafNodes" 
        :key="leaf.id" 
        :value="leaf.id"
      >
        {{ leaf.name }} ({{ leaf.type_name }})
      </option>
    </select>
    
    <button @click="submitPatient">Submit</button>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const patientName = ref('');
const issuingDepartmentId = ref('');
const leafNodes = ref([]);

// Load leaf nodes on mount
onMounted(async () => {
  const response = await fetch('http://localhost:8000/api/org-units/leaves');
  const data = await response.json();
  leafNodes.value = data.leaves;
});

async function submitPatient() {
  const patientData = {
    name: patientName.value,
    issuing_department_id: issuingDepartmentId.value
  };
  
  // Submit to backend...
}
</script>
```

### **Example 2: Report Configuration (Administrations Only)**

```vue
<template>
  <div class="report-config">
    <h2>Generate Report</h2>
    
    <!-- Report Scope (USE ADMINISTRATIONS) -->
    <select v-model="reportScope">
      <option value="all">All Administrations</option>
      <option 
        v-for="admin in administrations" 
        :key="admin.id" 
        :value="admin.id"
      >
        {{ admin.name }}
      </option>
    </select>
    
    <!-- Date Range -->
    <input type="date" v-model="startDate" />
    <input type="date" v-model="endDate" />
    
    <button @click="generateReport">Generate Report</button>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const reportScope = ref('all');
const administrations = ref([]);
const startDate = ref('');
const endDate = ref('');

// Load administrations on mount
onMounted(async () => {
  const response = await fetch('http://localhost:8000/api/org-units/administrations');
  const data = await response.json();
  administrations.value = data.administrations;
});

async function generateReport() {
  const reportConfig = {
    scope: reportScope.value,
    start_date: startDate.value,
    end_date: endDate.value
  };
  
  // Generate report...
}
</script>
```

### **Example 3: Department Filter (Departments Only)**

```vue
<template>
  <div class="department-filter">
    <h3>Filter by Department</h3>
    
    <!-- Department Multi-Select (USE DEPARTMENTS) -->
    <div class="checkbox-list">
      <label v-for="dept in departments" :key="dept.id">
        <input 
          type="checkbox" 
          :value="dept.id"
          v-model="selectedDepartments"
        />
        {{ dept.name }}
      </label>
    </div>
    
    <button @click="applyFilter">Apply Filter</button>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const departments = ref([]);
const selectedDepartments = ref([]);

// Load departments on mount
onMounted(async () => {
  const response = await fetch('http://localhost:8000/api/org-units/departments');
  const data = await response.json();
  departments.value = data.departments;
});

function applyFilter() {
  // Apply filter with selectedDepartments.value
  console.log('Selected:', selectedDepartments.value);
}
</script>
```

---

## ✅ **Action Checklist**

### **Frontend Tasks**:

- [ ] **Add Patient Form**: Update to use `/api/org-units/leaves`
- [ ] **Add Incident Form**: Update to use `/api/org-units/leaves`
- [ ] **Report Configuration**: Update to use `/api/org-units/administrations`
- [ ] **Monthly Reports**: Update to use `/api/org-units/administrations`
- [ ] **Department Filters**: Update to use `/api/org-units/departments`
- [ ] **User Assignment**: Use appropriate endpoint based on level
- [ ] **Test all forms**: Verify correct units appear in each dropdown

### **Keep Unchanged**:

- [ ] **Investigation Page**: Keep using `/api/investigation/hierarchy`
- [ ] **Dashboard Cascade**: Keep using `/api/investigation/hierarchy`

---

## 📊 **Endpoint Summary**

```
Base URL: http://localhost:8000

1. Leaf Nodes (216 units)
   GET /api/org-units/leaves

2. Administrations (9 units)
   GET /api/org-units/administrations

3. Departments (134 units)
   GET /api/org-units/departments

4. Sections (85 units)
   GET /api/org-units/sections

5. Unit with Breadcrumb
   GET /api/org-units/unit/{id}

6. Summary Stats
   GET /api/org-units/summary
```

---

## 🎯 **The Golden Rule**

| If User Needs To... | Use This Endpoint |
|---------------------|-------------------|
| **Report where incident happened** | `/api/org-units/leaves` |
| **Compare major divisions** | `/api/org-units/administrations` |
| **Select a department** | `/api/org-units/departments` |
| **Drill down through hierarchy** | `/api/investigation/hierarchy` |

---

**All endpoints are live and tested!** ✅ Start updating your UI components now.
