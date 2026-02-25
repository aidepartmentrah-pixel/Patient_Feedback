# 🎯 Organization Selector - Quick Reference Card

## 4 Endpoints for 4 Different Needs

```
┌─────────────────────────────────────────────────────────────────┐
│  NEED: Where did this incident ACTUALLY happen?                │
│  USE:  GET /api/org-units/leaves                               │
│  GETS: 216 leaf units (sections/departments with no children)  │
│                                                                 │
│  Example: Add Patient Form → "Issuing Department" dropdown     │
│                                                                 │
│  Code:                                                          │
│    fetch('/api/org-units/leaves')                              │
│      .then(r => r.json())                                      │
│      .then(data => {                                           │
│        dropdown.options = data.leaves.map(leaf => ({           │
│          value: leaf.id, label: leaf.name                      │
│        }));                                                     │
│      });                                                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  NEED: Report on "All Administrations"                         │
│  USE:  GET /api/org-units/administrations                      │
│  GETS: 9 administration units (top-level)                      │
│                                                                 │
│  Example: Monthly Report → "Report Scope" dropdown             │
│                                                                 │
│  Code:                                                          │
│    fetch('/api/org-units/administrations')                     │
│      .then(r => r.json())                                      │
│      .then(data => {                                           │
│        dropdown.options = [                                    │
│          { value: 'all', label: 'All Administrations' },       │
│          ...data.administrations.map(a => ({                   │
│            value: a.id, label: a.name                          │
│          }))                                                    │
│        ];                                                       │
│      });                                                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  NEED: Assign user to ONE administration                       │
│  USE:  GET /api/org-units/administrations (same endpoint!)     │
│  GETS: 9 administrations (no "All" option)                     │
│                                                                 │
│  Example: User Management → "Assign to Administration"         │
│                                                                 │
│  Code:                                                          │
│    fetch('/api/org-units/administrations')                     │
│      .then(r => r.json())                                      │
│      .then(data => {                                           │
│        dropdown.options = data.administrations.map(a => ({     │
│          value: a.id, label: a.name                            │
│        }));                                                     │
│        // Note: NO "All" option for user assignment            │
│      });                                                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  NEED: Filter/compare by department                            │
│  USE:  GET /api/org-units/departments                          │
│  GETS: 134 departments (mid-level units)                       │
│                                                                 │
│  Example: Dashboard → "Filter by Department" checkboxes        │
│                                                                 │
│  Code:                                                          │
│    fetch('/api/org-units/departments')                         │
│      .then(r => r.json())                                      │
│      .then(data => {                                           │
│        filters = data.departments.map(dept => ({               │
│          id: dept.id,                                          │
│          name: dept.name,                                      │
│          administrationId: dept.administration_id,             │
│          checked: false                                        │
│        }));                                                     │
│      });                                                        │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Quick Comparison Table

| Endpoint | Count | Use Case | Example UI |
|----------|-------|----------|------------|
| `/api/org-units/leaves` | 216 | Where incident happened | Add Patient dropdown |
| `/api/org-units/administrations` | 9 | Report scope with "All" | Report Config dropdown |
| `/api/org-units/administrations` | 9 | Single admin selection | User Assignment dropdown |
| `/api/org-units/departments` | 134 | Department filter | Multi-select checkboxes |

## 🧪 Test Commands

```bash
# Quick test all endpoints
python test_4_endpoints.py

# Interactive test with examples
python test_organization_selectors_interactive.py

# Test in browser
# 1. http://localhost:8000/api/org-units/leaves
# 2. http://localhost:8000/api/org-units/administrations
# 3. http://localhost:8000/api/org-units/departments
# 4. http://localhost:8000/docs
```

## 💡 Common Patterns

### Pattern 1: Simple Dropdown
```javascript
// Most common: Single-select dropdown
const response = await fetch('/api/org-units/leaves');
const data = await response.json();

dropdownOptions.value = data.leaves.map(item => ({
  value: item.id,
  label: item.name
}));
```

### Pattern 2: Dropdown with "All" Option
```javascript
// For reports: Add "All" option
const response = await fetch('/api/org-units/administrations');
const data = await response.json();

dropdownOptions.value = [
  { value: 'all', label: 'All Administrations' },
  ...data.administrations.map(item => ({
    value: item.id,
    label: item.name
  }))
];
```

### Pattern 3: Multi-select Checkboxes
```javascript
// For filters: Multiple selections
const response = await fetch('/api/org-units/departments');
const data = await response.json();

checkboxOptions.value = data.departments.map(dept => ({
  id: dept.id,
  name: dept.name,
  checked: false
}));
```

## ⚡ Response Formats

### Leaves (216 items)
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

### Administrations (9 items)
```json
{
  "administrations": [
    {
      "id": 1,
      "name": "الادارة العامة",
      "name_ar": "الادارة العامة"
    }
  ],
  "count": 9
}
```

### Departments (134 items)
```json
{
  "departments": [
    {
      "id": 5,
      "name": "دائرة المواد",
      "name_ar": "دائرة المواد",
      "administration_id": 1
    }
  ],
  "count": 134
}
```

## 🎯 Decision Tree

```
Start: I need to show organization selector
│
├─ Q: Is this an INSERT/ADD form?
│  └─ YES → Use /api/org-units/leaves (216 leaf units)
│
├─ Q: Is this for REPORTS?
│  └─ YES → Use /api/org-units/administrations (9 admins + "All" option)
│
├─ Q: Is this for USER ASSIGNMENT?
│  └─ YES → Use /api/org-units/administrations (9 admins, single-select)
│
└─ Q: Is this for FILTERING by department?
   └─ YES → Use /api/org-units/departments (134 depts, multi-select)
```

## ✅ Checklist Before Implementation

- [ ] Identified which selector type you need (1, 2, 3, or 4)
- [ ] Reviewed the code example for your type
- [ ] Tested the endpoint in browser or Postman
- [ ] Verified response format matches your needs
- [ ] Implemented error handling for network failures
- [ ] Added loading state while fetching data

## 📚 Full Documentation

For more details, see:
- `ORGANIZATION_SELECTOR_GUIDE.md` - Complete implementation guide
- `test_organization_selectors_interactive.py` - Live examples
- `http://localhost:8000/docs` - API documentation

---

**Last Updated**: February 11, 2026  
**Status**: ✅ All endpoints tested and working
