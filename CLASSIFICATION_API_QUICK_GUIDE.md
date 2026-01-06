# Classification Service API - Quick Visual Guide

## Request Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND APPLICATION                      │
│  (Streamlit UI / React / Vue / etc.)                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
    ┌─────────┐      ┌─────────┐      ┌──────────────┐
    │ NER API │      │ CLASS   │      │ REFERENCE    │
    │ EXTRACT │      │ CLASSIFY│      │ DATA         │
    └────┬────┘      └────┬────┘      └──────┬───────┘
         │                │                  │
         │ Extract        │ Classify         │ Get Dropdowns
         │ Entities       │ Text             │
         │                │                  │
         ▼                ▼                  ▼
    ┌──────────────────────────────────────────────────┐
    │          FASTAPI BACKEND ROUTERS                 │
    │                                                  │
    │  /api/ner/extract                                │
    │  /api/classification/classify                    │
    │  /api/reference/[domains|categories|...]         │
    └──────────────┬───────────────────────────────────┘
                   │
         ┌─────────┼──────────┐
         │         │          │
         ▼         ▼          ▼
    ┌────────┐ ┌────────┐ ┌──────────┐
    │  NER   │ │ CLASS  │ │ REFERENCE│
    │SERVICE │ │SERVICE │ │SERVICE   │
    └────┬───┘ └───┬────┘ └────┬─────┘
         │         │           │
         ▼         ▼           ▼
    ┌──────────────────────────────────┐
    │    ML MODELS & DATABASE           │
    │                                   │
    │  - NER Model (Arabic entities)    │
    │  - Classification Model (8-way)   │
    │  - Reference Data Tables          │
    └──────────────────────────────────┘
```

---

## Typical User Workflow

```
┌────────────────────────────────────────────────────────┐
│ 1. PAGE LOADS                                          │
│    └─> GET /api/reference/domains                    │
│    └─> GET /api/reference/severity-levels            │
│    └─> GET /api/reference/stages                     │
│    └─> GET /api/reference/harm-levels                │
└────────┬───────────────────────────────────────────────┘
         │ ✓ Dropdowns populated
         ▼
┌────────────────────────────────────────────────────────┐
│ 2. USER TYPES FEEDBACK TEXT                            │
│    └─> Display in text area                           │
└────────┬───────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ 3. USER CLICKS "AUTO-CLASSIFY"                         │
│    ├─> POST /api/ner/extract                          │
│    │   └─> Returns entities (patient, doctor, dept)   │
│    │       └─> Highlight in text                      │
│    │                                                   │
│    └─> POST /api/classification/classify              │
│        └─> Returns 8 classifications with IDs         │
│            ├─ domain_id = 1                           │
│            ├─ category_id = 12                        │
│            ├─ subcategory_id = 45                     │
│            ├─ classification_id = 102                 │
│            ├─ severity_level_id = 3                   │
│            ├─ stage_id = 2                            │
│            ├─ harm_level_id = 2                       │
│            └─ improvement_opportunity_type_id = 2     │
└────────┬───────────────────────────────────────────────┘
         │ ✓ Auto-fill dropdowns
         ▼
┌────────────────────────────────────────────────────────┐
│ 4. DROPDOWNS AUTO-FILLED                               │
│    ├─ Domain = "Clinical" (ID: 1)                     │
│    ├─ Category = "Medication Error" (ID: 12)          │
│    ├─ SubCategory = "Wrong Dosage" (ID: 45)           │
│    ├─ Classification = "Prescription Error" (ID: 102) │
│    ├─ Severity = "High" (ID: 3)                       │
│    ├─ Stage = "Care" (ID: 2)                          │
│    ├─ Harm Level = "Moderate" (ID: 2)                 │
│    └─ Opportunity Type = "Red Flag" (ID: 2)           │
└────────┬───────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ 5. USER CAN:                                           │
│    ├─ ACCEPT suggestions (use AI values)              │
│    ├─ MODIFY values (override AI with manual select)  │
│    └─ SAVE record (post to /api/records/add)          │
└────────────────────────────────────────────────────────┘
```

---

## API Endpoint Details

### 🔍 NER Extraction
```
POST /api/ner/extract

Request:
{
  "text": "المريض أحمد يشكو من ألم في البطن"
}

Response:
{
  "success": true,
  "entities": [
    { "text": "أحمد", "type": "PERSON", "label": "Patient Name" },
    { "text": "ألم في البطن", "type": "MEDICAL_CONDITION", "label": "Symptom" }
  ]
}
```

**Purpose:** Extract important entities from text (names, symptoms, departments)

---

### 🎯 Classification
```
POST /api/classification/classify

Request:
{
  "text": "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج",
  "explain": true
}

Response:
{
  "success": true,
  "classifications": {
    "domain_id": 1,              ← Pick from domain dropdown
    "domain": "Clinical",
    "category_id": 12,           ← Pick from category dropdown
    "category": "Medication Error",
    "subcategory_id": 45,        ← Pick from subcategory dropdown
    "subcategory": "Wrong Dosage",
    "classification_id": 102,    ← Pick from classification dropdown
    "classification": "Prescription Error",
    "severity_level_id": 3,      ← Pick from severity dropdown
    "severity_level": "High",
    "stage_id": 2,               ← Pick from stage dropdown
    "stage": "Care",
    "harm_level_id": 2,          ← Pick from harm level dropdown
    "harm_level": "Moderate",
    "improvement_opportunity_type_id": 2,  ← Pick from improvement type
    "improvement_opportunity_type": "Red Flag"
  ]
}
```

**Purpose:** Auto-classify text into 8 categories (get AI suggestions)

---

### 📋 Reference Data (Hierarchical)

```
LEVEL 1: Get Domains
GET /api/reference/domains
Response: [{ id: 1, name_en: "Clinical" }, ...]

        ↓ User selects domain_id = 1

LEVEL 2: Get Categories (filtered by domain)
GET /api/reference/categories?domain_id=1
Response: [{ id: 12, domain_id: 1, name_en: "Medication Error" }, ...]

        ↓ User selects category_id = 12

LEVEL 3: Get Subcategories (filtered by category)
GET /api/reference/subcategories?category_id=12
Response: [{ id: 45, category_id: 12, name_en: "Wrong Dosage" }, ...]

        ↓ User selects subcategory_id = 45

LEVEL 4: Get Classifications (filtered by subcategory)
GET /api/reference/classifications?subcategory_id=45
Response: [{ id: 102, subcategory_id: 45, name_en: "Prescription Error" }, ...]
```

**Purpose:** Provide hierarchical dropdown filtering

---

## Response Status Codes

| Code | Status | Meaning |
|------|--------|---------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid input (e.g., empty text) |
| 500 | Internal Error | Model/server error |

---

## Example: Complete Frontend Integration

### HTML Structure
```html
<div id="feedback-section">
  <!-- 1. Text Input -->
  <textarea id="feedback-text" placeholder="Enter Arabic feedback..."></textarea>
  
  <!-- 2. Buttons -->
  <button id="classify-btn">Extract & Classify</button>
  <button id="save-btn">Save Record</button>
  
  <!-- 3. Extracted Entities -->
  <div id="entities-display" style="display:none;">
    <h3>Extracted Information:</h3>
    <p>Patient: <span id="patient-name"></span></p>
    <p>Doctor: <span id="doctor-name"></span></p>
    <p>Department: <span id="department"></span></p>
  </div>
  
  <!-- 4. Classification Dropdowns -->
  <select id="domain" onchange="loadCategories()">
    <option>Select Domain...</option>
  </select>
  
  <select id="category" onchange="loadSubcategories()">
    <option>Select Category...</option>
  </select>
  
  <select id="subcategory" onchange="loadClassifications()">
    <option>Select Subcategory...</option>
  </select>
  
  <select id="classification">
    <option>Select Classification...</option>
  </select>
  
  <select id="severity">
    <option>Select Severity...</option>
  </select>
  
  <select id="stage">
    <option>Select Stage...</option>
  </select>
  
  <select id="harm-level">
    <option>Select Harm Level...</option>
  </select>
</div>
```

### JavaScript Logic
```javascript
// 1. Load static reference data on page load
async function initializeForm() {
  const domains = await fetch('/api/reference/domains').then(r => r.json());
  const severities = await fetch('/api/reference/severity-levels').then(r => r.json());
  const stages = await fetch('/api/reference/stages').then(r => r.json());
  const harms = await fetch('/api/reference/harm-levels').then(r => r.json());
  
  populateSelect('domain', domains.domains);
  populateSelect('severity', severities.severity_levels);
  populateSelect('stage', stages.stages);
  populateSelect('harm-level', harms.harm_levels);
}

// 2. Classification button click
document.getElementById('classify-btn').addEventListener('click', async () => {
  const text = document.getElementById('feedback-text').value;
  if (!text.trim()) return alert('Please enter text');
  
  // Extract entities
  const nerResult = await fetch('/api/ner/extract', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  }).then(r => r.json());
  
  if (nerResult.success) {
    displayEntities(nerResult.entities);
  }
  
  // Classify text
  const classResult = await fetch('/api/classification/classify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, explain: true })
  }).then(r => r.json());
  
  if (classResult.success) {
    const c = classResult.classifications;
    document.getElementById('domain').value = c.domain_id;
    loadCategories(); // Reload categories with new domain
    
    setTimeout(() => {
      document.getElementById('category').value = c.category_id;
      loadSubcategories();
    }, 100);
    
    setTimeout(() => {
      document.getElementById('subcategory').value = c.subcategory_id;
      loadClassifications();
    }, 200);
    
    setTimeout(() => {
      document.getElementById('classification').value = c.classification_id;
      document.getElementById('severity').value = c.severity_level_id;
      document.getElementById('stage').value = c.stage_id;
      document.getElementById('harm-level').value = c.harm_level_id;
    }, 300);
  }
});

// 3. Cascade filtering
async function loadCategories() {
  const domainId = document.getElementById('domain').value;
  if (!domainId) return;
  
  const result = await fetch(`/api/reference/categories?domain_id=${domainId}`)
    .then(r => r.json());
  populateSelect('category', result.categories);
}

async function loadSubcategories() {
  const categoryId = document.getElementById('category').value;
  if (!categoryId) return;
  
  const result = await fetch(`/api/reference/subcategories?category_id=${categoryId}`)
    .then(r => r.json());
  populateSelect('subcategory', result.subcategories);
}

async function loadClassifications() {
  const subcategoryId = document.getElementById('subcategory').value;
  if (!subcategoryId) return;
  
  const result = await fetch(`/api/reference/classifications?subcategory_id=${subcategoryId}`)
    .then(r => r.json());
  populateSelect('classification', result.classifications);
}

// Helper functions
function populateSelect(selectId, options) {
  const select = document.getElementById(selectId);
  const currentValue = select.value;
  select.innerHTML = '<option value="">Select...</option>';
  options.forEach(opt => {
    const optionEl = document.createElement('option');
    optionEl.value = opt.id;
    optionEl.textContent = opt.name_en || opt.name;
    select.appendChild(optionEl);
  });
}

function displayEntities(entities) {
  const display = document.getElementById('entities-display');
  
  const patient = entities.find(e => e.type === 'PERSON')?.text || 'N/A';
  const department = entities.find(e => e.type === 'LOCATION')?.text || 'N/A';
  
  document.getElementById('patient-name').textContent = patient;
  document.getElementById('department').textContent = department;
  
  display.style.display = 'block';
}
```

---

## Common Issues & Solutions

### Issue: Dropdown doesn't populate after classification
**Solution:** Add delay before setting value in cascade
```javascript
setTimeout(() => {
  document.getElementById('category').value = classResult.classifications.category_id;
}, 100); // Wait for options to load
```

### Issue: Classifications API returns empty list
**Solution:** Make sure you're passing `subcategory_id` as query parameter
```javascript
// ✓ Correct
const result = await fetch(`/api/reference/classifications?subcategory_id=45`);

// ✗ Wrong
const result = await fetch(`/api/reference/classifications`);
```

### Issue: NER/Classification returns error
**Solution:** Check response.ok first
```javascript
const response = await fetch('/api/classification/classify', { /* ... */ });

if (!response.ok) {
  const error = await response.json();
  console.error('Error:', error.detail.message_ar);
  return;
}

const result = await response.json();
```

---

## Summary

✅ **Fixed Parameter Name** in classification service (was `text_1`, now `patient_text`)
✅ **NER Endpoint** extracts entities (people, conditions, departments)  
✅ **Classification Endpoint** returns 8 categories with IDs
✅ **Reference Endpoints** provide hierarchical dropdown data with filtering
✅ **Frontend** should cascade through domain → category → subcategory → classification

---

**Last Updated:** January 5, 2026
**Status:** Ready for Frontend Integration
