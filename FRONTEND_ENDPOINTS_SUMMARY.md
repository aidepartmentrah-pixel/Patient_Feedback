# Backend Endpoints Summary for Frontend

Based on your logs, here's exactly what's happening:

---

## Your Request Sequence (From Logs)

```
INFO: 127.0.0.1:49361 - "POST /api/classification/classify HTTP/1.1" 200 OK
     ↓
     └─> User clicked "Extract & Classify" button
         └─> Called: POST /api/classification/classify
         └─> Returns: 8 classification categories with IDs
         └─> Frontend auto-fills domain dropdown
         
INFO: 127.0.0.1:49361 - "GET /api/reference/categories?domain_id=2 HTTP/1.1" 200 OK
     ↓
     └─> User changed domain dropdown OR domain auto-selected
     └─> Called: GET /api/reference/categories?domain_id=2
     └─> Returns: Categories for domain_id=2
     └─> Frontend populates category dropdown
     
INFO: 127.0.0.1:49361 - "GET /api/reference/subcategories?category_id=3 HTTP/1.1" 200 OK
     ↓
     └─> User changed category dropdown OR category auto-selected
     └─> Called: GET /api/reference/subcategories?category_id=3
     └─> Returns: Subcategories for category_id=3
     └─> Frontend populates subcategory dropdown
     
INFO: 127.0.0.1:49361 - "GET /api/reference/classifications?subcategory_id=30 HTTP/1.1" 200 OK
     ↓
     └─> User changed subcategory dropdown OR subcategory auto-selected
     └─> Called: GET /api/reference/classifications?subcategory_id=30
     └─> Returns: Classifications for subcategory_id=30
     └─> Frontend populates classification dropdown
     
INFO: 127.0.0.1:61651 - "POST /api/ner/extract HTTP/1.1" 200 OK
     ↓
     └─> User clicked "Extract & Classify" button (or auto-triggered)
     └─> Called: POST /api/ner/extract
     └─> Returns: Extracted entities (patient names, doctors, departments, etc.)
     └─> Frontend highlights or displays extracted information
```

---

## 5 Main Endpoints You're Using

### 1. 🎯 Classification (AI-Powered)
```
POST /api/classification/classify

Input:
{
  "text": "User's Arabic feedback text",
  "explain": true
}

Output:
{
  "success": true,
  "classifications": {
    "domain_id": 2,           ← Auto-fill domain dropdown
    "category_id": 3,         ← Auto-fill category dropdown
    "subcategory_id": 30,     ← Auto-fill subcategory dropdown
    "classification_id": 102, ← Auto-fill classification dropdown
    "severity_level_id": 3,
    "stage_id": 2,
    "harm_level_id": 2,
    "improvement_opportunity_type_id": 2
  }
}

When: User clicks "Extract & Classify" or types feedback
```

### 2. 🔍 Named Entity Recognition (NER)
```
POST /api/ner/extract

Input:
{
  "text": "User's Arabic feedback text"
}

Output:
{
  "success": true,
  "entities": [
    {
      "text": "أحمد",
      "type": "PERSON",
      "label": "Patient Name",
      "start": 8,
      "end": 12
    },
    {
      "text": "قسم الطوارئ",
      "type": "LOCATION",
      "label": "Department",
      "start": 45,
      "end": 56
    }
  ]
}

When: User clicks "Extract & Classify" or types feedback
```

### 3. 📋 Get Categories (Hierarchical)
```
GET /api/reference/categories?domain_id=2

Output:
{
  "categories": [
    { "id": 3, "domain_id": 2, "name_en": "Category Name", "name_ar": "اسم الفئة" },
    { "id": 4, "domain_id": 2, "name_en": "Another Category", "name_ar": "فئة أخرى" }
  ]
}

When: Domain dropdown changes (auto or user)
```

### 4. 📋 Get Subcategories (Hierarchical)
```
GET /api/reference/subcategories?category_id=3

Output:
{
  "subcategories": [
    { "id": 30, "category_id": 3, "name_en": "Subcategory", "name_ar": "فئة فرعية" },
    { "id": 31, "category_id": 3, "name_en": "Another Sub", "name_ar": "فئة فرعية أخرى" }
  ]
}

When: Category dropdown changes (auto or user)
```

### 5. 📋 Get Classifications (Hierarchical)
```
GET /api/reference/classifications?subcategory_id=30

Output:
{
  "classifications": [
    { "id": 102, "subcategory_id": 30, "name_en": "Classification", "name_ar": "تصنيف" },
    { "id": 103, "subcategory_id": 30, "name_en": "Another Class", "name_ar": "تصنيف آخر" }
  ]
}

When: Subcategory dropdown changes (auto or user)
```

---

## What Each Endpoint Does

| Endpoint | Purpose | Triggered By |
|----------|---------|--------------|
| POST `/api/classification/classify` | **AI Classification** - Classify text into 8 categories automatically | User types feedback + clicks "Extract & Classify" |
| POST `/api/ner/extract` | **Entity Extraction** - Find patient names, doctors, departments in text | User types feedback + clicks "Extract & Classify" |
| GET `/api/reference/categories?domain_id=X` | **Get Dropdown Options** - Fetch categories for selected domain | Domain dropdown changes or auto-selected by classification |
| GET `/api/reference/subcategories?category_id=X` | **Get Dropdown Options** - Fetch subcategories for selected category | Category dropdown changes or auto-selected by classification |
| GET `/api/reference/classifications?subcategory_id=X` | **Get Dropdown Options** - Fetch classifications for selected subcategory | Subcategory dropdown changes or auto-selected by classification |

---

## Frontend Implementation Checklist

✅ **Step 1: User Types Feedback**
- Display in textarea

✅ **Step 2: User Clicks "Extract & Classify"**
- Call: POST `/api/classification/classify` with text
- Call: POST `/api/ner/extract` with text (optional but recommended)

✅ **Step 3: Auto-Fill Dropdowns**
- Domain dropdown = classification_id from response
- Then call: GET `/api/reference/categories?domain_id=X`
- Category dropdown = category_id from response
- Then call: GET `/api/reference/subcategories?category_id=X`
- Subcategory dropdown = subcategory_id from response
- Then call: GET `/api/reference/classifications?subcategory_id=X`
- Classification dropdown = classification_id from response
- Severity dropdown = severity_level_id from response
- Stage dropdown = stage_id from response
- Harm Level dropdown = harm_level_id from response

✅ **Step 4: Display Extracted Entities** (Optional)
- Show patient name, doctor name, department from NER response

✅ **Step 5: Allow User to Override**
- User can change any dropdown value manually
- Each change triggers cascade (e.g., changing category reloads subcategories)

✅ **Step 6: Save Record**
- Collect all selected values
- POST to `/api/records/add` (insert endpoint)

---

## Example: Frontend Code Flow

```javascript
// ========== INITIALIZATION ==========
window.addEventListener('load', () => {
  // Load static data
  loadDomains();      // GET /api/reference/domains
  loadSeverities();   // GET /api/reference/severity-levels
  loadStages();       // GET /api/reference/stages
  loadHarmLevels();   // GET /api/reference/harm-levels
});

// ========== CLASSIFICATION TRIGGER ==========
document.getElementById('classify-btn').addEventListener('click', async () => {
  const text = document.getElementById('feedback-text').value;
  
  // REQUEST 1: Classification
  const classRes = await fetch('/api/classification/classify', {
    method: 'POST',
    body: JSON.stringify({ text, explain: true })
  }).then(r => r.json());
  
  if (classRes.success) {
    const c = classRes.classifications;
    // Auto-fill: domain, category, subcategory, classification, severity, stage, harm_level
    document.getElementById('domain').value = c.domain_id;
    
    // REQUEST 2: Get categories for this domain
    const catRes = await fetch(`/api/reference/categories?domain_id=${c.domain_id}`)
      .then(r => r.json());
    populateDropdown('category', catRes.categories);
    document.getElementById('category').value = c.category_id;
    
    // REQUEST 3: Get subcategories for this category
    const subRes = await fetch(`/api/reference/subcategories?category_id=${c.category_id}`)
      .then(r => r.json());
    populateDropdown('subcategory', subRes.subcategories);
    document.getElementById('subcategory').value = c.subcategory_id;
    
    // REQUEST 4: Get classifications for this subcategory
    const classesRes = await fetch(`/api/reference/classifications?subcategory_id=${c.subcategory_id}`)
      .then(r => r.json());
    populateDropdown('classification', classesRes.classifications);
    document.getElementById('classification').value = c.classification_id;
    
    // Fill remaining fields
    document.getElementById('severity').value = c.severity_level_id;
    document.getElementById('stage').value = c.stage_id;
    document.getElementById('harm-level').value = c.harm_level_id;
  }
  
  // REQUEST 5: NER Extraction (optional)
  const nerRes = await fetch('/api/ner/extract', {
    method: 'POST',
    body: JSON.stringify({ text })
  }).then(r => r.json());
  
  if (nerRes.success) {
    displayExtractedEntities(nerRes.entities);
  }
});

// ========== CASCADE FILTERING ==========
document.getElementById('domain').addEventListener('change', async (e) => {
  const domainId = e.target.value;
  if (!domainId) return;
  
  // REQUEST: Get categories for new domain
  const catRes = await fetch(`/api/reference/categories?domain_id=${domainId}`)
    .then(r => r.json());
  populateDropdown('category', catRes.categories);
  document.getElementById('category').value = '';
  document.getElementById('subcategory').value = '';
  document.getElementById('classification').value = '';
});

// Similar for category, subcategory...
```

---

## Summary

Your frontend is correctly using:

1. **Classification API** - Get AI predictions for 8 categories
2. **NER API** - Extract entities from text
3. **Cascade Reference APIs** - Filter dropdowns hierarchically (domain → category → subcategory → classification)

**The fix we made:**
- Changed parameter name in classification service from `text_1` to `patient_text`
- This allows the classification endpoint to work correctly

**Status:** ✅ All endpoints are working correctly (200 OK responses)

---

**Last Updated:** January 5, 2026
**Status:** Production Ready
