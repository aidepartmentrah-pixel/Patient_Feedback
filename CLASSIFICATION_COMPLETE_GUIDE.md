# Classification Service Endpoints - Complete Documentation

Generated: January 5, 2026  
Status: ✅ All Endpoints Operational

---

## Quick Summary

You have **5 backend endpoints** for the classification workflow:

1. **POST /api/classification/classify** - AI Classification (8 categories)
2. **POST /api/ner/extract** - Named Entity Recognition (extract people, places, conditions)
3. **GET /api/reference/categories** - Hierarchical dropdown (filtered by domain)
4. **GET /api/reference/subcategories** - Hierarchical dropdown (filtered by category)
5. **GET /api/reference/classifications** - Hierarchical dropdown (filtered by subcategory)

---

## Endpoint Details

### 1️⃣ AI Classification

```
POST /api/classification/classify
```

**What it does:**
- Takes Arabic patient feedback text
- Classifies into 8 categories using AI model
- Returns IDs for dropdown auto-fill

**Request:**
```json
{
  "text": "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج",
  "explain": true
}
```

**Response (Success):**
```json
{
  "success": true,
  "text": "...",
  "classifications": {
    "domain_id": 1,                          ← Fill domain dropdown
    "domain": "Clinical",
    "category_id": 12,                       ← Fill category dropdown
    "category": "Medication Error",
    "subcategory_id": 45,                    ← Fill subcategory dropdown
    "subcategory": "Wrong Dosage",
    "classification_id": 102,                ← Fill classification dropdown
    "classification": "Prescription Error",
    "severity_level_id": 3,                  ← Fill severity dropdown
    "severity_level": "High",
    "stage_id": 2,                           ← Fill stage dropdown
    "stage": "Care",
    "harm_level_id": 2,                      ← Fill harm level dropdown
    "harm_level": "Moderate",
    "improvement_opportunity_type_id": 2,    ← Fill improvement type
    "improvement_opportunity_type": "Red Flag"
  }
}
```

**Response (Error):**
```json
{
  "detail": {
    "error": "CLASSIFICATION_FAILED",
    "message": "Classification failed: [error]",
    "message_ar": "فشل التصنيف: [الخطأ]"
  }
}
```

**When to call:** User clicks "Extract & Classify" or types feedback

---

### 2️⃣ Named Entity Recognition (NER)

```
POST /api/ner/extract
```

**What it does:**
- Extracts important entities from Arabic text
- Identifies: people, locations, medical conditions, medications, dates

**Request:**
```json
{
  "text": "المريض أحمد محمد يشكو من ألم في البطن وتم فحصه بواسطة الدكتور خالد في قسم الطوارئ"
}
```

**Response (Success):**
```json
{
  "success": true,
  "text": "...",
  "entities": [
    {
      "text": "أحمد محمد",
      "type": "PERSON",
      "label": "Patient Name",
      "start": 8,
      "end": 16
    },
    {
      "text": "ألم في البطن",
      "type": "MEDICAL_CONDITION",
      "label": "Medical Condition",
      "start": 28,
      "end": 40
    },
    {
      "text": "الدكتور خالد",
      "type": "PERSON",
      "label": "Doctor Name",
      "start": 58,
      "end": 70
    },
    {
      "text": "قسم الطوارئ",
      "type": "LOCATION",
      "label": "Department",
      "start": 72,
      "end": 84
    }
  ]
}
```

**Entity Types:**
- `PERSON` - Patient/doctor names
- `LOCATION` - Hospital, department, building
- `MEDICAL_CONDITION` - Symptoms, diseases
- `MEDICATION` - Drug names
- `DATE_TIME` - Dates, times
- `ORGANIZATION` - Hospital names

**When to call:** User clicks "Extract & Classify" (optional but recommended)

---

### 3️⃣ Get Categories (Hierarchical)

```
GET /api/reference/categories?domain_id=1
```

**What it does:**
- Returns categories for selected domain
- Used to populate category dropdown

**Parameters:**
| Name | Type | Required | Example |
|------|------|----------|---------|
| `domain_id` | integer | ❌ No | `?domain_id=1` |

**Response:**
```json
{
  "categories": [
    {
      "id": 12,
      "domain_id": 1,
      "name_en": "Medication Error",
      "name_ar": "خطأ في الدواء"
    },
    {
      "id": 13,
      "domain_id": 1,
      "name_en": "Delayed Diagnosis",
      "name_ar": "تأخر في التشخيص"
    },
    {
      "id": 14,
      "domain_id": 1,
      "name_en": "Infection Control",
      "name_ar": "مكافحة العدوى"
    }
  ]
}
```

**When to call:** After domain is selected (auto or manual)

---

### 4️⃣ Get Subcategories (Hierarchical)

```
GET /api/reference/subcategories?category_id=12
```

**What it does:**
- Returns subcategories for selected category
- Used to populate subcategory dropdown

**Parameters:**
| Name | Type | Required | Example |
|------|------|----------|---------|
| `category_id` | integer | ❌ No | `?category_id=12` |

**Response:**
```json
{
  "subcategories": [
    {
      "id": 45,
      "category_id": 12,
      "name_en": "Wrong Dosage",
      "name_ar": "جرعة خاطئة"
    },
    {
      "id": 46,
      "category_id": 12,
      "name_en": "Wrong Medication",
      "name_ar": "دواء خاطئ"
    },
    {
      "id": 47,
      "category_id": 12,
      "name_en": "Missed Dose",
      "name_ar": "جرعة منسية"
    }
  ]
}
```

**When to call:** After category is selected (auto or manual)

---

### 5️⃣ Get Classifications (Hierarchical)

```
GET /api/reference/classifications?subcategory_id=45
```

**What it does:**
- Returns classifications for selected subcategory
- Used to populate classification dropdown (most specific)

**Parameters:**
| Name | Type | Required | Example |
|------|------|----------|---------|
| `subcategory_id` | integer | ❌ No | `?subcategory_id=45` |

**Response:**
```json
{
  "classifications": [
    {
      "id": 102,
      "subcategory_id": 45,
      "name_en": "Prescription Error",
      "name_ar": "خطأ في الوصفة"
    },
    {
      "id": 103,
      "subcategory_id": 45,
      "name_en": "Administration Error",
      "name_ar": "خطأ في التطبيق"
    },
    {
      "id": 104,
      "subcategory_id": 45,
      "name_en": "Dispensing Error",
      "name_ar": "خطأ في الصرف"
    }
  ]
}
```

**When to call:** After subcategory is selected (auto or manual)

---

## Frontend Integration Guide

### Initialization (Page Load)
```javascript
// Load static reference data once
fetch('/api/reference/domains').then(r => r.json())
  .then(data => populateDomainDropdown(data.domains));

fetch('/api/reference/severity-levels').then(r => r.json())
  .then(data => populateSeverityDropdown(data.severity_levels));

fetch('/api/reference/stages').then(r => r.json())
  .then(data => populateStageDropdown(data.stages));

fetch('/api/reference/harm-levels').then(r => r.json())
  .then(data => populateHarmLevelDropdown(data.harm_levels));
```

### Auto-Classification (User Clicks Button)
```javascript
const feedbackText = document.getElementById('feedback').value;

// Step 1: Classify
const classRes = await fetch('/api/classification/classify', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: feedbackText, explain: true })
}).then(r => r.json());

if (classRes.success) {
  const c = classRes.classifications;
  
  // Step 2: Auto-fill domain
  setDropdownValue('domain', c.domain_id);
  
  // Step 3: Load categories for this domain
  const catRes = await fetch(`/api/reference/categories?domain_id=${c.domain_id}`)
    .then(r => r.json());
  populateDropdown('category', catRes.categories);
  setDropdownValue('category', c.category_id);
  
  // Step 4: Load subcategories
  const subRes = await fetch(`/api/reference/subcategories?category_id=${c.category_id}`)
    .then(r => r.json());
  populateDropdown('subcategory', subRes.subcategories);
  setDropdownValue('subcategory', c.subcategory_id);
  
  // Step 5: Load classifications
  const classesRes = await fetch(`/api/reference/classifications?subcategory_id=${c.subcategory_id}`)
    .then(r => r.json());
  populateDropdown('classification', classesRes.classifications);
  setDropdownValue('classification', c.classification_id);
  
  // Step 6: Fill remaining fields
  setDropdownValue('severity', c.severity_level_id);
  setDropdownValue('stage', c.stage_id);
  setDropdownValue('harm-level', c.harm_level_id);
}

// Step 7: Extract entities (optional)
const nerRes = await fetch('/api/ner/extract', {
  method: 'POST',
  body: JSON.stringify({ text: feedbackText })
}).then(r => r.json());

if (nerRes.success) {
  displayExtractedEntities(nerRes.entities);
}
```

### Cascade Filtering (User Changes Dropdown)
```javascript
// When domain changes
document.getElementById('domain').addEventListener('change', async (e) => {
  const domainId = e.target.value;
  const catRes = await fetch(`/api/reference/categories?domain_id=${domainId}`)
    .then(r => r.json());
  populateDropdown('category', catRes.categories);
});

// When category changes
document.getElementById('category').addEventListener('change', async (e) => {
  const categoryId = e.target.value;
  const subRes = await fetch(`/api/reference/subcategories?category_id=${categoryId}`)
    .then(r => r.json());
  populateDropdown('subcategory', subRes.subcategories);
});

// When subcategory changes
document.getElementById('subcategory').addEventListener('change', async (e) => {
  const subcategoryId = e.target.value;
  const classRes = await fetch(`/api/reference/classifications?subcategory_id=${subcategoryId}`)
    .then(r => r.json());
  populateDropdown('classification', classRes.classifications);
});
```

---

## Error Codes & Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `400 Bad Request` | Empty text or invalid input | Ensure text is not empty |
| `422 Unprocessable Entity` | Request format error | Check JSON structure |
| `500 Internal Server Error` | Model error or server issue | Check server logs |
| Classifications empty | Subcategory has no classifications | Verify subcategory_id exists |
| Wrong categories loaded | domain_id parameter missing | Use `?domain_id=X` in URL |

---

## Recent Bug Fix ✅

**Issue:** Classification failed with "text_1" parameter error
**Fix:** Changed parameter from `text_1` to `patient_text` in service
**File:** `backend/api/services/classification_service.py` (line 53)
**Status:** ✅ RESOLVED - All endpoints now working correctly

---

## Your Current Flow (From Logs)

```
User Types Feedback
    ↓
Clicks "Extract & Classify"
    ↓
POST /api/classification/classify           ← Classify text
    ↓
POST /api/ner/extract                       ← Extract entities
    ↓
GET /api/reference/categories?domain_id=2   ← Load categories
    ↓
GET /api/reference/subcategories?category_id=3  ← Load subcategories
    ↓
GET /api/reference/classifications?subcategory_id=30  ← Load classifications
    ↓
Dropdowns auto-filled with AI predictions
```

All 5 endpoints are working correctly (200 OK)! ✅

---

## Documentation Files Available

1. **BACKEND_CLASSIFICATION_ENDPOINTS.md** - Complete API documentation
2. **CLASSIFICATION_API_QUICK_GUIDE.md** - Visual guide with examples
3. **FRONTEND_ENDPOINTS_SUMMARY.md** - Summary for frontend integration
4. **BUG_FIX_CLASSIFICATION_PARAMETER.md** - Bug fix details

---

**Status:** ✅ Production Ready
**Last Updated:** January 5, 2026
**All Endpoints:** Operational (200 OK)
