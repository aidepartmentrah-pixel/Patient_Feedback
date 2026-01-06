# Backend Classification Service Endpoints

Complete API documentation for the Classification and NER services used by the frontend.

---

## Overview

The backend provides 5 main endpoints for the classification workflow:

1. **Classification Endpoint** - Classify text into 8 categories
2. **NER Extraction Endpoint** - Extract named entities
3. **Reference Endpoints** - Get dropdown options (dynamic)

---

## 1. CLASSIFICATION ENDPOINT

### `POST /api/classification/classify`

Classifies Arabic patient feedback into 8 categories using AI models.

**Request:**
```json
{
  "text": "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج",
  "explain": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | ✅ Yes | Arabic patient feedback text (minimum 1 character) |
| `explain` | boolean | ✅ Yes | Whether to include explanation (default: true) |

**Response (Success - 200 OK):**
```json
{
  "success": true,
  "text": "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج",
  "classifications": {
    "domain_id": 1,
    "domain": "Clinical",
    "category_id": 12,
    "category": "Medication Error",
    "subcategory_id": 45,
    "subcategory": "Wrong Dosage",
    "classification_id": 102,
    "classification": "Prescription Error",
    "severity_level_id": 3,
    "severity_level": "High",
    "stage_id": 2,
    "stage": "Care",
    "harm_level_id": 2,
    "harm_level": "Moderate",
    "improvement_opportunity_type_id": 2,
    "improvement_opportunity_type": "Red Flag"
  }
}
```

**Response (Error - 400/500):**
```json
{
  "detail": {
    "error": "CLASSIFICATION_FAILED",
    "message": "Classification failed: [error message]",
    "message_ar": "فشل التصنيف: [الرسالة]"
  }
}
```

**What This Returns:**
- ✅ 8 classification categories with IDs and names
- ✅ Each category includes both English and Arabic names
- ✅ Used to populate form fields with AI predictions

**When to Call:**
- User enters patient feedback text
- Need to auto-populate classification dropdowns
- Want AI suggestions for categorization

---

## 2. NER EXTRACTION ENDPOINT

### `POST /api/ner/extract`

Extracts named entities (people, places, medications, etc.) from Arabic text.

**Request:**
```json
{
  "text": "المريض أحمد محمد يشكو من ألم في البطن وتم فحصه بواسطة الدكتور خالد في قسم الطوارئ"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | ✅ Yes | Arabic patient feedback text |

**Response (Success - 200 OK):**
```json
{
  "success": true,
  "text": "المريض أحمد محمد يشكو من ألم في البطن وتم فحصه بواسطة الدكتور خالد في قسم الطوارئ",
  "entities": [
    {
      "text": "أحمد محمد",
      "type": "PERSON",
      "start": 8,
      "end": 16,
      "label": "Patient Name"
    },
    {
      "text": "ألم في البطن",
      "type": "MEDICAL_CONDITION",
      "start": 28,
      "end": 40,
      "label": "Medical Condition"
    },
    {
      "text": "الدكتور خالد",
      "type": "PERSON",
      "start": 58,
      "end": 70,
      "label": "Doctor Name"
    },
    {
      "text": "قسم الطوارئ",
      "type": "LOCATION",
      "start": 72,
      "end": 84,
      "label": "Department"
    }
  ]
}
```

**Entity Types Extracted:**
- `PERSON` - Patient names, Doctor names
- `LOCATION` - Hospital departments, Buildings
- `MEDICAL_CONDITION` - Symptoms, Diseases
- `MEDICATION` - Drug names
- `DATE_TIME` - Dates and times
- `ORGANIZATION` - Hospital names, Departments

**Response (Error - 400/500):**
```json
{
  "detail": {
    "error": "NER_FAILED",
    "message": "NER extraction failed: [error message]",
    "message_ar": "فشل استخراج الكيانات: [الرسالة]"
  }
}
```

**When to Call:**
- User enters patient feedback text
- Need to extract important information (patient name, doctor name, department)
- Want to highlight or tag important entities in the text

---

## 3. REFERENCE DATA ENDPOINTS

These endpoints provide dropdown/lookup data. They can be called once at page load or when filtering.

### 3a. Get Domains

### `GET /api/reference/domains`

Gets all top-level classification domains.

**Request:**
```
GET /api/reference/domains
```

**Response:**
```json
{
  "domains": [
    {
      "id": 1,
      "name_en": "Clinical",
      "name_ar": "سريري"
    },
    {
      "id": 2,
      "name_en": "Management",
      "name_ar": "إداري"
    },
    {
      "id": 3,
      "name_en": "Administrative",
      "name_ar": "إداري"
    }
  ]
}
```

**Use For:**
- Populate domain dropdown (first level)
- No filtering needed

---

### 3b. Get Categories (Filtered by Domain)

### `GET /api/reference/categories?domain_id=1`

Gets categories for a specific domain.

**Request Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `domain_id` | integer | ❌ No | Filter by domain ID (if omitted, returns all) |

**Examples:**
```
GET /api/reference/categories
GET /api/reference/categories?domain_id=1
GET /api/reference/categories?domain_id=2
```

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

**Use For:**
- Populate category dropdown
- Called after domain is selected (use `domain_id` to filter)
- Example: User selects "Clinical" domain → fetch categories for domain_id=1

**How to Implement:**
```javascript
// After user selects domain
const domainId = selectedDomain.id;  // e.g., 1
const response = await fetch(`/api/reference/categories?domain_id=${domainId}`);
const data = await response.json();
populateCategoryDropdown(data.categories);
```

---

### 3c. Get Subcategories (Filtered by Category)

### `GET /api/reference/subcategories?category_id=12`

Gets subcategories for a specific category.

**Request Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `category_id` | integer | ❌ No | Filter by category ID (if omitted, returns all) |

**Examples:**
```
GET /api/reference/subcategories
GET /api/reference/subcategories?category_id=12
GET /api/reference/subcategories?category_id=13
```

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

**Use For:**
- Populate subcategory dropdown
- Called after category is selected
- Example: User selects "Medication Error" category → fetch subcategories for category_id=12

---

### 3d. Get Classifications (Filtered by Subcategory)

### `GET /api/reference/classifications?subcategory_id=45`

Gets classifications (most specific level) for a subcategory.

**Request Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `subcategory_id` | integer | ❌ No | Filter by subcategory ID (if omitted, returns all) |

**Examples:**
```
GET /api/reference/classifications
GET /api/reference/classifications?subcategory_id=45
GET /api/reference/classifications?subcategory_id=46
```

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

**Use For:**
- Populate classification dropdown (most specific)
- Called after subcategory is selected
- Example: User selects "Wrong Dosage" subcategory → fetch classifications for subcategory_id=45

---

## Complete Frontend Workflow

Here's how the frontend should use these endpoints:

### Step 1: Page Load
```javascript
// Load reference data at page initialization
const domains = await fetch('/api/reference/domains').then(r => r.json());
populateDomainDropdown(domains.domains);
```

### Step 2: User Enters Text
```javascript
const feedbackText = document.getElementById('feedback').value;

// Extract named entities (optional but recommended)
const nerResult = await fetch('/api/ner/extract', {
  method: 'POST',
  body: JSON.stringify({ text: feedbackText })
}).then(r => r.json());

// Show extracted entities to user (highlight names, conditions, etc.)
displayExtractedEntities(nerResult.entities);

// Classify the text
const classificationResult = await fetch('/api/classification/classify', {
  method: 'POST',
  body: JSON.stringify({ 
    text: feedbackText,
    explain: true 
  })
}).then(r => r.json());

// Auto-populate classification dropdowns
if (classificationResult.success) {
  selectDropdown('domain', classificationResult.classifications.domain_id);
  selectDropdown('category', classificationResult.classifications.category_id);
  // ... and so on for all 8 fields
}
```

### Step 3: User Selects Domain
```javascript
document.getElementById('domain').addEventListener('change', async (e) => {
  const domainId = e.target.value;
  const categories = await fetch(`/api/reference/categories?domain_id=${domainId}`)
    .then(r => r.json());
  populateCategoryDropdown(categories.categories);
});
```

### Step 4: User Selects Category
```javascript
document.getElementById('category').addEventListener('change', async (e) => {
  const categoryId = e.target.value;
  const subcategories = await fetch(`/api/reference/subcategories?category_id=${categoryId}`)
    .then(r => r.json());
  populateSubcategoryDropdown(subcategories.subcategories);
});
```

### Step 5: User Selects Subcategory
```javascript
document.getElementById('subcategory').addEventListener('change', async (e) => {
  const subcategoryId = e.target.value;
  const classifications = await fetch(`/api/reference/classifications?subcategory_id=${subcategoryId}`)
    .then(r => r.json());
  populateClassificationDropdown(classifications.classifications);
});
```

---

## Other Reference Endpoints (for dropdowns)

These return static reference data:

### Get Severity Levels
```
GET /api/reference/severity-levels
```
Response:
```json
{
  "severity_levels": [
    { "id": 1, "name_en": "Low", "name_ar": "منخفض" },
    { "id": 2, "name_en": "Medium", "name_ar": "متوسط" },
    { "id": 3, "name_en": "High", "name_ar": "عالي" }
  ]
}
```

### Get Stages
```
GET /api/reference/stages
```
Response:
```json
{
  "stages": [
    { "id": 1, "name_en": "Admission", "name_ar": "القبول" },
    { "id": 2, "name_en": "Care", "name_ar": "الرعاية" },
    { "id": 3, "name_en": "Discharge", "name_ar": "الخروج" }
  ]
}
```

### Get Harm Levels
```
GET /api/reference/harm-levels
```
Response:
```json
{
  "harm_levels": [
    { "id": 1, "name_en": "None", "name_ar": "لا يوجد" },
    { "id": 2, "name_en": "Moderate", "name_ar": "معتدل" },
    { "id": 3, "name_en": "Severe", "name_ar": "شديد" }
  ]
}
```

---

## Error Handling

All endpoints return errors in consistent format:

**400 Bad Request (Invalid Input):**
```json
{
  "detail": {
    "error": "CLASSIFICATION_FAILED",
    "message": "Text is required for classification",
    "message_ar": "النص مطلوب للتصنيف"
  }
}
```

**500 Internal Server Error:**
```json
{
  "detail": {
    "error": "INTERNAL_ERROR",
    "message": "An error occurred: [error details]",
    "message_ar": "حدث خطأ: [تفاصيل الخطأ]"
  }
}
```

**Frontend Error Handling:**
```javascript
try {
  const response = await fetch('/api/classification/classify', { /* ... */ });
  
  if (!response.ok) {
    const error = await response.json();
    console.error('Error:', error.detail.message);
    showErrorMessage(error.detail.message_ar);
    return;
  }
  
  const result = await response.json();
  if (!result.success) {
    showErrorMessage(result.message_ar);
    return;
  }
  
  processClassificationResult(result);
} catch (error) {
  showErrorMessage('Network error: ' + error.message);
}
```

---

## Batch Endpoints (Optional)

For processing multiple records at once:

### `POST /api/classification/classify-batch`
```json
{
  "texts": [
    "Text 1",
    "Text 2",
    "Text 3"
  ],
  "explain": false
}
```

### `POST /api/ner/extract-batch`
```json
{
  "texts": [
    "Text 1",
    "Text 2",
    "Text 3"
  ]
}
```

---

## Summary Table

| Endpoint | Method | Purpose | Parameters | Returns |
|----------|--------|---------|-----------|---------|
| `/api/classification/classify` | POST | Classify text into 8 categories | text, explain | 8 classifications + IDs |
| `/api/ner/extract` | POST | Extract named entities | text | Entities with types |
| `/api/reference/domains` | GET | Get all domains | - | Domain list |
| `/api/reference/categories` | GET | Get categories | domain_id (optional) | Category list |
| `/api/reference/subcategories` | GET | Get subcategories | category_id (optional) | Subcategory list |
| `/api/reference/classifications` | GET | Get classifications | subcategory_id (optional) | Classification list |
| `/api/reference/severity-levels` | GET | Get severity levels | - | Severity list |
| `/api/reference/stages` | GET | Get stages | - | Stage list |
| `/api/reference/harm-levels` | GET | Get harm levels | - | Harm level list |

---

## Best Practices for Frontend

1. **Cache Reference Data** - Load domains, severity levels, stages, harm levels once at app startup
2. **Use Filtering** - Use `domain_id`, `category_id`, `subcategory_id` parameters to get only relevant options
3. **Error Handling** - Always check `response.ok` and `result.success` before using data
4. **User Feedback** - Show loading spinner while waiting for classification/NER
5. **Default Values** - Use AI classification results as default/suggested values
6. **Manual Override** - Allow users to override AI suggestions with manual selection
7. **Optimize Calls** - Don't fetch data unnecessarily; cache and reuse

---

**Last Updated:** January 5, 2026
**API Version:** 1.0
**Status:** Production Ready
