# Insert Page Frontend Implementation - API Integration Guide

## Base URL
```
http://127.0.0.1:8000
```

## API Endpoints Summary

### 1. Create New Record
```
POST /api/records/add
Content-Type: application/json
```

**Required Fields:**
- `complaint_text` (string) - Full complaint description
- `feedback_received_date` (date: YYYY-MM-DD) - Date feedback was received
- `domain_id` (integer) - Domain ID
- `category_id` (integer) - Category ID  
- `severity_id` (integer) - Severity level ID

**Optional Fields:**
- `immediate_action`, `taken_action` (string)
- `issuing_department_id`, `target_department_id`, `source_id` (integer)
- `in_out` (string: "IN" or "OUT")
- `worker_type` (string)
- `patient_name`, `doctor_name` (string)
- `subcategory_id`, `classification_id`, `stage_id`, `harm_id` (integer)
- `improvement_type` (integer: 0 or 1)

**Example Request:**
```json
{
  "complaint_text": "تأخر كبير في تشخيص الحالة الطارئة",
  "feedback_received_date": "2024-12-15",
  "domain_id": 1,
  "category_id": 12,
  "severity_id": 2,
  "patient_name": "أحمد محمد"
}
```

**Success Response:**
```json
{
  "success": true,
  "message": "Record created successfully",
  "record_id": "REC-2024-0156",
  "id": 156,
  "status_id": 3,
  "created_at": "2024-12-17T15:30:00"
}
```

### 2. Get Reference Data (For Dropdowns)

All GET requests return JSON with array of objects containing `id`, `name_en`, `name_ar`.

```
GET /api/reference/departments       # All departments
GET /api/reference/sources           # Feedback sources
GET /api/reference/domains           # Top-level domains
GET /api/reference/severity-levels   # Severity levels
GET /api/reference/stages            # Care stages
GET /api/reference/harm-levels       # Harm levels
```

**Hierarchical Endpoints (with filtering):**
```
GET /api/reference/categories?domain_id=1           # Categories for domain 1
GET /api/reference/subcategories?category_id=12     # Subcategories for category 12
GET /api/reference/classifications?subcategory_id=45 # Classifications for subcategory 45
```

**Single Call Option:**
```
GET /api/reference/all  # Returns all reference data at once
```

### 3. NER Extraction (Optional - for auto-fill)

```
POST /api/ner/extract
Content-Type: application/json
```

**Request:**
```json
{
  "text": "المريض أحمد محمد يشكو من ألم"
}
```

**Response:**
```json
{
  "success": true,
  "patient_name": "أحمد محمد",
  "doctor_name": "",
  "entities": {...}
}
```

### 4. Classification (Optional - for AI suggestions)

```
POST /api/classification/classify
Content-Type: application/json
```

**Request:**
```json
{
  "text": "تأخر كبير في تشخيص الحالة الطارئة",
  "explain": false
}
```

**Response:**
```json
{
  "success": true,
  "classifications": {
    "domain": {"id": 1, "name": "Clinical", "confidence": 0.95},
    "category": {"id": 12, "name": "Delayed Diagnosis", "confidence": 0.89},
    "severity_level": {"id": 2, "name": "Medium", "confidence": 0.87}
  }
}
```

### 5. STT (Optional - for voice input)

```
POST /api/stt/transcribe
Content-Type: multipart/form-data
```

**Form Data:**
- `audio` (file) - Audio file (MP3, WAV, M4A, etc.)

**Response:**
```json
{
  "success": true,
  "text": "المريض يشكو من ألم شديد",
  "language": "ar"
}
```

## Implementation Flow

### Page Load:
1. Fetch all reference data: `GET /api/reference/all`
2. Populate all static dropdowns (departments, sources, domains, severity, stages, harm)

### User Interactions:
1. **Domain selection** → Fetch categories: `GET /api/reference/categories?domain_id={selected}`
2. **Category selection** → Fetch subcategories: `GET /api/reference/subcategories?category_id={selected}`
3. **Subcategory selection** → Fetch classifications: `GET /api/reference/classifications?subcategory_id={selected}`

### Optional AI Features:
1. **NER Button** → Call `/api/ner/extract` with complaint text → Pre-fill patient/doctor names
2. **Auto-classify Button** → Call `/api/classification/classify` → Suggest domain/category/severity
3. **Voice Input** → Record audio → Call `/api/stt/transcribe` → Set complaint text

### Form Submission:
1. Validate required fields (complaint_text, feedback_received_date, domain_id, category_id, severity_id)
2. POST to `/api/records/add`
3. On success: Show confirmation with `record_id` and redirect/reset form
4. On error: Display error message for specific field

## Key Implementation Notes

- **Hierarchical Cascading**: When domain changes, clear category/subcategory/classification selections
- **Date Format**: Use ISO format (YYYY-MM-DD) for `feedback_received_date`
- **Error Handling**: Backend returns `field` property indicating which field has error
- **Optional Fields**: All fields except the 5 required ones are optional
- **Status**: All new records auto-assigned `status_id = 3` (In Progress)
- **Arabic Support**: Ensure UTF-8 encoding for all Arabic text

## Error Response Format
```json
{
  "detail": {
    "error": "VALIDATION_ERROR",
    "message": "Category ID 999 does not exist",
    "message_ar": "الفئة رقم 999 غير موجودة",
    "field": "category_id"
  }
}
```

Display `message` or `message_ar` to user and highlight the `field` with error.
