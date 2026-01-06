# POST /api/records/add - Complete Endpoint Specification

## Endpoint Overview
```
POST http://0.0.0.0:8000/api/records/add
Content-Type: application/json
```

---

## 📋 REQUIRED FIELDS (Must be present and non-null)

| Field Name | Type | Constraints | Example |
|---|---|---|---|
| `complaint_text` | string | Min length: 1 | "تأخر كبير في التشخيص" |
| `feedback_received_date` | date | Format: YYYY-MM-DD | "2026-01-05" |
| `domain_id` | integer | > 0 | 1 |
| `category_id` | integer | > 0 | 6 |
| `severity_id` | integer | > 0 | 1 |

**Validation:**
- If ANY required field is missing → 400 error
- If ANY required field is null → 400 error
- If domain_id/category_id/severity_id don't exist in DB → 400 error

---

## 🎯 OPTIONAL FIELDS (Can be omitted or null)

### Text Content
| Field Name | Type | Constraints | Example |
|---|---|---|---|
| `immediate_action` | string | None | "تم توفير الرعاية الفورية" |
| `taken_action` | string | None | "تم اتخاذ إجراءات تصحيحية" |

### Classification Hierarchy (Optional but recommended)
| Field Name | Type | Constraints | Example |
|---|---|---|---|
| `subcategory_id` | integer | > 0 | 19 |
| `classification_id` | integer | > 0 | 132 |
| `stage_id` | integer | > 0 | 1 |
| `harm_id` | integer | > 0 | 1 |

**Validation:**
- If provided, subcategory_id must belong to category_id
- If provided, classification_id must belong to subcategory_id

### Patient/Entity Information
| Field Name | Type | Constraints | Example |
|---|---|---|---|
| `patient_name` | string | None | "أحمد محمد علي" |
| `doctors` | array of objects | None | [{"doctor_id": 45, "doctor_name": "د. خالد"}] |
| `employees` | array of objects | None | [{"employee_id": 12, "employee_name": "علي حسن"}] |

### Settings & Metadata
| Field Name | Type | Constraints | Example |
|---|---|---|---|
| `issuing_department_id` | integer | > 0 | 1 |
| `target_department_ids` | array of integers | None | [1, 2, 3] |
| `source_id` | integer | > 0 | 1 |
| `building_id` | integer | > 0 | 1 |
| `explanation_status_id` | integer | > 0 | 1 |
| `is_inpatient` | boolean | Default: true | true |
| `worker_type` | string | None | "Doctor" |
| `improvement_type` | integer | 0 or 1 | 0 |

### **NEW ML Training Fields** ⭐
| Field Name | Type | Constraints | Example |
|---|---|---|---|
| `feedback_type` | integer | 1-4 | 1 |
| `improvement_opportunity_type` | integer | 1-3 | 2 |
| `classification_ar` | float | 0.0-10.0 | 8.5 |
| `classification_en` | integer | >= 0 | 5 |

**Feedback Type Values:**
- `1` = Improvement Opportunity
- `2` = Notice
- `3` = Critique Suggestion
- `4` = Other

**Improvement Opportunity Type Values:**
- `1` = Ordinary
- `2` = Red Flag
- `3` = Never Event

---

## ✅ SUCCESS RESPONSE (HTTP 200)

```json
{
  "success": true,
  "message": "Record created successfully",
  "message_ar": "تم إنشاء السجل بنجاح",
  "record_id": "REC-2026-0053",
  "id": 53,
  "status_id": 3,
  "created_at": "2026-01-05T15:30:45.123456"
}
```

**Field Explanations:**
- `success`: Always true for success
- `message`: English success message
- `message_ar`: Arabic success message
- `record_id`: Human-readable record ID (format: REC-YYYY-XXXX)
- `id`: Database auto-increment ID
- `status_id`: Always 3 (In Progress)
- `created_at`: ISO timestamp when record was created

---

## ❌ ERROR RESPONSES

### 400 Bad Request - Validation Error
```json
{
  "detail": {
    "error": "VALIDATION_ERROR",
    "message": "Complaint Text is required",
    "message_ar": "نص الشكوى مطلوب",
    "field": "complaint_text"
  }
}
```

### 400 Bad Request - Invalid Reference
```json
{
  "detail": {
    "error": "INVALID_REFERENCE",
    "message": "Severity ID 999 does not exist",
    "message_ar": "معرف الخطورة 999 غير موجود",
    "field": "severity_id"
  }
}
```

### 400 Bad Request - Relationship Violation
```json
{
  "detail": {
    "error": "VALIDATION_ERROR",
    "message": "Selected subcategory does not belong to the selected category",
    "message_ar": "الفئة الفرعية المختارة لا تنتمي للفئة المختارة",
    "field": "subcategory_id"
  }
}
```

### 409 Conflict - Duplicate Request
```json
{
  "detail": {
    "error": "CONFLICT",
    "message": "A record with this ID already exists",
    "message_ar": "سجل بهذا المعرف موجود بالفعل"
  }
}
```

### 500 Internal Server Error
```json
{
  "detail": {
    "error": "INTERNAL_ERROR",
    "message": "An error occurred: [error details]",
    "message_ar": "حدث خطأ: [تفاصيل الخطأ]"
  }
}
```

---

## 📝 COMPLETE REQUEST EXAMPLE

```json
{
  "complaint_text": "تأخر كبير في تشخيص الحالة الطارئة الخطيرة",
  "feedback_received_date": "2026-01-05",
  "domain_id": 1,
  "category_id": 6,
  "severity_id": 1,
  "immediate_action": "تم توفير الرعاية الفورية",
  "taken_action": "تم اتخاذ إجراءات تصحيحية فورية",
  "patient_name": "أحمد محمد علي",
  "issuing_department_id": 1,
  "building_id": 1,
  "explanation_status_id": 1,
  "subcategory_id": 19,
  "classification_id": 132,
  "stage_id": 1,
  "harm_id": 1,
  "is_inpatient": true,
  "feedback_type": 1,
  "improvement_opportunity_type": 2,
  "classification_ar": 8.5,
  "classification_en": 5,
  "doctors": [
    {
      "doctor_id": 45,
      "doctor_name": "د. خالد حسن"
    }
  ],
  "target_department_ids": [1, 2]
}
```

---

## 📝 MINIMAL REQUEST EXAMPLE (Only Required Fields)

```json
{
  "complaint_text": "شكوى عامة",
  "feedback_received_date": "2026-01-05",
  "domain_id": 1,
  "category_id": 6,
  "severity_id": 1
}
```

---

## 🔍 FIELD VALIDATION DETAILS

### complaint_text
- Type: `string`
- Required: **YES**
- Min Length: 1
- Max Length: No limit
- Allows: Arabic, English, special characters
- Validation: Must not be empty or whitespace-only

### feedback_received_date
- Type: `date`
- Required: **YES**
- Format: `YYYY-MM-DD`
- Example: `"2026-01-05"`
- Validation: Must be valid date

### domain_id, category_id, severity_id
- Type: `integer`
- Required: **YES**
- Constraint: Must be > 0
- Validation: Must exist in respective lookup tables

### subcategory_id, classification_id (if provided)
- Type: `integer`
- Required: **NO**
- Constraint: Must be > 0
- Validation:
  - subcategory_id must belong to category_id
  - classification_id must belong to subcategory_id
  - If one is provided, the other is recommended

### feedback_type (NEW)
- Type: `integer`
- Required: **NO**
- Valid Values: 1, 2, 3, 4
- Validation: If provided, must be within 1-4

### improvement_opportunity_type (NEW)
- Type: `integer`
- Required: **NO**
- Valid Values: 1, 2, 3
- Validation: If provided, must be within 1-3

### classification_ar (NEW)
- Type: `float` (decimal)
- Required: **NO**
- Range: 0.0 - 10.0
- Example: `8.5`
- Validation: If provided, must be 0.0 to 10.0

### classification_en (NEW)
- Type: `integer`
- Required: **NO**
- Range: >= 0
- Example: `5`
- Validation: If provided, must be >= 0

### doctors
- Type: `array of objects`
- Required: **NO**
- Structure: `[{"doctor_id": int, "doctor_name": string}, ...]`
- Validation: doctor_id must exist in database

---

## 🧪 cURL EXAMPLES

### Minimal Request
```bash
curl -X POST "http://0.0.0.0:8000/api/records/add" \
  -H "Content-Type: application/json" \
  -d '{
    "complaint_text": "شكوى عامة",
    "feedback_received_date": "2026-01-05",
    "domain_id": 1,
    "category_id": 6,
    "severity_id": 1
  }'
```

### Complete Request with ML Fields
```bash
curl -X POST "http://0.0.0.0:8000/api/records/add" \
  -H "Content-Type: application/json" \
  -d '{
    "complaint_text": "تأخر في التشخيص",
    "feedback_received_date": "2026-01-05",
    "domain_id": 1,
    "category_id": 6,
    "severity_id": 1,
    "patient_name": "أحمد علي",
    "building_id": 1,
    "explanation_status_id": 1,
    "subcategory_id": 19,
    "classification_id": 132,
    "feedback_type": 1,
    "improvement_opportunity_type": 2,
    "classification_ar": 8.5,
    "classification_en": 5
  }'
```

---

## 🚨 HTTP STATUS CODES

| Status Code | Meaning | When Occurs |
|---|---|---|
| **200** | Success | Record created successfully |
| **400** | Bad Request | Missing/invalid required fields or validation failure |
| **409** | Conflict | Record already exists or constraint violation |
| **422** | Unprocessable Entity | Request body format invalid |
| **500** | Internal Server Error | Database error or unexpected exception |

---

## ✅ Frontend Implementation Checklist

- [ ] Validate `complaint_text` is not empty
- [ ] Validate `feedback_received_date` is in YYYY-MM-DD format
- [ ] Validate `domain_id`, `category_id`, `severity_id` are positive integers
- [ ] Include `Content-Type: application/json` header
- [ ] Handle 200 response and display `record_id` to user
- [ ] Handle 400/409/500 errors and display `message` or `message_ar`
- [ ] Display error `field` name if validation fails
- [ ] Send `subcategory_id` and `classification_id` if user selects them
- [ ] Send **NEW ML fields** if available (feedback_type, improvement_opportunity_type, etc.)
- [ ] Format date as YYYY-MM-DD (not timestamps)
- [ ] Send booleans as `true`/`false` (not "true"/"false" strings)
- [ ] Send arrays as actual arrays, not stringified JSON

---

## 🔗 Related Endpoints

### Search for valid lookup values:
- `GET /api/records/search/patients?q=name`
- `GET /api/records/search/doctors?q=name`
- `GET /api/records/search/employees?q=name`

### Reference data:
- `GET /api/reference/domains`
- `GET /api/reference/categories?domain_id=1`
- `GET /api/reference/subcategories?category_id=6`

---

## 📞 Support

If you get validation errors:
1. Check the `field` name in error response
2. Verify the value matches constraints in table above
3. For ID fields, use search endpoints to find valid IDs
4. Contact backend team if constraint still fails
