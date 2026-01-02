# API Documentation: Insert Record with 4 ML Type Fields

## Endpoint
```
POST /api/records/add
```

## Request Body

### CreateRecordRequest Schema

```json
{
  "complaint_text": "string (required)",
  "feedback_received_date": "string (required, format: YYYY-MM-DD)",
  "domain_id": "integer (required)",
  "category_id": "integer (required)",
  "sub_category_id": "integer (required)",
  "severity_id": "integer (required)",
  "stage_id": "integer (required)",
  "harm_level_id": "integer (required)",
  
  "immediate_action": "string (optional)",
  "taken_action": "string (optional)",
  "internal_id": "integer (optional)",
  
  "feedback_type": "integer (optional, range: 1-4)",
  "improvement_opportunity_type": "integer (optional, range: 1-3)",
  "classification_ar": "float (optional, range: 0-10)",
  "classification_en": "integer (optional, range: >= 0)"
}
```

## Field Descriptions

### Standard Fields (Already Supported)
- **complaint_text** [string, required]: The main complaint/feedback text in Arabic or English
- **feedback_received_date** [string, required]: Date feedback was received (YYYY-MM-DD format)
- **domain_id** [integer, required]: Classification domain ID (foreign key)
- **category_id** [integer, required]: Classification category ID (foreign key)
- **sub_category_id** [integer, required]: Classification sub-category ID (foreign key)
- **severity_id** [integer, required]: Severity level ID (foreign key)
- **stage_id** [integer, required]: Process stage ID (foreign key)
- **harm_level_id** [integer, required]: Harm level ID (foreign key)
- **immediate_action** [string, optional]: Immediate actions taken
- **taken_action** [string, optional]: Actions that were taken
- **internal_id** [integer, optional]: Internal reference ID

### NEW - 4 ML Training Type Fields

#### 1. feedback_type
- **Type:** Integer (1-4)
- **Required:** No (optional)
- **Valid Values:**
  - `1` = Improvement Opportunity
  - `2` = Notice
  - `3` = Critique Suggestion
  - `4` = Other
- **Purpose:** Categorizes the type of feedback received
- **Example:** `"feedback_type": 1`

#### 2. improvement_opportunity_type
- **Type:** Integer (1-3)
- **Required:** No (optional)
- **Valid Values:**
  - `1` = Ordinary (standard improvement opportunity)
  - `2` = RedFlag (concerning issue requiring urgent attention)
  - `3` = NeverEvent (critical patient safety issue)
- **Purpose:** Indicates severity level for improvement opportunities
- **Example:** `"improvement_opportunity_type": 2`

#### 3. classification_ar
- **Type:** Float (0-10)
- **Required:** No (optional)
- **Valid Range:** 0.0 to 10.0
- **Purpose:** Confidence score for Arabic text classification (from ML model)
- **Example:** `"classification_ar": 8.5`

#### 4. classification_en
- **Type:** Integer (≥0)
- **Required:** No (optional)
- **Valid Range:** 0 and above
- **Purpose:** Classification code for English text classification (from ML model)
- **Example:** `"classification_en": 5`

## Complete Example Request

```bash
curl -X POST "http://localhost:8000/api/records/add" \
  -H "Content-Type: application/json" \
  -d '{
    "complaint_text": "المريض يشتكي من الم في الراس مستمر لمدة ثلاثة ايام",
    "feedback_received_date": "2026-01-02",
    "domain_id": 1,
    "category_id": 1,
    "sub_category_id": 1,
    "severity_id": 1,
    "stage_id": 1,
    "harm_level_id": 1,
    "immediate_action": "تم إعطاء المريض دواء مسكن",
    "taken_action": "تم عمل فحوصات إضافية",
    "feedback_type": 1,
    "improvement_opportunity_type": 2,
    "classification_ar": 8.5,
    "classification_en": 5
  }'
```

## Response

### Success Response (201 Created)
```json
{
  "success": true,
  "message": "Record added successfully",
  "record_id": 42,
  "data": {
    "id": 42,
    "complaint_text": "المريض يشتكي من الم في الراس مستمر لمدة ثلاثة ايام",
    "feedback_received_date": "2026-01-02",
    "domain": "Patient Care",
    "category": "Clinical Issue",
    "sub_category": "Pain Management",
    "severity_level": "High",
    "stage": "Investigation",
    "harm_level": "Significant",
    "feedback_type": 1,
    "improvement_opportunity_type": 2,
    "classification_ar": 8.5,
    "classification_en": 5,
    "embedding_text": "3072-byte binary vector",
    "embedding_text123": "3072-byte binary vector",
    "embedding_text23": "3072-byte binary vector",
    "sentence_1_embedding": "3072-byte binary vector",
    "sentence_2_embedding": "3072-byte binary vector",
    "sentence_3_embedding": "3072-byte binary vector",
    "sentence_4_embedding": "3072-byte binary vector",
    "sentence_5_embedding": "3072-byte binary vector",
    "sentence_6_embedding": "3072-byte binary vector"
  }
}
```

### Error Response (400 Bad Request)
```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "feedback_type"],
      "msg": "Input should be greater than or equal to 1",
      "input": 0
    }
  ]
}
```

### Error Response (422 Unprocessable Entity)
```json
{
  "detail": [
    {
      "type": "less_than_or_equal",
      "loc": ["body", "improvement_opportunity_type"],
      "msg": "Input should be less than or equal to 3",
      "input": 5
    }
  ]
}
```

## Automatic Features

### 1. ID Auto-Increment
- The `id` field is automatically generated using `MAX(id) + 1` logic
- No need to provide ID in request

### 2. Automatic Embeddings
- When a record is inserted, embeddings are automatically generated from `complaint_text`
- Uses MPNet model to create 3072-byte vectors
- Generates 11 embedding fields:
  - `embedding_text`: Full text embedding
  - `embedding_text123`: Combined embedding (text, variations)
  - `embedding_text23`: Partial embedding (variations)
  - `sentence_1_embedding` through `sentence_6_embedding`: Individual sentence embeddings

### 3. Field Mapping
- Domain, category, and sub-category IDs are automatically mapped to text values
- Severity level, stage, and harm level IDs are automatically mapped to text values

## Data Persistence

All provided fields (including the 4 new ML type fields) are persisted in the database:

### SQL Query to Verify
```sql
SELECT 
  id,
  feedback_type,
  improvement_opportunity_type,
  classification_ar,
  classification_en,
  feedback_received_date,
  complaint_text
FROM patient_feedback_encoded
ORDER BY id DESC
LIMIT 1;
```

## Integration Points

### Frontend → API
```
POST /api/records/add with all 4 fields
```

### API → Database Layer
```
FastAPI CreateRecordRequest model validates fields
→ insert_service.create_record() receives data
→ add_corrected_record_to_ml() wrapper (auto-embeddings)
→ ml_insert_adapter._insert_row() (ID auto-increment)
→ SQLite Database
```

## Validation Rules

| Field | Type | Min | Max | Required | Default |
|-------|------|-----|-----|----------|---------|
| complaint_text | string | - | - | Yes | - |
| feedback_received_date | string | - | - | Yes | - |
| feedback_type | integer | 1 | 4 | No | None |
| improvement_opportunity_type | integer | 1 | 3 | No | None |
| classification_ar | float | 0 | 10 | No | None |
| classification_en | integer | 0 | ∞ | No | None |

## Status Codes

| Code | Meaning |
|------|---------|
| 201 | Record created successfully |
| 400 | Bad request (validation error) |
| 422 | Validation error (Pydantic) |
| 500 | Server error |

## Implementation Status

✅ **All 4 fields implemented and tested**
- feedback_type: Implemented
- improvement_opportunity_type: Implemented
- classification_ar: Implemented
- classification_en: Implemented

✅ **Auto-features working**
- ID auto-increment: Working
- Automatic embeddings: Working
- Field mapping: Working

✅ **Production ready**
- API validation: Complete
- Database schema: Updated
- End-to-end testing: Passed

---

**Last Updated:** January 2, 2025
**API Version:** 1.0
**Status:** Production Ready
