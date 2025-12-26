# Classification API - Testing Guide

## Base URL
```
http://127.0.0.1:8000
```

## Endpoints

### 1. Test Endpoint
**GET** `/api/classification/test`

Tests if the classification service is operational.

**Test URL:**
```
http://127.0.0.1:8000/api/classification/test
```

**Expected Response:**
```json
{
  "status": "operational",
  "service": "classification",
  "sample_result": {
    "success": true,
    "text": "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج",
    "classifications": {
      "domain": {...},
      "category": {...},
      "severity_level": {...}
    }
  }
}
```

---

### 2. Classify Single Text
**POST** `/api/classification/classify`

Classifies Arabic patient feedback text into 8 categories.

**Test with cURL:**
```bash
curl -X POST http://127.0.0.1:8000/api/classification/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج",
    "explain": true
  }'
```

**Test with Python:**
```python
import requests

url = "http://127.0.0.1:8000/api/classification/classify"
payload = {
    "text": "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج",
    "explain": True
}

response = requests.post(url, json=payload)
print(response.json())
```

**Test with JavaScript (Axios):**
```javascript
const axios = require('axios');

const url = 'http://127.0.0.1:8000/api/classification/classify';
const data = {
  text: 'المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج',
  explain: true
};

axios.post(url, data)
  .then(response => console.log(response.data))
  .catch(error => console.error(error));
```

**Request Body:**
```json
{
  "text": "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج",
  "explain": true
}
```

**Response:**
```json
{
  "success": true,
  "text": "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج",
  "classifications": {
    "domain": {
      "prediction": "السلامة الطبية",
      "confidence": 0.95,
      "explanation": "..."
    },
    "category": {
      "prediction": "التشخيص والعلاج",
      "confidence": 0.89
    },
    "subcategory": {...},
    "classification": {...},
    "severity_level": {
      "prediction": "عالية",
      "confidence": 0.92
    },
    "stage": {...},
    "harm_level": {...},
    "improvement_opportunity_type": {...}
  }
}
```

---

### 3. Classify Batch
**POST** `/api/classification/classify-batch`

Classifies multiple texts at once (up to 100).

**Test with cURL:**
```bash
curl -X POST http://127.0.0.1:8000/api/classification/classify-batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "المريض يشكو من ألم شديد",
      "تأخر في تقديم العلاج",
      "الطاقم الطبي محترم جداً"
    ],
    "explain": false
  }'
```

**Request Body:**
```json
{
  "texts": [
    "المريض يشكو من ألم شديد",
    "تأخر في تقديم العلاج",
    "الطاقم الطبي محترم جداً"
  ],
  "explain": false
}
```

**Response:**
```json
{
  "success": true,
  "total": 3,
  "successful": 3,
  "failed": 0,
  "results": [
    {
      "index": 0,
      "text": "المريض يشكو من ألم شديد",
      "result": {
        "success": true,
        "classifications": {...}
      }
    },
    ...
  ]
}
```

---

## Classification Categories

The model classifies into 8 categories:

1. **Domain (المجال)** - HCAT domain classification
2. **Category (التصنيف)** - Main category
3. **SubCategory (التصنيف الفرعي)** - Sub-category
4. **Classification (التصنيف الجديد)** - New classification system
5. **Severity Level (مستوى الخطورة)** - Low, Medium, High
6. **Stage (المرحلة)** - Process stage
7. **Harm Level (مستوى الضرر)** - Level of patient harm
8. **Improvement Opportunity Type (نوع فرصة التحسين)** - Type of improvement needed

---

## Error Responses

### Empty Text
```json
{
  "detail": {
    "error": "EMPTY_TEXT",
    "message": "Text is required for classification",
    "message_ar": "النص مطلوب للتصنيف"
  }
}
```

### Classification Failed
```json
{
  "detail": {
    "error": "CLASSIFICATION_FAILED",
    "message": "Classification failed: ...",
    "message_ar": "فشل التصنيف: ..."
  }
}
```

---

## Interactive Testing

### Using FastAPI Swagger UI
Open your browser and navigate to:
```
http://127.0.0.1:8000/docs
```

1. Find the **Classification** section
2. Click on any endpoint
3. Click **"Try it out"**
4. Enter your test data
5. Click **"Execute"**

### Using Postman
1. Create a new POST request
2. Set URL: `http://127.0.0.1:8000/api/classification/classify`
3. Set Headers: `Content-Type: application/json`
4. Set Body (raw JSON):
```json
{
  "text": "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج",
  "explain": true
}
```
5. Click **Send**

---

## Test Samples

### Sample 1: Complaint about pain
```json
{
  "text": "المريض يشكو من ألم شديد في البطن ولم يتم تقديم المسكنات في الوقت المناسب"
}
```

### Sample 2: Delay in service
```json
{
  "text": "تأخر طويل في قسم الطوارئ وعدم توفر الأطباء"
}
```

### Sample 3: Positive feedback
```json
{
  "text": "الطاقم الطبي محترم جداً والخدمة ممتازة"
}
```

### Sample 4: Medication error
```json
{
  "text": "تم إعطاء المريض دواء خاطئ مما تسبب في حساسية"
}
```

---

**Status:** ✅ Ready for Testing  
**Base URL:** http://127.0.0.1:8000  
**Documentation:** http://127.0.0.1:8000/docs
