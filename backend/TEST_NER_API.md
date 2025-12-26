# NER API - Testing Guide

## Base URL
```
http://127.0.0.1:8000
```

## Endpoints

### 1. Test Endpoint
**GET** `/api/ner/test`

Tests if the NER service is operational.

**Test URL:**
```
http://127.0.0.1:8000/api/ner/test
```

**Expected Response:**
```json
{
  "status": "operational",
  "service": "ner",
  "sample_result": {
    "success": true,
    "text": "المريض أحمد محمد يشكو من ألم في البطن وتم فحصه بواسطة الدكتور خالد في قسم الطوارئ",
    "entities": {
      "patient_names": ["أحمد محمد"],
      "doctor_names": ["خالد"],
      "departments": ["قسم الطوارئ"],
      ...
    }
  }
}
```

---

### 2. Extract Entities from Single Text
**POST** `/api/ner/extract`

Extracts named entities from Arabic patient feedback text.

**Test with cURL:**
```bash
curl -X POST http://127.0.0.1:8000/api/ner/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "المريض أحمد محمد يشكو من ألم في البطن وتم فحصه بواسطة الدكتور خالد في قسم الطوارئ"
  }'
```

**Test with Python:**
```python
import requests

url = "http://127.0.0.1:8000/api/ner/extract"
payload = {
    "text": "المريض أحمد محمد يشكو من ألم في البطن وتم فحصه بواسطة الدكتور خالد في قسم الطوارئ"
}

response = requests.post(url, json=payload)
print(response.json())
```

**Test with JavaScript (Axios):**
```javascript
const axios = require('axios');

const url = 'http://127.0.0.1:8000/api/ner/extract';
const data = {
  text: 'المريض أحمد محمد يشكو من ألم في البطن وتم فحصه بواسطة الدكتور خالد في قسم الطوارئ'
};

axios.post(url, data)
  .then(response => console.log(response.data))
  .catch(error => console.error(error));
```

**Request Body:**
```json
{
  "text": "المريض أحمد محمد يشكو من ألم في البطن وتم فحصه بواسطة الدكتور خالد في قسم الطوارئ"
}
```

**Response:**
```json
{
  "success": true,
  "text": "المريض أحمد محمد يشكو من ألم في البطن وتم فحصه بواسطة الدكتور خالد في قسم الطوارئ",
  "entities": {
    "patient_names": ["أحمد محمد"],
    "doctor_names": ["خالد"],
    "departments": ["قسم الطوارئ"],
    "conditions": ["ألم في البطن"],
    "locations": [],
    "medications": [],
    "dates": [],
    "organizations": []
  }
}
```

---

### 3. Extract Entities from Batch
**POST** `/api/ner/extract-batch`

Extracts entities from multiple texts at once (up to 100).

**Test with cURL:**
```bash
curl -X POST http://127.0.0.1:8000/api/ner/extract-batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "المريض أحمد يشكو من ألم",
      "الدكتور خالد في قسم الطوارئ",
      "تم إعطاء الباراسيتامول للمريض"
    ]
  }'
```

**Request Body:**
```json
{
  "texts": [
    "المريض أحمد يشكو من ألم",
    "الدكتور خالد في قسم الطوارئ",
    "تم إعطاء الباراسيتامول للمريض"
  ]
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
      "text": "المريض أحمد يشكو من ألم",
      "result": {
        "success": true,
        "entities": {
          "patient_names": ["أحمد"],
          "conditions": ["ألم"]
        }
      }
    },
    {
      "index": 1,
      "text": "الدكتور خالد في قسم الطوارئ",
      "result": {
        "success": true,
        "entities": {
          "doctor_names": ["خالد"],
          "departments": ["قسم الطوارئ"]
        }
      }
    },
    {
      "index": 2,
      "text": "تم إعطاء الباراسيتامول للمريض",
      "result": {
        "success": true,
        "entities": {
          "medications": ["الباراسيتامول"]
        }
      }
    }
  ]
}
```

---

## Entity Types Extracted

The NER model extracts the following entity types:

1. **Patient Names (أسماء المرضى)** - Names of patients
2. **Doctor Names (أسماء الأطباء)** - Names of doctors/physicians
3. **Departments (الأقسام)** - Hospital departments and units
4. **Medical Conditions (الحالات الطبية)** - Diseases, symptoms, conditions
5. **Medications (الأدوية)** - Drug names and medications
6. **Locations (المواقع)** - Physical locations within hospital
7. **Dates & Times (التواريخ والأوقات)** - Temporal references
8. **Organizations (المنظمات)** - Hospital names, external organizations

---

## Error Responses

### Empty Text
```json
{
  "detail": {
    "error": "EMPTY_TEXT",
    "message": "Text is required for NER",
    "message_ar": "النص مطلوب لاستخراج الكيانات"
  }
}
```

### NER Failed
```json
{
  "detail": {
    "error": "NER_FAILED",
    "message": "NER extraction failed: ...",
    "message_ar": "فشل استخراج الكيانات: ..."
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

1. Find the **NER** section
2. Click on any endpoint
3. Click **"Try it out"**
4. Enter your test data
5. Click **"Execute"**

### Using Postman
1. Create a new POST request
2. Set URL: `http://127.0.0.1:8000/api/ner/extract`
3. Set Headers: `Content-Type: application/json`
4. Set Body (raw JSON):
```json
{
  "text": "المريض أحمد محمد يشكو من ألم في البطن وتم فحصه بواسطة الدكتور خالد في قسم الطوارئ"
}
```
5. Click **Send**

---

## Test Samples

### Sample 1: Patient and doctor names
```json
{
  "text": "المريض أحمد محمد يشكو من ألم في البطن وتم فحصه بواسطة الدكتور خالد في قسم الطوارئ"
}
```

### Sample 2: Medication mention
```json
{
  "text": "تم إعطاء المريض محمد جرعة من الباراسيتامول والأموكسيسيلين"
}
```

### Sample 3: Multiple departments
```json
{
  "text": "تم تحويل المريض من قسم الطوارئ إلى قسم العناية المركزة بعد استشارة الدكتور عبدالله"
}
```

### Sample 4: Date and time mentions
```json
{
  "text": "حضر المريض يوم الأحد الماضي في الساعة الثالثة عصراً ولم يتم استقباله حتى الخامسة"
}
```

### Sample 5: Complex case
```json
{
  "text": "المريضة فاطمة علي عمرها 45 سنة تعاني من ارتفاع ضغط الدم والسكري وتم فحصها في مستشفى الملك فيصل بواسطة الدكتور أحمد والدكتورة سارة في قسم الباطنية"
}
```

---

## Use Cases

1. **Auto-fill patient information** - Extract patient names to pre-fill forms
2. **Doctor tracking** - Identify which doctors are mentioned in feedback
3. **Department analysis** - Analyze which departments receive most complaints
4. **Medication safety** - Extract medication names for safety analysis
5. **Privacy masking** - Identify names for anonymization
6. **Search enhancement** - Enable entity-based search and filtering

---

**Status:** ✅ Ready for Testing  
**Base URL:** http://127.0.0.1:8000  
**Documentation:** http://127.0.0.1:8000/docs
