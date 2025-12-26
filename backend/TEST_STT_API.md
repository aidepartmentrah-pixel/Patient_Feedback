# STT API - Testing Guide

## Base URL
```
http://127.0.0.1:8000
```

## Endpoints

### 1. Test Endpoint
**GET** `/api/stt/test`

Tests if the STT service is operational.

**Test URL:**
```
http://127.0.0.1:8000/api/stt/test
```

**Expected Response:**
```json
{
  "status": "operational",
  "service": "stt",
  "supported_formats": ["mp3", "wav", "m4a", "ogg", "flac", "aac"],
  "max_file_size": "50MB",
  "message": "Upload audio file to /api/stt/transcribe endpoint"
}
```

---

### 2. Transcribe Audio File
**POST** `/api/stt/transcribe`

Converts Arabic audio file to text.

**Content-Type:** `multipart/form-data`

**Supported Formats:**
- MP3
- WAV
- M4A
- OGG
- FLAC
- AAC

**Maximum File Size:** 50MB

**Test with cURL:**
```bash
curl -X POST http://127.0.0.1:8000/api/stt/transcribe \
  -F "audio=@/path/to/your/audio.mp3"
```

**Test with Python:**
```python
import requests

url = "http://127.0.0.1:8000/api/stt/transcribe"

# Open audio file
with open("/path/to/your/audio.mp3", "rb") as audio_file:
    files = {"audio": ("audio.mp3", audio_file, "audio/mpeg")}
    response = requests.post(url, files=files)
    print(response.json())
```

**Test with JavaScript (Axios):**
```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const url = 'http://127.0.0.1:8000/api/stt/transcribe';
const formData = new FormData();
formData.append('audio', fs.createReadStream('/path/to/audio.mp3'));

axios.post(url, formData, {
  headers: formData.getHeaders()
})
  .then(response => console.log(response.data))
  .catch(error => console.error(error));
```

**Response:**
```json
{
  "success": true,
  "text": "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج",
  "filename": "audio.mp3",
  "size_bytes": 524288
}
```

---

### 3. Transcribe from Server Path
**POST** `/api/stt/transcribe-path`

Transcribes audio from a file path on the server (internal use).

**Content-Type:** `application/x-www-form-urlencoded`

**Test with cURL:**
```bash
curl -X POST http://127.0.0.1:8000/api/stt/transcribe-path \
  -F "audio_path=/server/path/to/audio.mp3"
```

**Test with Python:**
```python
import requests

url = "http://127.0.0.1:8000/api/stt/transcribe-path"
data = {"audio_path": "/server/path/to/audio.mp3"}

response = requests.post(url, data=data)
print(response.json())
```

**Response:**
```json
{
  "success": true,
  "text": "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج",
  "audio_path": "/server/path/to/audio.mp3"
}
```

---

## Error Responses

### No Filename
```json
{
  "detail": {
    "error": "NO_FILENAME",
    "message": "Audio file must have a filename",
    "message_ar": "يجب أن يحتوي ملف الصوت على اسم"
  }
}
```

### Invalid Format
```json
{
  "detail": {
    "error": "INVALID_FORMAT",
    "message": "Unsupported audio format: .txt. Allowed: .mp3, .wav, .m4a, .ogg, .flac, .aac",
    "message_ar": "صيغة صوت غير مدعومة: .txt"
  }
}
```

### Empty File
```json
{
  "detail": {
    "error": "EMPTY_FILE",
    "message": "Audio file is empty",
    "message_ar": "ملف الصوت فارغ"
  }
}
```

### File Too Large
```json
{
  "detail": {
    "error": "FILE_TOO_LARGE",
    "message": "Audio file too large. Maximum size: 50MB",
    "message_ar": "ملف الصوت كبير جداً. الحد الأقصى: 50 ميجابايت"
  }
}
```

### Transcription Failed
```json
{
  "detail": {
    "error": "STT_FAILED",
    "message": "Transcription failed: ...",
    "message_ar": "فشل تحويل الصوت إلى نص: ..."
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

1. Find the **STT** section
2. Click on `/api/stt/transcribe`
3. Click **"Try it out"**
4. Click **"Choose File"** and select your audio file
5. Click **"Execute"**

### Using Postman
1. Create a new POST request
2. Set URL: `http://127.0.0.1:8000/api/stt/transcribe`
3. Go to **Body** tab
4. Select **form-data**
5. Add key `audio` with type **File**
6. Choose your audio file
7. Click **Send**

---

## Frontend Integration

### React Example with File Upload
```jsx
import React, { useState } from 'react';
import axios from 'axios';

function AudioTranscriber() {
  const [file, setFile] = useState(null);
  const [transcription, setTranscription] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('audio', file);

    try {
      const response = await axios.post(
        'http://127.0.0.1:8000/api/stt/transcribe',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      );
      setTranscription(response.data.text);
    } catch (error) {
      console.error('Transcription failed:', error);
      alert('فشل تحويل الصوت إلى نص');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input
        type="file"
        accept="audio/*"
        onChange={handleFileChange}
      />
      <button onClick={handleUpload} disabled={!file || loading}>
        {loading ? 'جاري التحويل...' : 'تحويل الصوت إلى نص'}
      </button>
      {transcription && (
        <div>
          <h3>النص المستخرج:</h3>
          <p>{transcription}</p>
        </div>
      )}
    </div>
  );
}

export default AudioTranscriber;
```

### HTML + Vanilla JavaScript
```html
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
  <meta charset="UTF-8">
  <title>اختبار تحويل الصوت إلى نص</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      max-width: 600px;
      margin: 50px auto;
      padding: 20px;
      direction: rtl;
    }
    .result {
      margin-top: 20px;
      padding: 15px;
      background: #f0f0f0;
      border-radius: 5px;
    }
    button {
      padding: 10px 20px;
      margin: 10px 0;
      font-size: 16px;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <h1>تحويل الصوت إلى نص</h1>
  
  <input type="file" id="audioFile" accept="audio/*">
  <br>
  <button onclick="transcribe()">تحويل</button>
  
  <div id="result" class="result" style="display:none;">
    <h3>النص المستخرج:</h3>
    <p id="transcription"></p>
  </div>

  <script>
    async function transcribe() {
      const fileInput = document.getElementById('audioFile');
      const file = fileInput.files[0];
      
      if (!file) {
        alert('الرجاء اختيار ملف صوتي');
        return;
      }

      const formData = new FormData();
      formData.append('audio', file);

      try {
        const response = await fetch('http://127.0.0.1:8000/api/stt/transcribe', {
          method: 'POST',
          body: formData
        });

        const data = await response.json();
        
        if (data.success) {
          document.getElementById('transcription').textContent = data.text;
          document.getElementById('result').style.display = 'block';
        } else {
          alert('فشل تحويل الصوت: ' + data.message_ar);
        }
      } catch (error) {
        console.error('Error:', error);
        alert('حدث خطأ أثناء تحويل الصوت');
      }
    }
  </script>
</body>
</html>
```

---

## Performance Notes

1. **Processing Time:** Depends on audio length
   - ~1 minute audio = ~5-10 seconds processing
   - ~5 minute audio = ~20-30 seconds processing

2. **Optimal Audio Quality:**
   - Sample rate: 16kHz or higher
   - Format: WAV or MP3
   - Clear speech, minimal background noise

3. **Language:** Optimized for Modern Standard Arabic (MSA) and Gulf dialects

---

## Use Cases

1. **Voice complaints** - Convert patient voice complaints to text
2. **Phone call transcription** - Transcribe recorded phone conversations
3. **Voice notes** - Convert doctor/nurse voice notes to text
4. **Accessibility** - Allow voice input for forms
5. **Documentation** - Transcribe medical recordings

---

## Sample Audio Files

For testing, you can:
1. Record your own Arabic audio using your phone
2. Use online text-to-speech services to generate Arabic audio
3. Convert the following text to speech:
   - "المريض يشكو من ألم شديد في البطن"
   - "تأخر طويل في قسم الطوارئ"
   - "الطاقم الطبي محترم والخدمة ممتازة"

---

**Status:** ✅ Ready for Testing  
**Base URL:** http://127.0.0.1:8000  
**Documentation:** http://127.0.0.1:8000/docs  
**Max File Size:** 50MB  
**Supported Formats:** MP3, WAV, M4A, OGG, FLAC, AAC
