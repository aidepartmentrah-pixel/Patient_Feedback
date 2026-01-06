# 🤖 Training Models Page - API Endpoints

Base URL: `http://0.0.0.0:8000/api/settings/training`

---

## 1️⃣ GET `/status` - Current Model Performance

**Get latest model metrics**

```bash
curl -X GET "http://0.0.0.0:8000/api/settings/training/status"
```

**Response:**
```json
{
  "last_run": "2026-01-02T11:43:00",
  "status": "completed",
  "models": [
    {
      "model_name": "Domain_Model",
      "num_records": 412,
      "accuracy": 0.8234,
      "precision": 0.8011,
      "recall": 0.7988,
      "f1": 0.7999,
      "last_trained": "2026-01-02T11:41:00"
    }
  ]
}
```

| Status | Meaning |
|--------|---------|
| `never_run` | No training executed yet |
| `running` | Training in progress |
| `completed` | Training finished successfully |
| `failed` | Last training failed |

---

## 2️⃣ POST `/run` - Start Training

**Trigger full model retraining (async - returns immediately)**

```bash
curl -X POST "http://0.0.0.0:8000/api/settings/training/run"
```

**Response:**
```json
{
  "status": "started",
  "run_id": "2026_01_05_1430"
}
```

⚠️ **Note:** Returns immediately. Training runs in background. Use `/status` to check progress.

**Error (409 Conflict):**
```json
{
  "detail": "Training already in progress. Please wait for completion."
}
```

---

## 3️⃣ GET `/history` - Training Run History

**Get past training runs**

```bash
curl -X GET "http://0.0.0.0:8000/api/settings/training/history"
```

**Response:**
```json
{
  "history": [
    {
      "run_id": "2026_01_02_1140",
      "started_at": "2026-01-02T11:40:00",
      "finished_at": "2026-01-02T11:43:00",
      "status": "completed",
      "models_trained": 12
    }
  ]
}
```

---

## 4️⃣ GET `/db-size` - Database Growth Chart

**Get ML database size over time (for graphing)**

```bash
curl -X GET "http://0.0.0.0:8000/api/settings/training/db-size"
```

**Response:**
```json
{
  "points": [
    { "date": "2026-01-01", "records": 153 },
    { "date": "2026-01-02", "records": 191 },
    { "date": "2026-01-03", "records": 248 }
  ]
}
```

---

## 🎯 Frontend Implementation Flow

1. **Page Load:**
   - Call `GET /status` → Display current metrics & status badge
   - Call `GET /db-size` → Plot database growth chart

2. **User Clicks "Retrain":**
   - Call `POST /run`
   - Show loading spinner with returned `run_id`

3. **Monitor Progress:**
   - Poll `GET /status` every 2-5 seconds while training
   - Update metrics when complete
   - Show timestamp of last training run

4. **History Tab:**
   - Call `GET /history`
   - Display list of past runs with timestamps

---

## 🔧 Response Codes

| Code | Scenario |
|------|----------|
| **200** | Success |
| **409** | Training already running |
| **500** | Server error |

---

## 📋 Checklist

- [ ] Display status badge (never_run / running / completed / failed)
- [ ] Show model performance metrics (accuracy, precision, recall, F1)
- [ ] Display last training timestamp
- [ ] "Retrain" button triggers POST /run
- [ ] Poll /status during training
- [ ] Show database growth chart from /db-size data
- [ ] Display training history in table
- [ ] Handle 409 conflict (training already in progress)
