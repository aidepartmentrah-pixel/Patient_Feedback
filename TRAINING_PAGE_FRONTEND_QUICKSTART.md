# 📱 Training Page - Frontend Dev Quick Start

## TL;DR: Copy This

### Base URL
```
http://0.0.0.0:8000/api/settings/training
```

### 4 Endpoints

```javascript
// 1. GET page data on load
fetch('/api/settings/training/status').then(r => r.json())
fetch('/api/settings/training/db-size').then(r => r.json())
fetch('/api/settings/training/history').then(r => r.json())

// 2. POST when user clicks "Retrain"
fetch('/api/settings/training/run', {method: 'POST'}).then(r => r.json())

// 3. POLL every 3 seconds during training
// (keep polling /status until status !== 'running')
```

---

## Response Examples

### GET /status
```json
{
  "last_run": "2026-01-02T11:43:00",
  "status": "completed",
  "models": [
    {
      "model_name": "Domain_Model",
      "accuracy": 0.8234,
      "precision": 0.8011,
      "recall": 0.7988,
      "f1": 0.7999,
      "num_records": 412
    }
  ]
}
```

### POST /run
```json
{
  "status": "started",
  "run_id": "2026_01_05_1430"
}
```

### GET /history
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

### GET /db-size
```json
{
  "points": [
    {"date": "2026-01-01", "records": 153},
    {"date": "2026-01-02", "records": 191}
  ]
}
```

---

## UI Checklist

- [ ] Show status badge (running/completed/failed)
- [ ] Display model metrics in table (accuracy, precision, recall, F1)
- [ ] Plot /db-size as line chart
- [ ] Show /history in table
- [ ] "Retrain" button → POST /run
- [ ] Poll /status every 3 sec while training
- [ ] Show spinner while `status === 'running'`
- [ ] Handle 409 error = "Training in progress"

Done! ✅
