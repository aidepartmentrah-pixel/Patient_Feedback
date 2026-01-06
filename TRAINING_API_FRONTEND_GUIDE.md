# Training API - Frontend Integration Guide

## Base URL
```
http://0.0.0.0:8000
```

---

## 4 Endpoints for Settings > Training Tab

### 1️⃣ GET `/api/settings/training/status`
**Purpose:** Display current model performance

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

**Status values:** `"never_run"` | `"running"` | `"completed"` | `"failed"`

---

### 2️⃣ GET `/api/settings/training/history`
**Purpose:** Show training run history

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
    },
    {
      "run_id": "2026_01_01_0900",
      "started_at": "2026-01-01T09:00:00",
      "finished_at": "2026-01-01T09:15:00",
      "status": "completed",
      "models_trained": 12
    }
  ]
}
```

---

### 3️⃣ GET `/api/settings/training/db-size`
**Purpose:** Get ML database growth for graph

**Response:**
```json
{
  "points": [
    { "date": "2026-01-01", "records": 153 },
    { "date": "2026-01-02", "records": 191 },
    { "date": "2026-01-03", "records": 248 },
    { "date": "2026-01-04", "records": 312 },
    { "date": "2026-01-05", "records": 371 }
  ]
}
```

---

### 4️⃣ POST `/api/settings/training/run`
**Purpose:** Trigger full retraining

**Request:**
```
POST /api/settings/training/run
(no body required)
```

**Immediate Response:**
```json
{
  "status": "started",
  "run_id": "2026_01_05_1430"
}
```

**Behavior:**
- Returns immediately (non-blocking)
- Training runs in background
- Check `/status` endpoint to monitor progress
- Results stored after completion

**Error (if already training):**
```json
{
  "detail": "Training already in progress. Please wait for completion.",
  "status": 409
}
```

---

## Frontend Implementation Checklist

### Display Requirements

**1. Status Card**
```javascript
// GET /api/settings/training/status
- Show "Last Run: <last_run>"
- Show "Status: <status>" (with color: green=completed, red=failed, yellow=running)
- List each model with accuracy/precision/recall/f1 in a table
```

**2. History Table**
```javascript
// GET /api/settings/training/history
- Display run_id, started_at, finished_at, status, models_trained
- Sort by started_at DESC (most recent first)
- Clickable row details (optional)
```

**3. Database Size Graph**
```javascript
// GET /api/settings/training/db-size
- Line chart: X-axis = date, Y-axis = records
- Update daily
- Show trend
```

**4. Train Button**
```javascript
// POST /api/settings/training/run
- Button: "Train All Models"
- On click: POST request
- Show confirmation: "Training started (Run ID: <run_id>)"
- Disable button while training
- Auto-refresh /status every 5 seconds during training
- Re-enable button when status != "running"
```

---

## Example JavaScript Code

```javascript
// Get status
async function getTrainingStatus() {
  const response = await fetch('http://0.0.0.0:8000/api/settings/training/status');
  return await response.json();
}

// Get history
async function getTrainingHistory() {
  const response = await fetch('http://0.0.0.0:8000/api/settings/training/history');
  return await response.json();
}

// Get DB size
async function getDBSize() {
  const response = await fetch('http://0.0.0.0:8000/api/settings/training/db-size');
  return await response.json();
}

// Start training
async function startTraining() {
  const response = await fetch('http://0.0.0.0:8000/api/settings/training/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });
  return await response.json();
}

// Monitor training
async function monitorTraining() {
  while (true) {
    const status = await getTrainingStatus();
    if (status.status !== 'running') break;
    await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 sec
  }
}
```

---

## Error Handling

| Status Code | Meaning |
|---|---|
| 200 | Success |
| 409 | Training already in progress |
| 500 | Server error |

---

## Notes

✅ All endpoints return immediately (no timeouts)  
✅ Training runs asynchronously in background  
✅ Results persisted to SQLite for history  
✅ No external dependencies on frontend  
✅ Supports multiple queries during training
