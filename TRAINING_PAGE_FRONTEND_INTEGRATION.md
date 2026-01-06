# 🔌 Training Page - Frontend Integration Guide

## Base URL
```
http://0.0.0.0:8000/api/settings/training
```

---

## 📡 4 API Endpoints

### 1. GET `/status` - Load Page / Monitor Training
**Use this for:**
- Page load (display current metrics)
- Poll every 2-5 sec while training runs

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

**Status values:** `never_run` | `running` | `completed` | `failed`

---

### 2. POST `/run` - Start Training
**Use this when:** User clicks "Retrain" button

**No body needed**
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

**Error (409 - training already running):**
```json
{
  "detail": "Training already in progress. Please wait for completion."
}
```

---

### 3. GET `/history` - Training Run History
**Use this for:** History/logs tab

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

### 4. GET `/db-size` - Database Growth Chart
**Use this for:** Chart showing records over time

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

**Use with:** Chart library (Chart.js, D3, etc.)

---

## ⚡ Quick Implementation

### On Page Load
```javascript
// Fetch current status & database size
fetch('http://0.0.0.0:8000/api/settings/training/status')
  .then(r => r.json())
  .then(data => {
    displayModels(data.models);
    displayStatus(data.status);
  });

fetch('http://0.0.0.0:8000/api/settings/training/db-size')
  .then(r => r.json())
  .then(data => {
    plotChart(data.points);
  });

fetch('http://0.0.0.0:8000/api/settings/training/history')
  .then(r => r.json())
  .then(data => {
    displayHistory(data.history);
  });
```

### User Clicks "Retrain"
```javascript
fetch('http://0.0.0.0:8000/api/settings/training/run', {
  method: 'POST'
})
  .then(r => r.json())
  .then(data => {
    showSpinner(); // Show loading
    startPolling(data.run_id); // Poll /status every 3 sec
  })
  .catch(e => {
    if (e.status === 409) {
      showError("Training already running");
    }
  });
```

### Poll During Training
```javascript
const pollInterval = setInterval(() => {
  fetch('http://0.0.0.0:8000/api/settings/training/status')
    .then(r => r.json())
    .then(data => {
      if (data.status !== 'running') {
        clearInterval(pollInterval);
        hideSpinner();
        displayModels(data.models);
      }
    });
}, 3000); // Poll every 3 seconds
```

---

## 🎨 Display Template

```html
<!-- Status Badge -->
<span class="badge" id="status">
  Running... (check back soon)
</span>

<!-- Model Metrics Table -->
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
      <th>Records</th>
      <th>Last Trained</th>
    </tr>
  </thead>
  <tbody id="models">
    <!-- Populated by JS -->
  </tbody>
</table>

<!-- Chart -->
<div id="chart" style="width: 100%; height: 300px;"></div>

<!-- Button -->
<button onclick="retrain()">Retrain Models</button>

<!-- History -->
<div id="history">
  <!-- Populated by JS -->
</div>
```

---

## ✅ Checklist

- [ ] Fetch `/status` on page load
- [ ] Display model metrics in table
- [ ] Show status badge with spinner while training
- [ ] Display last training timestamp
- [ ] "Retrain" button calls POST `/run`
- [ ] Poll `/status` every 2-5 seconds during training
- [ ] Stop polling when status changes from "running"
- [ ] Fetch `/db-size` and plot as line chart
- [ ] Fetch `/history` and show in table
- [ ] Handle 409 error (training already in progress)
- [ ] Display errors with user-friendly messages

---

## 🐛 Error Handling

| Status | Error | Action |
|--------|-------|--------|
| 200 | None | Success ✅ |
| 409 | Training already running | Show message "Please wait for training to complete" |
| 500 | Server error | Show "An error occurred, try again" |

---

## 📝 Response Field Reference

| Field | Type | Example |
|-------|------|---------|
| `accuracy` | float 0-1 | 0.8234 |
| `precision` | float 0-1 | 0.8011 |
| `recall` | float 0-1 | 0.7988 |
| `f1` | float 0-1 | 0.7999 |
| `num_records` | int | 412 |
| `model_name` | string | "Domain_Model" |
| `last_trained` | ISO datetime | "2026-01-02T11:41:00" |

---

## 🔗 Complete Example URL

```
GET  http://0.0.0.0:8000/api/settings/training/status
POST http://0.0.0.0:8000/api/settings/training/run
GET  http://0.0.0.0:8000/api/settings/training/history
GET  http://0.0.0.0:8000/api/settings/training/db-size
```

That's it! Copy-paste and integrate. 🚀
