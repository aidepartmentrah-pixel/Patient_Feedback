# System Settings Infrastructure - Implementation Complete

## 📋 Overview

A complete, extensible system settings infrastructure has been implemented to store and manage global configuration settings. This allows the Settings page to have dynamic, database-driven configuration values that can be updated through the UI without code changes.

---

## 🗄️ Database Structure

### Table: `APP_SystemSettings`

```sql
CREATE TABLE APP_SystemSettings (
    SettingKey NVARCHAR(100) PRIMARY KEY,
    SettingValue NVARCHAR(MAX) NOT NULL,
    SettingType NVARCHAR(20) NOT NULL,  -- 'int', 'bool', 'string', 'json'
    Description NVARCHAR(500) NULL,
    UpdatedAt DATETIME NOT NULL DEFAULT GETDATE(),
    UpdatedByUserID INT NULL
);
```

### Initial Seed Data

- **Key**: `ComplaintDelayDays`
- **Value**: `14`
- **Type**: `int`
- **Description**: "After this many days, a complaint is considered delayed"

**To create the table and seed data:**
```bash
# Execute the SQL script in SQL Server Management Studio or via command line
# File location: backend/sql_scripts/create_system_settings_table.sql
```

---

## 🏗️ Architecture

The implementation follows a clean 3-layer architecture:

```
API Layer (FastAPI Router)
    ↓
Service Layer (Business Logic)
    ↓
Database Layer (SQL Queries)
```

### Files Created

1. **SQL Script**: `backend/sql_scripts/create_system_settings_table.sql`
2. **Database Layer**: `backend/api/db_layer/system_settings_db.py`
3. **Service Layer**: `backend/api/services/system_settings_service.py`
4. **API Router**: `backend/api/routers/system_settings_router.py`
5. **Integration**: Updated `backend/main.py`

---

## 🔌 API Endpoints

Base URL: `/api/system-settings`

### 1. Get All Settings
```http
GET /api/system-settings
```

**Response:**
```json
[
  {
    "key": "ComplaintDelayDays",
    "value": "14",
    "type": "int",
    "description": "After this many days, a complaint is considered delayed",
    "updated_at": "2026-01-21T10:30:00",
    "updated_by_user_id": null,
    "parsed_value": 14,
    "parse_error": null
  }
]
```

### 2. Get Single Setting
```http
GET /api/system-settings/{key}
```

**Example:**
```http
GET /api/system-settings/ComplaintDelayDays
```

### 3. Update Setting
```http
PUT /api/system-settings/{key}
```

**Request Body:**
```json
{
  "value": "21",
  "updated_by_user_id": 1
}
```

**Response:**
```json
{
  "key": "ComplaintDelayDays",
  "value": "21",
  "type": "int",
  "parsed_value": 21,
  "updated_at": "2026-01-21T11:45:00",
  "updated_by_user_id": 1
}
```

### 4. Create Setting (Admin)
```http
POST /api/system-settings
```

**Request Body:**
```json
{
  "key": "MaxUploadSizeMB",
  "value": "50",
  "type": "int",
  "description": "Maximum file upload size in megabytes"
}
```

### 5. Delete Setting (Admin)
```http
DELETE /api/system-settings/{key}
```

---

## ✅ Validation Rules

The service layer automatically validates values based on their type:

| Type | Validation | Examples |
|------|-----------|----------|
| `int` | Must be numeric | `"14"`, `"100"`, `"-5"` |
| `bool` | Must be true/false | `"true"`, `"false"`, `"1"`, `"0"` |
| `string` | Any value | `"hello"`, `"any text"` |
| `json` | Valid JSON | `"{\"key\":\"value\"}"`, `"[1,2,3]"` |

**Example Validation Error:**
```json
{
  "detail": "Validation failed: Cannot parse 'not-a-number' as integer"
}
```

---

## 💡 Usage in Backend Code

### Get Setting Value (Type-Safe)

```python
from api.services.system_settings_service import SystemSettingsService

# Get parsed integer value
delay_days = SystemSettingsService.get_setting_value("ComplaintDelayDays")
# Returns: 14 (as int)

# Get full setting details
setting = SystemSettingsService.get_setting("ComplaintDelayDays")
# Returns: { key, value, type, parsed_value, ... }
```

### Update Setting Programmatically

```python
from api.services.system_settings_service import SystemSettingsService

# Update a setting
updated = SystemSettingsService.update_setting(
    key="ComplaintDelayDays",
    value="21",
    updated_by_user_id=1
)
```

---

## 🧪 Testing

A comprehensive test script has been created: `test_system_settings_api.py`

**To run tests:**
```bash
# 1. Make sure backend server is running
cd backend
uvicorn main:app --reload

# 2. In another terminal, run the test script
python test_system_settings_api.py
```

**Test Coverage:**
- ✅ Get all settings
- ✅ Get single setting
- ✅ Update setting with valid value
- ✅ Validation (reject invalid values)
- ✅ Error handling (404 for non-existent keys)
- ✅ Type parsing (string → int/bool/json)

---

## 🎨 Frontend Integration Guide

### Fetch Settings for Settings Page

```javascript
// Get all settings
fetch('http://localhost:8000/api/system-settings')
  .then(res => res.json())
  .then(settings => {
    console.log(settings);
    // Display in UI
  });
```

### Display in Settings Page (Arabic/English)

```jsx
// Example React component
function DelaySettingsTab() {
  const [delayDays, setDelayDays] = useState(14);
  
  useEffect(() => {
    fetch('http://localhost:8000/api/system-settings/ComplaintDelayDays')
      .then(res => res.json())
      .then(setting => setDelayDays(setting.parsed_value));
  }, []);
  
  const handleUpdate = async (newValue) => {
    const response = await fetch(
      'http://localhost:8000/api/system-settings/ComplaintDelayDays',
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          value: String(newValue),
          updated_by_user_id: currentUserId 
        })
      }
    );
    
    if (response.ok) {
      const updated = await response.json();
      setDelayDays(updated.parsed_value);
    }
  };
  
  return (
    <div>
      <label>📅 إعدادات التأخير (Delay Settings)</label>
      <input 
        type="number" 
        value={delayDays}
        onChange={(e) => handleUpdate(e.target.value)}
      />
      <p>After this many days, a complaint will be considered delayed</p>
    </div>
  );
}
```

---

## 🚀 Adding New Settings

To add new settings in the future:

### Option 1: Direct Database Insert
```sql
INSERT INTO APP_SystemSettings (SettingKey, SettingValue, SettingType, Description)
VALUES ('MaxUploadSizeMB', '50', 'int', 'Maximum file upload size in megabytes');
```

### Option 2: Via API
```bash
curl -X POST http://localhost:8000/api/system-settings \
  -H "Content-Type: application/json" \
  -d '{
    "key": "MaxUploadSizeMB",
    "value": "50",
    "type": "int",
    "description": "Maximum file upload size in megabytes"
  }'
```

### Future Setting Examples

```sql
-- Feature flags
('EnableMLPredictions', 'true', 'bool', 'Enable ML-based predictions'),

-- Thresholds
('HighPriorityThreshold', '7', 'int', 'Days before marking as high priority'),

-- Limits
('MaxReportsPerExport', '100', 'int', 'Maximum number of reports per export'),

-- SLA Values
('ResponseTimeSLA', '24', 'int', 'Response time SLA in hours'),

-- Complex config
('EscalationRules', '{"level1":3,"level2":7,"level3":14}', 'json', 'Escalation timer rules')
```

---

## ⚠️ Important Notes

1. **No Schema Changes Required**: Adding new settings never requires table schema changes
2. **Type Safety**: All values are validated before saving
3. **Audit Trail**: Every update records timestamp and user ID
4. **Error Handling**: Proper 404/400/500 responses for all error cases
5. **Extensibility**: Supports int, bool, string, and JSON types for complex configuration

---

## 📚 API Documentation

Once the server is running, access interactive API docs:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Look for the "System Settings" section to test endpoints directly.

---

## ✅ Checklist for Deployment

- [ ] Run SQL script to create table: `create_system_settings_table.sql`
- [ ] Verify seed data exists: `SELECT * FROM APP_SystemSettings`
- [ ] Restart backend server
- [ ] Test endpoints with `test_system_settings_api.py`
- [ ] Integrate with frontend Settings page
- [ ] Add user authentication to update endpoints (if not already present)

---

## 🎯 Summary

You now have a complete, production-ready system settings infrastructure that:
- ✅ Stores settings in database (not hardcoded)
- ✅ Supports multiple data types with validation
- ✅ Provides REST API for frontend
- ✅ Follows existing project architecture
- ✅ Is fully extensible for future settings
- ✅ Includes comprehensive testing

The first setting (`ComplaintDelayDays`) is ready to be integrated into the Settings page UI!
