# System Settings - Quick Start Guide

## 🚀 Quick Setup (5 Steps)

### Step 1: Create the Database Table
Run the SQL script in SQL Server Management Studio:

**File**: `backend/sql_scripts/create_system_settings_table.sql`

Or execute directly:
```sql
-- Run this in your SQL Server
-- Database: IncidentManager
-- Server: SOCIALMEDIA
```

### Step 2: Verify Database
```sql
-- Check table was created
SELECT * FROM APP_SystemSettings;

-- You should see one row:
-- ComplaintDelayDays | 14 | int | After this many days...
```

### Step 3: Start Backend Server
```bash
cd backend
uvicorn main:app --reload
```

The new router is already registered in [main.py](backend/main.py#L21)!

### Step 4: Test the API
```bash
python test_system_settings_api.py
```

Or test manually:
```bash
# Get all settings
curl http://localhost:8000/api/system-settings

# Get specific setting
curl http://localhost:8000/api/system-settings/ComplaintDelayDays

# Update setting
curl -X PUT http://localhost:8000/api/system-settings/ComplaintDelayDays \
  -H "Content-Type: application/json" \
  -d '{"value": "21", "updated_by_user_id": 1}'
```

### Step 5: Access API Documentation
Open in browser:
- Swagger UI: http://localhost:8000/docs
- Look for "System Settings" section

---

## 📱 Frontend Integration

### Fetch the Setting
```javascript
// Get ComplaintDelayDays setting
fetch('http://localhost:8000/api/system-settings/ComplaintDelayDays')
  .then(res => res.json())
  .then(data => {
    console.log(data.parsed_value);  // 14 (as number)
    console.log(data.description);   // "After this many days..."
  });
```

### Update the Setting
```javascript
// Update ComplaintDelayDays to 21
fetch('http://localhost:8000/api/system-settings/ComplaintDelayDays', {
  method: 'PUT',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    value: "21",
    updated_by_user_id: currentUserId  // Your user ID
  })
})
.then(res => res.json())
.then(data => {
  console.log('Updated:', data.parsed_value);  // 21
});
```

### React Component Example
```jsx
function DelaySettings() {
  const [delayDays, setDelayDays] = useState(14);
  const [loading, setLoading] = useState(false);
  
  // Load current value
  useEffect(() => {
    fetch('http://localhost:8000/api/system-settings/ComplaintDelayDays')
      .then(res => res.json())
      .then(data => setDelayDays(data.parsed_value));
  }, []);
  
  // Update value
  const handleSave = async () => {
    setLoading(true);
    const response = await fetch(
      'http://localhost:8000/api/system-settings/ComplaintDelayDays',
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          value: String(delayDays),
          updated_by_user_id: 1  // Replace with actual user ID
        })
      }
    );
    
    if (response.ok) {
      alert('✓ Setting saved!');
    } else {
      const error = await response.json();
      alert('Error: ' + error.detail);
    }
    setLoading(false);
  };
  
  return (
    <div className="setting-card">
      <h3>📅 إعدادات التأخير (Delay Settings)</h3>
      <label>
        After this many days, a complaint will be considered delayed:
      </label>
      <input 
        type="number" 
        value={delayDays}
        onChange={(e) => setDelayDays(e.target.value)}
        disabled={loading}
      />
      <button onClick={handleSave} disabled={loading}>
        {loading ? 'Saving...' : 'Save'}
      </button>
    </div>
  );
}
```

---

## 🔧 Backend Usage

### In Your Service Layer
```python
from api.services.system_settings_service import SystemSettingsService

# Get the delay threshold
delay_days = SystemSettingsService.get_setting_value("ComplaintDelayDays")
# Returns: 14 (as int, not string!)

# Check if complaint is delayed
if days_since_complaint > delay_days:
    status = "DELAYED"
else:
    status = "ON_TIME"
```

See [example_system_settings_usage.py](backend/example_system_settings_usage.py) for more examples!

---

## 🎯 Key Files

| File | Purpose |
|------|---------|
| [create_system_settings_table.sql](backend/sql_scripts/create_system_settings_table.sql) | Database table creation script |
| [system_settings_db.py](backend/api/db_layer/system_settings_db.py) | Database operations |
| [system_settings_service.py](backend/api/services/system_settings_service.py) | Business logic & validation |
| [system_settings_router.py](backend/api/routers/system_settings_router.py) | API endpoints |
| [test_system_settings_api.py](test_system_settings_api.py) | Automated tests |
| [example_system_settings_usage.py](backend/example_system_settings_usage.py) | Usage examples |

---

## ✅ Checklist

- [ ] SQL table created
- [ ] Backend server restarted
- [ ] API endpoints tested
- [ ] Frontend integrated
- [ ] Settings page displays value
- [ ] Settings page can update value
- [ ] Validation works (try invalid value)

---

## 🆘 Troubleshooting

### Error: "Table already exists"
✓ This is fine! The table is already created.

### Error: "Cannot connect to database"
Check connection string in [system_settings_db.py](backend/api/db_layer/system_settings_db.py#L12):
```python
SERVER=SOCIALMEDIA
DATABASE=IncidentManager
```

### Error: 404 when calling API
Make sure:
1. Backend server is running
2. Router is registered in main.py (already done!)
3. URL is correct: `/api/system-settings` (not `/api/settings`)

### Setting returns string instead of number
Use `parsed_value` field, not `value`:
```javascript
// Wrong:
data.value  // "14" (string)

// Correct:
data.parsed_value  // 14 (number)
```

---

## 📚 Next Steps

1. **Add more settings** as needed (no schema changes required!)
   ```sql
   INSERT INTO APP_SystemSettings 
   VALUES ('MaxUploadSizeMB', '50', 'int', 'Max upload size', GETDATE(), NULL);
   ```

2. **Add to Settings Page UI** - Create tabs for different setting categories

3. **Add Permissions** - Restrict who can update settings

4. **Add Audit Log** - Track all changes (UpdatedAt and UpdatedByUserID already captured!)

---

## 💡 Design Benefits

✅ **No Hardcoding** - All settings in database  
✅ **Type-Safe** - Automatic parsing & validation  
✅ **Extensible** - Add settings without code changes  
✅ **Auditable** - Track who changed what and when  
✅ **RESTful** - Standard API endpoints  
✅ **Clean Architecture** - 3-layer separation  

---

**Need Help?** Check [SYSTEM_SETTINGS_IMPLEMENTATION.md](SYSTEM_SETTINGS_IMPLEMENTATION.md) for detailed documentation!
