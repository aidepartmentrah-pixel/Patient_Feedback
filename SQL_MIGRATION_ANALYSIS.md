# SQL Database Migration Analysis & Plan

## Current Status: ✅ Software Installed + Database Backup Found
- SQL Server 2022 Express: Installed
- SSMS (SQL Server Management Studio): Installed
- **Database Backup**: IncidentManager.bak (30.22 MB) ✅ **FOUND**
- Last Updated: 5/4/2026 1:17:51 PM
- Ready for database migration

---

## Available Migration Files

### 🎯 PRIMARY: IncidentManager.bak ✅ **COMPLETE DATABASE BACKUP**
- **Location**: `C:\Users\Administrator\Downloads\SQL Queries\SQL Queries\`
- **Size**: 30.22 MB
- **Type**: Full SQL Server backup file (.bak)
- **Content**: Complete database with schema + all data + objects
- **Status**: ✅ Latest backup (5/4/2026)
- **Includes**:
  - All tables (lookup + transaction)
  - All views, stored procedures, triggers
  - All user permissions
  - Database settings and options
  - **All testing data** (random names as mentioned)

### Supporting Files (Optional Reference):

#### File 1: LookUpTables.sql ✅ **HAS DATA**
- **Location**: `C:\Users\Administrator\Downloads\SQL Queries\SQL Queries\New LookUpTables\`
- **Size**: 112 KB (672 lines)
- **Content**: Schema + Lookup Reference Data
- **Statistics**:
  - 26 tables with IDENTITY_INSERT (data being populated)
  - 199 INSERT statements with VALUES (actual data)

#### File 2: New_Tables_Only.sql 📋 **SCHEMA ONLY (NO DATA)**
- **Location**: `C:\Users\Administrator\Downloads\SQL Queries\SQL Queries\New Empty Tables\`
- **Size**: 94 KB (1,107 lines)
- **Content**: Empty table structures only
- **Statistics**:
  - 0 IDENTITY_INSERT lines
  - 0 INSERT statements

#### File 3: Testing_Database.sql ⚠️ **SCHEMA ONLY**
- **Location**: `C:\Users\Administrator\Downloads\SQL Queries\SQL Queries\Full_Testing_Script\`
- **Size**: 227 KB (2,691 lines)
- **Content**: Full database schema without data
- **Note**: Data is in the .bak backup file instead

---

## Recommended Migration Strategy

### ✅ OPTION 1: Restore from .bak Backup (RECOMMENDED - FASTEST & MOST RELIABLE)

**Why this is best**:
- ✅ One-step complete restore
- ✅ Preserves all data, users, permissions
- ✅ Fastest method (5-10 minutes)
- ✅ Guaranteed data consistency
- ✅ No manual schema recreation needed
- ✅ Built-in SQL Server feature

**Execution**:
1. Open SSMS on local machine
2. Right-click "Databases" → "Restore Database"
3. Select "Device" → Browse to `IncidentManager.bak`
4. Leave default database name as "IncidentManager"
5. Click "Restore" → Done!
6. Update application config to point to localhost

**Time**: ~5-10 minutes total

**Verification**:
- Check if tables exist: `SELECT COUNT(*) FROM sys.tables`
- Check data count: `SELECT COUNT(*) FROM APP_IncidentRequestCase`
- Test connection from application

---

### ⚠️ OPTION 2: Use SQL Scripts (IF Backup Fails)

**Only use if .bak restore has issues**

**Execution Order**:
1. Execute `New_Tables_Only.sql` → Creates empty structure
2. Execute `LookUpTables.sql` → Adds reference lookup data
3. Then manually add test data if needed

**Time**: ~15-20 minutes

**Pros**: Selective data loading, easier to debug
**Cons**: No transactional data, requires multiple steps

---

### ❌ OPTION 3: Testing_Database.sql (NOT RECOMMENDED)

**Reason**: Schema-only file, has no data inside
**Verdict**: Skip this file, use .bak backup instead

---

## Recommended Plan

### ✅ PRIMARY PLAN (Using .bak Backup - FASTEST)

```
STEP 1: Prepare for Restore
├─ Verify SQL Server is running
├─ Verify backup file exists at: C:\Users\Administrator\Downloads\SQL Queries\SQL Queries\IncidentManager.bak
└─ Note: Any old IncidentManager database will be overwritten

STEP 2: Restore Database in SSMS
├─ Open SQL Server Management Studio
├─ Connect to: (local) or localhost
├─ Right-click on "Databases" folder
├─ Select "Restore Database..."
├─ Select "Device" option
├─ Click [...] browse button
├─ Navigate to backup file: C:\Users\Administrator\Downloads\SQL Queries\SQL Queries\IncidentManager.bak
├─ Click OK
├─ Leave database name as: IncidentManager
├─ Leave Location/Path defaults
├─ Click "Restore" button
└─ Wait for completion (2-3 minutes)

STEP 3: Verify Restore Success
├─ Refresh Databases in Object Explorer
├─ IncidentManager should appear in the list
├─ Expand it to see all tables and objects
└─ Test query: SELECT TOP 1 * FROM APP_LOOKUP_DOMAIN

STEP 4: Update Application Configuration
├─ Edit: backend/config/db_settings.json
├─ Change:
│  ├─ "host": "localhost"  (or your actual VM IP)
│  ├─ "port": 1433
│  ├─ "database": "IncidentManager"
│  ├─ "use_windows_auth": false
│  ├─ "username": "HCAT_Insight"
│  └─ "password": "NewPassword2004"
└─ Save file

STEP 5: Test Application Connection
├─ Activate Python environment: venv\Scripts\Activate.ps1
├─ Run: python backend/test_db_connection.py
├─ Should see: ✅ Connection successful
└─ Record count details from each table

STEP 6: Final Verification
├─ Test application startup
├─ Test user login
├─ Test create incident
└─ Test database queries work correctly
```

**Total Time**: ~20-30 minutes (mostly waiting for restore)

---

### ALTERNATIVE PLAN (If Backup Fails)

```
STEP 1: Create Empty Database Structure
├─ Execute: New_Tables_Only.sql
├─ This creates all tables, constraints, indexes
└─ Run in SSMS against new IncidentManager database

STEP 2: Populate Lookup Reference Data
├─ Execute: LookUpTables.sql
├─ This fills in all lookup tables (domain, category, etc.)
└─ Verify records inserted

STEP 3: Add Test Data (Optional)
├─ Create sample users
├─ Create sample incidents
└─ Or import from old database using export/import

STEP 4: Proceed with Steps 4-6 above
```

**Total Time**: ~25-35 minutes

## Backend Code Analysis

I've reviewed your backend code and found:

### Connection Architecture ✅
- **File**: `backend/core/database.py`
- **Method**: Explicit TCP connections (recommended for stability)
- **Connection Format**: `tcp:<host>,<port>` (not Named Pipes)
- **Retry Logic**: 2 retries with 0.5s delay for transient errors

### Configuration System ✅
- **Source**: `backend/core/deployment_port.py` → `backend/core/config_loader.py`
- **Config File**: `backend/config/db_settings.json` (JSON-based)
- **Override**: Environment variables > JSON values
- **Features**: 
  - Auto-detect local IP for CORS
  - Supports Windows Auth OR SQL Auth
  - ODBC Driver 18 support
  - TRUST_SERVER_CERTIFICATE option

### Current Configuration
```json
{
    "host": "170.70.32.11",           ← Change to: "localhost"
    "port": 1433,
    "database": "IncidentManager",
    "driver": "ODBC Driver 18 for SQL Server",
    "use_windows_auth": false,         ← Keep false (SQL Auth)
    "username": "HCAT_Insight",
    "password": "NewPassword2004",
    "trust_server_certificate": true
}
```

### What Needs to Change
After backup restore, only **1 line** needs updating:

```diff
- "host": "170.70.32.11",
+ "host": "localhost",
```

Everything else stays the same!

### Backend Test Tools Available
```
✅ backend/test_db_connection.py        → Test connection
✅ backend/check_tables.py              → Verify tables exist
✅ backend/get_login_credentials.py     → Check user setup
✅ backend/query_lookup_tables.py       → Verify lookup data
```

Use these to validate migration success!

---

## Next Steps After Plan Selection

**Ready to Execute**: Simply follow PRIMARY PLAN above

**If Issues Occur**: 
- Use ALTERNATIVE PLAN with SQL scripts
- Check log files in `backend/logs/`
- Run validation scripts

**Post-Migration Checklist**:
- [ ] Backup file restored successfully
- [ ] IncidentManager database visible in SSMS
- [ ] `backend/config/db_settings.json` updated to localhost
- [ ] `python backend/test_db_connection.py` shows success
- [ ] Backend application starts without connection errors
- [ ] Can login to application
- [ ] Can create/view incidents
- [ ] Lookup tables populate dropdowns correctly

---

## Summary: What You Have

| Item | Status | Size | Format |
|------|--------|------|--------|
| IncidentManager.bak | ✅ Ready | 30.22 MB | SQL Server native backup |
| New_Tables_Only.sql | ✅ Backup | 94 KB | Schema only (schema reference) |
| LookUpTables.sql | ✅ Backup | 112 KB | Lookup data (reference) |
| Testing_Database.sql | ✅ Backup | 227 KB | Schema script (reference) |

**Recommended Path**: Use `.bak` backup (fastest, most reliable)

---

## Important Notes

1. **No Data Loss**: All data is in the .bak file (complete backup)
2. **User Credentials**: HCAT_Insight login credentials preserved in backup
3. **SQL Server Auth**: Using SQL authentication (not Windows Auth) - confirmed working
4. **Database Location**: Stays on same SQL Server instance (localhost:1433)
5. **Application Change**: Only config file needs update (1 parameter = host IP)

---

## ⏰ Timeline

- **Backup Restore**: 5-10 min
- **Config Update**: 2 min
- **Connection Test**: 2 min
- **Application Restart**: 2 min
- **Functional Testing**: 5-10 min

**Total: ~20-30 minutes to full operational status**
