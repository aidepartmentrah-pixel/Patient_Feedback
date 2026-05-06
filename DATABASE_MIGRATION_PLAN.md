# Professional Database Migration Execution Plan
**Status**: READY FOR REVIEW & APPROVAL  
**Date**: May 4, 2026  
**Version**: 1.0

---

## PART 1: UNDERSTANDING YOUR ARCHITECTURE

### ✅ Existing Smart Configuration System
Your system already has professional multi-environment support:

1. **JSON Configuration** (`backend/config/db_settings.json`)
   - Primary configuration source
   - Easy to read/edit
   - Version-controlled with application

2. **Environment Variable Overrides**
   - `DB_HOST` / `DB_SERVER` 
   - `DB_PORT`
   - `DB_DATABASE`
   - `DB_DRIVER`
   - `DB_USERNAME`
   - `DB_PASSWORD`
   - Override JSON values automatically

3. **Explicit TCP Connections**
   - Format: `tcp:<host>,<port>` 
   - Avoids Named Pipes ambiguity
   - Deterministic routing

4. **Auto-IP Detection**
   - CORS origins auto-generated from detected IP
   - Backend serves on `0.0.0.0:8000` (accepts all interfaces)
   - Already handles network flexibility

### ✅ Your Previous Work Achievements
- ✅ Eliminated hardcoded localhost
- ✅ Environment variable support for deployments
- ✅ JSON-based configuration
- ✅ TCP explicit routing
- ✅ CORS auto-generation

---

## PART 2: CURRENT SITUATION

### Before Migration (Original Setup)
```
Database Server: 170.70.32.11:1433 (Remote PC)
Application: Local machine (reading from remote)
Config: host = 170.70.32.11
```

### After Migration (New Setup)
```
Database Server: ??? (To be confirmed)
Options:
  a) Local SQL Server on same VM: host = localhost or 127.0.0.1
  b) Local SQL Server on different machine: host = <New_IP_Address>
  c) Continued use of remote: host = 170.70.32.11 (unchanged)
```

**CRITICAL QUESTION**: 
- Where is SQL Server 2022 Express installed?
  - ☐ On this same VM (user answer needed)
  - ☐ On different internal machine (user answer needed)
  - If different: What is its IP address?

---

## PART 3: PROFESSIONAL MIGRATION APPROACH

### Three Supported Methods (In Order of Preference)

#### METHOD 1: Environment Variable Deployment (RECOMMENDED FOR INTERNAL VM)
**Best For**: Multiple testers, different network environments, flexibility

**Advantages**:
- No code/config file changes needed
- Different machines can use different database IPs
- Easy to switch between development/test/production databases
- Already implemented in your codebase

**Implementation**:
```powershell
# Set these environment variables before running application
$env:DB_HOST = "<DATABASE_SERVER_IP>"
$env:DB_PORT = "1433"
$env:DB_USERNAME = "HCAT_Insight"
$env:DB_PASSWORD = "NewPassword2004"

# Then start the application
python backend/main.py
```

**Advantages**:
- No permanent changes to repository
- Each person can use their own database server
- Preserves your previous work
- Production-ready approach

---

#### METHOD 2: Update JSON Configuration (For Specific Deployment)
**Best For**: Dedicated internal VM, fixed configuration

**Implementation**:
```json
{
    "database": {
        "host": "10.x.x.x",           ← UPDATE TO ACTUAL SQL SERVER IP
        "port": 1433,
        "server": "10.x.x.x",          ← ALSO UPDATE (for backward compat)
        "database": "IncidentManager",
        "username": "HCAT_Insight",
        "password": "NewPassword2004",
        "trust_server_certificate": true
    }
}
```

**What Changes**:
- Only 2 lines change (host + server)
- Everything else stays identical
- Difference: 1 IP address update = 2 lines changed

**What Doesn't Change**:
- Credentials (HCAT_Insight / NewPassword2004)
- Database name (IncidentManager)
- Port (1433)
- All other configurations
- All Python code
- All application logic

---

#### METHOD 3: Deploy with Docker/Containers (FUTURE OPTION)
**Status**: Not needed now, but your architecture supports it
**Benefit**: Machine-independent, fully portable

---

## PART 4: IMMEDIATE ACTION PLAN

### STEP 0️⃣: Answer Critical Questions
**Before proceeding, confirm**:
1. Where is SQL Server 2022 Express installed?
   - [ ] On this same Windows machine (localhost)
   - [ ] On a different machine (specify IP: ___________)
2. Preferred approach:
   - [ ] Method 1 (Environment Variables - recommended)
   - [ ] Method 2 (Update JSON config)
   - [ ] Ask ChatGPT first before deciding

---

### STEP 1️⃣: Restore Database from Backup
**Tool**: SQL Server Management Studio (SSMS)

```
1. Open SSMS
2. Connect to: localhost (or database server IP)
3. Right-click "Databases" → "Restore Database..."
4. Select "Device"
5. Click [...] → Browse to: C:\Users\Administrator\Downloads\SQL Queries\SQL Queries\IncidentManager.bak
6. Click OK
7. Database name: leave as "IncidentManager"
8. Click "Restore"
9. Wait 5-10 minutes
10. Verify: Databases tree should show "IncidentManager"
```

**Output to Document**:
- ✓ Restore completed successfully
- ✓ Database appears in SSMS
- ✓ Can query: SELECT COUNT(*) FROM sys.tables

---

### STEP 2️⃣: Determine Database Server Location

**Query to Run**:
```sql
-- Run in SSMS in IncidentManager database
SELECT @@SERVERNAME AS ServerName
SELECT SERVERPROPERTY('MachineName') AS MachineName
SELECT SERVERPROPERTY('Edition') AS Edition
SELECT SERVERPROPERTY('ProductVersion') AS Version
```

**Document**:
- Server name: _____________________
- Machine name: _____________________
- IP Address (ping the server): _____________________

---

### STEP 3️⃣: Update Application Configuration

**Option A: Using Environment Variables (Recommended)**
```powershell
# Create a file: run_with_db.ps1
$env:DB_HOST = "10.x.x.x"  # ← Replace with actual database server IP
$env:DB_PORT = "1433"
$env:DB_USERNAME = "HCAT_Insight"
$env:DB_PASSWORD = "NewPassword2004"

# Activate Python environment
& "venv\Scripts\Activate.ps1"

# Start backend
python backend/main.py
```

**Advantages**:
- No permanent code changes
- Works for your previous architecture
- Flexible for different environments
- Professional approach

---

**Option B: Update JSON Configuration**

**File**: `backend/config/db_settings.json`

**Changes**:
```diff
{
    "database": {
-       "host": "170.70.32.11",
+       "host": "10.x.x.x",            ← DATABASE SERVER IP
-       "server": "170.70.32.11",
+       "server": "10.x.x.x",          ← SAME IP
        "port": 1433,
        "database": "IncidentManager",
        "driver": "ODBC Driver 18 for SQL Server",
        "use_windows_auth": false,
        "username": "HCAT_Insight",
        "password": "NewPassword2004",
        "trust_server_certificate": true
    }
}
```

**Impact Analysis**:
- ✓ Only 2 lines change
- ✓ No Python code changes needed
- ✓ No application logic changes needed
- ✓ Configuration auto-loads from JSON
- ✓ Environment variables can override if set

---

### STEP 4️⃣: Test Database Connection

**Run Test Script**:
```powershell
# Activate environment
& "venv\Scripts\Activate.ps1"

# Test connection
python backend/test_db_connection.py

# Expected output:
# ✅ Connection to IncidentManager successful
# ✅ Tables found: (list of all tables)
# ✅ Sample lookup data exists
```

**Verification Queries**:
```sql
-- Test 1: Database exists
SELECT DB_NAME() AS CurrentDatabase

-- Test 2: Tables exist
SELECT COUNT(*) AS TableCount FROM sys.tables

-- Test 3: Lookup data exists
SELECT COUNT(*) FROM APP_LOOKUP_DOMAIN
SELECT COUNT(*) FROM APP_LOOKUP_CATEGORY
SELECT COUNT(*) FROM APP_LOOKUP_SEVERITY

-- Test 4: User authentication works
SELECT * FROM APP_Users LIMIT 5
```

---

### STEP 5️⃣: Verify Application Functionality

```powershell
# Start backend
python backend/main.py

# Expected output:
# INFO: Uvicorn running on http://0.0.0.0:8000
# INFO: Application started
```

**Manual Tests**:
1. ✓ API responds: `http://localhost:8000/docs`
2. ✓ Can list incidents: `GET /api/incidents`
3. ✓ Lookup tables populate: Check dropdown values
4. ✓ User roles load correctly

---

### STEP 6️⃣: Document Final Configuration

**Create Log**:
```
Migration completed: [Date/Time]
Database Server: [IP address]
Database Name: IncidentManager
Tables Restored: [Count]
Connection Method: [JWT/Windows/SQL Auth]
Configuration: [Method 1/Method 2]
Tests Passed: [List]
Issues Found: [None/List]
```

---

## PART 5: FINAL SAFEGUARDS

### ✅ What WILL NOT Change
- Application code
- Python dependencies
- API endpoints
- Database schema
- User permissions
- Business logic
- Frontend code
- Any previous architecture work

### ✅ What WILL Change
- JSON: 2 lines (host + server IP address)
- OR: Environment variables (no file changes)

### ✅ Rollback Plan
If issues occur:
1. Restore database from clean backup: use original IncidentManager.bak
2. Revert config: restore from git (git checkout backend/config/db_settings.json)
3. Restart application
4. Everything back to original state

---

## PART 6: DECISIONS NEEDED FROM YOU

**Before I execute, please confirm**:

1. **Database Server Location?**
   - [ ] localhost (same VM)
   - [ ] Different IP: ____________
   
2. **Preferred Configuration Method?**
   - [ ] Method 1: Environment Variables (recommended)
   - [ ] Method 2: Update JSON file
   - [ ] Consult ChatGPT first
   
3. **Execute Order?**
   - [ ] Restore backup → Test → Configure
   - [ ] Wait for ChatGPT review first

---

## PART 7: EXECUTION CHECKLIST

Once approved:

```
PRE-EXECUTION:
☐ Backup exists at: C:\Users\Administrator\Downloads\SQL Queries\SQL Queries\IncidentManager.bak
☐ SSMS installed and can connect to SQL Server
☐ Original IncidentManager database backed up (if exists)
☐ Git status clean (no uncommitted changes)

EXECUTION:
☐ STEP 1: Restore database from backup (SSMS)
☐ STEP 2: Verify restore success (check database appears)
☐ STEP 3: Configure connection (Method 1 or 2)
☐ STEP 4: Test database connection (python test_db_connection.py)
☐ STEP 5: Start application (python backend/main.py)
☐ STEP 6: Manual functional tests (login, create incident, etc.)

POST-EXECUTION:
☐ Document final configuration
☐ Git commit if updating JSON: "feat: update database server configuration"
☐ Create new rollback backup
☐ Notify team of new database location
```

---

## ⚠️ QUESTIONS FOR CHATGPT REVIEW

Before I proceed, show ChatGPT:

1. **Is this professional and safe?**
   - Respects previous architecture work?
   - Flexible for future deployments?
   - Proper environment separation?

2. **Configuration method - which is better?**
   - Environment Variables (no code changes)
   - JSON updates (simpler initial setup)
   - Hybrid approach?

3. **Any security concerns?**
   - Credentials in JSON file (same as before)
   - Password in environment variables (temporary)
   - Trust certificate setting (already enabled)

4. **Anything missing?**
   - Error handling?
   - Logging?
   - Monitoring?

---

## READY FOR NEXT STEP

**Please provide**:
1. Database server location (localhost or IP)
2. Preferred configuration method (Env vars or JSON)
3. ChatGPT approval if desired
4. Then I will execute and test immediately

---

**Current Status**: 🔴 AWAITING YOUR DECISION + CHATGPT APPROVAL

