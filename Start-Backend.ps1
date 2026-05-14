# Patient Feedback Backend - Stable Startup Script
# Kills any existing instance, waits for SQL Server service + DB, then starts uvicorn.

$PYTHON  = "C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
$WORKDIR = "c:\Users\Administrator\Documents\GitHub\Patient_Feedback\backend"
$PORT    = 8000

# 1. Kill any existing backend on port 8000
$existing = netstat -ano | findstr "LISTEN" | findstr ":$PORT "
if ($existing) {
    $oldPid = ($existing -split "\s+")[-1]
    Write-Host "Stopping existing backend PID $oldPid..."
    Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
}

# 2a. Wait for MSSQL$SQLEXPRESS Windows SERVICE to reach Running state (up to 120 s)
# This is checked BEFORE the DB connection test because the service must be Running
# before any TCP connection is possible. At VM boot, SQL Server can take 30-90 seconds.
Write-Host "Waiting for SQL Server service to be Running..."
$svcReady = $false
for ($i = 0; $i -lt 60; $i++) {
    $svc = Get-Service 'MSSQL$SQLEXPRESS' -ErrorAction SilentlyContinue
    if ($svc -and $svc.Status -eq 'Running') {
        $svcReady = $true
        Write-Host "SQL Server service is Running (waited $($i * 2) s)."
        break
    }
    Start-Sleep -Seconds 2
}
if (-not $svcReady) {
    Write-Host "WARNING: SQL Server service did not reach Running state in 120 s."
    Write-Host "         Attempting to start it..."
    Start-Service 'MSSQL$SQLEXPRESS' -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 10
}

# 2b. Wait until SQL Server DB connection succeeds (up to 90 seconds)
Write-Host "Waiting for database connection..."
$dbReady = $false
for ($i = 0; $i -lt 45; $i++) {
    $test = & $PYTHON -c @"
import sys
sys.path.insert(0, r'$WORKDIR')
from core.database import get_connection
try:
    c = get_connection(); c.close(); print('OK')
except Exception as e:
    print('FAIL')
"@ 2>$null
    if ($test -eq "OK") { $dbReady = $true; Write-Host "Database connection OK (waited $($i * 2) s)."; break }
    Start-Sleep -Seconds 2
}
if (-not $dbReady) {
    Write-Host "WARNING: DB connection not established in 90 s - starting backend in bootstrap mode."
    Write-Host "         The backend will auto-retry every 15s and exit bootstrap mode when DB is ready."
}

# 3. Start uvicorn as a background process
Write-Host "Starting backend on port $PORT..."
Start-Process -FilePath $PYTHON `
    -ArgumentList "-m uvicorn main:app --host 0.0.0.0 --port $PORT" `
    -WorkingDirectory $WORKDIR `
    -WindowStyle Hidden

# 4. Wait and confirm bootstrap_mode
Start-Sleep -Seconds 15
try {
    $status = (Invoke-WebRequest -Uri "http://127.0.0.1:$PORT/api/status" -UseBasicParsing -TimeoutSec 5).Content | ConvertFrom-Json
    if ($status.bootstrap_mode -eq $false) {
        Write-Host "Backend started successfully. Database connected. bootstrap_mode=false"
    } else {
        Write-Host "WARNING: Backend started in bootstrap mode - DB connection failed at startup."
        Write-Host "         The background watcher will retry every 15 s automatically."
    }
} catch {
    Write-Host "Backend starting... (status check timed out - still loading ML models)"
}
