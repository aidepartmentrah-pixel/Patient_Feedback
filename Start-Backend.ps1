# Patient Feedback Backend — Stable Startup Script
# Kills any existing instance, waits for DB, then starts uvicorn.

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

# 2. Wait until SQL Server is reachable (up to 30 seconds)
Write-Host "Waiting for database..."
$dbReady = $false
for ($i = 0; $i -lt 15; $i++) {
    $test = & $PYTHON -c "
import sys
sys.path.insert(0, r'$WORKDIR')
from core.database import get_connection
try:
    c = get_connection(); c.close(); print('OK')
except: print('FAIL')
" 2>$null
    if ($test -eq "OK") { $dbReady = $true; Write-Host "Database ready."; break }
    Start-Sleep -Seconds 2
}
if (-not $dbReady) { Write-Host "WARNING: DB not ready — starting anyway (bootstrap mode may activate)." }

# 3. Start uvicorn as a background process
Write-Host "Starting backend on port $PORT..."
Start-Process -FilePath $PYTHON `
    -ArgumentList "-m uvicorn main:app --host 0.0.0.0 --port $PORT" `
    -WorkingDirectory $WORKDIR `
    -WindowStyle Hidden

# 4. Wait and confirm bootstrap_mode is false
Start-Sleep -Seconds 12
try {
    $status = (Invoke-WebRequest -Uri "http://127.0.0.1:$PORT/api/status" -UseBasicParsing -TimeoutSec 5).Content | ConvertFrom-Json
    if ($status.bootstrap_mode -eq $false) {
        Write-Host "Backend started successfully. Database connected. bootstrap_mode=false"
    } else {
        Write-Host "WARNING: Backend started in bootstrap mode — DB connection failed at startup."
    }
} catch {
    Write-Host "Backend starting... (status check timed out, may still be loading)"
}
