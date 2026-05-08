<#
.SYNOPSIS
    HCAT Patient Feedback System — New Server Setup Script
    Run this ONCE whenever the system moves to a new server or IP address.

.PARAMETER Domain
    The IP address or domain name that users will type in their browser.
    Examples:
        .\Setup-NewServer.ps1 -Domain "192.168.1.50"
        .\Setup-NewServer.ps1 -Domain "feedback.hospital.net"
        .\Setup-NewServer.ps1 -Domain "HCAT-SERVER"

.NOTES
    Must be run as Administrator.
    Requires IIS to be installed and running.
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$Domain
)

# ─── Require Administrator ───────────────────────────────────────────────────
if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]"Administrator")) {
    Write-Host ""
    Write-Host "  ERROR: This script must be run as Administrator." -ForegroundColor Red
    Write-Host "  Right-click PowerShell and choose 'Run as Administrator'." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

$ErrorActionPreference = "Stop"
$Domain = $Domain.Trim()

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║   HCAT Patient Feedback — New Server Setup           ║" -ForegroundColor Cyan
Write-Host "  ║   Domain/IP: $Domain" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir  = Join-Path $scriptDir "backend"
$configFile  = Join-Path $backendDir "config\db_settings.json"
$tempDir     = "C:\Temp\HCATSetup"
$certFile    = Join-Path $tempDir "HCAT_RootCA.cer"
$batFile     = Join-Path $tempDir "Install_HCAT_Certificate.bat"
$wwwroot     = "C:\inetpub\wwwroot"

New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

# ─── STEP 1: Create SSL Certificate ─────────────────────────────────────────
Write-Host "  [1/5] Creating SSL certificate for '$Domain'..." -ForegroundColor Yellow

# Determine if Domain is an IP address
$isIP = $Domain -match '^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'

if ($isIP) {
    $infContent = @"
[Version]
Signature="`$Windows NT`$"

[NewRequest]
Subject = "CN=HCAT Patient Feedback"
KeySpec = 1
KeyLength = 2048
Exportable = TRUE
MachineKeySet = TRUE
SMIME = False
PrivateKeyArchive = FALSE
UserProtected = FALSE
UseExistingKeySet = FALSE
ProviderName = "Microsoft RSA SChannel Cryptographic Provider"
ProviderType = 12
RequestType = Cert
KeyUsage = 0xa0

[EnhancedKeyUsageExtension]
OID=1.3.6.1.5.5.7.3.1

[Extensions]
2.5.29.17 = "{text}"
_continue_ = "dns=HCAT&"
_continue_ = "dns=localhost&"
_continue_ = "ipaddress=$Domain&"
_continue_ = "ipaddress=127.0.0.1"
"@
} else {
    $infContent = @"
[Version]
Signature="`$Windows NT`$"

[NewRequest]
Subject = "CN=HCAT Patient Feedback"
KeySpec = 1
KeyLength = 2048
Exportable = TRUE
MachineKeySet = TRUE
SMIME = False
PrivateKeyArchive = FALSE
UserProtected = FALSE
UseExistingKeySet = FALSE
ProviderName = "Microsoft RSA SChannel Cryptographic Provider"
ProviderType = 12
RequestType = Cert
KeyUsage = 0xa0

[EnhancedKeyUsageExtension]
OID=1.3.6.1.5.5.7.3.1

[Extensions]
2.5.29.17 = "{text}"
_continue_ = "dns=$Domain&"
_continue_ = "dns=HCAT&"
_continue_ = "dns=localhost&"
_continue_ = "ipaddress=127.0.0.1"
"@
}

$infPath = Join-Path $tempDir "hcat_cert.inf"
$infContent | Out-File -FilePath $infPath -Encoding ascii

$certOutput = certreq -new -machine $infPath $certFile 2>&1
$certThumb  = ($certOutput | Select-String "Thumbprint: ([A-Fa-f0-9]+)").Matches.Groups[1].Value

if (-not $certThumb) {
    Write-Host "  ERROR: Certificate creation failed." -ForegroundColor Red
    Write-Host $certOutput
    exit 1
}

Write-Host "      Certificate created. Thumbprint: $certThumb" -ForegroundColor Green

# ─── STEP 2: Bind Certificate to IIS Port 443 ────────────────────────────────
Write-Host "  [2/5] Binding certificate to HTTPS port 443..." -ForegroundColor Yellow

$appId = "{4dc3e181-e14b-4a21-b022-59fc669b0914}"
netsh http delete sslcert ipport=0.0.0.0:443 2>&1 | Out-Null
$bindResult = netsh http add sslcert ipport=0.0.0.0:443 certhash=$certThumb appid=$appId certstorename=MY 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Failed to bind certificate to port 443." -ForegroundColor Red
    Write-Host $bindResult
    exit 1
}

# Add to local Trusted Root so this server trusts itself
$cert = Get-Item "Cert:\LocalMachine\MY\$certThumb"
$rootStore = New-Object System.Security.Cryptography.X509Certificates.X509Store("Root", "LocalMachine")
$rootStore.Open("ReadWrite")
$rootStore.Add($cert)
$rootStore.Close()

Write-Host "      HTTPS bound and certificate trusted on this server." -ForegroundColor Green

# ─── STEP 3: Export .cer for distribution ────────────────────────────────────
Write-Host "  [3/5] Exporting certificate for user distribution..." -ForegroundColor Yellow

$certBytes = $cert.Export([System.Security.Cryptography.X509Certificates.X509ContentType]::Cert)
[System.IO.File]::WriteAllBytes($certFile, $certBytes)

if (Test-Path $wwwroot) {
    Copy-Item $certFile (Join-Path $wwwroot "HCAT_RootCA.cer") -Force
    Write-Host "      Certificate available at: http://$Domain/HCAT_RootCA.cer" -ForegroundColor Green
}

# ─── STEP 4: Create user installer script ────────────────────────────────────
Write-Host "  [4/5] Creating user certificate installer..." -ForegroundColor Yellow

$batContent = @"
@echo off
NET SESSION >nul 2>&1
if %ERRORLEVEL% neq 0 (
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)
echo.
echo  Installing HCAT certificate...
powershell -Command "try { (New-Object System.Net.WebClient).DownloadFile('http://$Domain/HCAT_RootCA.cer','%TEMP%\HCAT_RootCA.cer') } catch { Copy-Item '%~dp0HCAT_RootCA.cer' '%TEMP%\HCAT_RootCA.cer' -ErrorAction SilentlyContinue }"
certutil -addstore -f "Root" "%TEMP%\HCAT_RootCA.cer" >nul 2>&1
echo.
echo  Done! Please close and reopen your browser.
echo  Then go to: https://$Domain
echo.
del "%TEMP%\HCAT_RootCA.cer" >nul 2>&1
pause
"@

$batContent | Out-File -FilePath $batFile -Encoding ascii
if (Test-Path $wwwroot) {
    Copy-Item $batFile (Join-Path $wwwroot "Install_HCAT_Certificate.bat") -Force
}
Write-Host "      Installer available at: http://$Domain/Install_HCAT_Certificate.bat" -ForegroundColor Green

# ─── STEP 5: Update db_settings.json CORS ────────────────────────────────────
Write-Host "  [5/5] Updating CORS configuration in db_settings.json..." -ForegroundColor Yellow

if (Test-Path $configFile) {
    $json = Get-Content $configFile -Raw | ConvertFrom-Json

    # Build new cors_origins list
    $newCors = @(
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1",
        "https://localhost",
        "https://127.0.0.1"
    )

    if ($isIP) {
        $newCors += "http://$Domain"
        $newCors += "https://$Domain"
    } else {
        $newCors += "http://$Domain"
        $newCors += "https://$Domain"
    }

    $json.network.cors_origins = $newCors

    $json | ConvertTo-Json -Depth 10 | Set-Content $configFile -Encoding UTF8
    Write-Host "      CORS updated for: $Domain" -ForegroundColor Green
} else {
    Write-Host "      WARNING: db_settings.json not found at $configFile" -ForegroundColor Yellow
    Write-Host "      Manually add 'http://$Domain' and 'https://$Domain' to cors_origins." -ForegroundColor Yellow
}

# ─── Restart Backend Service ──────────────────────────────────────────────────
Write-Host ""
Write-Host "  Restarting backend service..." -ForegroundColor Yellow
try {
    & net stop PatientFeedbackAPI 2>&1 | Out-Null
    & net start PatientFeedbackAPI 2>&1 | Out-Null
    Write-Host "  Backend service restarted." -ForegroundColor Green
} catch {
    Write-Host "  Note: Could not restart PatientFeedbackAPI automatically." -ForegroundColor Yellow
    Write-Host "  Run manually: net stop PatientFeedbackAPI && net start PatientFeedbackAPI" -ForegroundColor Yellow
}

# Restart IIS
iisreset /noforce 2>&1 | Out-Null
Write-Host "  IIS restarted." -ForegroundColor Green

# ─── Summary ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "  ║   Setup Complete!                                    ║" -ForegroundColor Green
Write-Host "  ╠══════════════════════════════════════════════════════╣" -ForegroundColor Green
Write-Host "  ║   Application URL : https://$Domain" -ForegroundColor Green
Write-Host "  ║   Cert file       : $wwwroot\HCAT_RootCA.cer" -ForegroundColor Green
Write-Host "  ║   User installer  : http://$Domain/Install_HCAT_Certificate.bat" -ForegroundColor Green
Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "  NEXT STEPS:" -ForegroundColor Cyan
Write-Host "  1. Test on THIS machine: open https://$Domain in your browser" -ForegroundColor White
Write-Host "  2. For the 140 users: they go to http://$Domain/Install_HCAT_Certificate.bat" -ForegroundColor White
Write-Host "     download it, right-click → Run as Administrator" -ForegroundColor White
Write-Host "  3. After running the installer, tell users to reopen their browser" -ForegroundColor White
Write-Host "     and go to https://$Domain" -ForegroundColor White
Write-Host ""
