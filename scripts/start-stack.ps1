param(
  [ValidateSet("dev", "research")]
  [string]$Profile = "",
  [switch]$WithWatcher
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$npmCmd = "C:\Program Files\nodejs\npm.cmd"
$configPath = Join-Path $root "configs\runtime.profiles.json"

if (!(Test-Path $venvPython)) { throw ".venv python not found: $venvPython" }
if (!(Test-Path $npmCmd)) { throw "npm.cmd not found: $npmCmd" }
if (!(Test-Path $configPath)) { throw "profile config not found: $configPath" }

$cfg = Get-Content $configPath -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($Profile)) { $Profile = $cfg.default_profile }
$p = $cfg.profiles.$Profile
if ($null -eq $p) { throw "Profile '$Profile' not found in $configPath" }

function Stop-Port([int]$port) {
  $conns = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
  foreach ($c in $conns) {
    try { Stop-Process -Id $c.OwningProcess -Force } catch {}
  }
}

function Start-ServiceProc(
  [string]$Name,
  [string]$FilePath,
  [string[]]$ArgumentList,
  [string]$WorkDir,
  [string]$StdOut,
  [string]$StdErr,
  [hashtable]$Env
) {
  $cmdArgs = @()
  foreach ($k in $Env.Keys) {
    $v = [string]$Env[$k]
    $cmdArgs += @("-Command", "`$env:$k='$v'; & '$FilePath' " + ($ArgumentList -join " "))
    break
  }
  if ($Env.Keys.Count -gt 1) {
    $prefix = ""
    foreach ($k in $Env.Keys) { $prefix += "`$env:$k='" + [string]$Env[$k] + "'; " }
    $argString = $prefix + "& '$FilePath' " + ($ArgumentList -join " ")
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $argString -WorkingDirectory $WorkDir -RedirectStandardOutput $StdOut -RedirectStandardError $StdErr -WindowStyle Hidden | Out-Null
    return
  }
  if ($Env.Keys.Count -eq 1) {
    $psArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass")
    $psArgs += $cmdArgs
    Start-Process -FilePath "powershell.exe" -ArgumentList $psArgs -WorkingDirectory $WorkDir -RedirectStandardOutput $StdOut -RedirectStandardError $StdErr -WindowStyle Hidden | Out-Null
    return
  }
  Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -WorkingDirectory $WorkDir -RedirectStandardOutput $StdOut -RedirectStandardError $StdErr -WindowStyle Hidden | Out-Null
}

function To-Hashtable($obj) {
  $h = @{}
  if ($null -eq $obj) { return $h }
  foreach ($prop in $obj.PSObject.Properties) {
    $h[$prop.Name] = [string]$prop.Value
  }
  return $h
}

Stop-Port 3000
Stop-Port 8000
Stop-Port 8788

$agentOut = Join-Path $root "agent_server_out.txt"
$agentErr = Join-Path $root "agent_server_err.txt"
$backendOut = Join-Path $root "backend_out.txt"
$backendErr = Join-Path $root "backend_err.txt"
$webOut = Join-Path $root "web_dev_out.txt"
$webErr = Join-Path $root "web_dev_err.txt"

Start-ServiceProc -Name "agent" -FilePath $venvPython -ArgumentList @("agent/app.py") -WorkDir $root -StdOut $agentOut -StdErr $agentErr -Env (To-Hashtable $p.agent)
Start-ServiceProc -Name "backend" -FilePath $venvPython -ArgumentList @("backend/manage.py","runserver","127.0.0.1:8000") -WorkDir $root -StdOut $backendOut -StdErr $backendErr -Env (To-Hashtable $p.backend)
$webEnv = To-Hashtable $p.web
$webPrefix = ""
foreach ($k in $webEnv.Keys) { $webPrefix += "`$env:$k='" + $webEnv[$k] + "'; " }
$webCmd = $webPrefix + "& '$npmCmd' run dev:web"
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-Command",$webCmd -WorkingDirectory $root -RedirectStandardOutput $webOut -RedirectStandardError $webErr -WindowStyle Hidden | Out-Null

Start-Sleep -Seconds 4

if ($WithWatcher) {
  $watcher = Join-Path $root "scripts\watch-stack.ps1"
  Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File",$watcher,"-Profile",$Profile -WorkingDirectory $root -WindowStyle Hidden | Out-Null
}

Write-Host "Stack started with profile '$Profile'"
Write-Host "Run status check: .\scripts\status-stack.ps1"
