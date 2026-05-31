param(
  [ValidateSet("dev", "research")]
  [string]$Profile = "research"
)

$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$npmCmd = "C:\Program Files\nodejs\npm.cmd"
$configPath = Join-Path $root "configs\runtime.profiles.json"
$cfg = Get-Content $configPath -Raw | ConvertFrom-Json
$p = $cfg.profiles.$Profile

function To-Hashtable($obj) {
  $h = @{}
  if ($null -eq $obj) { return $h }
  foreach ($prop in $obj.PSObject.Properties) { $h[$prop.Name] = [string]$prop.Value }
  return $h
}

function Start-WithEnv([string]$file, [string[]]$args, [string]$stdout, [string]$stderr, [hashtable]$envMap) {
  $prefix = ""
  foreach ($k in $envMap.Keys) { $prefix += "`$env:$k='" + $envMap[$k] + "'; " }
  $cmd = $prefix + "& '$file' " + ($args -join " ")
  Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-Command",$cmd -WorkingDirectory $root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden | Out-Null
}

function Is-Up([int]$port) {
  $conn = Get-NetTCPConnection -State Listen -LocalPort $port
  return $null -ne $conn
}

while ($true) {
  if (-not (Is-Up 8788)) {
    Start-WithEnv $venvPython @("agent/app.py") (Join-Path $root "agent_server_out.txt") (Join-Path $root "agent_server_err.txt") (To-Hashtable $p.agent)
  }
  if (-not (Is-Up 8000)) {
    Start-WithEnv $venvPython @("backend/manage.py","runserver","127.0.0.1:8000") (Join-Path $root "backend_out.txt") (Join-Path $root "backend_err.txt") (To-Hashtable $p.backend)
  }
  if (-not (Is-Up 3000)) {
    $webEnv = To-Hashtable $p.web
    $webPrefix = ""
    foreach ($k in $webEnv.Keys) { $webPrefix += "`$env:$k='" + $webEnv[$k] + "'; " }
    $webCmd = $webPrefix + "& '$npmCmd' run dev:web"
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-Command",$webCmd -WorkingDirectory $root -RedirectStandardOutput (Join-Path $root "web_dev_out.txt") -RedirectStandardError (Join-Path $root "web_dev_err.txt") -WindowStyle Hidden | Out-Null
  }
  Start-Sleep -Seconds 5
}
