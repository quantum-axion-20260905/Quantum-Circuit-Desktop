$ports = 3000, 8000, 8788
$listening = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in $ports } | Sort-Object LocalPort

if (!$listening) {
  Write-Host "No stack services are listening on 3000/8000/8788."
  exit 0
}

$listening | Format-Table -AutoSize

function PingJson([string]$url) {
  try {
    return (Invoke-WebRequest -UseBasicParsing $url -TimeoutSec 4).Content
  } catch {
    return "ERROR: $($_.Exception.Message)"
  }
}

Write-Host "`nHealth checks:"
Write-Host "agent /health: $(PingJson 'http://127.0.0.1:8788/health')"
Write-Host "agent /hardware: $(PingJson 'http://127.0.0.1:8788/hardware')"
Write-Host "backend /api/projects: $(PingJson 'http://127.0.0.1:8000/api/projects/')"
