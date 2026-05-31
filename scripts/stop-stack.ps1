$ports = 3000, 8000, 8788
foreach ($port in $ports) {
  $conns = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
  foreach ($conn in $conns) {
    try { Stop-Process -Id $conn.OwningProcess -Force } catch {}
  }
}
Write-Host "Stack services stopped (ports 3000, 8000, 8788)."
