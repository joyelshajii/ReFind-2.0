$ErrorActionPreference = 'Stop'
$runKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
Remove-ItemProperty -Path $runKey -Name 'ReFindTray' -ErrorAction SilentlyContinue
Get-Process -Name 'ReFind.Tray' -ErrorAction SilentlyContinue | Stop-Process
Write-Output 'ReFind tray startup removed.'
