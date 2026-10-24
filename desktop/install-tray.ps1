$ErrorActionPreference = 'Stop'
$project = Join-Path $PSScriptRoot 'ReFind.Tray/ReFind.Tray.csproj'
$config = Join-Path $PSScriptRoot 'ReFind.Tray/NuGet.Config'
$executable = Join-Path $PSScriptRoot 'ReFind.Tray/bin/Release/net9.0-windows/ReFind.Tray.exe'
Get-Process -Name 'ReFind.Tray' -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -eq $executable } |
    Stop-Process
dotnet build $project -c Release --configfile $config
if ($LASTEXITCODE -ne 0) { throw 'ReFind tray build failed.' }

if (-not (Test-Path -LiteralPath $executable)) { throw 'ReFind tray executable was not created.' }

$runKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
New-ItemProperty -Path $runKey -Name 'ReFindTray' -PropertyType String -Value ('"' + $executable + '"') -Force | Out-Null

if (-not (Get-Process -Name 'ReFind.Tray' -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath $executable -WindowStyle Hidden
}

Write-Output "ReFind is running in the notification area and will start when you sign in. Press Alt+Space to search."
