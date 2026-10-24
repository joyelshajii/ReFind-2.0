$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'The backend virtual environment is missing. Install backend dependencies first.'
}
# One process keeps one GPU model resident. Reload is available separately for development.
$env:LOCAL_FAST_MODE = 'true'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
Set-Location -LiteralPath $PSScriptRoot
& $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port 8000
