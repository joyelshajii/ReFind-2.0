$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'The backend virtual environment is missing. From backend, run: python -m venv .venv; ./.venv/Scripts/python.exe -m pip install -r requirements.txt'
}
& $pythonPath -c 'import av'
if ($LASTEXITCODE -ne 0) {
    throw 'The backend media decoder is missing or broken. From backend, run: ./.venv/Scripts/python.exe -m pip install -r requirements.txt'
}
# One process keeps one GPU model resident. Reload is available separately for development.
$env:LOCAL_FAST_MODE = 'true'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
Set-Location -LiteralPath $PSScriptRoot
& $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port 8000
if ($LASTEXITCODE -ne 0) { throw 'ReFind backend failed to start. See the error above.' }
