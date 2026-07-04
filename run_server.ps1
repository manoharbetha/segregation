Set-Location -LiteralPath $PSScriptRoot

$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (Test-Path $pythonExe) {
    & $pythonExe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
} else {
    python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
}
