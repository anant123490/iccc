# One shared venv. Do not install TensorFlow/YOLO a second time.
Set-Location (Join-Path $PSScriptRoot "..")
if (Test-Path ".\.venv\Scripts\Activate.ps1") { . .\.venv\Scripts\Activate.ps1 }
Set-Location ".\app\backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
