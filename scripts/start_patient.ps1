Set-Location (Join-Path $PSScriptRoot "..")
if (Test-Path ".\.venv\Scripts\Activate.ps1") { . .\.venv\Scripts\Activate.ps1 }
streamlit run ".\app\frontend\streamlit\patient_app.py" --server.port 8502
