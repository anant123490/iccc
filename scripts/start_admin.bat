@echo off
REM Streamlit admin UI. ML runs on the backend process only.
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
streamlit run "app\frontend\streamlit\admin_app.py" --server.port 8503
