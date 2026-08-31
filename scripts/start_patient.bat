@echo off
REM Streamlit only talks to the FastAPI backend. No local TensorFlow/YOLO here.
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
streamlit run "app\frontend\streamlit\patient_app.py" --server.port 8502
