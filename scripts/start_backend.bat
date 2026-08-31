@echo off
REM One shared Python env. Do not create a second venv with TensorFlow/YOLO.
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
cd app\backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
