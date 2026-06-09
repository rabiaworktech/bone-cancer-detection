@echo off
echo Starting Bone Cancer Detection API...
cd "%~dp0backend"
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
pause
