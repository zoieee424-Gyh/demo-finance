@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=D:\AI\soft\conda\envs\python3.11\python.exe"

cd /d "%ROOT%"

echo Starting app on http://127.0.0.1:8010 ...
echo The FastAPI backend will serve both /api and the built frontend.
echo Keep this window open while using the app.
echo.

"%PYTHON%" backend\app\main.py

pause
