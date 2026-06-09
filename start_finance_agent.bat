@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "ROOT=%~dp0"
set "PYTHON=D:\AI\soft\conda\envs\python3.11\python.exe"
set "BACKEND_HOST=127.0.0.1"
set "BACKEND_PORT=8010"
set "FRONTEND_HOST=127.0.0.1"
set "FRONTEND_PORT=5173"

set "FIN_AGENT_HOST=%BACKEND_HOST%"
set "FIN_AGENT_PORT=%BACKEND_PORT%"
set "TEMP=%ROOT%.tmp"
set "TMP=%TEMP%"

if not exist "%TEMP%" mkdir "%TEMP%"

if not exist "%PYTHON%" (
  echo [ERROR] Python interpreter not found:
  echo   %PYTHON%
  echo Please update PYTHON in this script.
  pause
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] npm was not found in PATH.
  echo Please install Node.js or open this script from a terminal with npm available.
  pause
  exit /b 1
)

if not exist "%ROOT%frontend\node_modules" (
  echo [ERROR] frontend\node_modules not found.
  echo Please run this once before starting:
  echo   cd /d "%ROOT%frontend"
  echo   npm install
  pause
  exit /b 1
)

echo ==========================================
echo  Finance Agent Platform
echo ==========================================
echo Backend : http://%BACKEND_HOST%:%BACKEND_PORT%
echo Frontend: http://%FRONTEND_HOST%:%FRONTEND_PORT%
echo Project : %ROOT%
echo.

echo Starting backend...
start "finance-agent-backend" /D "%ROOT%" "%ComSpec%" /k ""%PYTHON%" "backend\app\main.py""

echo Starting frontend...
start "finance-agent-frontend" /D "%ROOT%frontend" "%ComSpec%" /k "npm run dev -- --host %FRONTEND_HOST% --port %FRONTEND_PORT%"

echo.
echo Waiting a few seconds before opening browser...
timeout /t 6 /nobreak >nul
start "" "http://%FRONTEND_HOST%:%FRONTEND_PORT%"

echo.
echo Started. Keep the backend/frontend command windows open while using the app.
pause
