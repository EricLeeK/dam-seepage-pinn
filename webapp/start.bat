@echo off
REM Start Dam Seepage PINN Web App (Windows)
REM Usage: double-click webapp\start.bat

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "BACKEND_DIR=%SCRIPT_DIR%backend"
set "FRONTEND_DIR=%SCRIPT_DIR%frontend"

set "BACKEND_VENV=%BACKEND_DIR%\.venv"
set "BACKEND_PYTHON=%BACKEND_VENV%\Scripts\python.exe"
set "BACKEND_PORT=8000"
set "FRONTEND_PORT=5173"

REM ── Check prerequisites ──
if not exist "%BACKEND_VENV%" (
    echo Backend venv not found. Creating...
    python -m venv "%BACKEND_VENV%"
    "%BACKEND_PYTHON%" -m pip install -r "%BACKEND_DIR%\requirements.txt"
)

if not exist "%FRONTEND_DIR%\node_modules" (
    echo Frontend node_modules not found. Installing...
    cd /d "%FRONTEND_DIR%"
    call npm install
)

REM ── Start backend ──
echo Starting backend on port %BACKEND_PORT%...
cd /d "%BACKEND_DIR%"
start "PINN Backend" "%BACKEND_PYTHON%" -m uvicorn main:app --host 0.0.0.0 --port %BACKEND_PORT% --reload

REM ── Start frontend ──
echo Starting frontend on port %FRONTEND_PORT%...
cd /d "%FRONTEND_DIR%"
start "PINN Frontend" npx vite --host

REM ── Wait and open browser ──
echo.
echo Waiting for servers...
timeout /t 5 /nobreak > nul
echo.
echo =========================================
echo   Dam Seepage PINN Web App is ready!
echo   Frontend: http://localhost:%FRONTEND_PORT%
echo   Backend:  http://localhost:%BACKEND_PORT%
echo =========================================
echo.
start http://localhost:%FRONTEND_PORT%

echo.
echo Press any key to stop all servers...
pause > nul

taskkill /FI "WindowTitle eq PINN Backend*" /F > nul 2>&1
taskkill /FI "WindowTitle eq PINN Frontend*" /F > nul 2>&1
echo Servers stopped.
