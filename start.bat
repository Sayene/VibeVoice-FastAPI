@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found!
    echo Please run: python install.py
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Load environment variables from .env if it exists
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "line=%%a"
        if not "!line:~0,1!"=="#" (
            set "%%a=%%b"
        )
    )
)

:: Default values
if not defined API_HOST set API_HOST=0.0.0.0
if not defined API_PORT set API_PORT=8001
if not defined API_WORKERS set API_WORKERS=1
if not defined LOG_LEVEL set LOG_LEVEL=info

:: Lowercase LOG_LEVEL for uvicorn (it rejects uppercase like INFO)
for /f %%i in ('python -c "print(''%LOG_LEVEL%''.lower())"') do set LOG_LEVEL_LOWER=%%i

echo ============================================================
echo Starting VibeVoice API Server
echo ============================================================
echo.
echo Server: http://%API_HOST%:%API_PORT%
echo API Docs: http://%API_HOST%:%API_PORT%/docs
echo Workers: %API_WORKERS%
echo Log Level: %LOG_LEVEL_LOWER%
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

:: Start the server
uvicorn api.main:app --host %API_HOST% --port %API_PORT% --workers %API_WORKERS% --log-level %LOG_LEVEL_LOWER%
