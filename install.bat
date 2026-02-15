@echo off
:: VibeVoice-FastAPI One-Click Installer for Windows
echo.
echo ============================================================
echo   VibeVoice-FastAPI One-Click Installer
echo ============================================================
echo.

:: Check for git
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: git is not installed.
    echo.
    echo Download from: https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

:: Check for Python 3
set PYTHON_CMD=
where python >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('python -c "import sys; print(sys.version_info.major)"') do set PY_MAJOR=%%i
    if "%PY_MAJOR%"=="3" set PYTHON_CMD=python
)
if not defined PYTHON_CMD (
    where python3 >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=python3
    )
)

if not defined PYTHON_CMD (
    echo ERROR: Python 3 is not installed.
    echo.
    echo Download from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo Using: git
echo Using: %PYTHON_CMD%
echo.

:: Clone or update repo
set INSTALL_DIR=VibeVoice-FastAPI
if exist "%INSTALL_DIR%" (
    echo Found existing %INSTALL_DIR% directory.
    echo Updating...
    git -C "%INSTALL_DIR%" pull
) else (
    echo Cloning VibeVoice-FastAPI...
    git clone https://github.com/ncoder-ai/VibeVoice-FastAPI.git "%INSTALL_DIR%"
)

echo.

:: Run interactive installer
cd "%INSTALL_DIR%"
%PYTHON_CMD% install.py

pause
