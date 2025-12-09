@echo off
REM Installation script for Computer Vision Detection System v2.0

echo ================================================
echo  Computer Vision Detection System v2.0
echo  Installation Script
echo ================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo ✓ Python found
echo.

REM Create virtual environment
echo Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)
echo.

REM Activate and install dependencies
echo Installing dependencies...
call venv\Scripts\activate.bat && pip install --upgrade pip && pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Installation failed!
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo ================================================
echo  ✓ Installation Complete!
echo ================================================
echo.
echo Next steps:
echo   1. Run: scripts\run.bat
echo   or
echo   2. python app.py
echo.
pause
