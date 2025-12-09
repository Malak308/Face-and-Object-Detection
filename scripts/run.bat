@echo off
REM Quick launcher for Computer Vision Detection System - Updated Structure

echo Starting Computer Vision Detection System...
echo.

REM Check if venv exists
if not exist venv (
    echo Virtual environment not found!
    echo Please run install.bat first.
    pause
    exit /b 1
)

REM Activate and run the new app.py
call venv\Scripts\activate.bat && python app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application failed to start!
    echo Make sure all dependencies are installed.
    pause
)
