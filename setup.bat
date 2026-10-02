@echo off
REM ============================================================
REM Stock Market Index Movement Forecasting - Windows Setup
REM ============================================================
REM Run this ONCE. It will:
REM   1. Check that Python is available
REM   2. Create a virtual environment (.venv) if it does not exist
REM   3. Activate the virtual environment
REM   4. Upgrade pip
REM   5. Install all required packages from requirements.txt
REM ============================================================

setlocal

echo.
echo === Step 1/4: Checking Python installation ===
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python was not found on your PATH.
    echo Please install Python 3.10 or 3.11 from https://www.python.org/downloads/
    echo During installation, make sure to check "Add Python to PATH".
    pause
    exit /b 1
)
python --version

echo.
echo === Step 2/4: Creating virtual environment (.venv) ===
if exist ".venv\Scripts\activate.bat" (
    echo Virtual environment already exists. Skipping creation.
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo.
echo === Step 3/4: Activating virtual environment and upgrading pip ===
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo.
echo === Step 4/4: Installing required packages (this may take a few minutes) ===
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Package installation failed. Check the error message above.
    echo If a specific package failed, try removing its version pin in requirements.txt
    echo and re-running this script.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Setup complete!
echo To start the application, run:  run.bat
echo ============================================================
pause
