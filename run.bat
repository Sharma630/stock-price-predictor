@echo off
REM ============================================================
REM Stock Market Index Movement Forecasting - Run the App
REM ============================================================
REM Run this every time you want to start the application.
REM Requires setup.bat to have been run at least once.
REM ============================================================

setlocal

if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo [ERROR] Virtual environment not found.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo Starting Streamlit app...
echo Once it starts, open your browser to: http://localhost:8501
echo Press CTRL+C in this window to stop the app.
echo.

streamlit run app.py

pause
