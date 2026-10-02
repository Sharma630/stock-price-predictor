# ============================================================
# Stock Market Index Movement Forecasting - Run the App (PowerShell)
# ============================================================
# Usage (from a PowerShell terminal in this folder):
#   .\run.ps1
#
# If you get an execution-policy error, run PowerShell as Administrator
# once and execute:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# ============================================================

$ErrorActionPreference = "Stop"

$venvActivate = ".\.venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvActivate)) {
    Write-Host ""
    Write-Host "[ERROR] Virtual environment not found." -ForegroundColor Red
    Write-Host "Please run setup.bat first (double-click it in File Explorer, or run it from cmd.exe)."
    exit 1
}

Write-Host ""
Write-Host "Activating virtual environment..."
& $venvActivate

Write-Host ""
Write-Host "Starting Streamlit app..."
Write-Host "Once it starts, open your browser to: http://localhost:8501"
Write-Host "Press CTRL+C in this window to stop the app."
Write-Host ""

streamlit run app.py
