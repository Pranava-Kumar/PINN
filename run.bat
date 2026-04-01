@echo off
TITLE SMART-DEORBIT System Runner

echo ============================================================
echo           SMART-DEORBIT System - Dashboard
echo ============================================================

if exist .venv (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate
) else (
    echo [WARN] .venv folder not found. Attempting to run globally.
)

echo [INFO] Launching Streamlit app...
streamlit run app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to start the app. Ensure dependencies are installed.
    pause
)
