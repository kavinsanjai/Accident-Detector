@echo off
echo ============================================================
echo    ACCIDENT DETECTION SIMULATOR - LAUNCHER
echo ============================================================
echo.
echo Starting the web server...
echo.
echo The simulator will open in your browser automatically.
echo.
echo Press Ctrl+C to stop the server when done.
echo ============================================================
echo.

cd /d "%~dp0"
python app.py

pause
