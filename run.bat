@echo off
title AeroPiano Launcher
echo.
echo  ========================================
echo   AeroPiano - Play Piano in the Air
echo  ========================================
echo.
python main.py
if errorlevel 1 (
    echo.
    echo  ERROR: Could not run. Make sure Python is installed
    echo  and dependencies are installed: pip install -r requirements.txt
    pause
)
