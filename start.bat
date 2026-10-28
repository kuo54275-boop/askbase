@echo off
REM AskBase one-click launcher for Windows.
REM All logic lives in start.ps1 (UTF-8 with BOM, so Chinese renders correctly
REM in both PowerShell 5.1 and 7). This wrapper only bypasses the execution policy.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
if errorlevel 1 (
    echo.
    echo Startup failed. Check the messages above.
    pause
)
