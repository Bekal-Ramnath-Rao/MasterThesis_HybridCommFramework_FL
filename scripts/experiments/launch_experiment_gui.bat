@echo off
setlocal EnableExtensions
rem Requires Python 3 on PATH (or use "py -3" instead of "python" below if you use the launcher).
cd /d "%~dp0"

echo Launching FL Experiment GUI...
echo.

python -c "import PyQt5" 2>nul
if errorlevel 1 (
    echo PyQt5 not found. Installing...
    python -m pip install -r "..\..\Network_Simulation\gui_requirements.txt"
    echo.
)

python "..\..\Network_Simulation\experiment_gui.py"
set "EXITCODE=%ERRORLEVEL%"

if %EXITCODE% equ 0 (
    echo GUI closed successfully
) else (
    echo GUI encountered an error
    exit /b 1
)
endlocal
