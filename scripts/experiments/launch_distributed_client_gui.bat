@echo off
rem Requires Python 3 on PATH (or replace "python" with "py -3" if you use the launcher).
rem Run this on a remote PC to connect a client to the central experiment server.
rem Equivalent to Network_Simulation\launch_distributed_client.sh

cd /d "%~dp0..\..\Network_Simulation"
if not exist "distributed_client_gui.py" (
    echo ERROR: distributed_client_gui.py not found. Expected at:
    echo   %CD%\distributed_client_gui.py
    exit /b 1
)

echo Starting Distributed FL Client GUI...
echo.

python -c "import PyQt5" 2>nul
if errorlevel 1 (
    echo PyQt5 not found. Installing...
    python -m pip install PyQt5
)

python "distributed_client_gui.py"
set "EXITCODE=%ERRORLEVEL%"

echo.
echo Distributed Client GUI closed
exit /b %EXITCODE%
