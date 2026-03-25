@echo off
rem Run on a remote PC to connect a client to the central experiment server.
rem Override Python: set PYTHON_CMD=path\to\python.exe before running.

cd /d "%~dp0..\..\Network_Simulation"
if not exist "distributed_client_gui.py" (
    echo ERROR: distributed_client_gui.py not found. Expected at:
    echo   %CD%\distributed_client_gui.py
    exit /b 1
)

call "%~dp0..\..\scripts\lib\resolve_python.bat"
if errorlevel 1 exit /b 1

echo Starting Distributed FL Client GUI...
echo.

%PYEXE% %PYFLAG% -c "import PyQt5" 2>nul
if errorlevel 1 (
    echo PyQt5 not found. Installing...
    %PYEXE% %PYFLAG% -m pip install PyQt5
)

%PYEXE% %PYFLAG% "distributed_client_gui.py"
set "EXITCODE=%ERRORLEVEL%"

echo.
echo Distributed Client GUI closed
exit /b %EXITCODE%
