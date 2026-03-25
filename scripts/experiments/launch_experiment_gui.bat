@echo off
setlocal EnableExtensions
cd /d "%~dp0"

call "%~dp0..\..\scripts\lib\resolve_python.bat"
if errorlevel 1 exit /b 1

echo Launching FL Experiment GUI...
echo.

%PYEXE% %PYFLAG% -c "import PyQt5" 2>nul
if errorlevel 1 (
    echo PyQt5 not found. Installing...
    %PYEXE% %PYFLAG% -m pip install -r "..\..\Network_Simulation\gui_requirements.txt"
    echo.
)

%PYEXE% %PYFLAG% "..\..\Network_Simulation\experiment_gui.py"
set "EXITCODE=%ERRORLEVEL%"

if %EXITCODE% equ 0 (
    echo GUI closed successfully
) else (
    echo GUI encountered an error
    exit /b 1
)
endlocal
