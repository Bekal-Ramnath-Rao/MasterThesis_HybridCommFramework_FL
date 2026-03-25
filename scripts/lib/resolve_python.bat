@echo off
rem Sets PYEXE and PYFLAG for callers (after "call" from another .bat).
rem Tries: PYTHON_CMD (full path or "python"), then "python", then "py -3".
rem Usage: call "%REPO%\scripts\lib\resolve_python.bat"

if defined PYTHON_CMD (
    set "PYEXE=%PYTHON_CMD%"
    set "PYFLAG="
    goto :done
)
where python >nul 2>nul && (
    set "PYEXE=python"
    set "PYFLAG="
    goto :done
)
where py >nul 2>nul && (
    set "PYEXE=py"
    set "PYFLAG=-3"
    goto :done
)
echo ERROR: Python not found. Install Python 3 or set PYTHON_CMD to your python.exe.
exit /b 1
:done
exit /b 0
