@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
python -m fibre_sight
if errorlevel 1 (
    echo.
    echo FibreSight could not start. Activate the fibre-sight environment, then run this file again.
    pause
)
endlocal

