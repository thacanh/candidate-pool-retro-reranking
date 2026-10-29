@echo off
setlocal EnableExtensions
cd /d "%~dp0\..\.."

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0package_inputs.ps1"
if errorlevel 1 (
  echo.
  echo Bundle creation failed. Review the error above.
  pause
  exit /b 1
)

echo.
echo Bundle creation completed. Check the machine_bundles folder.
pause
exit /b 0
