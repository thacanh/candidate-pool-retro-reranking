@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

if defined CONFORMER_PYTHON (
  set "PYTHON_EXE=%CONFORMER_PYTHON%"
) else if exist "%USERPROFILE%\anaconda3\envs\retrosynthesis-revision-benchmark-py310\python.exe" (
  set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\retrosynthesis-revision-benchmark-py310\python.exe"
) else if exist "%USERPROFILE%\miniconda3\envs\retrosynthesis-revision-benchmark-py310\python.exe" (
  set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\retrosynthesis-revision-benchmark-py310\python.exe"
) else (
  set "PYTHON_EXE=python"
)

set "PYTHONPATH=%CD%\src"
"%PYTHON_EXE%" -m rerank.experiments.run_shared_2d
if errorlevel 1 (
  echo Shared trained 2D comparator failed. Existing partial files were preserved.
  pause
  exit /b 1
)
echo Shared 2D comparator is complete. It is reused by every conformer seed.
pause
