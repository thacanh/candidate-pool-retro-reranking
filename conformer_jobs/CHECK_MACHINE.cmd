@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

if exist "%CD%\assets\unimol_weights\mol_pre_no_h_220816.pt" (
  set "UNIMOL_WEIGHT_DIR=%CD%\assets\unimol_weights"
)

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
echo Checking Python, hashes, disk and automatic CPU/GPU selection without computation...
"%PYTHON_EXE%" -m rerank.experiments.run_conformer_seed --seed 42 --dry-run
if errorlevel 1 (
  echo.
  echo MACHINE CHECK FAILED. Do not start a seed on this machine yet.
  pause
  exit /b 1
)

echo.
echo MACHINE CHECK PASSED. The JSON above shows the selected device, VRAM,
echo batch size and CPU threads. This did not generate a conformer.
pause
exit /b 0
