@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

if exist "%CD%\assets\unimol_weights\mol_pre_no_h_220816.pt" (
  set "UNIMOL_WEIGHT_DIR=%CD%\assets\unimol_weights"
)

set "CONFORMER_SEED=%~1"
if "%CONFORMER_SEED%"=="" (
  echo ERROR: missing conformer seed.
  goto :failed
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

echo ============================================================
echo Journal of Cheminformatics conformer run
echo Conformer seed: %CONFORMER_SEED%
echo Python: %PYTHON_EXE%
echo Output: outputs\jcheminform_revision\conformers\seed_%CONFORMER_SEED%
echo ============================================================

set "PYTHONPATH=%CD%\src"
"%PYTHON_EXE%" -m rerank.experiments.run_conformer_seed --seed %CONFORMER_SEED%
if errorlevel 1 goto :failed

echo.
echo SUCCESS: seed %CONFORMER_SEED% completed and validated.
echo The large seed-local atom cache was removed after validation.
echo Copy this folder back to the main machine:
echo   outputs\jcheminform_revision\conformers\seed_%CONFORMER_SEED%
echo.
pause
exit /b 0

:failed
echo.
echo FAILED: seed %CONFORMER_SEED% did not complete.
echo The temporary cache is preserved so clicking this file again can resume.
echo Check RUN_STATUS.json and logs inside the seed folder.
echo.
pause
exit /b 1
