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
if not exist "outputs\jcheminform_revision\shared_2d_legacy_fixed50\COMPLETED.json" (
  echo Missing the shared trained 2D comparator.
  echo Run conformer_jobs\RUN_SHARED_2D.cmd once before collecting results.
  pause
  exit /b 1
)
"%PYTHON_EXE%" -m rerank.data.collect_conformer_runs
if errorlevel 1 (
  echo.
  echo Collection failed. Check that seed_42 through seed_51 are all present.
  pause
  exit /b 1
)

"%PYTHON_EXE%" -m rerank.analysis.analyze_conformer_aggregate
if errorlevel 1 (
  echo.
  echo Aggregate analysis failed. No conformer cache will be regenerated.
  echo Inspect outputs\jcheminform_revision\conformer_aggregate and rerun only after review.
  pause
  exit /b 1
)

echo.
echo All ten seed folders, retained checksums, B1, B2, and B3 passed.
echo Compact tables, manifests, and averaged scalar caches were written under:
echo outputs\jcheminform_revision\conformer_aggregate
pause
exit /b 0
