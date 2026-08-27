@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

where conda >nul 2>nul
if errorlevel 1 (
  echo ERROR: Conda was not found. Install Miniconda or Anaconda first.
  pause
  exit /b 1
)

echo Creating the pinned retrosynthesis-revision-benchmark-py310 environment...
call conda env create -f environment-revision.yml
if errorlevel 1 (
  echo.
  echo Environment creation failed. If the environment already exists, run
  echo CHECK_MACHINE.cmd; otherwise inspect the Conda error above.
  pause
  exit /b 1
)

where nvidia-smi >nul 2>nul
if errorlevel 1 goto :gpu_done

echo.
echo NVIDIA GPU detected. Installing the pinned CUDA 11.8 PyTorch build...
call conda run -n retrosynthesis-revision-benchmark-py310 python -m pip install --force-reinstall --no-deps torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
  echo WARNING: CUDA PyTorch installation failed. The launcher will use CPU if
  echo the existing PyTorch installation still imports correctly.
) else (
  call conda run -n retrosynthesis-revision-benchmark-py310 python -c "import torch; print('PyTorch',torch.__version__,'CUDA runtime',torch.version.cuda,'available',torch.cuda.is_available()); assert torch.cuda.is_available()"
  if errorlevel 1 (
    echo WARNING: NVIDIA hardware was found, but PyTorch cannot use CUDA.
    echo CHECK_MACHINE.cmd will report the exact fallback before any run starts.
  )
)

:gpu_done

echo.
echo Environment created. Run CHECK_MACHINE.cmd before starting a seed.
echo GPU use is automatic when CUDA is available; otherwise the job uses CPU.
pause
exit /b 0
