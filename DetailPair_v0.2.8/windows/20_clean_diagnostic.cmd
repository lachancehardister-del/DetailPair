@echo off
setlocal
set "ENV_DIR=D:\ModelImprovement\envs\detailpair_cuda_v021"
set "CONFIG=%~1"
if not defined CONFIG set "CONFIG=%~dp0..\configs\balanced_recheck.yaml"
if not exist "%ENV_DIR%\python.exe" (
  echo Existing CUDA environment not found. No installation will be attempted.
  exit /b 1
)
cd /d "%~dp0.."
set "YOLO_CONFIG_DIR=%CD%\local_settings"
if not exist "%YOLO_CONFIG_DIR%\Ultralytics" mkdir "%YOLO_CONFIG_DIR%\Ultralytics"
set "WANDB_DISABLED=true"
echo GPU device 0, CUDA12.1, FP32. No training, no test evaluation.
echo Close QQ and other GPU compute tasks yourself before running.
"%ENV_DIR%\python.exe" timing_recheck.py --mode clean-diagnostic --config "%CONFIG%"
set "RESULT=%ERRORLEVEL%"
echo Exit code: %RESULT%. Full result ZIP is printed above.
exit /b %RESULT%
