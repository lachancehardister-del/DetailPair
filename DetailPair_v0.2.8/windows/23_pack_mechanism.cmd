@echo off
setlocal
set "ENV_DIR=D:\ModelImprovement\envs\detailpair_cuda_v021"
if not exist "%ENV_DIR%\python.exe" (
  echo Existing CUDA environment not found. No automatic installation.
  exit /b 1
)
cd /d "%~dp0.."
set "YOLO_CONFIG_DIR=%CD%\local_settings"
if not exist "%YOLO_CONFIG_DIR%\Ultralytics" mkdir "%YOLO_CONFIG_DIR%\Ultralytics"
set "WANDB_DISABLED=true"
"%ENV_DIR%\python.exe" mechanism_diagnose.py --pack-only
exit /b %ERRORLEVEL%
