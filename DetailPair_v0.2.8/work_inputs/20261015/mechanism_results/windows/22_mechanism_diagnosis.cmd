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
set "CONFIG=%~1"
if not defined CONFIG set "CONFIG=%~dp0..\configs\mechanism_diagnosis.yaml"
echo GPU device0 FP32 endpoint diagnostics, fixed BN, no parameter updates.
echo 64 fixed train images, 32 batches per model. No training or test evaluation.
"%ENV_DIR%\python.exe" mechanism_diagnose.py --config "%CONFIG%"
set "RESULT=%ERRORLEVEL%"
echo Exit code: %RESULT%. Full result ZIP is printed above.
exit /b %RESULT%
