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
if not defined CONFIG set "CONFIG=%~dp0..\configs\matched_checkpoint_eval.yaml"
echo Inventory existing old40 B5/B6 checkpoints, then GPU0 FP32 evaluation.
echo Latest common completed epoch up to38. Only clean and Gaussian.03 val. No training.
"%ENV_DIR%\python.exe" matched_checkpoint_eval.py --config "%CONFIG%"
set "RESULT=%ERRORLEVEL%"
echo Exit code: %RESULT%. Full result ZIP is printed above.
exit /b %RESULT%
