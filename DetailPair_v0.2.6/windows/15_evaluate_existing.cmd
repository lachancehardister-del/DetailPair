@echo off
setlocal
set "ENV_DIR=D:\ModelImprovement\envs\detailpair_cuda_v021"
set "CONFIG=%~1"
if not defined CONFIG set "CONFIG=%~dp0..\configs\realtime_comparison.yaml"
if not exist "%ENV_DIR%\python.exe" (
  echo Existing environment not found: "%ENV_DIR%\python.exe"
  exit /b 1
)
cd /d "%~dp0.."
set "YOLO_CONFIG_DIR=%CD%\local_settings"
set "WANDB_DISABLED=true"
echo Evaluation only. No training or dependency installation.
echo Stop other GPU computations yourself before running this entry.
"%ENV_DIR%\python.exe" realtime_eval.py --config "%CONFIG%"
set "RESULT=%ERRORLEVEL%"
echo Exit code: %RESULT%. Result ZIP path is printed above, including failures.
exit /b %RESULT%
