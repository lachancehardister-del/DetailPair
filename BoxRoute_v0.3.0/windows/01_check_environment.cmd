@echo off
setlocal
set "ENV_DIR=D:\ModelImprovement\envs\detailpair_cuda_v021"
if not exist "%ENV_DIR%\python.exe" (
  echo Existing CUDA environment missing. No automatic installation.
  exit /b 1
)
set "PATH=%ENV_DIR%;%ENV_DIR%\Scripts;%ENV_DIR%\Library\bin;%PATH%"
set "CONDA_PREFIX=%ENV_DIR%"
cd /d "%~dp0.."
set "YOLO_CONFIG_DIR=%CD%\local_settings"
if not exist "%YOLO_CONFIG_DIR%\Ultralytics" mkdir "%YOLO_CONFIG_DIR%\Ultralytics"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
set "WANDB_DISABLED=true"
echo Checking existing GPU0 CUDA environment and pinned source identity.
"%ENV_DIR%\python.exe" run_boxroute.py env
set "RESULT=%ERRORLEVEL%"
echo Exit code: %RESULT%. The result ZIP path is printed above.
exit /b %RESULT%
