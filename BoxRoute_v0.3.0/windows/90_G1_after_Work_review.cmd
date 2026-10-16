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
if "%~1"=="" (
  echo G1 requires a real G0 reviewed by Work. Supply the approved review-record JSON as argument1.
  echo Do not run G1 before review. No training started.
  exit /b 2
)
set "PATHS=%~2"
if not defined PATHS set "PATHS=%CD%\configs\local_paths.yaml"
"%ENV_DIR%\python.exe" run_boxroute.py g1 --Work-review "%~1" --paths "%PATHS%"
set "RESULT=%ERRORLEVEL%"
echo Exit code: %RESULT%. The result ZIP path is printed above.
exit /b %RESULT%
