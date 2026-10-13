@echo off
setlocal
if "%~1"=="" (
  echo Requires a frozen test record; read REALTIME_EVALUATION_PROTOCOL.md first.
  exit /b 1
)
set "ENV_DIR=D:\ModelImprovement\envs\detailpair_cuda_v021"
if not exist "%ENV_DIR%\python.exe" exit /b 1
set "CONFIG=%~2"
if not defined CONFIG set "CONFIG=%~dp0..\configs\realtime_comparison.yaml"
cd /d "%~dp0.."
"%ENV_DIR%\python.exe" realtime_eval.py --config "%CONFIG%" --final-test --freeze-record "%~1"
exit /b %ERRORLEVEL%
