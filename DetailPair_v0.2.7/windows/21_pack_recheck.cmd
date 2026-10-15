@echo off
setlocal
set "ENV_DIR=D:\ModelImprovement\envs\detailpair_cuda_v021"
if not exist "%ENV_DIR%\python.exe" exit /b 1
cd /d "%~dp0.."
"%ENV_DIR%\python.exe" timing_recheck.py --pack-only
exit /b %ERRORLEVEL%
