@echo off
setlocal
cd /d "%~dp0.."
set "DP_GPU_PYTHON=D:\ModelImprovement\envs\detailpair_cuda_v021\python.exe"
if not exist "%DP_GPU_PYTHON%" exit /b 1
"%DP_GPU_PYTHON%" gpu_check.py
if errorlevel 1 exit /b 1
"%DP_GPU_PYTHON%" run.py synthetic --device 0 --variant B6
if errorlevel 1 exit /b 1
"%DP_GPU_PYTHON%" run.py pack
exit /b %errorlevel%
