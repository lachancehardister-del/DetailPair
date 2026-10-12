@echo off
setlocal
cd /d "%~dp0.."
set "DP_PYTHON=D:\ModelImprovement\envs\detailpair_cuda_v021\python.exe"
set "DP_PILOT=D:\ModelImprovement\DetailPair_v0.2.2\runs\pilot__20261011_112235_644745"
if not "%~1"=="" set "DP_PILOT=%~1"
if not exist "%DP_PYTHON%" exit /b 1
set "DP_EXIT=0"
"%DP_PYTHON%" checkpoint_debug.py inspect --pilot "%DP_PILOT%"
if errorlevel 1 goto failed
"%DP_PYTHON%" checkpoint_debug.py replay --pilot "%DP_PILOT%" --device cpu
if errorlevel 1 set "DP_EXIT=1"
set "CUDA_LAUNCH_BLOCKING=1"
"%DP_PYTHON%" checkpoint_debug.py replay --pilot "%DP_PILOT%" --device 0
if errorlevel 1 set "DP_EXIT=1"
goto pack
:failed
set "DP_EXIT=1"
:pack
"%DP_PYTHON%" run.py pack
if errorlevel 1 set "DP_EXIT=1"
exit /b %DP_EXIT%
