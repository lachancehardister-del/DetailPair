@echo off
setlocal
cd /d "%~dp0.."
set "DP_PYTHON=D:\ModelImprovement\envs\detailpair_cuda_v021\python.exe"
set "DP_DATA=D:\datasets\NEU-DET_DetailPair_s0_20261011_104211_157196\data.yaml"
if not "%~1"=="" set "DP_DATA=%~1"
if not exist "%DP_PYTHON%" exit /b 1
"%DP_PYTHON%" compare_long.py --data "%DP_DATA%" --accept-reviewed-near-duplicates
exit /b %errorlevel%
