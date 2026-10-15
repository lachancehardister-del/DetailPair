@echo off
setlocal
cd /d "%~dp0.."
set "DP_GPU_PYTHON=D:\ModelImprovement\envs\detailpair_cuda_v021\python.exe"
set "DP_DATA=D:\datasets\NEU-DET_DetailPair_s0_20261011_104211_157196\data.yaml"
if not "%~1"=="" set "DP_DATA=%~1"
set "DP_EXIT=0"
if not exist "%DP_GPU_PYTHON%" (
 echo GPU environment not found. Existing CPU environment remains unchanged.
 exit /b 1
)
"%DP_GPU_PYTHON%" gpu_check.py
if errorlevel 1 goto failed
"%DP_GPU_PYTHON%" run.py pilot --data "%DP_DATA%" --device 0 --config configs\convergence.yaml --accept-near-duplicates-for-pilot
if errorlevel 1 goto failed
"%DP_GPU_PYTHON%" run.py collect
if errorlevel 1 goto failed
goto pack
:failed
set "DP_EXIT=1"
:pack
"%DP_GPU_PYTHON%" run.py pack
if errorlevel 1 set "DP_EXIT=1"
exit /b %DP_EXIT%
