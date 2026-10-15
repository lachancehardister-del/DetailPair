@echo off
setlocal
cd /d "%~dp0.."
if not exist "D:\ModelImprovement\envs\detailpair_v02\python.exe" exit /b 1
"D:\ModelImprovement\envs\detailpair_v02\python.exe" gpu_setup.py
exit /b %errorlevel%
