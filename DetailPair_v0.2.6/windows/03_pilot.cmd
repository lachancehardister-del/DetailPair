@echo off
cd /d "%~dp0.."
if "%~1"=="" (
 echo Usage: windows\03_pilot.cmd "D:\actual_data\data.yaml" [cpu or 0]
 exit /b 2
)
set "DETAILPAIR_DEVICE=%~2"
if "%DETAILPAIR_DEVICE%"=="" set "DETAILPAIR_DEVICE=cpu"
python run.py pilot --data "%~1" --device %DETAILPAIR_DEVICE% --config configs\pilot.yaml
exit /b %errorlevel%
