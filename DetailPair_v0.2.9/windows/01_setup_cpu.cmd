@echo off
cd /d "%~dp0.."
if not exist install_logs mkdir install_logs
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu >install_logs\torch_cpu.log 2>&1
if errorlevel 1 goto failed
python -m pip install -r requirements.txt >install_logs\requirements.log 2>&1
if errorlevel 1 goto failed
python run.py env
exit /b %errorlevel%
:failed
echo Installation failed. Logs retained in install_logs. Run python run.py pack.
exit /b 1
