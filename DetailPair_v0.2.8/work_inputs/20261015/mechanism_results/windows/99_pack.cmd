@echo off
cd /d "%~dp0.."
python run.py pack
exit /b %errorlevel%
