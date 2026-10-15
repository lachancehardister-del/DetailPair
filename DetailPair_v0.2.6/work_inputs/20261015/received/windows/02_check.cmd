@echo off
cd /d "%~dp0.."
if not exist verification mkdir verification
python run.py env
if errorlevel 1 exit /b 1
python run.py smoke
if errorlevel 1 exit /b 1
python tests\check_invariants.py >verification\windows_invariants.log 2>&1
if errorlevel 1 exit /b 1
python tests\check_novelty_controls.py >verification\windows_controls.log 2>&1
if errorlevel 1 exit /b 1
python run.py synthetic --device cpu --variant B6
exit /b %errorlevel%
