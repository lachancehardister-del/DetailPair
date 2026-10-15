@echo off
set "DETAILPAIR_ENV=D:\ModelImprovement\envs\detailpair_v02"
if exist "%DETAILPAIR_ENV%\python.exe" (
 echo Environment already exists: %DETAILPAIR_ENV%
 echo Verify its Python/dependencies before reuse. No files overwritten.
 exit /b 0
)
call conda create --prefix "%DETAILPAIR_ENV%" python=3.12 -y
exit /b %errorlevel%
