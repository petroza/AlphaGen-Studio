@echo off
cd /d "%~dp0"
echo Automated tests (video I/O, sync, segmentation, alpha export):
.venv\Scripts\python.exe -m pytest %*
pause
