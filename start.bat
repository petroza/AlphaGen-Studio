@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo The application is not installed - running setup.bat
  call setup.bat || exit /b 1
)
if /i "%~1"=="debug" (
  .venv\Scripts\python.exe -m alphagen_studio
  pause
  exit /b
)
start "" ".venv\Scripts\pythonw.exe" -m alphagen_studio %*
