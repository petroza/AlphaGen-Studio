@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title AlphaGen Studio - setup

echo ==========================================================
echo   AlphaGen Studio - setup (Windows 11, NVIDIA CUDA)
echo ==========================================================
echo.

rem ---------------------------------------------------------------- uv
set "UV=uv"
where uv >nul 2>nul
if errorlevel 1 (
  echo [..] uv not found - installing via pip ^(py launcher^)...
  py -3 -m pip install --user uv
  if errorlevel 1 (
    echo [ERROR] Install Python 3.10+ from python.org or uv from https://docs.astral.sh/uv/
    goto :fail
  )
  set "UV=py -3 -m uv"
)
%UV% --version || goto :fail

rem ---------------------------------------------------------------- git
where git >nul 2>nul
if errorlevel 1 (
  if not exist "vendor\LTX-2\pyproject.toml" (
    echo [ERROR] git and the vendor\LTX-2 folder are missing. Install Git for Windows: winget install Git.Git
    goto :fail
  )
)

rem ---------------------------------------------------------------- FFmpeg
where ffmpeg >nul 2>nul
if errorlevel 1 (
  if not exist "tools\ffmpeg\bin\ffmpeg.exe" (
    echo [..] FFmpeg not found. Trying winget install Gyan.FFmpeg ...
    winget install --id Gyan.FFmpeg -e --accept-source-agreements
    if errorlevel 1 (
      echo [ERROR] Install FFmpeg manually ^(winget install Gyan.FFmpeg^) or extract it to tools\ffmpeg\bin
      goto :fail
    )
    echo [i] After installing FFmpeg you may need to open a new window to reload PATH.
  )
)

rem ---------------------------------------------------------------- NVIDIA
where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo [WARNING] nvidia-smi not found - AlphaGen will not run without an NVIDIA GPU and driver.
) else (
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
)

rem ---------------------------------------------------------------- LTX-2 (oficialni repo)
if not exist "vendor\LTX-2\pyproject.toml" (
  echo [..] Cloning the official Lightricks/LTX-2 repository ...
  git clone https://github.com/Lightricks/LTX-2.git vendor\LTX-2 || goto :fail
  rem tested commit (2026-11-12) - 'setup.bat update' moves to the latest
  git -C vendor\LTX-2 checkout --quiet f0f2f8607189c78a72420b2b63cb1975b5eca179 || goto :fail
)
if /i "%~1"=="update" (
  echo [..] Updating vendor\LTX-2 to the latest version ^(untested combination!^)
  git -C vendor\LTX-2 checkout --quiet main && git -C vendor\LTX-2 pull --ff-only || goto :fail
)

echo [..] Installing the LTX-2 inference environment ^(torch + CUDA, several GB^) ...
set UV_LINK_MODE=copy
pushd vendor\LTX-2
%UV% sync --package ltx-pipelines --python 3.12
if errorlevel 1 ( popd & goto :fail )
popd

rem ---------------------------------------------------------------- GUI prostredi
if not exist ".venv\Scripts\python.exe" (
  echo [..] Creating the application environment .venv ...
  %UV% venv .venv --python 3.12 || goto :fail
)
%UV% pip install --python .venv\Scripts\python.exe -r requirements.txt || goto :fail

rem ---------------------------------------------------------------- kontrola
echo.
echo [..] Environment check:
.venv\Scripts\python.exe -m alphagen_studio.cli check
echo.
echo ==========================================================
echo   Done. Run start.bat. On first launch the app will
echo   offer to download the models (~67.5 GB, Hugging Face token).
echo ==========================================================
pause
exit /b 0

:fail
echo.
echo [ERROR] Setup failed - see the output above.
pause
exit /b 1
