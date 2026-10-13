@echo off
rem Hand-tracking square filter. First run installs what it needs (internet needed once).
setlocal
cd /d "%~dp0"
set "PY=py -3"
where py >nul 2>nul || set "PY=python"
%PY% --version >nul 2>nul || ( echo Python 3 was not found. Install Python 3.10, 3.11 or 3.12 from python.org and tick "Add to PATH". & pause & exit /b 1 )
if not exist .venv (
  echo Creating a private Python environment...
  %PY% -m venv .venv || ( echo Could not create the environment. & pause & exit /b 1 )
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip >nul
  pip install -r requirements.txt || ( echo Install failed. See README.txt. & pause & exit /b 1 )
) else (
  call .venv\Scripts\activate.bat
)
python square_filter.py %*
pause
