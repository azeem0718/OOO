@echo off
setlocal EnableExtensions
title Employee Salary Prediction System
rem ==================================================================
rem  Employee Salary Prediction System - one-click Windows launcher
rem  Double-click this file. First run installs everything (Python,
rem  packages, database, trained model); later runs open the menu.
rem ==================================================================
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PY%" goto setup
if not exist "db_settings.ini" goto setup
if not exist "models\salary_model.joblib" goto setup
goto menu

rem ------------------------------------------------------------------
:setup
echo.
echo  ==================================================================
echo    FIRST-TIME SETUP  (takes 2-5 minutes, needs internet once)
echo  ==================================================================
echo.
echo  [1/5] Looking for Python 3.9 or newer...
call :find_python
if not defined PY call :install_python
if not defined PY goto no_python
echo        Using: %PY%

if exist "%VENV_PY%" goto have_venv
echo  [2/5] Creating a private Python environment (.venv)...
%PY% -m venv .venv
if errorlevel 1 goto fail
:have_venv

echo  [3/5] Installing packages (numpy, pandas, scikit-learn, flask, mysql-connector)...
"%VENV_PY%" -m pip install --upgrade pip --disable-pip-version-check -q
"%VENV_PY%" -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 goto fail

if exist "db_settings.ini" goto have_settings
echo  [4/5] Database settings...
"%VENV_PY%" configure_db.py
if errorlevel 1 goto fail
:have_settings

echo  [5/5] Creating tables, loading employee data and training the model...
"%VENV_PY%" setup_db.py
if errorlevel 1 goto fail
echo.
echo  Setup complete!

rem ------------------------------------------------------------------
:menu
echo.
echo  ==================================================================
echo    EMPLOYEE SALARY PREDICTION SYSTEM   (Python + MySQL + ML)
echo  ==================================================================
echo    1. Web version           (opens in your browser)
echo    2. Command-line version
echo    3. Change MySQL settings
echo    4. Reset database and retrain model
echo    5. Reinstall packages
echo    0. Exit
echo.
set "CH="
set /p "CH=  Choose an option and press Enter: "
if "%CH%"=="1" goto web
if "%CH%"=="2" goto cli
if "%CH%"=="3" goto settings
if "%CH%"=="4" goto reset
if "%CH%"=="5" goto reinstall
if "%CH%"=="0" exit /b 0
echo  Invalid option.
goto menu

:web
echo.
echo  Starting the web app at http://127.0.0.1:5000 ...
echo  Keep this window open. Press Ctrl+C here to stop the server.
set "OPEN_BROWSER=1"
"%VENV_PY%" web_app.py
set "OPEN_BROWSER="
goto menu

:cli
"%VENV_PY%" cli_app.py
goto menu

:settings
"%VENV_PY%" configure_db.py
if errorlevel 1 goto menu
"%VENV_PY%" setup_db.py
goto menu

:reset
set "OK="
set /p "OK=  This deletes all employees and predictions and reloads sample data. Type YES to continue: "
if /i not "%OK%"=="YES" goto menu
"%VENV_PY%" setup_db.py --reset
goto menu

:reinstall
"%VENV_PY%" -m pip install --upgrade -r requirements.txt --disable-pip-version-check
goto menu

rem ------------------------------------------------------------------
:find_python
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if defined PY goto :eof
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=python"
if defined PY goto :eof
for %%V in (313 312 311 310) do if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" if not defined PY set PY="%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
goto :eof

:install_python
echo        Python not found. Installing Python 3.12 automatically (winget)...
where winget >nul 2>&1
if errorlevel 1 goto :eof
winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
call :find_python
goto :eof

:no_python
echo.
echo  Python could not be installed automatically.
echo  1. Download Python from https://www.python.org/downloads/
echo  2. In the installer, TICK "Add python.exe to PATH"
echo  3. Double-click START_HERE.bat again.
start "" "https://www.python.org/downloads/"
pause
exit /b 1

:fail
echo.
echo  Something went wrong (see the messages above).
echo  Check your internet connection and run START_HERE.bat again.
pause
exit /b 1
