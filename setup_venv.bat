@echo off
REM Create and activate a Python virtual environment for this project.
REM Run this from the project root: setup_venv.bat

if not exist requirements.txt (
    echo requirements.txt not found. Please ensure it exists in the project root.
    exit /b 1
)

set VENV_DIR=.venv
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not on PATH. Please install Python 3.10+ and try again.
    exit /b 1
)
echo Creating virtual environment in %VENV_DIR%...
python -m venv %VENV_DIR%
if errorlevel 1 (
    echo Failed to create virtual environment.
    exit /b 1
)
necho Upgrading pip...
%VENV_DIR%\Scripts\python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    exit /b 1
)
necho Installing required packages from requirements.txt...
%VENV_DIR%\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies. Please review the error messages above.
    exit /b 1
)
necho Virtual environment created successfully.
echo Activate it with:
echo    %VENV_DIR%\Scripts\activate
exit /b 0
