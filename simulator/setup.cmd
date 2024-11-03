@echo off
:: may need to give exec permission to this file
set VENV_DIR=venv

:: Check if the virtual environment directory exists
if not exist %VENV_DIR% (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo Failed to create virtual environment
        exit /b 1
    )
    echo Virtual environment created at %VENV_DIR%
)

:: Activate the virtual environment
if exist %VENV_DIR%\Scripts\activate (
    call %VENV_DIR%\Scripts\activate
    echo Virtual environment activated
) else (
    echo Activation script not found
    exit /b 1
)

:: Install required packages
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install required packages
    exit /b 1
)

:: install windows specific packages
pip install -r requirements-windows.txt
if errorlevel 1 (
    echo Failed to install windows specific packages
    exit /b 1
)

deactivate

echo Virtual environment setup complete.