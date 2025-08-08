@echo off
echo ==========================================
echo StyleLens Environment Setup Script
echo ==========================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please run install_python.bat first.
    pause
    exit /b 1
)
echo Python found:
python --version
echo.

:setup_env
:: Create a new virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    goto :error
)

:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Set PIP_NO_VERIFY to true to skip SSL verification
set PIP_NO_VERIFY=*

:: Upgrade pip
echo Upgrading pip...
python -m pip install pip==25.0

:: Install required packages
echo Installing required packages...

:: pip install certifi
pip install -U certifi

python -m pip install --upgrade pip
pip install --upgrade certifi

:: Uninstall existing torch packages
pip uninstall -y torch torchvision torchaudio
:: pip install opencv-python

:: Install CPU-only versions of torch packages
pip install torch torchvision torchaudio
if %errorlevel% neq 0 (
    echo Failed to install torch packages.
    goto :error
)

:: Install other required packages
pip install flask faiss-cpu pillow numpy tqdm werkzeug
if %errorlevel% neq 0 (
    echo Failed to install other required packages.
    goto :error
)

echo All packages installed successfully.

:: Run the Flask app
echo Sparkling up your iJewelMatch...
python ijewelmatch.py
pause
exit /b 0

:error
echo An error occurred.
pause
exit /b 1

echo.
echo ==========================================
echo Environment setup completed successfully!
echo ==========================================
echo Virtual environment location: venv
echo.
echo Starting StyleLens application...
echo Sparkling up your StyleLens...
python ijewelmatch.py
pause