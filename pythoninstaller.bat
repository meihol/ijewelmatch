@echo off
:: Check if Python is already installed
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Python is already installed.
    goto setup_env
)

:: If Python is not installed, download Python 3.12.4
echo Python not found. Downloading Python 3.12.4...

:: Use full path to PowerShell to download Python installer
%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -Command "Try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest 'https://www.python.org/ftp/python/3.12.4/python-3.12.4-amd64.exe' -OutFile 'python_installer.exe' } Catch { Exit 1 }"
if %errorlevel% neq 0 (
    echo Failed to download Python installer.
    goto error
)

:: Verify the installer exists
if not exist python_installer.exe (
    echo Python installer not found.
    goto error
)

:: Install Python silently
echo Installing Python 3.12.4...
python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
if %errorlevel% neq 0 (
    echo Failed to install Python.
    goto error
)

:: Clean up
del python_installer.exe

:: Set Python path explicitly for this session
set "PATH=%PATH%;C:\Program Files\Python312;C:\Program Files\Python312\Scripts"

:: Verify Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python installation failed.
    goto error
)

echo Python installed successfully.
goto setup_env

:setup_env
:: SETUP ENVIRONMENT
echo Setting up environment...

:: Create a new virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    goto error
)

:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Set PIP_NO_VERIFY to skip SSL checks (optional)
set PIP_NO_VERIFY=*

:: Upgrade pip
echo Upgrading pip...
python -m pip install pip==25.0

:: Install certifi
echo Installing certifi...
pip install -U certifi

:: Upgrade pip again
python -m pip install --upgrade pip
pip install --upgrade certifi

:: Uninstall existing torch packages
echo Uninstalling existing torch packages...
pip uninstall -y torch torchvision torchaudio

:: Install CPU-only versions of torch packages
echo Installing torch CPU-only packages...
pip install torch torchvision torchaudio
if %errorlevel% neq 0 (
    echo Failed to install torch packages.
    goto error
)

:: Install other required packages
echo Installing other required packages...
pip install flask faiss-cpu pillow numpy tqdm werkzeug
if %errorlevel% neq 0 (
    echo Failed to install other required packages.
    goto error
)

echo All packages installed successfully.

:: Run the Flask app
echo Sparkling up your iJewelMatch...
python ijewelmatch.py
pause
exit /b 0

:error
echo ===========================================
echo ERROR: Python installation or setup failed.
echo ===========================================
pause
goto end

:end
echo Script finished.
pause
