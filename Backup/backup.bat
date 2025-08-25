@echo off

:: Get the current directory and use it as working directory
cd /d "%~dp0"

:: Check if Python is already installed
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Python is already installed.
    goto :setup_env
)

:: If Python is not installed, download Python 3.12.4
echo Python not found. Downloading Python 3.12.4...
%SYSTEMROOT%\System32\WindowsPowerShell\v1.0\powershell.exe -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-RestMethod https://www.python.org/ftp/python/3.12.4/python-3.12.4-amd64.exe -OutFile python_installer.exe"
if %errorlevel% neq 0 (
    echo Failed to download Python installer.
    goto :error
)

:: Verify the installer exists
if not exist python_installer.exe (
    echo Python installer not found.
    goto :error
)

:: Install Python
echo Installing Python 3.12.4...
python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
if %errorlevel% neq 0 (
    echo Failed to install Python.
    goto :error
)

:: Clean up
del python_installer.exe

:: Set Python path explicitly
set PATH=%PATH%;C:\Program Files\Python312;C:\Program Files\Python312\Scripts

:: Verify Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python installation failed.
    goto :error
)
echo Python installed successfully.

:setup_env
:: Create a new virtual environment in current directory
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
python -m pip install pip==25.2

:: Install required packages
echo Installing required packages...

:: pip install certifi
pip install -U certifi

python -m pip install --upgrade pip
pip install --upgrade certifi

:: Uninstall existing torch packages
pip uninstall -y torch torchvision torchaudio
pip install opencv-python

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














:: 22/08/2025


@echo off

:: Check if Python is already installed
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Python is already installed.
    goto :setup_env
)

:: If Python is not installed, download Python 3.12.4
echo Python not found. Downloading Python 3.12.4...
%SYSTEMROOT%\System32\WindowsPowerShell\v1.0\powershell.exe -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-RestMethod https://www.python.org/ftp/python/3.12.4/python-3.12.4-amd64.exe -OutFile python_installer.exe"
if %errorlevel% neq 0 (
    echo Failed to download Python installer.
    goto :error
)

:: Verify the installer exists
if not exist python_installer.exe (
    echo Python installer not found.
    goto :error
)

:: Install Python
echo Installing Python 3.12.4...
python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
if %errorlevel% neq 0 (
    echo Failed to install Python.
    goto :error
)

:: Clean up
del python_installer.exe

:: Set Python path explicitly
set PATH=%PATH%;C:\Program Files\Python312;C:\Program Files\Python312\Scripts

:: Verify Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python installation failed.
    goto :error
)
echo Python installed successfully.

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
python -m pip install pip==25.2

:: Install required packages
echo Installing required packages...

:: pip install certifi
pip install -U certifi==2025.8.3

python -m pip install --upgrade pip
pip install --upgrade certifi

:: Uninstall existing torch packages
pip uninstall -y torch torchvision torchaudio
:: pip install opencv-python

:: Install CPU-only versions of torch packages
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
if %errorlevel% neq 0 (
    echo Failed to install torch packages.
    goto :error
)

:: Install other required packages
pip install flask==3.1.1 faiss-cpu==1.12.0 pillow==11.3.0 numpy==2.3.2 tqdm==4.67.1 werkzeug==3.1.3
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
