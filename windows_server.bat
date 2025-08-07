:: https://stackoverflow.com/questions/25981703/pip-install-fails-with-connection-error-ssl-certificate-verify-failed-certi

@echo off

:: Download and install Visual C++ Redistributable
echo Downloading Visual C++ Redistributable...
%SYSTEMROOT%\System32\WindowsPowerShell\v1.0\powershell.exe -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-RestMethod https://aka.ms/vs/17/release/vc_redist.x64.exe -OutFile vc_redist.x64.exe"
if %errorlevel% neq 0 (
    echo Failed to download Visual C++ Redistributable.
    goto :error
)

echo Installing Visual C++ Redistributable...
start /wait vc_redist.x64.exe /quiet /norestart
if %errorlevel% neq 0 (
    echo Failed to install Visual C++ Redistributable.
    goto :error
)

del vc_redist.x64.exe

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