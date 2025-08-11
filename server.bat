@echo off
:: ================================
:: INSTALL VISUAL C++ REDISTRIBUTABLE
:: ================================
echo Downloading Visual C++ Redistributable...
%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -Command "Try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile 'vc_redist.x64.exe' } Catch { Exit 1 }"
if %errorlevel% neq 0 (
    echo Failed to download Visual C++ Redistributable.
    goto error
)

echo Installing Visual C++ Redistributable...
start /wait vc_redist.x64.exe /quiet /norestart
if %errorlevel% neq 0 (
    echo Failed to install Visual C++ Redistributable.
    goto error
)

del vc_redist.x64.exe

:: ================================
:: CHECK IF PYTHON IS INSTALLED
:: ================================
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Python is already installed.
    goto setup_env
)

:: ================================
:: INSTALL PYTHON 3.12.4
:: ================================
echo Python not found. Downloading Python 3.12.4...

%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -Command "Try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest 'https://www.python.org/ftp/python/3.12.4/python-3.12.4-amd64.exe' -OutFile 'python_installer.exe' } Catch { Exit 1 }"
if %errorlevel% neq 0 (
    echo Failed to download Python installer.
    goto error
)

if not exist python_installer.exe (
    echo Python installer not found.
    goto error
)

echo Installing Python 3.12.4...
python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
if %errorlevel% neq 0 (
    echo Failed to install Python.
    goto error
)

del python_installer.exe

:: Add Python to session path (in case it's not picked up)
set "PATH=%PATH%;C:\Program Files\Python312;C:\Program Files\Python312\Scripts"

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python installation failed.
    goto error
)

echo Python installed successfully.
goto setup_env

:: ================================
:: SETUP ENVIRONMENT & INSTALL PACKAGES
:: ================================
:setup_env
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

:: Disable SSL verification (optional - for custom cert setups)
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

:: Uninstall any existing torch packages
echo Uninstalling existing torch packages...
pip uninstall -y torch torchvision torchaudio

:: Install CPU-only torch packages
echo Installing torch CPU-only packages...
pip install torch torchvision torchaudio
if %errorlevel% neq 0 (
    echo Failed to install torch packages.
    goto error
)

:: Install other dependencies
echo Installing other required packages...
pip install flask faiss-cpu pillow numpy tqdm werkzeug
if %errorlevel% neq 0 (
    echo Failed to install other required packages.
    goto error
)

echo All packages installed successfully.

:: ================================
:: RUN FLASK APP
:: ================================
echo Sparkling up your iJewelMatch...
python ijewelmatch.py
pause
exit /b 0

:: ================================
:: ERROR HANDLER
:: ================================
:error
echo ===========================================
echo ERROR: Installation or setup failed.
echo ===========================================
pause
goto end

:end
echo Script finished.
pause
