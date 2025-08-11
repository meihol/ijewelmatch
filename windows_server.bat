@echo off

:: Check if Python is already installed
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Python is already installed.
    goto :check_vc
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

:check_vc
:: Check if Visual C++ Redistributable is already installed
echo Checking for Visual C++ Redistributable...
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1
if %errorlevel% equ 0 (
    echo Visual C++ Redistributable already installed.
    goto :setup_env
)

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
    echo Warning: Visual C++ Redistributable installation may have failed, but continuing...
)

del vc_redist.x64.exe

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

:: Set environment variables to handle SSL certificate issues
set PIP_TRUSTED_HOST=pypi.org pypi.python.org files.pythonhosted.org
set PIP_NO_VERIFY=1
set PYTHONHTTPSVERIFY=0
set SSL_CERT_VERIFICATION=0

:: Upgrade pip with SSL trust settings
echo Upgrading pip...
python -m pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --upgrade pip
if %errorlevel% neq 0 (
    echo Failed to upgrade pip.
    goto :error
)

:: Install certifi and requests for SSL handling
echo Installing SSL certificates and requests...
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org certifi requests urllib3
if %errorlevel% neq 0 (
    echo Failed to install SSL packages.
    goto :error
)

:: Uninstall existing torch packages (ignore errors if they don't exist)
echo Removing existing torch packages...
pip uninstall -y torch torchvision torchaudio 2>nul

:: Install CPU-only versions of torch packages with trusted hosts
echo Installing PyTorch packages...
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo PyTorch installation failed, trying alternative method...
    pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org torch torchvision torchaudio
    if %errorlevel% neq 0 (
        echo Failed to install torch packages.
        goto :error
    )
)

:: Install other required packages with trusted hosts
echo Installing other required packages...
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org flask faiss-cpu pillow numpy tqdm werkzeug
if %errorlevel% neq 0 (
    echo Failed to install other required packages.
    goto :error
)

echo All packages installed successfully.

:: Test basic imports
echo Testing package imports...
python -c "import torch; print('PyTorch:', torch.__version__)" || echo PyTorch import failed
python -c "import flask; print('Flask imported successfully')" || echo Flask import failed
python -c "import faiss; print('FAISS imported successfully')" || echo FAISS import failed
python -c "import numpy; print('NumPy imported successfully')" || echo NumPy import failed

:: Check if the main script exists
if not exist ijewelmatch.py (
    echo Warning: ijewelmatch.py not found in the current directory.
    echo Please make sure the script is in the same folder as this batch file.
    pause
    goto :error
)

:: Run the Flask app
echo.
echo ================================================
echo Sparkling up your iJewelMatch...
echo SSL certificate verification is disabled to prevent connection errors
echo ================================================
echo.

:: Run the application
python ijewelmatch.py
pause
exit /b 0

:error
echo.
echo ============================================
echo Installation failed!
echo ============================================
echo.
echo Troubleshooting steps:
echo 1. Run this script as Administrator
echo 2. Check your internet connection
echo 3. Temporarily disable antivirus/firewall
echo 4. If behind corporate firewall, contact IT support
echo.
echo Manual installation commands:
echo   python -m venv venv
echo   venv\Scripts\activate
echo   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org torch flask faiss-cpu pillow numpy tqdm werkzeug
echo.
pause
exit /b 1