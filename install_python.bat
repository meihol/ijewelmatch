@echo off

:: Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo This script requires administrator privileges.
    echo Right-click the script and select "Run as administrator"
    pause
    exit /b 1
)

:: Get current directory for virtual environment
set CURRENT_DIR=%cd%

:: Check if Python is already installed
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Python is already installed.
    goto :setup_env
)

:: If Python is not installed, download Python 3.12.4
echo Python not found. Downloading Python 3.12.4...
echo Please wait, this may take a few minutes...

:: Use curl instead of PowerShell for better reliability
curl -L https://www.python.org/ftp/python/3.12.4/python-3.12.4-amd64.exe -o python_installer.exe
if %errorlevel% neq 0 (
    echo Failed to download Python installer. Trying alternative method...
    %SYSTEMROOT%\System32\WindowsPowerShell\v1.0\powershell.exe -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest https://www.python.org/ftp/python/3.12.4/python-3.12.4-amd64.exe -OutFile python_installer.exe } catch { exit 1 }"
    if %errorlevel% neq 0 (
        echo Failed to download Python installer.
        goto :error
    )
)

:: Verify the installer exists
if not exist python_installer.exe (
    echo Python installer not found after download.
    goto :error
)

:: Install Python with more detailed logging
echo Installing Python 3.12.4...
echo This may take several minutes. Please wait...

:: Try installation with logging
python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0 /log python_install.log

:: Wait a moment for installation to complete
timeout /t 10 /nobreak >nul

:: Check if installation was successful by looking for Python executable
if exist "C:\Program Files\Python312\python.exe" (
    echo Python installation appears successful.
) else if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe" (
    echo Python installation appears successful (user install).
) else (
    echo Python installation may have failed. Check python_install.log for details.
    if exist python_install.log (
        echo Last few lines of install log:
        powershell -Command "Get-Content python_install.log | Select-Object -Last 10"
    )
    goto :error
)

:: Clean up installer
if exist python_installer.exe del python_installer.exe

:: Refresh environment variables
call :refresh_env

:: Verify Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python installation verification failed. Trying to refresh PATH...
    :: Try to find Python and add to PATH
    for /f "tokens=*" %%i in ('dir "C:\Program Files\Python312\python.exe" /s /b 2^>nul') do (
        set "PYTHON_PATH=%%~dpi"
        goto :found_python
    )
    for /f "tokens=*" %%i in ('dir "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe" /s /b 2^>nul') do (
        set "PYTHON_PATH=%%~dpi"
        goto :found_python
    )
    echo Could not locate Python installation.
    goto :error
    
    :found_python
    set "PATH=%PYTHON_PATH%;%PYTHON_PATH%Scripts;%PATH%"
    python --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Python installation failed verification.
        goto :error
    )
)

echo Python installed successfully.
python --version

:setup_env
:: Create virtual environment in current directory instead of Program Files
echo Creating virtual environment in current directory...
set VENV_PATH=%CURRENT_DIR%\venv

:: Remove existing venv if it exists
if exist "%VENV_PATH%" (
    echo Removing existing virtual environment...
    rmdir /s /q "%VENV_PATH%"
)

python -m venv "%VENV_PATH%"
if %errorlevel% neq 0 (
    echo Failed to create virtual environment in %VENV_PATH%
    echo Trying alternative location...
    set VENV_PATH=%TEMP%\StyleLens_venv
    python -m venv "%VENV_PATH%"
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        goto :error
    )
)

echo Virtual environment created at: %VENV_PATH%

:: Activate the virtual environment
echo Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    goto :error
)

:: Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Warning: Failed to upgrade pip, continuing anyway...
)

:: Install required packages with better error handling
echo Installing required packages...

:: Install packages one by one for better error tracking
echo Installing torch packages...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo Failed to install torch packages. Trying without index URL...
    pip install torch torchvision torchaudio
    if %errorlevel% neq 0 (
        echo Failed to install torch packages.
        goto :error
    )
)

echo Installing other required packages...
pip install flask
if %errorlevel% neq 0 (
    echo Failed to install flask.
    goto :error
)

pip install faiss-cpu
if %errorlevel% neq 0 (
    echo Failed to install faiss-cpu.
    goto :error
)

pip install pillow numpy tqdm werkzeug
if %errorlevel% neq 0 (
    echo Failed to install remaining packages.
    goto :error
)

echo All packages installed successfully.

:: Verify ijewelmatch.py exists
if not exist ijewelmatch.py (
    echo Warning: ijewelmatch.py not found in current directory.
    echo Make sure the Python script is in the same folder as this batch file.
    pause
    goto :error
)

:: Run the Flask app
echo Starting iJewelMatch...
python ijewelmatch.py
pause
exit /b 0

:refresh_env
:: Refresh environment variables
for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH') do set "SYSTEM_PATH=%%b"
for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set "USER_PATH=%%b"
set "PATH=%SYSTEM_PATH%;%USER_PATH%"
goto :eof

:error
echo.
echo Installation failed. Please check the following:
echo 1. Make sure you're running as administrator
echo 2. Check your internet connection
echo 3. Temporarily disable antivirus software
echo 4. Make sure you have enough disk space
echo.
if exist python_install.log (
    echo Python installation log has been created. Check python_install.log for more details.
)
pause
exit /b 1