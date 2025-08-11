@echo off
setlocal enabledelayedexpansion

:: Get OS version
for /f "tokens=4-5 delims=. " %%i in ('ver') do set VERSION=%%i.%%j

:: Check if it's Windows 10
if "%VERSION%" == "10.0" (
    echo Detected Windows 10 or Windows Server with version 10.0
    call windows.bat
    goto :END
)

:: Check if it's Windows Server 2012 R2
if "%VERSION%" == "6.3" (
    call windows_server.bat
    goto :END
)

:: If neither condition is met
echo This is not Windows 10 or Windows Server 2012 R2
echo OS Version: %VERSION%
echo No installation file will be run.

:END
echo Installation process completed.
pause