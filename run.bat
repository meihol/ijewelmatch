@echo off

:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Run the Flask app
:: pip install transformers
echo Sparkling up your iJewelMatch...
python ijewelmatch.py
pause