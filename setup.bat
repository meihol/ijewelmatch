@echo off
:: Create folders only if they don't exist
if not exist StyleImages mkdir StyleImages
if not exist ijewelmatch_data mkdir ijewelmatch_data
if not exist ijewelmatch_logs mkdir ijewelmatch_logs
if not exist upload mkdir upload

echo Folders created. Now:
echo 1. Copy your jewelry images to StyleImages\
echo 2. Run: docker-compose up -d
echo 3. Open http://localhost:5002
echo 4. Train the model with path: /app/StyleImages
pause
