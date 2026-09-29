$ErrorActionPreference = "Stop"

Write-Host "Checking dependencies..." -ForegroundColor Cyan
python -m pip check

Write-Host "Running Ruff..." -ForegroundColor Cyan
python -m ruff check src tests

Write-Host "Compiling source..." -ForegroundColor Cyan
python -m compileall -q src

Write-Host "Running tests..." -ForegroundColor Cyan
python -m pytest -q

Write-Host "Local verification passed." -ForegroundColor Green
