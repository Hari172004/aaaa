# run_agniv.ps1 — Agni-V PowerShell Launcher
# This script automatically uses the project's virtual environment to run the bot.

$VenvPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (Test-Path $VenvPath) {
    Write-Host "🚀 Starting Agni-V Bot in Virtual Environment..." -ForegroundColor Green
    & $VenvPath run_bot.py
} else {
    Write-Error "❌ Virtual environment not found at .venv\Scripts\python.exe"
    Write-Host "Please ensure you have run the setup and the .venv folder exists." -ForegroundColor Yellow
}
