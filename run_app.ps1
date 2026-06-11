# ─────────────────────────────────────────────────────────────────────────────
# run_app.ps1  —  Shared launcher for all conservation Streamlit projects
# Usage:  .\run_app.ps1 <project_folder>
# Example: .\run_app.ps1 elephant_conflict
#          .\run_app.ps1 fence_vulnerability
# ─────────────────────────────────────────────────────────────────────────────

param(
    [Parameter(Mandatory=$true)]
    [string]$Project
)

$Root    = $PSScriptRoot
$EnvPy   = "$Root\conservation_env\Scripts\streamlit.exe"
$AppPath = "$Root\$Project\app.py"

if (-not (Test-Path $AppPath)) {
    Write-Error "Could not find app.py at: $AppPath"
    exit 1
}

Write-Host ""
Write-Host "  Conservation Data Platform" -ForegroundColor Cyan
Write-Host "  Launching project : $Project" -ForegroundColor Green
Write-Host "  Python env        : conservation_env (Python 3.12)" -ForegroundColor Gray
Write-Host "  App path          : $AppPath" -ForegroundColor Gray
Write-Host ""

& $EnvPy run $AppPath
