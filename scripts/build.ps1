# Build Script for Root Mask and Skeletons Application
# This script automates the executable building process

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Root Mask & Skeletons - Build Script" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & ".\venv\Scripts\Activate.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to activate virtual environment" -ForegroundColor Red
        Write-Host "Please ensure venv exists at .\venv\" -ForegroundColor Red
        exit 1
    }
}

Write-Host "Virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green
Write-Host ""

# Check if PyInstaller is installed
Write-Host "Checking for PyInstaller..." -ForegroundColor Yellow
$pyinstallerInstalled = pip list | Select-String "pyinstaller"
if (-not $pyinstallerInstalled) {
    Write-Host "PyInstaller not found. Installing..." -ForegroundColor Yellow
    pip install pyinstaller
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to install PyInstaller" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "PyInstaller is already installed" -ForegroundColor Green
}
Write-Host ""

# Clean previous builds
Write-Host "Cleaning previous build artifacts..." -ForegroundColor Yellow
if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
    Write-Host "Removed build/ directory" -ForegroundColor Green
}
if (Test-Path "dist") {
    Remove-Item -Recurse -Force "dist"
    Write-Host "Removed dist/ directory" -ForegroundColor Green
}
Write-Host ""

# Verify required files exist
Write-Host "Verifying required files..." -ForegroundColor Yellow
$requiredFiles = @(
    "main.py",
    "resources\themes\dark_theme.qss",
    "checkpoints\mask_weights\best_mask_model_V5.pth",
    "checkpoints\skeletonizer\latest_net_G.pth",
    "config\normalization_defaults.json"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $missingFiles += $file
        Write-Host "  [MISSING] $file" -ForegroundColor Red
    } else {
        Write-Host "  [OK] $file" -ForegroundColor Green
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "Warning: Some required files are missing!" -ForegroundColor Red
    Write-Host "The executable may not work correctly." -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne "y") {
        Write-Host "Build cancelled." -ForegroundColor Yellow
        exit 0
    }
}
Write-Host ""

# Build the executable
Write-Host "Building executable..." -ForegroundColor Cyan
Write-Host "This may take 5-15 minutes on first build..." -ForegroundColor Yellow
Write-Host ""

pyinstaller build_executable.spec

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=====================================" -ForegroundColor Green
    Write-Host "Build completed successfully!" -ForegroundColor Green
    Write-Host "=====================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Your executable is located at:" -ForegroundColor Cyan
    Write-Host "  dist\RootMaskAndSkeletons\RootMaskAndSkeletons.exe" -ForegroundColor White
    Write-Host ""
    Write-Host "To run the application:" -ForegroundColor Yellow
    Write-Host "  cd dist\RootMaskAndSkeletons" -ForegroundColor White
    Write-Host "  .\RootMaskAndSkeletons.exe" -ForegroundColor White
    Write-Host ""
    Write-Host "To distribute: Compress the entire 'dist\RootMaskAndSkeletons\' folder" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "=====================================" -ForegroundColor Red
    Write-Host "Build failed!" -ForegroundColor Red
    Write-Host "=====================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check the error messages above." -ForegroundColor Yellow
    Write-Host "See BUILD_EXECUTABLE.md for troubleshooting tips." -ForegroundColor Yellow
    exit 1
}
