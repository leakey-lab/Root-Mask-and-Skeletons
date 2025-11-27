@echo off
REM Build Script for Root Mask and Skeletons Application

echo =====================================
echo Root Mask and Skeletons - Build Script
echo =====================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
call .\venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    echo Please ensure venv exists at .\venv\
    pause
    exit /b 1
)
echo Virtual environment activated
echo.

REM Check if PyInstaller is installed
echo Checking for PyInstaller...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
    if errorlevel 1 (
        echo Error: Failed to install PyInstaller
        pause
        exit /b 1
    )
) else (
    echo PyInstaller is already installed
)
echo.

REM Clean previous builds
echo Cleaning previous build artifacts...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
echo Previous builds cleaned
echo.

REM Build the executable
echo Building executable...
echo This may take 5-15 minutes on first build...
echo.

pyinstaller build_executable.spec

if errorlevel 0 (
    echo.
    echo =====================================
    echo Build completed successfully!
    echo =====================================
    echo.
    echo Your executable is located at:
    echo   dist\RootMaskAndSkeletons\RootMaskAndSkeletons.exe
    echo.
    echo To run the application:
    echo   cd dist\RootMaskAndSkeletons
    echo   RootMaskAndSkeletons.exe
    echo.
    echo To distribute: Compress the entire 'dist\RootMaskAndSkeletons\' folder
    echo.
) else (
    echo.
    echo =====================================
    echo Build failed!
    echo =====================================
    echo.
    echo Please check the error messages above.
    echo See BUILD_EXECUTABLE.md for troubleshooting tips.
    echo.
)

pause
