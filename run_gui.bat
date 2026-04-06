@echo off
REM Deep Past GUI Launcher for Windows

echo.
echo ============================================================
echo    Deep Past - Akkadian Translation GUI
echo ============================================================
echo.

REM Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Warning: No virtual environment found. Using system Python.
)

REM Install/update dependencies
echo Checking dependencies...
pip install -q gradio>=4.0.0

echo.
echo Starting GUI...
echo.

python gui/app.py %*

pause
