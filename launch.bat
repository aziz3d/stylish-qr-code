@echo off
echo ============================================================
echo  AI QR Code Generator - Local Launcher
echo ============================================================
echo.
echo  Options you can add:
echo    --share       expose a public Gradio URL
echo    --port 8080   use a different port (default: 7860)
echo    --cpu         run on CPU only (no GPU required, slow)
echo.
echo  Example: launch.bat --share
echo           launch.bat --port 8080
echo ============================================================
echo.

:: Activate the virtual environment
call "%~dp0venv\Scripts\activate.bat"

:: Run the local launcher (models will auto-download on first run)
python "%~dp0run_local.py" %*

pause
