@echo off
echo ============================================================
echo  AI QR Code Generator - Local Setup
echo  Python 3.12 required (found in venv)
echo ============================================================
echo.

:: Activate venv
call "%~dp0venv\Scripts\activate.bat"

echo [1/3] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/3] Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [3/3] Installing remaining requirements...
pip install -r "%~dp0requirements.txt"

echo.
echo ============================================================
echo  Setup complete! Run launch.bat to start the app.
echo ============================================================
pause
