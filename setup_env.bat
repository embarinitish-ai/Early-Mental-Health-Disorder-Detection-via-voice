@echo off
REM ============================================================
REM  Depression Detection Pipeline v4 — Environment Setup
REM  Creates Conda env "Nirvana" with Python 3.11 + CUDA 11.8
REM ============================================================

echo [1/3] Creating Conda environment "Nirvana" with Python 3.11 ...
conda create -n Nirvana python=3.11 -y

echo [2/3] Installing PyTorch + CUDA 11.8 ...
conda run -n Nirvana pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

echo [3/3] Installing remaining dependencies ...
conda run -n Nirvana pip install transformers datasets accelerate librosa soundfile pandas openpyxl scikit-learn numpy

echo.
echo ============================================================
echo  Environment "Nirvana" is ready!
echo  Activate with:   conda activate Nirvana
echo ============================================================
pause
