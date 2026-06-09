@echo off
echo =======================================================
echo     Bone Vision AI - Model Training Pipeline
echo =======================================================
echo.
echo This script will execute the PyTorch Transfer Learning
echo process using the Kaggle dataset.
echo.
echo Note: Depending on your system (CPU vs GPU), this may 
echo take several minutes to complete. Please do not close 
echo this window until it says "Training complete".
echo.
cd "%~dp0backend"
python ml\train.py
echo.
pause
