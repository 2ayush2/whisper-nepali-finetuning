@echo off
echo Starting Whisper Fine-Tuning Setup...

echo Installing Dependencies...
pip install -r requirements.txt

echo Starting Training...
python train.py %*

echo Done!
pause
