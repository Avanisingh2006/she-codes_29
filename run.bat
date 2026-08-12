@echo off
REM MoveWise launcher. Each PC runs its own instance so it uses its own webcam.
cd /d "%~dp0"
python -m streamlit run app.py --server.port 8501 --server.headless false
