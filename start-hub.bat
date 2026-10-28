@echo off
REM FlowLens Hub - one site view for many cameras. Extra options are passed through, e.g. start-hub.bat --pin 4821
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python hub\hub.py %*
