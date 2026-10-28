@echo off
REM FlowLens camera node. Examples:
REM   start-camera.bat
REM   start-camera.bat --name "Hall A - entrance" --hub http://192.168.1.20:9000 --hub-token YOUR_TOKEN
REM   start-camera.bat --source rtsp://user:pass@192.168.1.31/stream1 --port 8001 --data cams\cam1 --name "Cam 1"
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python app.py %*
