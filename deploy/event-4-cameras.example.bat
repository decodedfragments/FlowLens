@echo off
REM Example: one GPU machine running four IP cameras + the hub for an event.
REM Copy this file, change names / addresses / token, then double-click it.
cd /d "%~dp0\.."
set HUB=http://127.0.0.1:9000
set TOKEN=PASTE_THE_HUB_TOKEN_HERE
start "FlowLens Hub"  cmd /k start-hub.bat --event "My Event"
timeout /t 5 >nul
start "Cam 1" cmd /k start-camera.bat --source rtsp://192.168.1.31/stream1 --port 8001 --data cams\cam1 --name "Main entrance" --hub %HUB% --hub-token %TOKEN%
start "Cam 2" cmd /k start-camera.bat --source rtsp://192.168.1.32/stream1 --port 8002 --data cams\cam2 --name "Hall A"        --hub %HUB% --hub-token %TOKEN%
start "Cam 3" cmd /k start-camera.bat --source rtsp://192.168.1.33/stream1 --port 8003 --data cams\cam3 --name "Lounge"        --hub %HUB% --hub-token %TOKEN%
start "Cam 4" cmd /k start-camera.bat --source rtsp://192.168.1.34/stream1 --port 8004 --data cams\cam4 --name "Checkout"      --hub %HUB% --hub-token %TOKEN%
