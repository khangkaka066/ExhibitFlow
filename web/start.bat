@echo off
cd /d "%~dp0"
echo ExhibitFlow Web: http://127.0.0.1:8765/index.html
echo (Dong cua so nay de tat server)
start "" cmd /c "timeout /t 2 >nul & start http://127.0.0.1:8765/index.html"
python -m http.server 8765 --bind 127.0.0.1 || py -m http.server 8765 --bind 127.0.0.1
