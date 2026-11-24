@echo off
cd /d "%~dp0"
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8765/index.html"
where py >nul 2>nul
if %errorlevel%==0 (
  py -m http.server 8765 --bind 127.0.0.1
) else (
  python -m http.server 8765 --bind 127.0.0.1
)
pause
