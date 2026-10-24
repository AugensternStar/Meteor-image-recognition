@echo off
cd /d "%~dp0"
set PY=C:\Users\17275\anaconda3\python.exe
if not exist "%PY%" (
  echo [ERROR] Python not found: %PY%
  echo Please edit this .bat file and set PY to your python.exe path.
  pause
  exit /b 1
)
echo Starting meteor image filter service...
echo Open http://127.0.0.1:8734 in your browser after "service ready" appears.
echo Keep this window open while using the service.
echo.
"%PY%" service.py
echo.
echo Service exited. If you see an error above, send it to us.
pause
