@echo off
rem Starts the demo. With .ngrok.env filled in you get the SAME public https link on every start;
rem without it you get a temporary link (python app.py --share). See "Phone access" in README.md.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The virtual environment .venv was not found. Follow "Quick start" in README.md first.
  pause
  exit /b 1
)
if exist ".ngrok.env" for /f "usebackq eol=# tokens=1,* delims==" %%a in (".ngrok.env") do set "%%a=%%b"
set "FIXED="
if defined NGROK_DOMAIN if defined NGROK_AUTHTOKEN set "FIXED=1"
if defined FIXED (
  echo Starting with the fixed public link: https://%NGROK_DOMAIN%
  echo Keep this window open while you show the app. Press Ctrl+C to stop it.
  ".venv\Scripts\python.exe" app.py
) else (
  echo No ngrok settings found, starting with a temporary public link.
  echo Wait for the line "Running on public URL: https://....gradio.live" and keep this window open.
  ".venv\Scripts\python.exe" app.py --share
)
pause
