#!/usr/bin/env bash
# Starts the demo. With .ngrok.env filled in you get the SAME public https link on every start;
# without it you get a temporary link (python app.py --share). See "Phone access" in README.md.
cd "$(dirname "$0")"

PY=.venv/Scripts/python.exe                      # Windows (Git Bash)
[ -x "$PY" ] || PY=.venv/bin/python              # Linux / macOS
[ -x "$PY" ] || { echo 'No .venv found. Follow "Quick start" in README.md first.'; exit 1; }

if [ -f .ngrok.env ]; then
  while IFS='=' read -r key value; do
    value=${value%$'\r'}
    case "$key" in NGROK_AUTHTOKEN|NGROK_DOMAIN) [ -n "$value" ] && export "$key=$value" ;; esac
  done < .ngrok.env
fi

if [ -n "$NGROK_DOMAIN" ] && [ -n "$NGROK_AUTHTOKEN" ]; then
  echo "Starting with the fixed public link: https://$NGROK_DOMAIN"
  echo "Keep this window open while you show the app. Ctrl+C stops it."
  exec "$PY" app.py
else
  echo 'No ngrok settings found, starting with a temporary public link.'
  echo 'Wait for the line "Running on public URL: https://....gradio.live" and keep this window open.'
  exec "$PY" app.py --share
fi
