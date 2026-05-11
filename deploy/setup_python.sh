#!/bin/bash
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt"
[ -f "$APP_DIR/config/secrets.env" ] || touch "$APP_DIR/config/secrets.env"
echo "Setup complete."
