#!/bin/bash
set -e
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$APP_DIR/venv"
MODE="${1:-paper}"
mkdir -p "$APP_DIR/logs"
cd "$APP_DIR"

pkill -f "run_dashboard.py" 2>/dev/null || true
pkill -f "run_paper.py"     2>/dev/null || true
pkill -f "run_live.py"      2>/dev/null || true
sleep 1

nohup "$VENV/bin/python" scripts/run_dashboard.py >> logs/dashboard.log 2>&1 &
echo "Dashboard started (PID $!)"
sleep 3

BOT_SCRIPT="scripts/run_paper.py"
[ "$MODE" = "live" ] && BOT_SCRIPT="scripts/run_live.py --no-confirm"

(while true; do
    "$VENV/bin/python" $BOT_SCRIPT >> logs/bot.log 2>&1
    [ $? -eq 0 ] && break
    sleep 5
done) &
echo $! > logs/watchdog.pid
echo "Bot started in $MODE mode (watchdog PID $(cat logs/watchdog.pid))"
