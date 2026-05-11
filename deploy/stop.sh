#!/bin/bash
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$APP_DIR/logs/watchdog.pid" ] && kill "$(cat $APP_DIR/logs/watchdog.pid)" 2>/dev/null && rm -f "$APP_DIR/logs/watchdog.pid"
pkill -f "run_paper.py"     2>/dev/null || true
pkill -f "run_live.py"      2>/dev/null || true
pkill -f "run_dashboard.py" 2>/dev/null || true
echo "All SO processes stopped."
