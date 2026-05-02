#!/bin/bash
# Run once on a fresh Ubuntu 22.04 DigitalOcean droplet as root
# Usage: bash setup_droplet.sh

set -e

APP_DIR=/root/MyTradingBot-SO

# 1. System packages + Docker
apt-get update -y
apt-get install -y ca-certificates curl git

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 2. Clone repo
if [ ! -d "$APP_DIR" ]; then
    git clone https://github.com/srivishnucharan/MyTradingBot-SO.git "$APP_DIR"
else
    cd "$APP_DIR" && git pull
fi

# 3. Create secrets file from template if missing
if [ ! -f "$APP_DIR/config/secrets.env" ]; then
    echo "# Add your credentials below" > "$APP_DIR/config/secrets.env"
    echo "DHAN_CLIENT_ID=" >> "$APP_DIR/config/secrets.env"
    echo "DHAN_ACCESS_TOKEN=" >> "$APP_DIR/config/secrets.env"
    echo "TELEGRAM_BOT_TOKEN=" >> "$APP_DIR/config/secrets.env"
    echo "TELEGRAM_CHAT_ID=" >> "$APP_DIR/config/secrets.env"
    echo ""
    echo ">>> ACTION REQUIRED: Edit $APP_DIR/config/secrets.env with your credentials"
    echo ""
fi

# 4. Build and start
cd "$APP_DIR"
docker compose build
docker compose up -d

echo ""
echo "Setup complete."
echo "  Monitor:    docker compose logs -f"
echo "  Dashboard:  http://<your-droplet-ip>:8002"
echo "  Backtest:   docker compose run --rm bot python scripts/run_backtest.py"
