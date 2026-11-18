#!/bin/bash
# Make Again backend deployment script for CentOS 8
# Run as root: bash deploy.sh

set -e

APP_DIR="/opt/make-again/backend"
DATA_DIR="/opt/make-again/data"
UPLOADS_DIR="/opt/make-again/uploads"
NGINX_CONF="/etc/nginx/conf.d/10-makeagain-edsionc-top.conf"

echo "=== 1. Install Python 3.12 (if not available) ==="
if ! command -v python3.12 &> /dev/null; then
    echo "Python 3.12 not found. Installing via Docker instead..."
    if ! command -v docker &> /dev/null; then
        echo "ERROR: Neither Python 3.12 nor Docker found. Please install one."
        exit 1
    fi
    echo "Docker found, will use container deployment."
    USE_DOCKER=1
else
    USE_DOCKER=0
    echo "Python 3.12 found."
fi

echo "=== 2. Create directories ==="
mkdir -p "$APP_DIR" "$DATA_DIR" "$UPLOADS_DIR"
chmod 750 "$APP_DIR" "$DATA_DIR" "$UPLOADS_DIR"

echo "=== 3. Copy application code ==="
# Assumes code is already uploaded to $APP_DIR
cd "$APP_DIR"

echo "=== 4. Set up environment ==="
if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Copy env.production.example to .env and fill in values."
    exit 1
fi

echo "=== 5. Install FFmpeg (if needed) ==="
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing FFmpeg..."
    dnf install -y epel-release 2>/dev/null || true
    dnf install -y ffmpeg 2>/dev/null || yum install -y ffmpeg 2>/dev/null || echo "FFmpeg not available via package manager. Install manually."
fi

if [ "$USE_DOCKER" = "1" ]; then
    echo "=== 6a. Docker deployment ==="
    docker compose down 2>/dev/null || true
    docker compose up -d --build
    echo "Waiting for health check..."
    sleep 5
    curl -s http://127.0.0.1:8000/api/health
else
    echo "=== 6b. Native deployment ==="
    python3.12 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python preflight.py
    python admin_cli.py migrate
    # Start with systemd or screen
    echo "Run: source venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000"
fi

echo "=== 7. Nginx config ==="
if [ -f deploy/nginx-makeagain.conf ]; then
    cp deploy/nginx-makeagain.conf "$NGINX_CONF"
    echo "Nginx config copied. Run: nginx -t && systemctl reload nginx"
fi

echo "=== 8. SSL certificate ==="
echo "Run: certbot certonly --nginx -d makeagain.edsionc.top"
echo "Then: nginx -t && systemctl reload nginx"

echo "=== 9. Start worker ==="
if [ "$USE_DOCKER" = "1" ]; then
    echo "Worker started via docker-compose (make-again-worker service)."
else
    echo "Run: source venv/bin/activate && python admin_cli.py worker --forever --poll-seconds 3"
fi

echo ""
echo "=== Deployment complete ==="
echo "Health check: curl https://makeagain.edsionc.top/api/health"
echo "Admin init: python admin_cli.py init-admin --username admin"