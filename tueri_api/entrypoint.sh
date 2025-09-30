#!/bin/bash

APP_WORKERS=${APP_WORKERS:-1}
CONFIG_FILE=${CONFIG_FILE:-./config/app_config.yml}

# Start the API in the background
uvicorn app.app:create_app --host=0.0.0.0 --port=8000 --factory --workers="$APP_WORKERS" --forwarded-allow-ips="*" --proxy-headers --timeout-keep-alive="2"

# Wait for API to be ready
# echo "Waiting for API to be ready..."
# until python3 -c "import urllib.request; urllib.request.urlopen('http://0.0.0.0:8001/healthz')" 2>/dev/null; do
#   sleep 2
# done
# echo "API is ready!"

# # Download models
# echo "Downloading models..."
# python3 -c "import urllib.request; urllib.request.urlopen('http://l0.0.0.0:8001/download/models', data=b'')"

# # Keep the API running in foreground
# wait
