#!/usr/bin/env bash
set -euo pipefail

echo "== Starting Gunicorn App =="

# ------------------------------
# Configurable defaults
# ------------------------------
APP_MODULE=${APP_MODULE:-"sam2_endpoint:app"}     # Example: FastAPI/Flask app object
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-4}
LOG_LEVEL=${LOG_LEVEL:-"info"}

# Activate virtualenv automatically if found
if [ -d ".venv" ]; then
  echo "→ Activating virtualenv .venv"
  source .venv/bin/activate
fi

# ------------------------------
# Start Gunicorn
# ------------------------------
echo "→ Starting Gunicorn:"
echo "    Module:  $APP_MODULE"
echo "    Host:    $HOST"
echo "    Port:    $PORT"
echo "    Workers: $WORKERS"
echo "    LogLevel:$LOG_LEVEL"

exec gunicorn "$APP_MODULE" \
  --bind "$HOST:$PORT" \
  --workers "$WORKERS" \
  --log-level "$LOG_LEVEL" \
  --timeout 120 \
  --preload