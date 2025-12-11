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
# Start Uvicorn
# ------------------------------
echo "→ Starting Uvicorn:"
echo "    Module:  $APP_MODULE"
echo "    Host:    $HOST"
echo "    Port:    $PORT"
echo "    Workers: $WORKERS"
echo "    LogLevel:$LOG_LEVEL"

exec uvicorn "$APP_MODULE" \
  --host "$HOST" \
  --port "$PORT" \
  --log-level "$LOG_LEVEL" \
  --workers "$WORKERS"