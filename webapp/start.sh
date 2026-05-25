#!/usr/bin/env bash
# Start Dam Seepage PINN Web App (macOS / Linux)
# Usage: bash webapp/start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

BACKEND_VENV="$BACKEND_DIR/.venv"
BACKEND_PYTHON="$BACKEND_VENV/bin/python"
BACKEND_PORT=8000
FRONTEND_PORT=5173

# ── Cleanup on exit ──
cleanup() {
  echo ""
  echo "Shutting down..."
  if [[ -n "$BACKEND_PID" ]]; then kill "$BACKEND_PID" 2>/dev/null && echo "  Backend (PID $BACKEND_PID) stopped"; fi
  if [[ -n "$FRONTEND_PID" ]]; then kill "$FRONTEND_PID" 2>/dev/null && echo "  Frontend (PID $FRONTEND_PID) stopped"; fi
  exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# ── Check prerequisites ──
if [[ ! -d "$BACKEND_VENV" ]]; then
  echo "Backend venv not found. Creating..."
  python3 -m venv "$BACKEND_VENV"
  "$BACKEND_PYTHON" -m pip install -r "$BACKEND_DIR/requirements.txt"
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Frontend node_modules not found. Installing..."
  (cd "$FRONTEND_DIR" && npm install)
fi

# ── Start backend ──
echo "Starting backend on port $BACKEND_PORT..."
cd "$BACKEND_DIR"
"$BACKEND_PYTHON" -m uvicorn main:app --host 0.0.0.0 --port $BACKEND_PORT --reload &
BACKEND_PID=$!

# ── Start frontend ──
echo "Starting frontend on port $FRONTEND_PORT..."
cd "$FRONTEND_DIR"
npx vite --host &
FRONTEND_PID=$!

# ── Wait for servers to be ready ──
echo ""
echo "Waiting for servers..."
for i in $(seq 1 30); do
  if curl -s http://localhost:$BACKEND_PORT/health > /dev/null 2>&1 && \
     curl -s http://localhost:$FRONTEND_PORT > /dev/null 2>&1; then
    echo ""
    echo "========================================="
    echo "  Dam Seepage PINN Web App is ready!"
    echo "  Frontend: http://localhost:$FRONTEND_PORT"
    echo "  Backend:  http://localhost:$BACKEND_PORT"
    echo "========================================="
    echo ""
    open http://localhost:$FRONTEND_PORT 2>/dev/null || true
    break
  fi
  sleep 1
done

# ── Wait for either process to exit ──
wait
