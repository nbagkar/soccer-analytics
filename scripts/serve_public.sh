#!/usr/bin/env bash
#
# Serve the dashboard publicly via a free Cloudflare "quick tunnel".
#
# The app runs on THIS machine, so it uses your full local data (every league, the
# Champions League, fixtures, StatsBomb) with nothing to re-download. Cloudflare hands back
# a public https://<random>.trycloudflare.com URL that points at it. The tunnel is only up
# while this script runs -- i.e. while your Mac is on and awake.
#
# A password gate protects the public URL, which matters because the Home page can trigger
# data downloads on your machine. Set SOCCER_DASHBOARD_PASSWORD ahead of time, or you'll be
# prompted for one.
#
# Usage:  ./scripts/serve_public.sh
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8501}"
PY="$REPO/.venv/bin/python"
STREAMLIT="$REPO/.venv/bin/streamlit"

for bin in "$PY" "$STREAMLIT"; do
  [[ -x "$bin" ]] || { echo "Missing $bin -- create the .venv and install '.[dashboard]' first." >&2; exit 1; }
done
command -v cloudflared >/dev/null || { echo "cloudflared not found -- run: brew install cloudflared" >&2; exit 1; }

# Password gate for the public URL (refuse to expose the write actions unprotected).
if [[ -z "${SOCCER_DASHBOARD_PASSWORD:-}" ]]; then
  read -rsp "Set a password for the public dashboard: " SOCCER_DASHBOARD_PASSWORD; echo
  [[ -n "$SOCCER_DASHBOARD_PASSWORD" ]] || { echo "Refusing to expose the app with no password." >&2; exit 1; }
fi
export SOCCER_DASHBOARD_PASSWORD

# Pin the resolved data dir so the tunneled app reads the same store as your CLI, even
# though Streamlit runs from the dashboard dir (so its .streamlit/config.toml theme loads).
export SOCCER_DATA_DIR="$("$PY" -c 'from soccer.config import get_settings; print(get_settings().data_dir)')"

echo "Starting dashboard on 127.0.0.1:$PORT (data: $SOCCER_DATA_DIR)"
( cd "$REPO/src/soccer/dashboard" && exec "$STREAMLIT" run app.py \
    --server.port "$PORT" --server.address 127.0.0.1 --server.headless true \
    >/tmp/soccer_streamlit.log 2>&1 ) &
STREAMLIT_PID=$!
trap 'kill "$STREAMLIT_PID" 2>/dev/null || true' EXIT INT TERM

# Wait for Streamlit to answer before opening the tunnel.
for _ in $(seq 1 40); do
  curl -fsS "http://127.0.0.1:$PORT/" >/dev/null 2>&1 && break
  sleep 0.5
done

echo
echo "Opening a free public Cloudflare tunnel. Share the https://…trycloudflare.com URL below."
echo "Press Ctrl-C to stop — the tunnel and the app both go down."
echo
cloudflared tunnel --url "http://127.0.0.1:$PORT"
