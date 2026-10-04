#!/usr/bin/env bash
#
# Bootstrap this repo on a fresh Ubuntu 24.04 VM and install a systemd timer that
# runs `soccer serve --once` a few times a day (see soccer-serve.timer for the
# schedule). Idempotent -- safe to re-run after a `git pull` to pick up code changes.
#
# Usage (on the VM): curl -fsSL .../deploy/setup_vm.sh | bash
#   or:              git clone <repo> && ./deploy/setup_vm.sh
#
set -euo pipefail

REPO_URL="https://github.com/nbagkar/soccer-analytics.git"
APP_DIR="$HOME/soccer-analytics"
UNIT_DIR="/etc/systemd/system"

sudo apt-get update -y
sudo apt-get install -y python3.12 python3.12-venv git

if [[ -d "$APP_DIR/.git" ]]; then
  git -C "$APP_DIR" pull
else
  git clone "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo
  echo "Created $APP_DIR/.env from the template -- edit it now and set"
  echo "SOCCER_FOOTBALL_DATA_ORG_TOKEN before the timer runs, e.g.:"
  echo "  nano $APP_DIR/.env"
  echo
fi

sed -e "s#__APP_DIR__#$APP_DIR#g" -e "s#__USER__#$(whoami)#g" \
  deploy/soccer-serve.service | sudo tee "$UNIT_DIR/soccer-serve.service" >/dev/null
sudo cp deploy/soccer-serve.timer "$UNIT_DIR/soccer-serve.timer"

sudo systemctl daemon-reload
sudo systemctl enable --now soccer-serve.timer

echo "Installed. Useful commands:"
echo "  systemctl list-timers soccer-serve.timer   # next scheduled run"
echo "  sudo systemctl start soccer-serve.service   # trigger a run now"
echo "  journalctl -u soccer-serve.service -f       # watch logs"
