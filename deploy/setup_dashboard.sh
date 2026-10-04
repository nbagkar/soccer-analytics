#!/usr/bin/env bash
#
# Run this after setup_vm.sh, and after you've given the VM a STATIC public IP in
# Azure (Networking -> IP configuration -> Static). A dynamic IP would break the
# nip.io hostname -- and the TLS certificate for it -- on the next reboot.
#
# Puts Streamlit behind Caddy so the app itself only listens on localhost; Caddy
# terminates TLS on 443 and gets its certificate automatically from Let's Encrypt,
# using a nip.io hostname that resolves to the VM's own IP (no domain needed).
#
# Usage: ./deploy/setup_dashboard.sh <static-public-ip>
#
set -euo pipefail

IP="${1:?Usage: $0 <static-public-ip>}"
HOSTNAME="${IP}.nip.io"
APP_DIR="$HOME/soccer-analytics"
UNIT_DIR="/etc/systemd/system"

cd "$APP_DIR"
.venv/bin/pip install -e ".[dashboard]"

if ! grep -q "^SOCCER_DASHBOARD_PASSWORD=.\+" .env 2>/dev/null; then
  echo "SOCCER_DASHBOARD_PASSWORD is not set -- once this is public, anyone with the"
  echo "URL could trigger data downloads on your VM. Set a password now:"
  read -rsp "Dashboard password: " DASH_PW; echo
  [[ -n "$DASH_PW" ]] || { echo "Refusing to expose the dashboard with no password." >&2; exit 1; }
  if grep -q "^SOCCER_DASHBOARD_PASSWORD=" .env; then
    sed -i "s#^SOCCER_DASHBOARD_PASSWORD=.*#SOCCER_DASHBOARD_PASSWORD=$DASH_PW#" .env
  else
    echo "SOCCER_DASHBOARD_PASSWORD=$DASH_PW" >>.env
  fi
fi

sed -e "s#__APP_DIR__#$APP_DIR#g" -e "s#__USER__#$(whoami)#g" \
  deploy/soccer-dashboard.service | sudo tee "$UNIT_DIR/soccer-dashboard.service" >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now soccer-dashboard.service

if ! command -v caddy >/dev/null; then
  sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' |
    sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' |
    sudo tee /etc/apt/sources.list.d/caddy-stable.list
  sudo apt-get update -y
  sudo apt-get install -y caddy
fi

sed "s#__HOSTNAME__#$HOSTNAME#g" deploy/Caddyfile | sudo tee /etc/caddy/Caddyfile >/dev/null
sudo systemctl restart caddy

echo
echo "Dashboard will be live at: https://$HOSTNAME"
echo "(needs ports 80 and 443 open inbound in the VM's Azure NSG, and $IP must be"
echo " the VM's static public IP or the certificate will fail to renew on reboot)"
