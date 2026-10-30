#!/bin/bash
# Read a tunnel token on stdin and run the tunnel from RAM. The token is not on
# disk after this. A reboot empties /run, so after a reboot the tunnel stays
# down and the box must be destroyed and created again. That is deliberate.
set -eu
install -d -m 700 /run/bake-tunnel
umask 077
cat >/run/bake-tunnel/token
[ -s /run/bake-tunnel/token ] || { echo "empty token" >&2; exit 1; }
install -d -m 755 /etc/systemd/system/cloudflared-bake.service.d
cat >/etc/systemd/system/cloudflared-bake.service.d/20-ram-token.conf <<'UNIT'
[Service]
ExecStart=
ExecStart=/usr/bin/cloudflared --no-autoupdate tunnel run --token-file /run/bake-tunnel/token
UNIT
rm -f /etc/cloudflared/tunnel-token
systemctl daemon-reload
nohup sh -c 'sleep 3; systemctl restart cloudflared-bake' >/dev/null 2>&1 &
