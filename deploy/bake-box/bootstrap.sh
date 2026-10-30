#!/bin/bash
# First boot of the vivijure bake box. Vultr passes this file as user-data.
# bakebox.sh fills the token and key placeholders. Do not add set -x: the tunnel
# token is in this file until bakebox.sh rotates it and clears the user-data.
set -eu
exec >>/var/log/bakebox-bootstrap.log 2>&1
echo "bootstrap start $(date -u +%FT%TZ)"
export DEBIAN_FRONTEND=noninteractive

# Tunnel first, so a later failure in this script can still be read over SSH.
install -d -m 700 /etc/cloudflared
umask 077
cat >/etc/cloudflared/tunnel-token <<'TOKEN'
@@TUNNEL_TOKEN@@
TOKEN
umask 022

# User and keys. sshd keeps its default listeners until harden.sh runs, so the
# firewall proof runs against a port that really listens.
id bake >/dev/null 2>&1 || useradd -m -s /bin/bash bake
install -d -m 700 -o bake -g bake /home/bake/.ssh
cat >/home/bake/.ssh/authorized_keys <<'KEYS'
@@SSH_KEYS@@
KEYS
chown bake:bake /home/bake/.ssh/authorized_keys
chmod 600 /home/bake/.ssh/authorized_keys
echo 'bake ALL=(ALL) NOPASSWD:ALL' >/etc/sudoers.d/90-bake
chmod 440 /etc/sudoers.d/90-bake
cat >/etc/ssh/sshd_config.d/10-bakebox.conf <<'SSHD'
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
SSHD
systemctl reload ssh || systemctl restart ssh

# Vultr images can hold a dpkg lock for a short time after first boot.
apt_wait() { apt-get -o DPkg::Lock::Timeout=300 "$@"; }
apt_wait update -y
apt_wait install -y ca-certificates curl gnupg git jq unzip ufw

install -d -m 755 /usr/share/keyrings /etc/apt/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
  -o /usr/share/keyrings/cloudflare-main.gpg
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' \
  >/etc/apt/sources.list.d/cloudflared.list
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
. /etc/os-release
echo "deb [signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  >/etc/apt/sources.list.d/docker.list
apt_wait update -y
apt_wait install -y cloudflared

cat >/etc/systemd/system/cloudflared-bake.service <<'UNIT'
[Unit]
Description=cloudflared tunnel for the vivijure bake box
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/bin/cloudflared --no-autoupdate tunnel run --token-file /etc/cloudflared/tunnel-token
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now cloudflared-bake

apt_wait install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
usermod -aG docker bake
curl -fsSL https://rclone.org/install.sh | bash

# The RAM directory for registry and R2 credentials during a bake. See the runbook.
cat >/etc/tmpfiles.d/bakebox.conf <<'TMPF'
d /run/bake-creds 0700 bake bake -
TMPF
systemd-tmpfiles --create /etc/tmpfiles.d/bakebox.conf

touch /var/lib/bakebox-ready
echo "bootstrap done $(date -u +%FT%TZ)"
