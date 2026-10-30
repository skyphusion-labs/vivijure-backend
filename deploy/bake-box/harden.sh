#!/bin/bash
# Second layer, run once through the tunnel after the firewall proof. It makes
# sshd listen on loopback only, turns on ufw with no allow rule, and removes the
# copies of the boot token that cloud-init keeps. The Vultr firewall stays the
# main control: Docker published ports go around ufw.
set -eu
cat >/etc/ssh/sshd_config.d/20-bakebox-loopback.conf <<'SSHD'
ListenAddress 127.0.0.1
SSHD
sshd -t
# The Vultr image ships ufw active with allow rules for 22. Reset removes them.
ufw --force reset >/dev/null
ufw --force default deny incoming >/dev/null
ufw --force default allow outgoing >/dev/null
ufw --force enable >/dev/null
passwd -l root >/dev/null
# tmpfs pages can move to swap, and swap is disk. Keep RAM-only data in RAM.
swapoff -a
sed -i '/[[:space:]]swap[[:space:]]/d' /etc/fstab
rm -f /swapfile
# cloud-init keeps the user-data in these files. Run this after the tunnel
# rotation has made that token dead; this removes the dead copies as well.
rm -f /var/lib/cloud/instance/user-data.txt /var/lib/cloud/instance/user-data.txt.i \
  /var/lib/cloud/instance/obj.pkl /var/lib/cloud/instance/scripts/part-001
# Ubuntu 24.04 uses socket activation, and the generator reads ListenAddress at
# daemon-reload. Run the restart after this session has had time to close.
systemctl daemon-reload
nohup sh -c 'sleep 3; systemctl restart ssh.socket; systemctl restart ssh' >/dev/null 2>&1 &
