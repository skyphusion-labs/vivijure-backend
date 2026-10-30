# The bake box

The bake box is a Vultr host that builds vivijure RunPod images. It is a build
host. It is not a crew seat and it is not a CI runner. Crew operate from the
laptop and reach the box through a tunnel.

Conrad set this lane on 2026-10-30. The change request is
[#538](https://github.com/skyphusion-labs/vivijure-backend/issues/538).

| Item | Value |
| --- | --- |
| Label | `vivijure-bake` |
| Plan | `vhf-8c-32gb`: 8 vCPU, 32 GB RAM, 512 GB NVMe |
| Region | `ewr` |
| OS | Ubuntu 24.04 LTS |
| Access | `ssh-vivijure-bake.skyphusion.org`, through Cloudflare Access only |
| SSH user | `bake`, with passwordless sudo and the `docker` group |

Read the instance id and IP address from the Vultr API. This repo is public, so
they are not written here.

## Cost

Read from the Vultr plan API on 2026-10-30: **$0.263 per hour, capped at $192 per
month**. Backups are off. Vultr bills a stopped instance at the same rate, so
stopping the box does not stop the bill. **Only destroy stops the bill.**

The account also has a monthly-fee limit that Vultr set. A create that goes over
it fails. That is a spend decision for Conrad. Do not work around it.

A 3 hour bake on a box that exists for that bake costs about $0.80. Destroy the
box when the bake is done, unless another bake is next.

## Access rule

Every estate box is reachable only through a cloudflared tunnel. For this box:

- The Vultr firewall group has two sentinel rules (TCP port 1 from documentation
  addresses, one v4 and one v6) and no other rule. **An empty Vultr group admits
  everything.** The sentinels are what make the group deny. `bakebox.sh up` reads
  them back and creates no instance unless both exist.
- sshd listens on 127.0.0.1 only, and ufw has no allow rule. These are the second
  layer. They do not replace the group: a published Docker port goes around ufw.
- The tunnel is outbound from the box. Do not add an allow rule to get in. If the
  tunnel is down, destroy the box and create it again.

## Create

Run from a seat that has `VULTR_API_KEY` and `CLOUDFLARE_API_TOKEN`. The Cloudflare
token must be able to edit tunnels, DNS, Access apps and Access service tokens. The
Vultr API key allows only listed client addresses.

1. Read the price and confirm that no objects exist:

   ```
   scripts/bakebox.sh plan
   ```

2. Write one public key per line to a file, for each person who needs the box.
   Create everything:

   ```
   BAKEBOX_SSH_KEYS=<file> scripts/bakebox.sh up
   ```

3. Mint a service token for each holder. The command writes the secret to stdout
   and nowhere else, so send it to a mode 600 file. Never send it to a terminal.

   ```
   (umask 077; scripts/bakebox.sh token <holder> > ~/.cf-ssh-bake.env)
   ```

4. Wait for the tunnel to read `healthy`, then wait for `/var/lib/bakebox-ready` on
   the box. Boot took about 2 minutes on 2026-10-30.

5. **Prove the firewall now, while sshd still listens on all addresses.** Later,
   when sshd is on loopback, a closed port no longer shows that the group works.
   Probe from a vantage that no rule admits, with a control that can answer:

   ```
   nc -z -G 8 -w 8 <main_ip> 22      # must fail
   nc -z -G 8 -w 8 github.com 22     # must pass, or the probe proves nothing
   ```

   On the box, `ss -tln` must show `0.0.0.0:22`, and `ufw status` must show the image
   default that allows 22. A failed probe then has one cause: the Vultr group.

6. Remove the boot token. The token that `up` passed in user-data can be read from
   the Vultr API and from the metadata address, and a Docker build can reach that
   address. Rotate first, then clear:

   ```
   scripts/bakebox.sh clear-user-data
   scripts/bakebox.sh rotate-tunnel | <ssh to the box> \
     'sudo tee /etc/cloudflared/tunnel-token >/dev/null; (sudo nohup sh -c "sleep 3; systemctl restart cloudflared-bake" >/dev/null 2>&1 &)'
   ```

7. Run the second layer through the tunnel:

   ```
   <ssh to the box> 'sudo bash -s' < deploy/bake-box/harden.sh
   ```

   Run it after step 6. It removes the cloud-init copies of the boot token.

8. Probe again from outside, and connect through the tunnel again.

## Connect

Use a service token and your own key. cloudflared reads the token from its
environment, so the secret stays off the command line. A clean `HOME` for
cloudflared means a success can only come from the service token.

```
set -a; . ~/.cf-ssh-bake.env; set +a
export TUNNEL_SERVICE_TOKEN_ID="$CF_ACCESS_CLIENT_ID" TUNNEL_SERVICE_TOKEN_SECRET="$CF_ACCESS_CLIENT_SECRET"
CH=$(mktemp -d)
ssh -F none -o BatchMode=yes -i <your key> -o IdentitiesOnly=yes \
  -o ProxyCommand="env HOME=$CH cloudflared access ssh --hostname %h" \
  bake@ssh-vivijure-bake.skyphusion.org
```

A person can also sign in with a `skyphusion.org` address through Access.

`SSH_CONNECTION` on the box reads `127.0.0.1`. That is the sign that cloudflared
carried the session. To add a holder, mint a token with `bakebox.sh token <holder>`
and append the public key to `/home/bake/.ssh/authorized_keys`.

## Bake

The box has Docker with buildx, rclone, git and cloudflared. **It holds no GHCR or
R2 credential until a bake needs one.**

| Credential | Needed for | Source | Where it goes on the box |
| --- | --- | --- | --- |
| GHCR push token, classic PAT with `write:packages` only and a short expiry | Any push to `ghcr.io/skyphusion-labs/vivijure-backend` | A dedicated PAT. Conrad mints it, because a PAT needs the GitHub web UI. Do not use a crew PAT: those carry admin and delete scopes. | `DOCKER_CONFIG=/run/bake-creds/docker`, by `docker login --password-stdin` over the SSH stdin |
| R2 key: `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT` | **Only** a seed stage (`deploy/stage_bake_seed.py`). `deploy/runtime.Dockerfile` pulls the seed image, so a runtime or thin build needs no R2 key. | A per-function R2 key for this lane. Read-only is enough to read the mirror; `stage_bake_seed.py` also writes the seed, so it needs write on the `vivijure` bucket. | rclone environment for that one command: `RCLONE_CONFIG_R2_TYPE=s3` and the matching keys. No `rclone.conf`. |
| Tunnel token | Always | Written by `up`, replaced in step 6 | `/etc/cloudflared/tunnel-token`, mode 600. This one is on disk because cloudflared must read it at each start. |

`/run/bake-creds` is a RAM directory, owned by `bake`, mode 700. A reboot empties it.
`docker login` writes a base64 password to `config.json`, so point `DOCKER_CONFIG` at
that directory and never log in with the default path:

```
export DOCKER_CONFIG=/run/bake-creds/docker
mkdir -p "$DOCKER_CONFIG"
docker login ghcr.io -u <user> --password-stdin   # the token arrives on stdin over SSH
# build and push
rm -rf "$DOCKER_CONFIG"
```

The buildx plugin is in `/usr/libexec/docker/cli-plugins`, so a new `DOCKER_CONFIG`
still finds it. No RunPod key is needed to build. Do not pass a credential as a
`--build-arg`, because it stays in the image history.

Build from a clean clone at a commit on `main`:

```
git clone https://github.com/skyphusion-labs/vivijure-backend && cd vivijure-backend
git checkout <sha>
docker buildx build -f deploy/runtime.Dockerfile --target toolchain --provenance=false \
  -t ghcr.io/skyphusion-labs/vivijure-backend:<tag> --push .
```

Read the digest from the registry after the push, and pin that digest. See
[operations.md](operations.md) and `.github/workflows/runpod-bake.yml` for the tag names
and the repin. The image contents are not set by this document.

The seed image is public and can be pulled with no login. Measured from the box on
2026-10-30: 76 MB/s from GHCR, for the first 2 GB of the largest seed layer. Upload
to GHCR and transfer to or from R2 were not measured. `df` read 447 GB free on `/`.
The older seed image is about 87 GB and the runtime built on it was about 101 GB,
so disk is not the limit for one build. Run `docker builder prune` between bakes
if you keep the box.

## Destroy

```
scripts/bakebox.sh plan
BAKEBOX_DESTROY=vivijure-bake scripts/bakebox.sh down
```

`down` finds each object by its exact label, name or hostname. It deletes the
instance, DNS record, Access app, service tokens and tunnel, then the firewall
group. It ends by running `plan`, and every line must be empty. Delete the local
`~/.cf-ssh-bake.env` files too, and revoke the GHCR PAT if no bake is next.

Then remove the declarations from `fleet-chezmoi`. See the section below.

## Declarations in fleet-chezmoi

The tunnel hostname, the Access app and the firewall group are declared in
`fleet-chezmoi`, in `system/cloudflare/dns/zones/skyphusion.org.json`,
`system/cloudflare/access/access-declared.json` and
`system/vultr/firewalls/firewalls.declared.json`. Those files feed the estate drift
checks. Create and destroy must both update them, or the checks read wrong.
