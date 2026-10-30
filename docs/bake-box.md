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

## Custody rule

Conrad keeps his data in his physical custody. **This box holds build inputs only:
public source and model weights.** Build outputs are public images, pushed to GHCR.

Never put these on the box:

- the memory store, or any crew or lead notes
- a clone of a private repo, or a crew working clone
- customer or tenant data
- a long-lived credential, or a forwarded SSH agent (`ssh -A`)

Clone over anonymous HTTPS only. If a build seems to need a private input, stop and
ask the lead. Do not copy the input over.

## Cost

Read from the Vultr plan API on 2026-10-30: **$0.263 per hour, capped at $192 per
month**. Backups are off. Vultr bills a stopped instance at the same rate, so
stopping the box does not stop the bill. **Only destroy stops the bill.**

The account also has a monthly-fee limit that Vultr set. A create that goes over
it fails. That is a spend decision for Conrad. Do not work around it.

A 3 hour bake on a box that exists for that bake costs about $0.80.

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
   address. Clear the record, then rotate:

   ```
   scripts/bakebox.sh clear-user-data
   <ssh to the box> 'cat > /tmp/install-token.sh' < deploy/bake-box/install-token.sh
   scripts/bakebox.sh rotate-tunnel | <ssh to the box> 'sudo bash /tmp/install-token.sh'
   ```

   `install-token.sh` keeps the new token in `/run`, which is RAM, and removes the disk
   copy. **After a reboot the tunnel stays down. Destroy the box and create it again.**

7. Run the second layer through the tunnel:

   ```
   <ssh to the box> 'sudo bash -s' < deploy/bake-box/harden.sh
   ```

   Run it after step 6. It removes the cloud-init copies of the boot token, turns
   off swap, and leaves sshd on loopback.

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
R2 credential between runs.** Each credential lives on the laptop, reaches the box
over SSH stdin for one run, and stays in RAM. Swap is off, so RAM stays RAM.

| Credential | Needed for | Source and life | On the box |
| --- | --- | --- | --- |
| GHCR push token | Any push to `ghcr.io/skyphusion-labs/vivijure-backend` | A classic PAT with `write:packages` only, at the shortest expiry GitHub offers. Conrad mints it, because a PAT needs the GitHub web UI. **Revoke it when the bake ends.** Never use a crew PAT: those carry admin and delete scopes. | `DOCKER_CONFIG=/run/bake-creds/docker`, removed at the end of the run |
| R2 temporary credential | **Only** a seed stage (`deploy/stage_bake_seed.py`). `deploy/runtime.Dockerfile` pulls the public seed image, so a runtime or thin build needs no R2 access. | Minted on the laptop with `POST /accounts/{id}/r2/temp-access-credentials`, scoped to the `vivijure` bucket and the prefixes the stage uses, with a TTL that fits the run. The parent R2 key stays on the laptop. | `RCLONE_CONFIG_R2_*` in the environment of that one command, with `SESSION_TOKEN`. No `rclone.conf`. |
| Tunnel token | Always | Only runs this tunnel's connector. Rotate it with `rotate-tunnel` and `install-token.sh` if the box is kept between bakes. | `/run/bake-tunnel/token`, RAM, mode 600 |

No RunPod key is needed to build. Do not pass a credential as a `--build-arg`, because
it stays in the image history.

Run a bake like this. The token arrives on the SSH stdin and is not typed on the box:

```
export DOCKER_CONFIG=/run/bake-creds/docker
mkdir -p "$DOCKER_CONFIG"
docker login ghcr.io -u <user> --password-stdin
git clone https://github.com/skyphusion-labs/vivijure-backend ~/build && cd ~/build
git checkout <sha on main>
docker buildx build -f deploy/runtime.Dockerfile --target toolchain --provenance=false \
  -t ghcr.io/skyphusion-labs/vivijure-backend:<tag> --push .
rm -rf "$DOCKER_CONFIG"
```

The buildx plugin is in `/usr/libexec/docker/cli-plugins`, so a new `DOCKER_CONFIG`
still finds it. Read the digest from the registry after the push, and pin that digest.
See [operations.md](operations.md) and `.github/workflows/runpod-bake.yml` for the tag
names and the repin. The image contents are not set by this document.

Check the end state before you leave the box: `ls -A /run/bake-creds` must be empty and
`find / -xdev -name config.json -path '*.docker*'` must print nothing.

The seed image is public and can be pulled with no login. Measured from the box on
2026-10-30: 76 MB/s from GHCR, for the first 2 GB of the largest seed layer. Upload
to GHCR and transfer to or from R2 were not measured. `df` read 447 GB free on `/`.
The seed image is about 87 GB and the runtime built on it was about 101 GB, so disk is
not the limit for one build.

## After a bake

**Destroy is the default.** See the next section. Destroy is the only step that stops
the bill, and it removes the disk from our account.

If another bake with the same inputs is next, and you keep the box, wipe it:

```
<ssh to the box> 'bash -s' < deploy/bake-box/wipe.sh
```

`wipe.sh` removes the build tree, the credentials in RAM, and all Docker images and
cache. It is housekeeping, not erasure: blocks stay on the disk until overwritten. That
is acceptable because the disk holds only public inputs under the custody rule. If
anything else ever reached the disk, destroy the box and tell the lead. Revoke the GHCR
PAT in both cases.

## Destroy

```
scripts/bakebox.sh plan
BAKEBOX_DESTROY=vivijure-bake scripts/bakebox.sh down
```

`down` finds each object by its exact label, name or hostname. It deletes the
instance, DNS record, Access app, service tokens and tunnel, then the firewall
group. It ends by running `plan`, and every line must be empty. Delete the local
`~/.cf-ssh-bake.env` files too, and revoke the GHCR PAT.

Then remove the declarations from `fleet-chezmoi`. See the section below.

## Declarations in fleet-chezmoi

The tunnel hostname, the Access app and the firewall group are declared in
`fleet-chezmoi`, in `system/cloudflare/dns/zones/skyphusion.org.json`,
`system/cloudflare/access/access-declared.json` and
`system/vultr/firewalls/firewalls.declared.json`. Those files feed the estate drift
checks. Create and destroy must both update them, or the checks read wrong.
