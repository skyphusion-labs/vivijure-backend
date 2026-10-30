#!/bin/bash
# Create, inspect and destroy the vivijure bake box. Runbook: docs/bake-box.md.
#
#   bakebox.sh plan                read the plan price and every object found
#   bakebox.sh up                  firewall, tunnel, DNS, Access, instance
#   bakebox.sh token <name>        mint one service token, env lines on stdout
#   bakebox.sh rotate-tunnel       new tunnel secret, new token on stdout
#   bakebox.sh clear-user-data     erase the boot token from the Vultr record
#   bakebox.sh down                delete what plan lists (needs BAKEBOX_DESTROY=vivijure-bake)
#
# Needs VULTR_API_KEY and CLOUDFLARE_API_TOKEN in the environment. This script
# never prints a credential except the two commands that exist to emit one,
# and those write only to stdout, for a pipe. Never send them to a terminal.
set -euo pipefail

LABEL=vivijure-bake
PLAN=vhf-8c-32gb
REGION=${BAKEBOX_REGION:-ewr}
OS_ID=2284 # Ubuntu 24.04 LTS x64
ZONE_NAME=skyphusion.org
HOST=ssh-vivijure-bake.$ZONE_NAME
TUNNEL=vivijure-bake
APP_NAME="vivijure bake box (SSH over Access)"
FW_DESC="vivijure-bake: sentinel rules only, no allow rule; SSH through the cloudflared tunnel"
TOKEN_PREFIX=ssh-over-access-vivijure-bake
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VULTR=https://api.vultr.com/v2
CF=https://api.cloudflare.com/client/v4

vultr() { # method path [json]
  curl -fsS -X "$1" -H "Authorization: Bearer $VULTR_API_KEY" \
    -H 'Content-Type: application/json' ${3:+--data "$3"} "$VULTR$2"
}
cf() { # method path [json]
  local out
  out=$(curl -sS -X "$1" -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
    -H 'Content-Type: application/json' ${3:+--data "$3"} "$CF$2")
  if [ "$(jq -r '.success' <<<"$out")" != true ]; then
    echo "cloudflare $1 $2 failed: $(jq -c '.errors' <<<"$out")" >&2
    return 1
  fi
  printf '%s\n' "$out"
}

instance_id() { vultr GET '/instances?per_page=500' | jq -r --arg l "$LABEL" '.instances[]|select(.label==$l)|.id'; }
firewall_id() { vultr GET '/firewalls?per_page=500' | jq -r --arg d "$FW_DESC" '.firewall_groups[]|select(.description==$d)|.id'; }
tunnel_id() { cf GET "/accounts/$ACCOUNT/cfd_tunnel?is_deleted=false&name=$TUNNEL" | jq -r --arg n "$TUNNEL" '.result[]|select(.name==$n)|.id'; }
app_id() { cf GET "/accounts/$ACCOUNT/access/apps?per_page=200" | jq -r --arg d "$HOST" '.result[]|select(.domain==$d)|.id'; }
dns_id() { cf GET "/zones/$ZONE/dns_records?name=$HOST" | jq -r '.result[].id'; }
token_rows() { cf GET "/accounts/$ACCOUNT/access/service_tokens?per_page=200" | jq -r --arg p "$TOKEN_PREFIX" '.result[]|select(.name==$p or (.name|startswith($p+"-")))|[.id,.name]|@tsv'; }

cmd_plan() {
  vultr GET "/plans?type=vhf&per_page=500" | jq -r --arg p "$PLAN" --arg r "$REGION" \
    '.plans[]|select(.id==$p)|"plan \(.id) vcpu=\(.vcpu_count) ram_mb=\(.ram) disk_gb=\(.disk) hourly=\(.hourly_cost) monthly=\(.monthly_cost) in_\($r)=\(.locations|index($r)!=null)"'
  echo "instance: $(instance_id)"
  echo "firewall: $(firewall_id)"
  echo "tunnel:   $(tunnel_id)"
  echo "dns:      $(dns_id)"
  echo "app:      $(app_id)"
  echo "tokens:"; token_rows
}

cmd_up() {
  : "${BAKEBOX_SSH_KEYS:?file of public keys for the bake user}"
  [ -s "$BAKEBOX_SSH_KEYS" ] || { echo "BAKEBOX_SSH_KEYS is empty" >&2; exit 1; }
  [ -z "$(instance_id)" ] || { echo "an instance labelled $LABEL exists; refusing" >&2; exit 1; }

  # An empty Vultr group admits everything. The sentinels go in before any
  # instance can attach, and the read-back is the gate.
  local fw; fw=$(firewall_id)
  if [ -z "$fw" ]; then
    fw=$(vultr POST /firewalls "$(jq -nc --arg d "$FW_DESC" '{description:$d}')" | jq -r .firewall_group.id)
    vultr POST "/firewalls/$fw/rules" '{"ip_type":"v4","protocol":"tcp","subnet":"192.0.2.1","subnet_size":32,"port":"1","notes":"sentinel, an empty group allows all"}' >/dev/null
    vultr POST "/firewalls/$fw/rules" '{"ip_type":"v6","protocol":"tcp","subnet":"2001:db8::1","subnet_size":128,"port":"1","notes":"sentinel, an empty group allows all"}' >/dev/null
  fi
  local rules
  rules=$(vultr GET "/firewalls/$fw/rules" | jq -r '[.firewall_rules[]|select(.subnet=="192.0.2.1" or .subnet=="2001:db8::1")]|length')
  [ "$rules" = 2 ] || { echo "firewall $fw has $rules sentinels, need 2; no instance created" >&2; exit 1; }
  echo "firewall $fw sentinels=2"

  local t; t=$(tunnel_id)
  if [ -z "$t" ]; then
    t=$(cf POST "/accounts/$ACCOUNT/cfd_tunnel" \
      "$(jq -nc --arg n "$TUNNEL" --arg s "$(openssl rand -base64 32)" '{name:$n,tunnel_secret:$s,config_src:"cloudflare"}')" | jq -r .result.id)
  fi
  cf PUT "/accounts/$ACCOUNT/cfd_tunnel/$t/configurations" \
    "$(jq -nc --arg h "$HOST" '{config:{ingress:[{hostname:$h,service:"ssh://127.0.0.1:22"},{service:"http_status:404"}]}}')" >/dev/null
  echo "tunnel $t"

  if [ -z "$(dns_id)" ]; then
    cf POST "/zones/$ZONE/dns_records" \
      "$(jq -nc --arg n "$HOST" --arg c "$t.cfargotunnel.com" '{type:"CNAME",name:$n,content:$c,proxied:true,ttl:1,comment:"vivijure bake box SSH over Access, outbound tunnel only (vivijure-backend#538)"}')" >/dev/null
  fi
  local app; app=$(app_id)
  if [ -z "$app" ]; then
    app=$(cf POST "/accounts/$ACCOUNT/access/apps" \
      "$(jq -nc --arg n "$APP_NAME" --arg d "$HOST" '{type:"self_hosted",name:$n,domain:$d,session_duration:"8h",app_launcher_visible:false,allowed_idps:[],auto_redirect_to_identity:false,http_only_cookie_attribute:true}')" | jq -r .result.id)
    cf POST "/accounts/$ACCOUNT/access/apps/$app/policies" \
      '{"name":"skyphusion.org operators","decision":"allow","precedence":1,"include":[{"email_domain":{"domain":"skyphusion.org"}}]}' >/dev/null
  fi
  echo "access app $app"

  local boot token keys
  token=$(cf GET "/accounts/$ACCOUNT/cfd_tunnel/$t/token" | jq -r .result)
  keys=$(cat "$BAKEBOX_SSH_KEYS")
  boot=$(TOKEN="$token" KEYS="$keys" python3 -c 'import os,sys,base64
s=open(sys.argv[1]).read().replace("@@TUNNEL_TOKEN@@",os.environ["TOKEN"]).replace("@@SSH_KEYS@@",os.environ["KEYS"])
assert "@@" not in s
print(base64.b64encode(s.encode()).decode())' "$HERE/../deploy/bake-box/bootstrap.sh")
  # Project to the id. The create response carries the default root password.
  local id
  id=$(jq -nc --arg r "$REGION" --arg p "$PLAN" --argjson o "$OS_ID" --arg l "$LABEL" --arg f "$fw" --arg u "$boot" \
    '{region:$r,plan:$p,os_id:$o,label:$l,hostname:$l,firewall_group_id:$f,user_data:$u,backups:"disabled",enable_ipv6:false,activation_email:false,tags:[$l]}' \
    | curl -fsS -X POST -H "Authorization: Bearer $VULTR_API_KEY" -H 'Content-Type: application/json' --data @- "$VULTR/instances" | jq -r .instance.id)
  [ -n "$id" ] && [ "$id" != null ] || { echo "instance create returned no id" >&2; exit 1; }
  echo "instance $id"
}

cmd_token() {
  local name=${1:?token suffix}
  local app; app=$(app_id); [ -n "$app" ] || { echo "no Access app" >&2; exit 1; }
  local out id
  out=$(cf POST "/accounts/$ACCOUNT/access/service_tokens" \
    "$(jq -nc --arg n "$TOKEN_PREFIX-$name" '{name:$n,duration:"8760h"}')")
  id=$(jq -r .result.id <<<"$out")
  # The stdout below is the only copy of the secret.
  jq -r '"CF_ACCESS_CLIENT_ID=\(.result.client_id)\nCF_ACCESS_CLIENT_SECRET=\(.result.client_secret)"' <<<"$out"
  # One non_identity policy lists every token of this lane.
  local include policy
  include=$(token_rows | cut -f1 | jq -Rsc 'split("\n")|map(select(length>0))|map({service_token:{token_id:.}})')
  policy=$(cf GET "/accounts/$ACCOUNT/access/apps/$app/policies" | jq -r '.result[]|select(.decision=="non_identity")|.id')
  local body; body=$(jq -nc --argjson i "$include" '{name:"vivijure-bake service tokens",decision:"non_identity",precedence:2,include:$i}')
  if [ -n "$policy" ]; then
    cf PUT "/accounts/$ACCOUNT/access/apps/$app/policies/$policy" "$body" >/dev/null
  else
    cf POST "/accounts/$ACCOUNT/access/apps/$app/policies" "$body" >/dev/null
  fi
  echo "minted $TOKEN_PREFIX-$name id=$id" >&2
}

cmd_rotate_tunnel() {
  local t; t=$(tunnel_id); [ -n "$t" ] || { echo "no tunnel" >&2; exit 1; }
  cf PATCH "/accounts/$ACCOUNT/cfd_tunnel/$t" \
    "$(jq -nc --arg s "$(openssl rand -base64 32)" '{tunnel_secret:$s}')" >/dev/null
  cf GET "/accounts/$ACCOUNT/cfd_tunnel/$t/token" | jq -r .result
}

cmd_clear_user_data() {
  local id; id=$(instance_id); [ -n "$id" ] || { echo "no instance" >&2; exit 1; }
  vultr PATCH "/instances/$id" '{"user_data":""}' >/dev/null
  vultr GET "/instances/$id/user-data" | jq -r '"user_data_bytes=\(.user_data.data|length)"'
}

cmd_down() {
  [ "${BAKEBOX_DESTROY:-}" = "$LABEL" ] || { echo "set BAKEBOX_DESTROY=$LABEL to destroy" >&2; exit 1; }
  local id fw t app d
  id=$(instance_id); fw=$(firewall_id); t=$(tunnel_id); app=$(app_id); d=$(dns_id)
  if [ -n "$id" ]; then
    vultr DELETE "/instances/$id" >/dev/null; echo "instance $id deleted"
    while [ -n "$(instance_id)" ]; do sleep 5; done
  fi
  [ -z "$d" ] || { cf DELETE "/zones/$ZONE/dns_records/$d" >/dev/null; echo "dns $d deleted"; }
  [ -z "$app" ] || { cf DELETE "/accounts/$ACCOUNT/access/apps/$app" >/dev/null; echo "app $app deleted"; }
  token_rows | while IFS=$'\t' read -r tid tname; do
    cf DELETE "/accounts/$ACCOUNT/access/service_tokens/$tid" >/dev/null; echo "token $tname deleted"
  done
  if [ -n "$t" ]; then
    cf DELETE "/accounts/$ACCOUNT/cfd_tunnel/$t/connections" >/dev/null || true
    cf DELETE "/accounts/$ACCOUNT/cfd_tunnel/$t" >/dev/null; echo "tunnel $t deleted"
  fi
  # The group goes last. Vultr refuses it while an instance is attached.
  [ -z "$fw" ] || { vultr DELETE "/firewalls/$fw" >/dev/null; echo "firewall $fw deleted"; }
  cmd_plan
}

: "${VULTR_API_KEY:?}" "${CLOUDFLARE_API_TOKEN:?}"
ZONE_ROW=$(cf GET "/zones?name=$ZONE_NAME" | jq -r '.result[0]|[.id,.account.id]|@tsv')
ZONE=$(cut -f1 <<<"$ZONE_ROW")
ACCOUNT=$(cut -f2 <<<"$ZONE_ROW")
[ -n "$ZONE" ] && [ "$ZONE" != null ] || { echo "cannot read zone $ZONE_NAME" >&2; exit 1; }
case "${1:-}" in
  plan) cmd_plan ;;
  up) cmd_up ;;
  token) shift; cmd_token "$@" ;;
  rotate-tunnel) cmd_rotate_tunnel ;;
  clear-user-data) cmd_clear_user_data ;;
  down) cmd_down ;;
  *) sed -n 2,14p "${BASH_SOURCE[0]}" >&2; exit 2 ;;
esac
