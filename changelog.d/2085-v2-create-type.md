**Fix: `deploy.sh` sends the v2 routing type on a first create (fleet-chezmoi#2085).**

A v2 endpoint create requires `type` (`QUEUE` here), along with `name` and `scaling`, so the first run of `deploy.sh` on an account with no endpoint was refused. The create body now carries it. The in-place update takes no type and is unchanged.
