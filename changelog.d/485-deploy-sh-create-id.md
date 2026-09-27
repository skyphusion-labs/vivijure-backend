**Fix: `deploy.sh` first-time create reads the created template and endpoint ids (#486).**

The template and endpoint POSTs ran `rp` in a pipeline subshell, so its `RP_BODY` assignment was
lost and `json_field id` read the previous call's body; a first-time create died with "could not
determine the template id" right after creating the template. Both POSTs now run `rp` in the
current shell. PATCH branches are unchanged.
