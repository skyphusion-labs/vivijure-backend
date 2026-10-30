**Fix: stills load from container disk, and say what they are loading.**

A stills job on a global volume now copies the weights it reads to container disk once per worker.
The volume read was about 26 MB/s, and one job read the UNet twice.
Each file reports its MB/s.
Each model load now emits `load_start`, `load_heartbeat` and `load_done` with elapsed seconds.
