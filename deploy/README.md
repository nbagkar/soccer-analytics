# deploy/ — shelved

Scripts for running the platform 24/7 on a cloud VM: unattended ingestion
(`soccer serve --once` via a systemd timer) plus a password-protected public
dashboard behind Caddy (auto-HTTPS).

**Status: shelved (2026-09).** Azure for Students was blocked by the NYU tenant;
the effort was paused rather than moved to another provider. The scripts are
provider-agnostic and should work as-is on any Ubuntu 24.04 box with SSH + systemd.
Oracle Cloud Free Tier is the standing fallback if this is revisited.

To resume: provision the VM, clone the repo, then run `setup_vm.sh` (ingestion)
and `setup_dashboard.sh` (public dashboard).

For ad-hoc sharing from your own machine instead, see `scripts/serve_public.sh`.
