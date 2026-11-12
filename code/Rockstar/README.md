# Rockstar changes for Sahyadri

The changed Rockstar files kept here (an overlay on upstream commit 36ce9eea36ee: curvature, w0-wa, the pkdgrav3
`.pkc` reader) are now maintained as git history in https://github.com/shadaba/rockstar (branch `sahyadri`, one commit
per change with the original authors; its source is byte-identical to this overlay applied to 36ce9ee). New
Rockstar changes go there. `scripts/pkdgrav3_pipeline/setup_pawna.sh` builds from that repository and uses this
overlay only when it cannot fetch it.
