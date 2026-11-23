# Handover: pride / leap / flock / raft probe (session in worktree sahyadri-sandbox-pride, 20–23 Nov 2026)

These notes hand the probe over to the session that manages the full programme (worktree ../sahyadri-sandbox, branch
run-manager). Everything here is on branch `pride-probe` (pushed; HEAD 1261a53). The branch is **not merged** into run-manager.
Merging needs the user's approval. The project memory file `pride-probe-brief.md` has the same facts in short form.

## 1. Live now

- **Run:** `leap:Sahyadri_L200N2048/default2048/r100` (seed 471913). This is a production 2048^3 run on leap.
  - The user approved it on 23 Nov.
  - The pawna manager submitted it as job **945009.leap**. It is queued with "Not enough free nodes available".
  - It waits for compute-4-4 and 4-8. Another user's job 945003 holds them (with 4-3 and 4-12), 72 h wall, ends by about 24 Nov 19:00.
  - Run dir: `/user1/shadab/sahyadri_scratch_leap/sims/Sahyadri_L200N2048/default2048/r100/`.
  - Its run.conf is identical to swarm r7 apart from the seed, the paths and the site resources. That gives 2126 steps, 101 outputs and iCheckInterval 100.
  - Resources:
    - `SIM_SELECT` names 6 nodes: compute-4-2, 4-4, 4-5, 4-6, 4-8, 4-9, each 32 cores, 175gb, mpiprocs=2.
    - `place=scatter:excl`, 72 h, `-r n`.
  - Expect about 1.1–1.4× swarm's s/step and about 10 or more 72 h restarts.
- **pawna manager changes (23 Nov).** The manager still runs run-manager code c69928e. No code was changed or restarted.
  - `~/.sahyadri_mgr/config.json`:
    - Added `sites.leap`, which the running manager reloaded on its own.
    - Settings: `login/data shadab@leap.theory.tifr.res.in`, `scratch_root /user1/shadab/sahyadri_scratch_leap`,
      `python /user1/shadab/sahyadri_pride/codes/venv/bin/python`, `sandbox /user1/shadab/sahyadri_c7/sahyadri-sandbox`,
      `mirror /user/shadab/sahyadri_mirror/leap`.
    - Limits: `max_sim_running 1`, `max_sim_queued 1`, **`max_sim_restarts 15`**, **`max_rockstar 0`** (no halo stages on leap),
      `scratch_reserve_gb 2000`, `max_pbs_queued 4`.
    - Backup: `config.json.bak_leap_20261123`.
  - `queues.txt`: added the line `leap workq open`.
  - r100 was registered with `manager.py add leap:<run.conf>` (unpaused), then `manager.py schedule`.
- **Restarts and fixed node names.** Restarts reuse the fixed `SIM_SELECT`. If one of the six named nodes goes down,
  the restart waits forever. In that case edit `SIM_SELECT` in the leap run.conf, or rerun
  `setup_production.py --site leap ... --force` after updating `LEAP_FABRIC_NODES`, then `manager.py` mirror/reload.
  The fabric nodes that work were 4-2, 4-4, 4-5, 4-6, 4-8, 4-9, 4-11 and 4-12 on 23 Nov.
- **Halo stages for r100 are not set up anywhere yet.** The user said not to build Rockstar on leap. Pride can run
  them (section 5), but the manager ties one run to one site. Options:
  - (a) A second run.conf on pride that points at leap's sim dir (same `/user1`), with halo stages started by hand or registered as a separate run.
  - (b) Manager support for a separate post-processing site per run (code change in run-manager).
  - The user has not chosen. It matters once r100 writes snapshots, from about 24–25 Nov.
- **Still queued: flock 1409297** (`bflock4`, 4-node 512^3 benchmark). It is harmless (40 min) and can be deleted with `qdel 1409297` on flock if not wanted.

## 2. Branch pride-probe (on run-manager 485225c; run-manager has since moved to 9b9127a)

| Commit | What |
|---|---|
| 1bd8d18, b99c91c | pride site: `install_production.sh pride`, `pride/env_pride.sh`, `setup_production.py --site pride` (resources_pride), manager `sites.pride` |
| 35b7012 | merge of the pre-rebase copies of the same two commits (a force-push was denied); tree diff 0 |
| de04e5d | leap site: `leap/env_leap.sh`, `leap/ssh_noconda`, `leap/build_c7.sh`, `resources_leap`, `SIM_SELECT` in `submit_pipeline.sh`, manager `sites.leap`, `DEFAULT_QUEUE` pride/leap |
| e0cf4ef | `submit_pipeline.sh`: empty arrays under `set -u` on bash 4.2 (CentOS 7) |
| ee2de15 | `LEAP_FABRIC_NODES` from the 22 Nov node states |
| e6d156b | `memmon.sh` on CentOS 7: `ps -o pid= -o ppid=`; `/proc/<pid>/smaps` when there is no `smaps_rollup` |
| eacf4ab | pride consistent-trees on 20 cores above 512^3 (was 2) |
| 1261a53 | pride vahc 245gb above 512^3 (measured 238.9 GB) |

The leap checkout `/user1/shadab/sahyadri_c7/sahyadri-sandbox` and the pride checkout `/user1/shadab/sahyadri_pride/sahyadri-sandbox`
are both on pride-probe at 1261a53/e6d156b. Leap's jobs run from that checkout. **Do not switch it to run-manager before merging**,
or `SIM_SELECT` and the bash 4.2 fixes are lost for r100's restarts. When merged, the pawna manager also gets `sites.leap`/`pride`
from its code defaults, and the config.json entry stays as an override.

## 3. Sites

| | pride | leap | flock | raft |
|---|---|---|---|---|
| Nodes | 16 × 20 Haswell (E5-2660 v3), ~252 GB (2-2: 189, 2-15: 220) | 12 × 32 Gold 6142, 187 GB | 24-core Gold 6136 | 48-core Gold 6240R |
| Network | 1 GbE | hfi1 100 Gb/s (OPA) | 10 GbE (p4p1) | – |
| OS | Ubuntu, gcc 11 | CentOS 7 (glibc 2.17, bash 4.2, kernel 3.10) | CentOS 7 | CentOS 7 |
| PBS | workq, 72 h | workq, 72 h | workq | workq |
| Down (23 Nov) | 2-1, 2-4, 2-5, 2-9, 2-13, 2-16 | 4-1, 4-10 offline; 4-7 down; 4-3 hfi1 port down | firewalld on 3-3, 3-7, 3-8, 3-11, 3-12, 3-13 blocks MPI | multi-node jobs never start (mom join, Obit 41) |
| Use | halo stages; sim ≤1024^3 | 2048^3 sim on 6 nodes | possible sim site, untested beyond 2 nodes | single node only |

- **All sites:** compute nodes cannot qsub, so submit from the login node. All share `/user1` (swarm's Lustre, about 88 TB free on 23 Nov).
  The TIFR clusters share known_hosts, so ssh between nodes needs `StrictHostKeyChecking=no UserKnownHostsFile=/dev/null`
  (`leap/ssh_noconda`).
- **pride installs:** `/user1/shadab/sahyadri_pride/{codes (Rockstar, consistent-trees, venv), pkdgrav3_io, sahyadri-sandbox}`.
  Scratch: `/user1/shadab/sahyadri_scratch_pride`.
- **leap/flock/raft install:** pkdgrav3 a99f6142 built in a glibc-2.17 conda env (`CONDA_OVERRIDE_GLIBC=2.17`,
  `sysroot_linux-64=2.17`), skylake-avx512: `/user1/shadab/sahyadri_c7/pkdgrav3_io` (`leap/build_c7.sh`). The leap login
  profile sets `ARCH=x86_64`, so compile without `bash -l`.
- **leap MPI:** `--mca pml ucx`, `UCX_TLS=rc_verbs,ud_verbs,self,sm`, `UCX_NET_DEVICES=hfi1_0:1` (`env_leap.sh`). With the defaults the ranks
  pick different PMLs and MPI_Init fails.
- **flock MPI:** `--mca pml ob1 --mca btl tcp,self,vader --mca btl_tcp_if_include p4p1 --prtemca oob_tcp_if_include p4p1`. Every node has the same virbr0.

## 4. Simulation benchmarks (512^3 L200, restart from the step-2000 checkpoint, 2 ranks/node, s/step from snap.timing)

| | 1 node | 2 | 4 | 6 |
|---|---|---|---|---|
| pride | 63.3 | 40.7 | 25.5 | – |
| leap | 34.3 | 20.0 | 10.5 | 7.8 |
| flock | 36.6 | ~21 | queued | – |
| raft | 16.5–21.5 | – | – | – |
| swarm (32 c/node) | 22–24.5 | 11.9 | – | – |

- **leap test:** 128^3 on 2 nodes through the production `job_sim.pbs` passed (945005: 27 min, 101/101 outputs, CHECK OK).
- **pride test:** a 128^3 end-to-end chain passed (sim, Rockstar, ctrees, vahc, FITS 770 checks, Parquet).

## 5. Pride 2048^3 post-processing timing test (23 Nov)

- **Input:** pawna r3 snapshots 01542 and 01566 (z 0.37/0.35, 75 GB each), copied from the pawna head node by rsync (27 min, 92 MB/s).
- **Run dir:** `/user1/shadab/sahyadri_scratch_pride/timing2048` (sims/halos `Sahyadri_L200N2048/default2048/r3`). The output list was cut to 2,
  `NUM_SNAPS = 2`, and `CT_NCPU=20`; the originals are kept as `*.full101` and `run.conf.orig`. It is not a production run and is never archived. It takes 185 GB.

| Stage (job) | Resources | Wall | Peak memory (memmon PSS) |
|---|---|---|---|
| Rockstar (764187) | 3 nodes × 20 writers, 176gb each | 6863 s for 2 snaps (~57 min/snap; pawna ~23 min/snap) | 141 GB on the server node (other nodes not sampled; clients 7.4 GB RSS); 11.9 M halos |
| consistent-trees (764188) | 20 c, 128gb, BOX_DIVISIONS 5 | 1918 s (2 snaps, not representative) | 43 GB |
| vahc (764189) | 20 c, 2 procs, grid 1024, vpot+pk | 8990 s for 2 snaps (read+density ~5200 s, add_value ~3700 s each) | 238.9 GB (asked 230gb; now 245gb) |

- **Full-run estimates:**
  - Rockstar about 95 h, in batches of ~9.5 h. That keeps pace with leap's month-long simulation.
  - ctrees about 15–30 h (a guess).
  - vahc about 100–125 h, so it needs 2 × 72 h (the job skips finished snapshots on a rerun) or a split over nodes.
  - FITS and Parquet were not tested.
- **To do:** when pawna's Rockstar reaches r3 snapshots 01542/01566, compare halo counts with this test, then delete `timing2048`.

## 6. For the admins

leap compute-4-3 hfi1 port down and compute-4-7 down; pride 6 nodes down; raft multi-node PBS jobs never start; firewalld on
flock 3-3, 3-7, 3-8, 3-11, 3-12, 3-13.

## 7. Rules carried over

- Simulations above 512^3 need the user's approval. r100 is approved; others are not.
- Merges into run-manager need the user's approval.
- Use `ssh -o BatchMode=yes`; on the clusters put `/opt/pbs/bin` on PATH.
