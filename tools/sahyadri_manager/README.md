# Sahyadri run manager

Runs the pkdgrav3 → Rockstar → consistent-trees → vahc → FITS pipeline for many runs from the **pawna head node**,
archives the kept data to `/storage` and deletes the working copy only after you approve it. A web page shows
the progress and has the controls.

All files that control what the manager does (settings, plan file, queues file, per-run files): see
[CONTROL_FILES.md](CONTROL_FILES.md).

Compute nodes on pawna cannot submit, release or alter PBS jobs (`qsub: Bad UID for job execution`,
`qrls: Unauthorized Request`, tested 12 Nov 2026). The manager therefore runs on the head node, where it submits
every job.

## Start

On pawna (head node):

```bash
tools/sahyadri_manager/start_manager.sh            # detached screen session "sahyadri_mgr"
tools/sahyadri_manager/manager.py add <run.conf>    # register a run (or use the page); --paused to hold it
tools/sahyadri_manager/start_manager.sh status
```

Set up runs and queue them in one step (they are registered with the manager, which submits the simulations as
the limits below allow):

```bash
cd scripts/pkdgrav3_pipeline/production
python setup_production.py --cosmo default --N 512 --L 200 --real 1-2 --schedule        # priority 0
python setup_production.py --cosmo 'var_*' --real 1 --schedule 5                              # several cosmologies
```

`--cosmo` takes a name, a comma list, shell patterns or `all` (files `cosmologies/<name>.cosmo`).

It starts by itself after a reboot of pawna: the crontab of shadab on pawna has
`@reboot /user/shadab/sahyadri_prod/sahyadri-sandbox/tools/sahyadri_manager/start_manager.sh start` (installed 20 Nov
2026; cron is enabled on pawna, Ubuntu 22.04). It has not yet run at a real boot; on 26 Nov the same command started the
manager in cron's environment (`env -i HOME=... PATH=/usr/bin:/bin`, no terminal, `setsid`). Optional, safer at boot
(NFS and PBS up first, output kept; cron's mail goes nowhere on pawna):
`@reboot sleep 120; <same path>/start_manager.sh start >> /user/shadab/.sahyadri_mgr/cron_start.log 2>&1`.
After a reboot check `start_manager.sh status`.

On the Mac:

```bash
tools/sahyadri_manager/open_gui.sh                  # ssh tunnel + opens http://localhost:8765
```

The page is served on a Unix socket, `~/.sahyadri_mgr/gui.sock`. The socket's directory is mode 0700, so other
users on the head node cannot reach it. The tunnel forwards it to the Mac. If you close the Mac, only the page goes
away: the manager keeps working.

## How it waits

- **Events.** Every job writes its own event file, `<SCRATCH>/state/<stem>/r<real>/events/*.jsonl`. It records
  `start`, `end` with the exit code, `snap_ready` for each output once it is complete in the scratch directory,
  `rockstar_done`, `ctrees_done`, `vahc_done`, `fits_done`, `verify_done` and `archive_done`. `/user` is a local
  disk of the head node, so one `inotifywait -m -r` there sees these writes from the compute nodes as they happen.
  The manager acts only when a line arrives.
- **`qstat`.** It runs `qstat` only every 30 min (`heartbeat_s` in `~/.sahyadri_mgr/config.json`), at start, from
  the page's "Check now" button, and about 3 min after a job seems to have vanished from PBS.
- **State.** The run's state lives in files: the event files written by the jobs and `manager.json` written by
  the manager. Stopping or restarting the manager loses nothing.

## Simulation queue and limits

A run is *scheduled* from the page, with `manager.py schedule <stem/rN> [--priority N] [--at "YYYY-MM-DD HH:MM"]`
or with `setup_production.py --schedule [N]`. The manager submits scheduled simulations, lowest priority number
first, then oldest, when all of these hold:

- running + waiting-in-PBS simulations < `max_sim_running` (default 3) and waiting-in-PBS < `max_sim_queued` (1);
- the `--at` time has passed;
- the run fits in scratch: free space − (this run's estimate) − (what the running simulations will still write) −
  `scratch_reserve_gb` (2000) ≥ 0. The estimate is 10.5 bytes per particle per snapshot (measured at 128³ and
  512³), +15 % for halo catalogues, trees and FITS, plus `SIM_KEEPCHK` checkpoints at 48 bytes per particle, all
  +10 %.
  The page shows the numbers; a run that does not fit waits (the *Submit simulation* button can override).

The start order is on the page: the *Upcoming* tab begins with a **Start order** table (per site: position, run,
priority, when scheduled, what it waits for), its run cards are sorted in that order, and every scheduled run's card
carries a badge `#k of n to start on <site>`. Order in the plan file does not matter, only the priority column.

The plan file's current lines decide what is waiting. A line is set up (setup_production.py, then registered and
queued) only when it is next to start: its site has a simulation slot that no registered run is waiting for
(`max_sim_running`/`max_sim_queued` and the site PBS limit, computed by the scheduler), it is first among that site's
planned lines by priority then line number, its `not_before` has passed and its queue is open. Until then the plan
card lists it as *planned*, with its place on the site. `hold` lines are set up at once (registered paused). A run the
plan set up is forgotten at the next plan check if it never started (no job and no event of it; phase scheduled or not
started) and its line was removed or changed to another site, or its site has no free slot (it is set up again on
its turn; the set-up files are reused). It leaves the manager without being retired, so a line for it can come back later; its set-up files
(run.conf, parameter files) stay on its site. A run that started is never forgotten; the plan card lists it as
*kept*. Runs registered before 26 Nov 2026 count as set up by the plan when the manager log has their
"plan line N: <name> set up" line.

A simulation that stops cleanly at its wall time (checkpoint written, outputs missing) is queued again with
priority −1 and restarts from its newest checkpoint, at most `max_sim_restarts` (5) times. A simulation that
fails stops the run as usual.

Change the limits on the page or with `manager.py limits max_sim_running=4 max_rockstar=2 ...` (keys:
`max_sim_running`, `max_sim_queued`, `max_sim_restarts`, `max_rockstar`, `scratch_reserve_gb`, `heartbeat_s`).

When `max_rockstar` Rockstar jobs are already running, the run whose next ready snapshot has waited longest gets
the next turn, so one large run does not hold back the others.

Each registered run's `run.conf` is copied to `~/.sahyadri_mgr/confs/` when it is added.

## When storage is in trouble

* **Archive paused by hand:** `manager.py queue pawna archive closed` (or the Summary tab). Verified runs wait with
  "archive paused" instead of failing; their data stays in scratch. `... archive open` archives them, one at a time.
* **Storage that hangs** (NFS hard mount): before touching a run's files the manager runs a small read/write test on
  its state directory in a thread (at most once a minute). No answer within `storage_timeout_s` (30 s) -> the run is
  "storage not responding": the manager does not read its files, submit for it or accept page actions on it, and
  stops watching its directories, so the other runs (e.g. on pegasus) go on. An `ARCHIVE` that does not answer
  holds that run's archive the same way. All resumes by itself once the test returns.

## Maintenance windows

A planned shutdown of a site goes in `~/.sahyadri_mgr/maintenance.txt`, one line per window:
`site start end [# note]`, times `YYYY-MM-DDTHH:MM` in local time (IST), `end` `-` when not known. Use the CLI
(the file is reread before every scheduling pass; the Summary tab lists the windows):

    manager.py maint pawna 2026-11-27T08:00 2026-11-27T20:00 cluster maintenance
    manager.py maint swarm 2026-11-27T08:00 -
    manager.py maint                     # list
    manager.py maint swarm clear         # remove the site's windows (e.g. to give an end time: clear, add again)

What the manager does for the run's site (for archive copies and pulls also when the window is pawna's, the head
node running them):

* **Before the start:** a job goes in only if it is expected to end `maint_guard_min` (30) before the start. The
  expected time is 1.25 x the longest earlier job of that stage in the run (else in runs of the same site and N;
  Rockstar per snapshot x the batch size), at most the requested wall time, which is also the estimate without
  earlier jobs. A job that fits but whose requested wall time runs past the start is submitted with its wall time
  cut to the start (`submit_pipeline.sh --wall`). Everything else waits, with the reason on the run's card.
* **Simulations:** a new one or a restart is submitted only with at least `maint_min_sim_h` (4) h of computing
  before the STOP time; its wall time is cut to the start. At `maint_stop_h` before the start (pawna 6 h: the
  2048^3 checkpoint must still drain to NFS; other sites 2 h, `sites.<name>.maint_stop_h`) every running
  simulation of the site gets a `STOP` file in its run directory: pkdgrav3 finishes the step, writes a checkpoint
  and exits cleanly (it deletes the file). Simulation jobs still waiting in PBS then are deleted (qdel) and
  scheduled again. All of them restart after the window, not counted in `max_sim_restarts`.
* **During the window** (until `maint_settle_min` (30) after its end; without an end until the line is changed):
  nothing is submitted to the site.
* **Back early:** every `maint_check_min` (5) minutes during a window the manager looks at the site: its login
  node's boot time (`/proc/stat`), whether PBS answers (`qstat -B`) and how many compute nodes PBS has up
  (`pbsnodes -a`: free or busy). A site that was shut down (booted after the window's start, or no answer, or
  PBS down, at two checks in a row) and then stays up with PBS and at least one compute node for
  `maint_settle_min` is back: its window is over then, and qstat is read again before the next scheduling pass.
  maintenance.txt is not changed; the page shows "over (back <time>)". A site that never goes down (the work was
  done without a shutdown) keeps the window until its end. For a site that rebooted, the time up counts from its
  boot (the site's clock), or from the manager's last check that found it down if that is later; so a manager
  started after the site came back (a reboot of pawna itself) does not wait another `maint_settle_min`. A site
  that went down without rebooting counts from the first check that finds it up. The state is in memory.
* **After it:** jobs that were running or waiting when the window started and failed or vanished are not
  failures: a simulation is scheduled to restart from its newest checkpoint, other stages are resubmitted
  automatically (the "retry"), archive batches are simply done again by the next batch.

When pawna itself is down the manager stops with it; the crontab's `@reboot` line starts it again (see Start;
`start_manager.sh status` to check, `start_manager.sh start` if it is not running); it then catches up as described.

## Node health (pawna compute nodes)

pawna's nodes can be degraded in ways that only show up hours later as a stalled job: an NFS client that hangs
(node1, 23-25 Nov 2026: memory pressure "full" ~90 %, Writeback stuck, a kernel flush thread in D state; the r4
restart and two Rockstar batches stalled), commits that never complete (nodes 2 and 4 since 20 Nov, other I/O still
working), or a local disk filled by what killed jobs left in `/var/tmp`. `scripts/pkdgrav3_pipeline/node_check.sh`
checks a node in ~15 s, with no probe able to block it:

| Result | Check |
|---|---|
| bad | write + fsync + read of a small file on `/user` (the run's state directory) and `/storage` not done in 20 s; memory pressure full avg60 > 20 %; MemAvailable below 90 % of the job's memory request; `/var/tmp` free space below what the job needs (staged 2048^3 checkpoint ~430 GB; Rockstar local output ~65 GB per 2048^3 snapshot, bad below half of that) |
| warn | Writeback unchanged and > 0 over 10 s (a hung NFS commit; reboot due); processes in D state for more than 10 min; `/var/tmp` below the Rockstar estimate |

It is used in three places:

* **Every PBS job on pawna** runs it first (`pipeline_env.sh`; `NODE_CHECK=0` in run.conf turns it off, other sites
  have it off unless `NODE_CHECK=1`). On a bad node the job writes a `node_bad` event and exits with code 75 before
  touching anything. The manager does not count that as a failure: the node is avoided for `node_bad_h` (6) h or
  until a probe passes, a simulation is scheduled again (not counted in `max_sim_restarts`), and any other stage is
  resubmitted at once on another node. More than `node_retry_max` (3) such exits of one run in 24 h stop its
  automation.
* **The manager probes every node** that PBS lists over ssh every `node_probe_min` minutes (default 30, 0 = off).
  Set this in the Summary tab's Nodes card ("probe every ... min", Save), which also has "Probe now". A node whose
  newest probe is bad is avoided. A probe the node cut off without an ssh error (rc 255, nothing on stderr) while a
  bare `ssh <node> true` works is "unknown", not bad, and the node is not avoided: with PBS MoM `$restrict_user on`
  (pawna's nodes since the 27 Nov 2026 maintenance) a node ends any process of a user who has no job there, so the
  check cannot run; each job's own node check still runs inside PBS.
* **Avoiding a node:** while a bad node is usable in PBS, the manager pins each new single-node job
  (`submit_pipeline.sh --host`) to the healthy node with the most free cores that has the job's cores and memory
  free. If none has, the job waits with the reason on the run's card. A node that is down or offline in PBS needs no
  pinning.

The Nodes card also lists our directories in each node's `/var/tmp` (staging, Rockstar output) with size, age and
whether a job still uses them: Rockstar output of a job still in PBS, or the stage of a run whose simulation runs
there. A leftover can be deleted with its button. You type the directory name to confirm, the manager checks it is
not in use and runs `rm -rf` on the node. A stage set aside after a killed job (`*.left.<job>`) may hold outputs
that were never drained; look before deleting it.

Restarts take the newest *complete* checkpoint (`job_sim.pbs`, `scripts/pkdgrav3_pipeline/chk_complete.py`):
the restart file `.chk`, its `.pkl`, the data files `.chk.0` .. `.chk.<n-1>` all there and non-empty (as many
`.chk.<i>.dec`), and their total size an exact multiple of the particle count stored in the `.pkl` (48 bytes per
particle on pawna, 36 with integer positions). The last test catches missing pieces that matching counts miss:
r4's step 1000 had 166 of 191 of each. An incomplete one is skipped (event `chk_skipped`, with the reason) and
the one before it is used. The `sim_rewind` that follows handles the outputs after it.

## Core-hour budgets (pegasus)

A pegasus job is charged to the user's budget (regular/debug queues; **monthly**, 50,000 core-h on the 1st, nothing
carried over) or, with `PBS_PROJECT`, to the project's (half-year, shared with the project's other users). The
manager reads `amgr checkbalance` every `budget_check_min` minutes (default 30; Summary → Core-hour budgets: "read
every … min", "Read now"). amgr needs a token from `ssh -t calam.shadab@pegasus.iucaa.in amgr login` (password,
about 3 h; `amgr sshlogin` is refused for this account), so between logins the manager works from its last reading
(`budgets.json`): reading − what it submitted since − our jobs waiting in PBS on that budget. Refunds are not
counted, so the estimate errs low. A new month without a reading counts as `budget.monthly_coreh`; a project reading
older than `budget.project_stale_h` (6 h) is not used. A submission whose cores × requested wall does not fit in the
estimate minus `budget.reserve_coreh` waits (not a failure), with the reason on the run card. Log in on pegasus and
press "Read now" when the page says the balance is not known.

**One job on another queue.** `manager.py queue-once <stem/rN> project hpc2502016` sends the run's next simulation
job (restart) to the pegasus project queue, charged to the project budget; the setting clears itself after that
one submission, and later jobs use the run.conf's queue again (`manager.py queue-once <stem/rN>` cancels; the run
card shows it with a cancel button). submit_pipeline.sh takes `--queue Q --project P` for this.

## Watchdog (alerts)

Red **ALERT** lines on a run's card, also written to `manager.log` and the run's actions once when they start and
once when they end. The watchdog only looks: it never kills, resubmits or deletes anything. *Dismiss* hides an alert
until it ends and comes back. Limits in `config.json`:

| alert | when | limit |
|---|---|---|
| `sim_stalled` | a running simulation's `run.<job>.log` has not changed | max(`sim_stall_s` 2 h, `sim_stall_steps` 5 × its step time) |
| `drain_stalled` | a staged snapshot or checkpoint is "finished" in `drain.<job>.log` but not "drained" (it exists only on the node's disk) | `drain_stall_s` 3 h |
| `drain_direct` | the drain points pkdgrav3 straight at NFS (node disk below `--min-free-gb`) | — |
| `swap` | the node is swapping: more swap in use than `swap_alert_gb` and pages moving in or out of swap at `swap_alert_mb_s` or more between two `node_mem` events (`job_sim.pbs`, every 30 min; jobs started before 24 Nov 2026 lack the counters), or less memory available than `swap_alert_avail_gb`. Swap merely in use is no alert: a 2048³ run keeps ~120 GB of idle pages in swap with no swap traffic | `swap_alert_gb` 16, `swap_alert_mb_s` 1, `swap_alert_avail_gb` 2 |
| `stage_leftover` | a new simulation job found a stage of an earlier job on its node and set it aside (`<stage>.left.<job>`) | — |
| `rs_redo` | Rockstar output of snapshots that a restart rewrote; Rockstar runs on them again | — |

Why (20–22 Nov 2026, pawna 2048³ r3, r4, r5): the nodes ran with 2–8 GB of memory available and ~125 GB swapped;
the drains' NFS writes hung (`wait_on_commit`) on all three nodes, 1–2 days of snapshots stayed on the local disks,
and r5's simulation hung for 29 h. None of it was visible on the page.

## Restarts after a crash (rewinds)

A simulation restarted from the checkpoint of step K (`--restart latest` = the newest `checkpoints/*.chk`, whose
restart file the drain copies last, so it is complete) writes the outputs after K again. `job_sim.pbs` therefore:

* moves what an earlier job wrote after step K — snapshots, incomplete checkpoints, and stale drain `.part` files —
  to `<sim>/lost_after_chk<K>.<job>/` (a rename; delete it when the run is checked), and sends `sim_rewind step=K`;
* announces as ready (for Rockstar) only the outputs up to K that are complete;
* sets aside a stage directory left on its node by an earlier job (`stage_leftover`) instead of draining it again;
* with staging, also announces snapshots written straight to NFS (the drain switched) once all their files check.

The manager ignores everything announced before a restart for outputs after K; Rockstar results for those outputs
count only if Rockstar started after their new version was ready (otherwise `rs_redo`, and the Rockstar batch
covers them again); an archive batch counts only up to the output before the first one rewritten, so the next batch
(or the final archive: it skips only files of the same size and time) copies the new versions.

## Runs on another site (pegasus)

`manager.py add pegasus:<run.conf on pegasus>` registers a run set up there (`setup_production.py --site pegasus`).
The manager keeps a mirror of the run's small files (events, output list, logs, P(k) plot) under
`sites.pegasus.mirror` (`/user/shadab/sahyadri_mirror/pegasus`), refreshed every `sync_s` (120 s) with rsync from
the data node `data.iucaa.in`. The run's jobs are submitted and watched over ssh on `pegasus.iucaa.in`, within the
site's own limits (`manager.py limits pegasus.max_sim_running=2 ...`, or the pegasus row of the queue panel). After
verify, a process on the pawna head node pulls the kept files from the data node into the archive and checks
them against the checksums the verify job wrote (`KEPT_FILES.json`). Trash, undo and purge run on pegasus. Details:
guide section 13.5.

## Halo stages on another cluster (POST_SITE)

A swarm or leap run can have its halo stages (rockstar ... verify) on pride, which shares `/user1`:
`post_site_conf.py <run.conf> --site pride` (run on pride) writes `run.post_pride.conf` and marks the run.conf with
`POST_SITE`/`POST_CONF`; then `manager.py reload-conf swarm:<stem>/r<R>`. For a new run, the plan option `post_site=pride`
(swarm or leap lines) does both: the manager runs `post_site_conf.py` on pride after `setup_production.py` and
before it registers the run, e.g. `leap workq default 2048 200 101 0 post_site=pride`. The manager submits those stages on pride
with that file and pride's install; the simulation and the archive stay with the run's site. Because the post
configuration keeps the run's SCRATCH, the pride jobs write their products, logs and events into the run's own
directories, and everything downstream (mirror, deletion checks, archive) works unchanged. Limits (`max_rockstar`,
`max_pbs_queued`), queue states and maintenance windows are those of the site where the stage runs. The
maintenance gate's estimate of a Rockstar batch uses the jobs of that site, else `sites.<name>.rs_snap_s`
({NPART: s per snapshot}; pride `{"2048": 3432}`). In production since 25-26 Nov: swarm r6, r7, leap r100 (pride
`max_rockstar` 2). Guide section 19.

## What it does

| When | Action |
|---|---|
| you press *Submit simulation*, or the queue submits a scheduled run | `submit_pipeline.sh --stages sim`; *Restart from checkpoint* = `--restart latest` |
| `RS_BATCH_MIN` consecutive snapshots after the last Rockstar batch are ready | Rockstar batch `--rs-snaps A-B` (at most `max_rockstar` Rockstar jobs over all runs, default 1) |
| simulation finished | Rockstar batch with all the remaining snapshots |
| Rockstar covers every snapshot | `ctrees → vahc → fits → verify` (one afterok chain) |
| verify passed | `archive` |
| archive passed | waits: *Move scratch data to the trash* (button) |
| a job fails or vanishes | automation of that run stops (red); *Retry* resubmits that stage (a Rockstar batch continues after its newest finished snapshot), *Acknowledge* clears it without resubmitting. A failed Rockstar batch is first resubmitted by itself `rs_auto_retry` times (1; counted per batch start, page: Simulation queue); only a further failure stops the run |

## What is kept

The kept data are set in `run.conf` (written by `setup_production.py`). `scripts/pkdgrav3_pipeline/retention.py
run.conf` prints the resolved plan:

```bash
KEEP_FULL_Z="0 0.05 0.1 0.15 0.2 0.3 0.5 0.7137 0.8 1.0 1.9821 2.9563 4.9089"     # full snapshots: the planned output nearest each redshift
KEEP_FULL_SNAPS=""                # ... or explicit Rockstar snapshot numbers
KEEP_SUBSAMPLE_PCT=1              # % of the particles of every other snapshot: 1, 4, 9, 18, ... (leading SUBSAMPLES)
KEEP_HALO_TABLES="basic,extended,tens,vahc"
KEEP_TREES=1; KEEP_CHECKPOINT=0
ARCHIVE_FILE_MODE=0444; ARCHIVE_DIR_MODE=0555
```

## Checks before anything is deleted

1. **`verify`** (`verify_run.py`, written to `verify.json`) checks:
   - **Snapshots.** Every snapshot: all `.pkc` files complete, N³ particles, the planned redshift, the
     subsamples, the potential. The kept full snapshots are also run through `pkc_check`, which confirms every
     particle ID is present exactly once.
   - **1% subsample.** It holds the configured fraction of the particles.
   - **Rockstar.** Every snapshot is covered by a successful batch.
   - **Merger trees.** Present for every snapshot after the first one that has halos.
   - **vahc / tens / vpot.** The same halo IDs, in the same order, as the `.trees` catalogue.
   - **FITS.** The FITS stage passed its read-back check, after the last vahc run, and every kept FITS file is
     present.
   - **P(k)** (`pk_check.py`; spectra from the vahc stage: `pk_measure.py` for every output, CLASS linear and
     halofit at every output redshift from `class_pk.py`). On the bins with k ≤ `PK_KLIN` (0.1 h/Mpc):
     - *absolute*: R = ⟨P_mm / P_ref⟩ over modes, P_ref = P_CLASS(z_start) [D(z)/D(z_start)]² × halofit/linear,
       D the no-radiation growth that pkdgrav3 uses. FAIL if |R − 1| > 4σ + 0.02, σ = √(2/N_modes);
     - *growth*: G = ⟨[P(z)/P(z_first)] / [P_ref(z)/P_ref(z_first)]⟩ over the bins with Δ²_lin(k, z) ≤ 0.01
       (`PK_DELTA2_LIN`). The same modes at both times, so the realisation's scatter cancels. FAIL if |G − 1| > 0.03;
     - CLASS at z_start against the table the ICs were made from: FAIL above 2 %;
     - the ratio to CLASS at z itself (CLASS includes radiation) and the halo bias b1 per mass bin: reported.
     P(k) is the raw P_mm, without 1/n̄ subtraction: particles start on a lattice, so on these scales their
     discreteness noise is far below Poisson. The summary plot `<halo dir>/pk/pk_summary.png` is shown on the
     page and kept in the archive.
2. **`archive`** (`archive_run.py`):
   - refuses unless verify passed after the last change to any kept file;
   - copies each file while computing its SHA-256;
   - reads every copy back from the disk with `O_DIRECT` and compares the checksums;
   - writes `MANIFEST.json` and `SHA256SUMS` (`cd $ARCHIVE && sha256sum -c state/<stem>/r<real>/SHA256SUMS`);
   - makes the archive read-only.
3. **Move to trash** (manager, after you click):
   - checks that no job of the run is active, that verify passed and archive passed after it, that every manifest
     entry is in the archive with the right size and checksum, and that every currently kept file is in the
     manifest and unchanged since;
   - then renames `sims/<stem>/r<real>` and `halos/<stem>/r<real>` to `<SCRATCH>/.trash/…`. This can be undone.
4. **Purge** (after you type the run name): checks the archive copies again, then deletes the trash directory.
   Only paths inside `<SCRATCH>/.trash/` are deleted.
