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

To start it again after a reboot, add this line with `crontab -e`:
`@reboot <repo>/tools/sahyadri_manager/start_manager.sh`.

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

A simulation that stops cleanly at its wall time (checkpoint written, outputs missing) is queued again with
priority −1 and restarts from its newest checkpoint, at most `max_sim_restarts` (5) times. A simulation that
fails stops the run as usual.

Change the limits on the page or with `manager.py limits max_sim_running=4 max_rockstar=2 ...` (keys:
`max_sim_running`, `max_sim_queued`, `max_sim_restarts`, `max_rockstar`, `scratch_reserve_gb`, `heartbeat_s`).

When `max_rockstar` Rockstar jobs are already running, the run whose next ready snapshot has waited longest gets
the next turn, so one large run does not hold back the others.

Each registered run's `run.conf` is copied to `~/.sahyadri_mgr/confs/` when it is added.

## Runs on another site (pegasus)

`manager.py add pegasus:<run.conf on pegasus>` registers a run set up there (`setup_production.py --site pegasus`).
The manager keeps a mirror of the run's small files (events, output list, logs, P(k) plot) under
`sites.pegasus.mirror` (`/user/shadab/sahyadri_mirror/pegasus`), refreshed every `sync_s` (120 s) with rsync from
the data node `data.iucaa.in`. The run's jobs are submitted and watched over ssh on `pegasus.iucaa.in`, within the
site's own limits (`manager.py limits pegasus.max_sim_running=2 ...`, or the pegasus row of the queue panel). After
verify, a process on the pawna head node pulls the kept files from the data node into the archive and checks
them against the checksums the verify job wrote (`KEPT_FILES.json`). Trash, undo and purge run on pegasus. Details:
guide section 13.5.

## What it does

| When | Action |
|---|---|
| you press *Submit simulation*, or the queue submits a scheduled run | `submit_pipeline.sh --stages sim`; *Restart from checkpoint* = `--restart latest` |
| `RS_BATCH_MIN` consecutive snapshots after the last Rockstar batch are ready | Rockstar batch `--rs-snaps A-B` (at most `max_rockstar` Rockstar jobs over all runs, default 1) |
| simulation finished | Rockstar batch with all the remaining snapshots |
| Rockstar covers every snapshot | `ctrees → vahc → fits → verify` (one afterok chain) |
| verify passed | `archive` |
| archive passed | waits: *Move scratch data to the trash* (button) |
| a job fails or vanishes | automation of that run stops (red); *Retry* resubmits that stage (a Rockstar batch continues after its newest finished snapshot), *Acknowledge* clears it without resubmitting |

## What is kept

The kept data are set in `run.conf` (written by `setup_production.py`). `scripts/pkdgrav3_pipeline/retention.py
run.conf` prints the resolved plan:

```bash
KEEP_FULL_Z="0 0.5 1 2 3 5 7"     # full snapshots: the planned output nearest each redshift
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
