# Sahyadri run manager

Runs the pkdgrav3 → Rockstar → consistent-trees → vahc → FITS pipeline for many runs from the **pawna head node**,
archives the kept data to `/storage` and deletes the working copy only after you approve it. A web page shows
the progress and has the controls.

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

## What it does

| When | Action |
|---|---|
| you press *Submit simulation* | `submit_pipeline.sh --stages sim` (never automatic); *Restart from checkpoint* = `--restart latest` |
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
