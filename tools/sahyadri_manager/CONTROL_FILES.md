# Files that control the run manager

Every file that decides what the Sahyadri run manager does, grouped by where it lives. Paths are those of the
production setup: the manager runs on the pawna head node from `/user/shadab/sahyadri_prod/sahyadri-sandbox`, and the
pegasus install is `~/sahyadri_pipeline` there.

**In practice you edit only `planned_runs.txt` and `queues.txt`, and change limits on the page or with
`manager.py limits`.** Edit a run's `run.conf` only for something special to that one run.

## 1. Manager settings: `~/.sahyadri_mgr/` on pawna

| File | What it controls | Edit? |
|---|---|---|
| `config.json` | Global limits (`max_sim_running`, `max_sim_queued`, `max_sim_restarts`, `max_rockstar`, `scratch_reserve_gb`, `heartbeat_s`, `max_pulls`); paths (`local_python`, `local_codes`, `scratch_root`, `plan_file`, `queues_file`); per-site block `sites.pegasus` (login, data node, scratch, python, sandbox, mirror, `sync_s`, its own limits, `max_pbs_queued`, budget user and `reserve_coreh`); the list of registered runs | Limits: use the page or `manager.py limits`. Anything else by hand, then restart the manager (`start_manager.sh stop; start_manager.sh start`) |
| `planned_runs.txt` | Runs to set up, register and schedule: site, queue, cosmology, N, L, realisations, priority, options (`hold`, `not_before=`, `keep_full_z=`, `keep_pct=`, `root=`, `archive=`). Reread every `plan_s` seconds (300) | Yes |
| `queues.txt` | Which PBS queues take jobs: `open`, `drain` (no new runs; restarts and post-processing go on) or `closed` (nothing submitted), with optional per-queue `max_running` and `max_waiting`. Reread before every scheduling pass. A queue not listed is open | Yes, or the Summary tab buttons, or `manager.py queue <site> <queue> open\|drain\|closed` |
| `confs/` | The manager's saved copy of each run.conf, used when the run's own copy has been purged | No |
| `manager.log`, `screen.log` | What the manager did | Read only |
| `gui.sock`, `ssh-*` | Socket for the page and CLI; ssh connection sockets | No |

`planned_runs.txt` and `queues.txt` are created from the templates in
`scripts/pkdgrav3_pipeline/production/` the first time the manager starts. After that the copies in
`~/.sahyadri_mgr` are the ones it reads, and they live outside the git checkout, so updating the install does not
touch them.

## 2. Per run

Written by `setup_production.py` (directly, or by the manager for a plan-file line).

| File | What it controls |
|---|---|
| `<SCRATCH>/sims/<stem>/r<N>/run.conf` | Everything about one run: N, L, cosmology, `PBS_QUEUE`, wall times, cores/nodes, `RS_BATCH` / `RS_BATCH_MIN`, kept redshifts and subsample (`KEEP_*`), P(k) checks (`PP_PK`, `PK_*`), `ARCHIVE`, `SANDBOX` / `CODES` / `PKD_IO_ROOT`, `SIM_KEEPCHK`; for pegasus runs `SITE` and `REMOTE_*`. Takes effect for the run's next submitted job |
| `<SCRATCH>/sims/<stem>/r<N>/output_steps.txt`, `snapnames.txt` | The planned outputs; the manager counts progress against them |
| `<SCRATCH>/state/<stem>/r<N>/manager.json` | The manager's state for the run: scheduled and priority, paused, failed, acknowledged jobs, restarts, trash, purge. Change it through the page or CLI, not by hand |
| `<SCRATCH>/state/<stem>/r<N>/events/*.jsonl` | Events written by the jobs (started, snapshot ready, done, failed, ...). Every decision of the manager follows from these |
| `<SCRATCH>/state/<stem>/r<N>/KEPT_FILES.json` | Pegasus runs: checksums written after verify; the archive pull on pawna is checked against them |
| `/user/shadab/sahyadri_mirror/pegasus/confs/*.conf` | Mirror configs of pegasus runs (pawna paths plus `REMOTE_*`), rewritten each time the manager loads them |

## 3. Code and templates

In the production clone `/user/shadab/sahyadri_prod/sahyadri-sandbox` (branch `run-manager`); the pegasus
copy is in `~/sahyadri_pipeline/sahyadri-sandbox`.

| File | Role |
|---|---|
| `tools/sahyadri_manager/manager.py` | The manager. `DEFAULT_CFG` holds the defaults; keys in `config.json` override them |
| `tools/sahyadri_manager/ui.html` | The page (Summary, Active, Upcoming, Finished tabs) |
| `tools/sahyadri_manager/start_manager.sh` | Start, stop and status in the `screen` session `sahyadri_mgr` |
| `tools/sahyadri_manager/open_gui.sh` | ssh tunnel from the Mac to the page socket |
| `scripts/pkdgrav3_pipeline/production/setup_production.py` | Writes run.conf: site defaults (root, archive, codes, queue), pegasus queue wall times, resources per N |
| `scripts/pkdgrav3_pipeline/production/cosmologies/*.cosmo` | The cosmology names usable in the plan file |
| `scripts/pkdgrav3_pipeline/production/planned_runs.txt`, `queues.txt` | Templates; copied to `~/.sahyadri_mgr` only when no copy exists there |
| `scripts/pkdgrav3_pipeline/submit_pipeline.sh`, `job_*.pbs`, `pipeline_env.sh` | What a submitted stage actually runs |
| `scripts/pkdgrav3_pipeline/retention.py` | Which files are kept and archived (reads run.conf) |
| `scripts/pkdgrav3_pipeline/archive_run.py`, `verify_run.py`, `scratch_ops.py` | Archive copy and pegasus pull, verification, pegasus trash/undo/purge |

## 4. Outside state the manager reads

- **PBS:** `qstat` on pawna, and over ssh on the pegasus login node.
- **Core-hour budgets** on pegasus (`amgr`); a job is submitted only if cores × requested wall time fits in the
  balance minus `reserve_coreh`.
- **Free space** on each scratch file system, kept above `scratch_reserve_gb`.
