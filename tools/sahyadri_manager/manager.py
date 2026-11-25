#!/usr/bin/env python3
"""sahyadri_manager: runs the Sahyadri pipeline for many runs from the pawna head node, with a web page.

  manager.py serve                 the manager (in `screen`; start_manager.sh); page on ~/.sahyadri_mgr/gui.sock
  manager.py add <run.conf> [--paused]   register a run (the running manager picks it up at once)
  manager.py add <site>:<run.conf>       register a run computed on another site (pegasus): conf path there
  manager.py remove <stem/rN>      forget a run (nothing is deleted); its plan line is then not set up again
  manager.py unretire <name>       set up and schedule a removed/deleted run's plan line again
  manager.py delete <stem/rN> [--yes]   show (and with --yes do) the complete clean-up of a run: removed from the manager,
                                   its scratch data deleted (sim, halo, state, trash; mirror for a remote run); the
                                   archive is never touched, only shown
  manager.py list                  registered runs
  manager.py schedule <stem/rN> [--priority N] [--at "YYYY-MM-DD HH:MM"]   queue the simulation (lower priority first)
  manager.py unschedule <stem/rN>
  manager.py queue <site> <queue> open|drain|closed   change a queue's state in the queues file (~/.sahyadri_mgr/queues.txt)
  manager.py queue-once <stem/rN> <queue> [project]   the run's next simulation job only on that queue (pegasus: project hpc2502016); no queue: clear
  manager.py maint [<site> <start> <end|-> [note] | <site> clear]   list / add / remove maintenance windows
                                   (~/.sahyadri_mgr/maintenance.txt; YYYY-MM-DDTHH:MM local time; see below)
  manager.py limits [max_sim_running=3] [max_sim_queued=1] [max_sim_restarts=5] [max_rockstar=1] [scratch_reserve_gb=2000]
  On the Mac: open_gui.sh  (ssh tunnel to the socket, opens http://localhost:8765)

What it does, per run (state = files in <SCRATCH>/state/<stem>/r<real>/: events/*.jsonl written by the jobs,
manager.json written here; nothing is lost when the manager stops, it catches up when started again):
  * waits for events: one `inotifywait -m -r` on the state directories (and ~/.sahyadri_mgr for config changes);
    `qstat` only every HEARTBEAT_S (default 30 min), at start, and shortly after a job seems to have vanished
  * the simulation is submitted from the page (button), or by the simulation queue: a scheduled run (page, `manager.py
    schedule`, `setup_production.py --schedule`) is submitted when the limits (max_sim_running, max_sim_queued) and
    the scratch space allow, by priority then age, not before its --at time; a simulation that stops cleanly at its
    wall time is queued again to restart from its newest checkpoint (at most max_sim_restarts times)
  * Rockstar in batches (RS_BATCH=1): when RS_BATCH_MIN consecutive snapshots after the last batch are ready, or the
    simulation has finished, one batch job; at most MAX_ROCKSTAR Rockstar jobs over all runs (default 1), taken in
    turn between runs (the run whose next ready snapshot has waited longest goes first)
  * then ctrees -> vahc -> fits -> verify (one chain, afterok), then archive (copy, check, read-only) automatically,
    as a process on this head node (archive_on_head, default on: /storage at ~430 MB/s from here against ~25-30 MB/s
    from a compute node, and no compute node taken), one at a time with the pulls of remote runs (max_pulls)
  * archive batches (archive_batches, default on): while the run is computed, each time a snapshot kept in full and
    all earlier outputs are written, their kept particle files (the bulk of a run) are copied (pulled from another
    site) and checked; the final archive skips them. A failed batch only stops further batches of that run
  * a failed job stops the run's automation (shown in red; retry from the page)
  * maintenance windows (maintenance.txt: site start end): before one, jobs go in only if they are expected to end
    maint_guard_min before it (earlier jobs of the stage, else the wall time; a simulation's wall time is cut to the
    start and it needs maint_min_sim_h to compute); maint_stop_h before the start each running simulation of the site
    gets a STOP file (checkpoint, clean exit) and waiting simulation jobs are taken out (qdel); after the window
    (+ maint_settle_min) the simulations restart (not counted in max_sim_restarts) and jobs killed by the shutdown
    are resubmitted instead of stopping the automation
  * deletion of the run's scratch data only after you approve it in the page, in two steps: move to
    <SCRATCH>/.trash/ (instant, can be undone), then purge; both re-check the archive manifest first

Runs on another site (cfg sites.<name>: pegasus, swarm; run.conf written there by setup_production.py --site <name>):
  * the manager keeps a mirror of the run's small files (state/ with the events, output_steps.txt, snapnames.txt,
    the pkdgrav3 and job logs, the P(k) plot) under sites.<name>.mirror, refreshed with rsync over ssh from the
    site's data node every sites.<name>.sync_s seconds; the mirror config (confs/ there) is the run's run.conf with
    SCRATCH, LOGS, SANDBOX and CODES pointing to pawna (REMOTE_* keep the site's paths)
  * qsub/qstat over ssh on the site's login node; its own limits (max_sim_running, max_sim_queued, max_sim_restarts,
    max_rockstar, scratch_reserve_gb) under sites.<name>
  * archive: after verify (which there also writes KEPT_FILES.json, the checksums), a process on this host pulls the
    kept files from the data node into ARCHIVE (archive_run.py with PULL_HOST; pawna's compute nodes cannot reach
    it), at most max_pulls at a time; trash/undo/purge on the site with scratch_ops.py over ssh
"""
import datetime
import glob
import http.server
import json
import os
import queue
import re
import shlex
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'pkdgrav3_pipeline'))
import retention  # noqa: E402
import scratch_ops  # noqa: E402

HOME = os.environ.get('SAHYADRI_MGR_HOME', os.path.expanduser('~/.sahyadri_mgr'))
CONFIG = os.path.join(HOME, 'config.json')
SOCK = os.path.join(HOME, 'gui.sock')
LOGF = os.path.join(HOME, 'manager.log')
PBS = '/opt/pbs/bin'
# scratch_reserve_gb: kept free on the scratch file system (shared home) when deciding whether a new simulation fits
# simulation queue: a scheduled run is submitted when running + waiting-in-PBS simulations < max_sim_running and
# waiting-in-PBS < max_sim_queued (and it fits in scratch); a simulation that stops cleanly at its wall time
# (checkpoint, not all outputs) is scheduled again to restart, at most max_sim_restarts times
DEFAULT_CFG = dict(runs=[], heartbeat_s=1800, max_rockstar=1, lost_grace_s=180, scratch_reserve_gb=2000,
                   max_sim_running=3, max_sim_queued=1, max_sim_restarts=5, max_pulls=1, archive_on_head=True, archive_batches=True,
                   # a file system (run SCRATCH, ARCHIVE) that does not answer a small read/write test within
                   # storage_timeout_s is 'not responding': its runs are left alone until it answers again
                   storage_timeout_s=30,
                   # watchdog (alerts on the page and in the log; nothing is killed or resubmitted by it):
                   # a running simulation whose log has not changed for max(sim_stall_s, sim_stall_steps x its step
                   # time); a staged snapshot or checkpoint finished but not drained to NFS after drain_stall_s, or
                   # the drain writing straight to NFS (node disk full); the node swapping: above swap_alert_gb in
                   # swap and pages moving in or out of swap at swap_alert_mb_s or more between two node_mem events
                   # (job_sim.pbs, every 30 min), or less than swap_alert_avail_gb of memory available; a stage left
                   # on a node by an earlier job (stage_leftover)
                   sim_stall_s=7200, sim_stall_steps=5, drain_stall_s=10800, swap_alert_gb=16, swap_alert_mb_s=1.0,
                   swap_alert_avail_gb=2,
                   local_codes='/user/shadab/sahyadri_prod/codes',
                   # planned runs: set up, registered and scheduled by the manager; queues: which PBS queues take
                   # jobs (open | drain | closed) and their own limits. Both in the manager's directory (edited there,
                   # outside the git checkout), first copied from the templates in production/
                   plan_file=os.path.join(HOME, 'planned_runs.txt'), queues_file=os.path.join(HOME, 'queues.txt'),
                   plan_s=300, local_python='/user/shadab/sahyadri_prod/codes/venv/bin/python',
                   # maintenance windows of the sites (maintenance.txt in the manager's directory; see read_maint):
                   # before a window a job is submitted only if it is expected to end maint_guard_min before the
                   # start (simulations: with at least maint_min_sim_h to compute, their wall time cut to the start);
                   # a running simulation gets a STOP file maint_stop_h (per site) before the start (checkpoint and
                   # clean exit) and restarts after the window, maint_settle_min after its end; jobs killed by the
                   # shutdown are resubmitted then instead of stopping the run's automation
                   maintenance_file=os.path.join(HOME, 'maintenance.txt'), maint_guard_min=30, maint_min_sim_h=4,
                   maint_stop_h=6, maint_settle_min=30,
                   # node health (this site's compute nodes; scripts/pkdgrav3_pipeline/node_check.sh): probed over ssh
                   # every node_probe_min minutes (0 = off; page: Nodes); a node that fails (probe or a job's own check,
                   # exit 75) is avoided for node_bad_h hours or until a probe passes: jobs are pinned to a healthy node
                   # with room; a run's job is resubmitted after a failed node check at most node_retry_max times a day
                   node_probe_min=30, node_bad_h=6, node_retry_max=3, node_probe_paths=None,
                   # core-hour budgets of the sites with one (pegasus amgr; class Budgets) read every
                   # budget_check_min minutes (0 = only when a job is submitted or on the page's "read now")
                   budget_check_min=30,
                   scratch_root='/user/shadab/sahyadri_scratch',
                   sites=dict(pegasus=dict(login='calam.shadab@pegasus.iucaa.in', data='calam.shadab@data.iucaa.in',
                                           scratch_root='/data/project/hpc2502016/shadab/sahyadri_scratch',
                                           python='/mnt/home/project/calam.shadab/sahyadri_pipeline/codes/venv/bin/python',
                                           sandbox='/mnt/home/project/calam.shadab/sahyadri_pipeline/sahyadri-sandbox',
                                           mirror='/user/shadab/sahyadri_mirror/pegasus', sync_s=120,
                                           max_sim_running=2, max_sim_queued=1, max_sim_restarts=10, max_rockstar=1,
                                           scratch_reserve_gb=2000, max_pbs_queued=6,
                                           # core-hour budgets (amgr, class Budgets): PBS_PROJECT set -> the
                                           # project's, else the user's (monthly, monthly_coreh on the 1st); a job
                                           # is submitted only if cores x requested wall fits in the estimate minus
                                           # reserve_coreh; projects: also shown on the page
                                           budget=dict(user='calam.shadab', reserve_coreh=2000, monthly_coreh=50000,
                                                       projects=['hpc2502016'], project_stale_h=6)),
                             # swarm (TIFR): login, data and PBS on the same host; no core-hour budgets
                             swarm=dict(login='shadab@swarm.theory.tifr.res.in', data='shadab@swarm.theory.tifr.res.in',
                                        scratch_root='/user1/shadab/sahyadri_scratch',
                                        python='/user1/shadab/sahyadri_pipeline/codes/venv/bin/python',
                                        sandbox='/user1/shadab/sahyadri_pipeline/sahyadri-sandbox',
                                        mirror='/user/shadab/sahyadri_mirror/swarm', sync_s=120,
                                        max_sim_running=1, max_sim_queued=1, max_sim_restarts=10, max_rockstar=1,
                                        scratch_reserve_gb=2000, max_pbs_queued=4),
                             # pride (TIFR theory): like swarm (same Lustre /user1, own scratch and install); compute
                             # nodes cannot qsub (Bad UID): jobs are submitted on the login node
                             pride=dict(login='shadab@pride.theory.tifr.res.in', data='shadab@pride.theory.tifr.res.in',
                                        scratch_root='/user1/shadab/sahyadri_scratch_pride',
                                        python='/user1/shadab/sahyadri_pride/codes/venv/bin/python',
                                        sandbox='/user1/shadab/sahyadri_pride/sahyadri-sandbox',
                                        mirror='/user/shadab/sahyadri_mirror/pride', sync_s=120,
                                        max_sim_running=1, max_sim_queued=1, max_sim_restarts=10, max_rockstar=1,
                                        scratch_reserve_gb=2000, max_pbs_queued=4),
                             # leap (TIFR theory): same Lustre /user1 (own scratch); simulation only so far (pkdgrav3
                             # from leap/build_c7.sh, pride's venv for the Python steps; the halo stages need builds
                             # for glibc 2.17); compute nodes cannot qsub
                             leap=dict(login='shadab@leap.theory.tifr.res.in', data='shadab@leap.theory.tifr.res.in',
                                       scratch_root='/user1/shadab/sahyadri_scratch_leap',
                                       python='/user1/shadab/sahyadri_pride/codes/venv/bin/python',
                                       sandbox='/user1/shadab/sahyadri_c7/sahyadri-sandbox',
                                       mirror='/user/shadab/sahyadri_mirror/leap', sync_s=120,
                                       max_sim_running=1, max_sim_queued=1, max_sim_restarts=10, max_rockstar=0,
                                       scratch_reserve_gb=2000, max_pbs_queued=4)))
LIMIT_KEYS = ('max_sim_running', 'max_sim_queued', 'max_sim_restarts', 'max_rockstar', 'scratch_reserve_gb', 'heartbeat_s',
              'node_probe_min', 'budget_check_min')
ZERO_OK = ('max_sim_queued', 'max_sim_restarts', 'scratch_reserve_gb', 'node_probe_min', 'budget_check_min')   # limits that may be 0
# max_pbs_queued: our jobs waiting in the site's PBS at most (pegasus queued_jobs_threshold: project 6, debug 2 per
# user); with it, the post-processing stages are submitted one at a time, each after the previous one finished
SITE_LIMIT_KEYS = ('max_sim_running', 'max_sim_queued', 'max_sim_restarts', 'max_rockstar', 'scratch_reserve_gb',
                   'max_pbs_queued')
# requested wall time of a stage in run.conf (and submit_pipeline.sh's default when the key is not set)
STAGE_WALL = dict(sim=('SIM_WALL', '24:00:00'), rockstar=('RS_WALL', '12:00:00'), ctrees=('CT_WALL', '08:00:00'),
                  vahc=('PP_WALL', '24:00:00'), vpot=('PP_WALL', '24:00:00'), fits=('FITS_WALL', '06:00:00'),
                  parquet=('PQ_WALL', '02:00:00'), xcorr=('XC_WALL', '24:00:00'), ximult=('XM_WALL', '06:00:00'),
                  vprof=('VPROF_WALL', '48:00:00'), verify=('VF_WALL', '12:00:00'), archive=('AR_WALL', '72:00:00'),
                  archb=('AR_WALL', '72:00:00'))
# cores and memory of a stage's job (run.conf keys and submit_pipeline.sh defaults), for placing it on a node
STAGE_RES = dict(sim=('SIM_NCPU', 'SIM_MEM', 192, '700gb'), rockstar=('RS_NCPU', 'RS_MEM', 64, '480gb'),
                 ctrees=('CT_NCPU', 'CT_MEM', 2, '32gb'), vahc=('PP_NCPU', 'PP_MEM', 16, '400gb'),
                 vpot=('PP_NCPU', 'PP_MEM', 16, '400gb'), fits=('FITS_NCPU', 'FITS_MEM', 16, '100gb'),
                 parquet=('PQ_NCPU', 'PQ_MEM', 8, '16gb'), xcorr=('XC_NCPU', 'XC_MEM', 16, '16gb'),
                 ximult=('XM_NCPU', 'XM_MEM', 32, '32gb'), vprof=('VPROF_NCPU', 'VPROF_MEM', 32, '160gb'),
                 verify=('VF_NCPU', 'VF_MEM', 8, '32gb'))
NODE_CHECK = os.path.join(REPO, 'scripts', 'pkdgrav3_pipeline', 'node_check.sh')
NODE_RC = 75     # exit code of a job whose node failed its check (pipeline_env.sh)


def mem_gb(text):
    """PBS memory ``700gb`` / ``792078740kb`` / ``1tb`` -> GB (float); 0 if not understood."""
    m = re.match(r'^\s*(\d+(?:\.\d+)?)\s*([kmgt]?b)?\s*$', str(text or '').lower())
    if not m:
        return 0.0
    return float(m.group(1)) * {'b': 2**-30, 'kb': 2**-20, 'mb': 2**-10, 'gb': 1, 'tb': 1024}[m.group(2) or 'b']
MAINT_STOP_H_REMOTE = 2    # sites without node staging: the checkpoint goes straight to Lustre (minutes)
POST = ['ctrees', 'vahc', 'fits', 'verify']


def post_of(E):
    """Post-processing stages of a run, in order: with PARQUET=1 in run.conf, stage parquet (Parquet copy of the FITS,
    archived and verified with them) between fits and verify."""
    return ['ctrees', 'vahc', 'fits', 'parquet', 'verify'] if E.get('PARQUET', '0') == '1' else POST
LOCAL = 'pawna'
SSH_OPTS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30', '-o', 'ServerAliveInterval=30',
            '-o', 'ControlMaster=auto', '-o', 'ControlPersist=600']


def scratch_estimate(E, n_out):
    """Bytes a run will occupy in SCRATCH at most (before its deletion is approved): every snapshot in full
    (10.5 B/particle measured with the potential at z <= 0.5: 128^3 and 512^3 runs), SIM_KEEPCHK checkpoints
    (SIM_CHK_BPP: 48 B/particle, 36 with integer positions), halo catalogues/trees/FITS (~0.5 of the snapshots at 64^3, a small fraction at large N;
    0.15 used), plus 10 %."""
    n3 = float(E.get('NPART', 0)) ** 3
    snaps = 10.5 * n3 * n_out
    chk = float(E.get('SIM_CHK_BPP', 48) or 48) * n3 * int(E.get('SIM_KEEPCHK', '2') or 2)   # 36 with integer positions
    return int(1.10 * (snaps * 1.15 + chk))
def archive_estimate(E, plan):
    """Bytes a run's archive will hold (retention plan): full snapshots 10.5 B/particle each, the other outputs at the
    subsample fractions of that, halo catalogues + trees as FITS 106 B/particle and as Parquet 88 B/particle more,
    48 B/particle with the checkpoint. Calibrated on today's 512^3 archives (L200/L500/L1000: estimate 55.5 / 54 / 52
    GB, archived 53.0 / 54.1 / 48.9 GB; halo products from L200, the largest)."""
    if not plan:
        return None
    n3 = float(E.get('NPART', 0)) ** 3
    nf, no = len(plan['full']), len(plan['outputs'])
    b = 10.5 * n3 * (nf + (no - nf) * sum(plan['subsamples']) / 100.0)
    b += 106.0 * n3 + (88.0 * n3 if plan.get('parquet') else 0)
    if plan.get('checkpoint'):
        b += float(E.get('SIM_CHK_BPP', 48) or 48) * n3
    return int(b)


STAGES = ['sim', 'rockstar', 'ctrees', 'vahc', 'fits', 'parquet', 'verify', 'archive']
# actions that read the archive on /storage (NFS, can take minutes when it is busy): run in a worker thread, the
# run marked busy (no automation for it meanwhile), so events of the other runs are still handled
SLOW = ('check_trash', 'approve_trash', 'undo_trash', 'approve_purge')


def now():
    """Current time in seconds since the epoch (``time.time``; one place to patch in tests)."""
    return time.time()


def iso(t=None):
    """Local time ``t`` (default: now) as ``YYYY-MM-DD HH:MM:SS``, the format of the manager log."""
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t or now()))


def log(msg):
    """Print ``msg`` with a time stamp and append it to the manager log ``LOGF``.

    Writing the log file is best effort: an unwritable log never stops the manager.
    """
    line = '[%s] %s' % (iso(), msg)
    print(line, flush=True)
    try:
        with open(LOGF, 'a') as f:
            f.write(line + '\n')
    except OSError:
        pass


def load_cfg():
    """Return the manager configuration: ``DEFAULT_CFG`` updated with ``config.json`` in ``HOME``.

    Top-level keys replace the defaults; each entry of ``sites`` is merged key by key over the default site
    settings, so a site in ``config.json`` only needs the keys it changes. An unreadable file is logged and the
    defaults are used.
    """
    c = json.loads(json.dumps(DEFAULT_CFG))
    if os.path.isfile(CONFIG):
        try:
            u = json.load(open(CONFIG))
            sites = u.pop('sites', {})
            c.update(u)
            for k, v in sites.items():          # site settings merged over the defaults
                c['sites'].setdefault(k, {}).update(v)
        except ValueError as e:
            log('config.json unreadable (%s); using defaults' % e)
    return c


def ssh(host, cmd, timeout=300, check=False):
    """Run `cmd` (a shell string) on `host`; one shared connection per host (ControlPath in HOME)."""
    r = subprocess.run(['ssh'] + SSH_OPTS + ['-o', 'ControlPath=%s/ssh-%%r@%%h' % HOME, host, cmd],
                       capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError('ssh %s %r: rc %d: %s' % (host, cmd[:200], r.returncode, (r.stderr or r.stdout)[-500:]))
    return r


class BudgetWait(Exception):
    """Not enough core hours left on the site's budget for this submission (wait, not a failure)."""


def hours(wall):
    """Convert a PBS walltime ``HH[:MM[:SS]]`` to hours (float)."""
    h, m, sec = (list(map(int, str(wall).split(':'))) + [0, 0])[:3]
    return h + m / 60.0 + sec / 3600.0


def job_coreh(E, stage):
    """Core hours PBS reserves for one job of `stage` (cores x requested wall time), from the run config."""
    n = lambda k, d: int(float(E.get(k) or d))     # noqa: E731
    if stage == 'sim':
        return n('SIM_NODES', 1) * n('SIM_NCPU', 1) * hours(E.get('SIM_WALL', '0'))
    pre = dict(rockstar='RS', ctrees='CT', vahc='PP', vpot='PP', fits='FITS', parquet='PQ', verify='VF', xcorr='XC',
               archive='AR')[stage]
    dflt = dict(VF=(8, '12:00:00'), XC=(16, '02:00:00'), AR=(2, '72:00:00'), PQ=(8, '02:00:00')).get(pre, (1, '0'))
    nodes = n('RS_NODES', 1) if stage == 'rockstar' else 1    # Rockstar over several nodes: RS_NCPU per node
    return nodes * n(pre + '_NCPU', dflt[0]) * hours(E.get(pre + '_WALL') or dflt[1])


class Budgets:
    """Core-hour balances of the sites' budgets (pegasus ``amgr``), read every ``budget_check_min`` minutes.

    ``amgr checkbalance`` needs a login token that the user makes with ``amgr login`` (password; it expires after
    ~3 h, and ``amgr sshlogin`` is not allowed for our account), so most reads fail with "token expired". The last
    good reading is kept (``HOME/budgets.json``, survives restarts) and turned into an estimate:

        left = last amgr balance - core hours of jobs submitted since that reading (cores x requested wall)
               - core hours PBS will hold for our jobs still waiting in the queue (held only when a job starts)

    Refunds of unused wall time are not counted, so the estimate is low, never high. A user budget is monthly
    (``2026.mNN``, ``monthly_coreh`` credited on the 1st, nothing carried over): a reading from an earlier month is
    replaced by ``monthly_coreh``. A project budget is shared with the project's other users, so an estimate older
    than ``project_stale_h`` hours is not trusted (submissions to it wait for a fresh reading).
    """
    FRESH = 300          # an amgr reading younger than this is used without asking again

    def __init__(self, path=None):
        """Load the saved readings ``{key: entry}`` (key ``site|kind|name|period``) from ``path``."""
        self.path = path or os.path.join(HOME, 'budgets.json')
        self.lock = threading.Lock()
        self.st = {}              # key -> dict(read, t, month, debit, queued, queued_t, err, err_t, ok_t)
        try:
            with open(self.path) as f:
                self.st = json.load(f)
        except (OSError, ValueError):
            pass

    def save(self):
        """Write the readings (best effort)."""
        try:
            tmp = self.path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(self.st, f, indent=1)
            os.replace(tmp, self.path)
        except OSError:
            pass

    @staticmethod
    def month(t=None):
        """amgr's monthly period of time ``t``: ``YYYY.mN``."""
        lt = time.localtime(t or now())
        return '%d.m%d' % (lt.tm_year, lt.tm_mon)

    @staticmethod
    def which(cfg, run, project=None):
        """Budget key of a run, or None when its site has no budget.

        A run with ``PBS_PROJECT`` is charged to that project in the current half-year period (``YYYY.h1``/``h2``),
        otherwise to the site's ``budget.user``. Returns ``(site, 'project'|'user', name, period)``.
        """
        B = cfg['sites'][run.site].get('budget')
        if not B:
            return None
        proj = (run.E.get('PBS_PROJECT') or '') if project is None else project
        return Budgets.project_key(run.site, proj) if proj else (run.site, 'user', B['user'], '')

    @staticmethod
    def project_key(site, proj):
        """Key of project ``proj`` on ``site`` in the current half-year."""
        t = time.localtime(now())
        return (site, 'project', proj, '%d.h%d' % (t.tm_year, 1 if t.tm_mon <= 6 else 2))

    @staticmethod
    def k(key):
        """String form of a key tuple (the JSON file's keys)."""
        return '|'.join(key)

    def keys(self, cfg, runs):
        """The budgets to read: each budget site's user and its ``budget.projects``, and the budgets of the runs."""
        out = []
        for site, sc in cfg['sites'].items():
            B = sc.get('budget')
            if not B:
                continue
            out.append((site, 'user', B['user'], ''))
            out += [self.project_key(site, p) for p in B.get('projects', [])]
        for run in runs:
            if run.site in cfg['sites']:
                key = self.which(cfg, run)
                if key and key not in out:
                    out.append(key)
        return out

    def read(self, cfg, key):
        """Ask amgr for the balance of ``key`` (and qstat for our waiting jobs); store it. True if amgr answered."""
        site, kind, name, period = key
        sc = cfg['sites'][site]
        cmd = 'amgr checkbalance %s -n %s%s < /dev/null 2>&1' % (kind, shlex.quote(name),
                                                                 ' -p %s' % period if period else '')
        try:
            r = ssh(sc['login'], cmd, timeout=120)
            out = (r.stdout or '') + (r.stderr or '')
        except Exception as e:  # noqa: BLE001
            out = 'ssh: %s' % e
        m = re.search(r"'cost':\s*([-0-9.eE+]+)", out)
        q = self.queued(sc, kind, name)
        with self.lock:
            e = self.st.setdefault(self.k(key), {})
            if q is not None:
                e['queued'], e['queued_t'] = q, now()
            if m:
                e.update(read=float(m.group(1)), t=now(), month=self.month(), debit=0.0, err=None, err_t=None)
            else:
                msg = ' '.join(out.split())[-200:] or 'no output'
                if 'token' in msg.lower() or 'log in' in msg.lower() or 'login' in msg.lower():
                    msg = 'amgr login needed (%s)' % msg
                if e.get('err') != msg:
                    log('budget %s: %s' % (self.k(key), msg))
                e.update(err=msg, err_t=now())
        self.save()
        return bool(m)

    @staticmethod
    def queued(sc, kind, name):
        """Core hours (cores x requested wall) of our jobs waiting in the site's PBS on this budget, or None."""
        user = sc['login'].split('@')[0]
        try:
            r = ssh(sc['login'], '/opt/pbs/bin/qstat -f -F json -u %s 2>/dev/null' % shlex.quote(user), timeout=60)
            J = json.loads(r.stdout or '{}', strict=False).get('Jobs', {})
        except Exception:  # noqa: BLE001
            return None
        tot = 0.0
        for j in J.values():
            if j.get('job_state') not in ('Q', 'H', 'W'):
                continue
            proj = j.get('project', '_pbs_project_default')
            mine = (proj == name) if kind == 'project' else proj in ('', '_pbs_project_default')
            if mine and j.get('job_owner', j.get('Job_Owner', '')).split('@')[0] == user:
                rl = j.get('Resource_List', {})
                tot += float(rl.get('ncpus', 0) or 0) * hours(rl.get('walltime', '0'))
        return tot

    def estimate(self, cfg, key):
        """``(core hours left, state, note)``; state ``ok`` (fresh), ``estimate`` (from an older reading),
        ``assumed`` (monthly credit, no reading this month) or ``unknown`` (no usable reading)."""
        site, kind = key[0], key[1]
        B = cfg['sites'][site].get('budget') or {}
        e = self.st.get(self.k(key), {})
        queued = e.get('queued', 0.0) or 0.0
        if 'read' not in e:
            return None, 'unknown', 'never read'
        if kind == 'user' and e.get('month') != self.month():
            credit = B.get('monthly_coreh')
            if not credit:
                return None, 'unknown', 'last reading is from %s' % e.get('month')
            return float(credit) - e.get('debit_m', 0.0) - queued, 'assumed', \
                'monthly credit %.0f (no amgr reading in %s yet)' % (float(credit), self.month())
        left = e['read'] - e.get('debit', 0.0) - queued
        age = now() - e['t']
        if kind == 'project' and age > float(B.get('project_stale_h', 6)) * 3600:
            return left, 'unknown', 'project reading %.1f h old (shared budget; needs a fresh amgr reading)' % (age / 3600)
        return left, 'ok' if age < self.FRESH else 'estimate', ''

    def check(self, cfg, run, stages, project=None):
        """Raise ``BudgetWait`` if the budget of ``run`` cannot pay for ``stages`` (comma separated).

        The need is the sum of :func:`job_coreh` over the stages; the site's ``reserve_coreh`` is kept back. A
        reading older than ``FRESH`` s is renewed first (if the token still works). On success the need is debited,
        so submissions between two amgr readings cannot overspend.
        """
        key = self.which(cfg, run, project)     # project: this submission's -P (None: the run.conf's)
        if key is None:
            return
        e = self.st.get(self.k(key), {})
        if now() - e.get('t', 0) > self.FRESH:
            self.read(cfg, key)
        need = sum(job_coreh(run.E, st) for st in stages.split(','))
        reserve = float(cfg['sites'][run.site]['budget'].get('reserve_coreh', 0))
        left, state, note = self.estimate(cfg, key)
        name = '%s%s' % (key[2], ' ' + key[3] if key[3] else '')
        if left is None or state == 'unknown':
            raise BudgetWait('%s budget %s: balance not known (%s); run "amgr login" on %s' % (
                run.site, name, note, cfg['sites'][run.site]['login']))
        if left - reserve < need:
            raise BudgetWait('%s budget %s: %.0f core-h left (%s%s), reserve %.0f, this submission (%s) needs %.0f'
                             % (run.site, name, left, state, ', ' + note if note else '', reserve, stages, need))
        with self.lock:
            e = self.st.setdefault(self.k(key), {})
            e['debit'] = e.get('debit', 0.0) + need
            if e.get('month') != self.month():          # debits of this month, for the monthly-credit estimate
                if e.get('debit_month') != self.month():
                    e['debit_m'], e['debit_month'] = 0.0, self.month()
                e['debit_m'] = e.get('debit_m', 0.0) + need
        self.save()

    def refresh(self, cfg, runs):
        """Read every budget (``keys``); called by the manager's budget thread."""
        for key in self.keys(cfg, runs):
            self.read(cfg, key)

    def view(self, cfg):
        """The budgets for the page: site, kind, name, period, coreh (estimate), state, note, amgr reading and
        its time, our queued core hours, the last error and the login command."""
        out = []
        for ks, e in list(self.st.items()):
            key = tuple(ks.split('|'))
            if len(key) != 4 or key[0] not in cfg['sites']:
                continue
            left, state, note = self.estimate(cfg, key)
            out.append(dict(site=key[0], kind=key[1], name=key[2], period=key[3] or self.month(), coreh=left,
                            state=state, note=note, read=e.get('read'), t=e.get('t'), debit=e.get('debit', 0.0),
                            queued=e.get('queued'), err=e.get('err'), err_t=e.get('err_t'),
                            login='ssh -t %s amgr login' % cfg['sites'][key[0]]['login']))
        return out


BUDGETS = Budgets()


class Probe:
    """Is a file system answering? A small test (fn) in a thread, waited for at most `timeout` s; a test that has
    not returned by then marks the key down until it returns (NFS hard mounts hang instead of failing). Errors count
    as answering: only a hang is 'not responding'. Results younger than FRESH s are reused."""
    FRESH = 60

    def __init__(self):
        """Empty per-key probe state (last success, running test thread, start time, down since)."""
        self.st = {}

    def ok(self, key, fn, timeout):
        """Non-blocking health check: True if the storage behind ``key`` answered recently.

        ``fn`` is a test (listdir, small write) run in a daemon thread; a success within ``FRESH`` seconds counts as
        up. If a test is still running after ``timeout`` seconds the storage is marked down (logged once) and False is
        returned, so a hung NFS mount never blocks the manager's main loop.
        """
        s = self.st.setdefault(key, dict(t_ok=0.0, th=None, t0=0.0, down=None))
        th = s['th']
        if th is None or not th.is_alive():
            if now() - s['t_ok'] < self.FRESH:
                return True

            def test():
                """Run the probe; record the success time and log when a storage that was down answers again."""
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass
                s['t_ok'] = now()
                if s['down'] is not None:
                    log('storage %s answers again (not responding for %.0f s)' % (key, now() - s['down']))
                    s['down'] = None
            s['t0'] = now()
            th = s['th'] = threading.Thread(target=test, daemon=True)
            th.start()
        wait = s['t0'] + timeout - now()
        if wait > 0:
            th.join(wait)
        if th.is_alive():
            if s['down'] is None:
                s['down'] = s['t0']
                log('storage %s not responding (no answer in %d s); its runs are left alone until it answers' % (key, timeout))
            return False
        return True

    def down(self, key):
        """Down since (time) or None, without starting a test."""
        s = self.st.get(key)
        return s and s['down']


PROBE = Probe()


def storage_test(d):
    """List directory d and write, fsync and remove a small file in it."""
    def fn():
        """Probe a scratch directory: list it and write, fsync and remove a small file."""
        os.listdir(d)
        f = os.path.join(d, '.mgr_probe')
        with open(f, 'w') as fh:
            fh.write(str(now()))
            fh.flush()
            os.fsync(fh.fileno())
        os.unlink(f)
    return fn


def archive_test(d):
    """Stat the archive root (or its nearest existing parent) and list it."""
    def fn():
        """Probe an archive path: list the deepest existing parent directory (read only)."""
        p = d
        while p and not os.path.isdir(p) and p != '/':
            p = os.path.dirname(p)
        os.listdir(p)
    return fn


def fmt_l(L):
    """Box size for run names: ``200`` for 200.0, ``12p5`` for 12.5 (no dot in file names)."""
    L = float(L)
    return ('%d' % L) if L.is_integer() else ('%g' % L).replace('.', 'p')


def reals_of(spec):
    """Expand a realisation list like ``'1-3,7'`` to ``[1, 2, 3, 7]``."""
    out = []
    for part in str(spec).split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def swap_alert(mem, cfg):
    """Swap alert text from a job's node_mem events (oldest first), or None. Swap merely in use is not an alert: a
    2048^3 run on pawna keeps ~120 GB of idle pages in swap with no swap traffic and runs normally (24 Nov 2026). Alert
    when swap is above swap_alert_gb and pages moved in or out of swap at swap_alert_mb_s or more (average between the
    last two samples; needs the pswpin/pswpout counters of job_sim.pbs from 24 Nov 2026 on), or when less than
    swap_alert_avail_gb of memory is available (20-22 Nov 2026: hangs with 2-8 GB available)."""
    e = mem[-1]
    swap = float(e.get('swap_used_mb') or 0) / 1024
    avail = float(e.get('avail_mb') or 0) / 1024
    when = e.get('time', '')[:16]
    if 'avail_mb' in e and avail < float(cfg.get('swap_alert_avail_gb', 2)):
        return ('%.1f GB memory available, %.0f GB swap in use (%s): the simulation and its drain may hang'
                % (avail, swap, when))
    if swap <= float(cfg.get('swap_alert_gb', 16)) or len(mem) < 2:
        return None
    p = mem[-2]
    if p.get('host') != e.get('host') or any(k not in x for x in (p, e) for k in ('pswpin', 'pswpout')):
        return None                     # job_sim.pbs without the counters (jobs started before 24 Nov 2026)
    dt = float(e.get('t', 0)) - float(p.get('t', 0))
    pages = (int(e['pswpin']) - int(p['pswpin'])) + (int(e['pswpout']) - int(p['pswpout']))
    if dt <= 0 or pages < 0:
        return None
    rate = pages * 4096 / dt / 1e6
    if rate < float(cfg.get('swap_alert_mb_s', 1.0)):
        return None
    return ('swapping: %.1f MB/s in+out of swap over the last %.0f min, %.0f GB swap in use, %.1f GB memory available '
            '(%s): the simulation and its drain may hang' % (rate, dt / 60, swap, avail, when))


def read_plan(path):
    """planned_runs.txt -> [dict(site, queue, cosmo, N, L, real, priority, opts, line, lineno, name)]; malformed lines
    come back with 'error'."""
    out = []
    if not os.path.isfile(path):
        return out
    for i, raw in enumerate(open(path), 1):
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        try:
            f = shlex.split(line)
            if len(f) < 7:
                raise ValueError('needs 7 columns: site queue cosmology N L realisations priority [options]')
            site, queue, cosmo, N, L, reals, prio = f[:7]
            opts = {}
            for o in f[7:]:
                k, _, v = o.partition('=')
                opts[k] = v if _ else True
            for r in reals_of(reals):
                stem = 'Sahyadri_L%sN%d/%s%d' % (fmt_l(L), int(N), cosmo, int(N))
                out.append(dict(site=site, queue=queue, cosmo=cosmo, N=int(N), L=float(L), real=r, priority=int(prio),
                                opts=opts, line=line, lineno=i, stem=stem,
                                name=('' if site == LOCAL else site + ':') + '%s/r%d' % (stem, r)))
        except ValueError as e:
            out.append(dict(line=line, lineno=i, error=str(e)))
    return out


TEMPLATES = os.path.join(REPO, 'scripts', 'pkdgrav3_pipeline', 'production')
QSTATES = ('open', 'drain', 'closed')
DEFAULT_QUEUE = {LOCAL: 'workq', 'pegasus': 'project', 'swarm': 'workq', 'pride': 'workq', 'leap': 'workq'}


def read_queues(path):
    """queues.txt -> ({(site, queue): dict(state, max_running, max_waiting, lineno)}, [errors]). Columns:
    site queue state [max_running|-] [max_waiting|-] [# comment]. A queue not listed is open, without limits of its own."""
    out, errs = {}, []
    if not path or not os.path.isfile(path):
        return out, errs
    for i, raw in enumerate(open(path), 1):
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        f = line.split()
        try:
            if len(f) < 3 or f[2] not in QSTATES:
                raise ValueError('needs: site queue open|drain|closed [max_running] [max_waiting]')
            lim = [None if x == '-' else int(x) for x in (f[3:5] + ['-', '-'])[:2]]
            out[(f[0], f[1])] = dict(state=f[2], max_running=lim[0], max_waiting=lim[1], lineno=i,
                                     note=raw.split('#', 1)[1].strip() if '#' in raw else '')
        except ValueError as e:
            errs.append('line %d: %s (%s)' % (i, e, line))
    return out, errs


def set_queue_state(path, site, q, state):
    """Rewrite the state column of site/queue in queues.txt (appends a line when the queue is not listed)."""
    if state not in QSTATES:
        raise ValueError('state must be one of %s' % ', '.join(QSTATES))
    lines = open(path).read().splitlines(True) if os.path.isfile(path) else []
    done = False
    for i, raw in enumerate(lines):
        body, hsh, com = raw.rstrip('\n').partition('#')
        f = body.split()
        if len(f) >= 3 and f[0] == site and f[1] == q:
            f[2] = state
            lines[i] = '  '.join('%-8s' % x for x in f) + ('  #' + com if hsh else '') + '\n'
            done = True
    if not done:
        lines.append('%-8s  %-8s  %-8s\n' % (site, q, state))
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        fh.writelines(lines)
    os.replace(tmp, path)


class MaintWait(Exception):
    """A submission refused because of a maintenance window of the site (the message says which and why)."""


class NodeWait(MaintWait):
    """A submission held back: bad nodes are usable in PBS and no healthy node has room for the job now."""


def parse_when(text):
    """``YYYY-MM-DDTHH:MM`` (or with a space or ``_`` instead of ``T``; local time) -> epoch seconds."""
    return time.mktime(time.strptime(re.sub(r'[ _]', 'T', text.strip())[:16], '%Y-%m-%dT%H:%M'))


def read_maint(path):
    """maintenance.txt -> ({site: [dict(start, end, note, lineno)] sorted by start}, [errors]).

    Columns: ``site start end [# note]``, times ``YYYY-MM-DDTHH:MM`` in local time (IST on pawna); ``end`` ``-``
    when not known: the window then lasts until the line is removed or given an end. Lines stay after a window
    has passed (they are ignored then, except that jobs killed during it are recognised as such)."""
    out, errs = {}, []
    if not path or not os.path.isfile(path):
        return out, errs
    for i, raw in enumerate(open(path), 1):
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        f = line.split()
        try:
            if len(f) != 3:
                raise ValueError('needs: site start end (YYYY-MM-DDTHH:MM, end - if not known)')
            t0 = parse_when(f[1])
            t1 = None if f[2] == '-' else parse_when(f[2])
            if t1 is not None and t1 <= t0:
                raise ValueError('end before start')
            out.setdefault(f[0], []).append(dict(start=t0, end=t1, note=raw.split('#', 1)[1].strip() if '#' in raw else '',
                                                 lineno=i))
        except ValueError as e:
            errs.append('line %d: %s (%s)' % (i, e, line))
    for v in out.values():
        v.sort(key=lambda w: w['start'])
    return out, errs


def set_maint(path, site, start=None, end=None, note=''):
    """Add a window ``site start end`` to maintenance.txt, or (``start`` None) remove all lines of ``site``."""
    lines = open(path).read().splitlines(True) if os.path.isfile(path) else [
        '# Maintenance windows of the sites, read by the run manager before every scheduling pass\n',
        '# (manager.py maint SITE START END [note] | manager.py maint SITE clear). Local time, YYYY-MM-DDTHH:MM;\n',
        '# end - if not known (the window lasts until the line is removed or given an end).\n',
        '#\n', '# site     start              end                # note\n']
    if start is None:
        lines = [l for l in lines if l.split('#', 1)[0].split()[:1] != [site]]
    else:
        t0 = parse_when(start)
        if end not in (None, '-'):
            if parse_when(end) <= t0:
                raise ValueError('end before start')
        lines.append('%-8s  %-17s  %-17s%s\n' % (site, re.sub(r'[ _]', 'T', start.strip())[:16],
                                                 '-' if end in (None, '-') else re.sub(r'[ _]', 'T', end.strip())[:16],
                                                 ('  # ' + note) if note else ''))
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        fh.writelines(lines)
    os.replace(tmp, path)


def hms(sec):
    """Seconds -> a PBS walltime ``HH:MM:SS`` (rounded down to whole minutes)."""
    m = max(0, int(sec) // 60)
    return '%02d:%02d:00' % (m // 60, m % 60)


def run_queue(run):
    """The PBS queue of a run: ``PBS_QUEUE`` from its run.conf, else the site's default queue."""
    return run.E.get('PBS_QUEUE') or DEFAULT_QUEUE.get(run.site, 'workq')


def site_cfg(cfg, site):
    """Limits of a site: the top-level ones for this host (pawna), sites.<name> for the others."""
    if site == LOCAL:
        return cfg
    return cfg['sites'][site]


def mirror_conf(cfg, site, remote_conf):
    """Fetch a remote run.conf and write its mirror config here; returns the mirror config's path."""
    S = cfg['sites'][site]
    text = ssh(S['login'], 'cat %s' % shlex.quote(remote_conf), check=True).stdout
    tmp = os.path.join(HOME, 'tmp_remote.conf')
    with open(tmp, 'w') as f:
        f.write(text)
    R = retention.load_conf(tmp)
    os.remove(tmp)
    stem, real = R['STEM'], R['REAL']
    mroot = S['mirror']
    path = os.path.join(mroot, 'confs', '%s_r%s.conf' % (stem.replace('/', '__'), real))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    over = '\n'.join([
        '', '# ---- mirror on %s of a run computed on %s (written by manager.py %s); the lines above are %s:%s' % (
            socket.gethostname(), site, iso(), site, remote_conf),
        'SITE=%s' % site, 'REMOTE_CONF=%s' % remote_conf, 'REMOTE_SCRATCH=%s' % R['SCRATCH'],
        'REMOTE_SANDBOX=%s' % R['SANDBOX'], 'REMOTE_CODES=%s' % R['CODES'], 'REMOTE_LOGS=%s' % R.get('LOGS', ''),
        'PULL_HOST=%s' % S['data'],
        'SCRATCH=%s' % mroot, 'LOGS=%s/sims/%s/r%s/logs' % (mroot, stem, real),
        'SANDBOX=%s' % REPO, 'CODES=%s' % cfg.get('local_codes', ''), ''])
    with open(path + '.tmp', 'w') as f:
        f.write(text.rstrip('\n') + '\n' + over)
    os.replace(path + '.tmp', path)
    return path


def refresh_mirror_conf(path, cfg):
    """The local part of a mirror config (after the '# ---- mirror' line): SANDBOX and CODES of this manager's install,
    so archive pulls run the code the manager runs (configs written by an older install pointed elsewhere)."""
    text = open(path).read()
    i = text.find('\n# ---- mirror')
    if i < 0:
        return
    head, tail = text[:i], text[i:]
    new = re.sub(r'(?m)^SANDBOX=.*$', 'SANDBOX=%s' % REPO, tail)
    new = re.sub(r'(?m)^CODES=.*$', 'CODES=%s' % cfg.get('local_codes', ''), new)
    if new != tail:
        with open(path + '.tmp', 'w') as f:
            f.write(head + new)
        os.replace(path + '.tmp', path)


def sync_site(cfg, site, runs, timeout=1800):
    """rsync the small files of the site's runs into its mirror (from the data node). rc 23/24 (some files absent,
    e.g. before the simulation starts) is normal."""
    S = cfg['sites'][site]
    if not runs:
        return 0
    srcs = []
    for run in runs:
        rs = run.E['REMOTE_SCRATCH'].rstrip('/')
        st, r = run.E['STEM'], run.E['REAL']
        for rel in ('state/%s/r%s' % (st, r), 'sims/%s/r%s/output_steps.txt' % (st, r), 'sims/%s/r%s/snapnames.txt' % (st, r),
                    'sims/%s/r%s/run.*.log' % (st, r), 'sims/%s/r%s/drain.*.log' % (st, r), 'sims/%s/r%s/logs' % (st, r), 'halos/%s/r%s/pk' % (st, r)):
            srcs.append('%s/./%s' % (rs, rel))
    os.makedirs(S['mirror'], exist_ok=True)
    cmd = (['rsync', '-rtR', '--partial', '-e', 'ssh ' + ' '.join(SSH_OPTS + ['-o', 'ControlPath=%s/ssh-%%r@%%h' % HOME])]
           + ['%s:%s' % (S['data'], srcs[0])] + [':%s' % x for x in srcs[1:]] + [S['mirror'].rstrip('/') + '/'])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode not in (0, 23, 24):
        log('sync %s: rsync rc %d: %s' % (site, r.returncode, r.stderr.strip()[-300:]))
    return r.returncode


def save_cfg(c):
    """Write the manager configuration to ``config.json`` atomically (temporary file + rename)."""
    os.makedirs(HOME, mode=0o700, exist_ok=True)
    tmp = CONFIG + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(c, f, indent=1)
    os.replace(tmp, CONFIG)


def atomic_json(path, obj):
    """Write ``obj`` as JSON to ``path`` atomically (temporary file + ``os.replace``)."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def jtime(j):
    """When a job was submitted (the manager's clock), else started: orders PBS jobs and head-node processes
    (L<epoch>) alike, which job numbers cannot."""
    return j.get('submitted') or j.get('start') or 0


def jobnum(j):
    """PBS job number; local processes (archive pulls) are 'L<epoch>'."""
    try:
        return int(str(j).split('.')[0].lstrip('L'))
    except ValueError:
        return -1


def short(job):
    """PBS job id without the server suffix: ``'2539127.pawna'`` -> ``'2539127'``."""
    return str(job).split('.')[0]


# ====================================================================================================== one run
class Run:
    """One simulation realisation known to the manager, described by its run.conf.

    Holds the configuration (``E``), the derived paths (``P``, from ``retention.paths``), the manager's own state
    file ``ms`` (paused, scheduled, failed, trash, acknowledged jobs, action history), the events written by the
    pipeline jobs (``state/events/*.jsonl``) and caches of the simulation progress. ``derive`` turns events and the
    PBS queue into the run's current status; ``submit`` starts pipeline stages. Remote runs (another site) keep a
    local mirror of their state directory and are submitted over ssh.
    """
    CFG = None          # the manager's configuration (sites), set by Manager.load_runs
    GATE = None         # Manager.maint_gate, set by the manager: every submission passes it
    PLACE = None        # Manager.place: the node for a job when some nodes are bad

    def __init__(self, conf, paused=False):
        """Load the run from ``conf`` (absolute path); ``paused`` marks a newly added run paused."""
        self.conf = os.path.abspath(conf)
        self.ev = []
        self.offsets = {}
        self.samples = []          # (t, step) of the simulation, for its rate
        self.simcache = (0, None)
        self.scratch_bytes = None
        self.err = None
        self.busy = None
        self.seen_ready = {}       # snapshot number -> when the manager first saw it ready (its own clock)
        self.load()
        if paused and not os.path.isfile(self.msfile):
            self.ms['paused'] = True
            self.save()

    def load(self):
        # the manager keeps a copy of every run.conf: a purged run's own copy is gone with its scratch directory
        """(Re)read run.conf and derive paths, output list and the manager state file.

        A copy of run.conf is kept in ``HOME/confs`` so that a run whose scratch directory was purged can still be
        shown; if neither the file nor the copy exists, FileNotFoundError is raised.
        """
        saved = os.path.join(HOME, 'confs', self.conf.strip('/').replace('/', '__'))
        if os.path.isfile(self.conf):
            os.makedirs(os.path.dirname(saved), exist_ok=True)
            shutil.copyfile(self.conf, saved)
            self.E = retention.load_conf(self.conf)
        elif os.path.isfile(saved):
            self.E = retention.load_conf(saved)
            self.E['CONF'] = self.conf
        else:
            raise FileNotFoundError('%s (and no saved copy)' % self.conf)
        self.P = retention.paths(self.E)
        if not os.path.isdir(self.E['SCRATCH']):
            # an unmounted file system looks like a missing directory: do not create anything there or start afresh
            raise FileNotFoundError('SCRATCH %s does not exist (file system not mounted?)' % self.E['SCRATCH'])
        self.site = self.E.get('SITE') or LOCAL
        self.remote = self.site != LOCAL and bool(self.E.get('REMOTE_CONF'))
        # runs of another site are named <site>:<stem>/r<real> (the same run name may exist on both sites)
        self.name = '%s%s/r%s' % (self.site + ':' if self.remote else '', self.E['STEM'], self.E['REAL'])
        self.outs = retention.planned_outputs(self.E, self.P)
        self.step_index = {o[1]: o[0] for o in self.outs}
        os.makedirs(os.path.join(self.P['state'], 'events'), exist_ok=True)
        self.msfile = os.path.join(self.P['state'], 'manager.json')
        self.ms = dict(submitted=[], failed=None, paused=False, trash=None, purged=None, actions=[])
        if os.path.isfile(self.msfile):
            self.ms.update(json.load(open(self.msfile)))
        try:
            self.plan = retention.plan(self.E)
            self.err = None
        except ValueError as e:
            # after "move to trash"/purge the run's output_steps.txt is gone with its directory: not an error then
            self.plan = None
            self.err = None if (self.ms.get('trash') or self.ms.get('purged')) else str(e)

    def save(self):
        """Write the manager state of this run (``ms``) to its state file atomically."""
        atomic_json(self.msfile, self.ms)

    def note(self, msg):
        """Log ``msg`` for this run and append it to its action history (last 200 kept)."""
        log('%s: %s' % (self.name, msg))
        self.ms['actions'] = (self.ms.get('actions', []) + [[now(), msg]])[-200:]
        self.save()

    # ------------------------------------------------------------------ events
    def read_events(self):
        """Read new complete lines from the run's event files (``state/events/*.jsonl``).

        Each file is read from the offset reached last time; an incomplete last line is left for the next call.
        Returns the number of new events appended to ``self.ev``.
        """
        new = 0
        for f in sorted(glob.glob(os.path.join(self.P['state'], 'events', '*.jsonl'))):
            off = self.offsets.get(f, 0)
            try:
                size = os.path.getsize(f)
            except OSError:
                continue
            if size <= off:
                continue
            with open(f, 'rb') as fh:
                fh.seek(off)
                data = fh.read()
            cut = data.rfind(b'\n') + 1      # only complete lines
            for line in data[:cut].splitlines():
                try:
                    e = json.loads(line)
                except ValueError:   # numbers with leading zeros (step 00124) from jobs before pipeline_env fix
                    try:
                        e = json.loads(re.sub(rb'(:\s*)(-?0\d+)(\s*[,}])', rb'\1"\2"\3', line))
                    except ValueError:
                        continue
                for k in ('step',):
                    if k in e:
                        e[k] = str(e[k])
                self.ev.append(e)
                new += 1
                if e.get('ev') == 'snap_ready' and e.get('step') in self.step_index:
                    self.seen_ready.setdefault(self.step_index[e['step']], now())
            self.offsets[f] = off + cut
        if new:
            self.ev.sort(key=lambda e: e.get('t', 0))
        return new

    # ------------------------------------------------------------------ derived state
    def jobs(self, qjobs, qtime):
        """job id -> dict(stage, status, rc, first, last, host, start, end, submitted)."""
        J = {}
        for s in self.ms['submitted']:
            J[s['job']] = dict(job=s['job'], stage=s['stage'], first=s.get('first'), last=s.get('last'),
                               submitted=s['t'], status='submitted')
        for e in self.ev:
            j = J.setdefault(e['job'], dict(job=e['job'], stage=e['stage'], status='?'))
            if e['ev'] == 'start':
                j['start'] = e['t']
                j['host'] = e.get('host')
                j['log'] = e.get('log')
                if 'first' in e:
                    j['first'], j['last'] = e['first'], e['last']
            elif e['ev'] == 'end':
                j['end'] = e['t']
                j['rc'] = e['rc']
            elif e['ev'] == 'rockstar_done':
                j['first'], j['last'], j['ok'] = e['first'], e['last'], int(e.get('ok', 0))
        pids = {s['job']: s['pid'] for s in self.ms['submitted'] if s.get('pid')}
        for jid, j in J.items():
            q = qjobs.get(jid)
            if jid in pids and 'end' not in j:     # a process on this host (archive pull of a remote run)
                try:
                    os.kill(pids[jid], 0)
                    j['status'] = 'running'
                except OSError:
                    j['status'] = 'gone'
                continue
            if 'end' in j:
                j['status'] = 'done' if j['rc'] == 0 else 'failed'
            elif q:
                j['status'] = {'Q': 'queued', 'H': 'held', 'R': 'running', 'E': 'exiting'}.get(q['state'], q['state'])
                j['node'] = q.get('node')
                j.pop('gone', None)
            elif j.get('submitted', 0) > qtime or 'start' in j and j['start'] > qtime:
                j['status'] = 'queued' if 'start' not in j else 'running'   # newer than the last qstat
            else:
                j['status'] = 'gone'
        return J

    def derive(self, qjobs, qtime):
        """Status of the run from its events and the PBS queue, as a dict for the scheduler and the page.

        ``qjobs``/``qtime``: the site's qstat snapshot. Keys include ``jobs`` (all jobs with stage and status),
        ``active`` (queued or running jobs), ``ready`` (outputs whose snapshot is complete), ``rs_done``/``rs_redo``
        (Rockstar), ``archb``/``archive``, ``fits``, ``parquet``, ``verify``, ``sim_done``, ``sim_clean_stop``,
        ``sim_info`` (step, redshift, rate) and ``rewinds``. Outputs rewritten by a restart from an earlier
        checkpoint (``rewind_cuts``) only count events after that restart.
        """
        D = dict(name=self.name, conf=self.conf, n_out=len(self.outs), err=self.err)
        J = self.jobs(qjobs, qtime)
        D['jobs'] = sorted(J.values(), key=lambda j: jobnum(j['job']))
        # Restarts from a checkpoint (rewinds): a simulation job restarted from the checkpoint of step K rewrites the
        # outputs after K. Whatever was announced, run through Rockstar or archived for those outputs before the
        # restart belongs to a run that no longer exists (a job killed after its last checkpoint). From the job's
        # sim_rewind event, or its sim_info restart path (jobs before sim_rewind existed).
        cut = self.rewind_cuts(J)
        D['rewinds'] = cut['rewinds']
        tcut = cut['tcut']                       # output index -> time of the newest restart that rewrites it (0: none)
        ready, since = set(), {}
        for e in self.ev:
            if e['ev'] == 'snap_ready' and e.get('step') in self.step_index:
                i = self.step_index[e['step']]
                if e['t'] < tcut.get(i, 0):
                    continue                     # announced before a restart that rewrites it
                k = cut['job_step'].get(e['job'])
                if e.get('earlier') and k is not None and int(e['step'], 10) > k:
                    continue                     # restart announcement of an output after the checkpoint (old jobs)
                ready.add(i)
                since[i] = min(since.get(i, e['t']), e['t'])
        D['ready'] = sorted(ready)
        D['ready_since'] = since
        D['sim_done'] = any(e['ev'] == 'sim_outputs' and int(e.get('ndone', -1)) == int(e.get('nout', -2))
                            for e in self.ev)
        cov, D['rs_redo'] = set(), []
        for e in self.ev:
            if e['ev'] == 'rockstar_done' and int(e.get('ok', 0)) == 1:
                t0 = J.get(e['job'], {}).get('start', e['t'])
                for i in range(int(e['first']), int(e['last']) + 1):
                    # an output that a restart rewrote counts only if Rockstar ran after its new version was ready
                    # (10 min for clocks of different nodes)
                    if tcut.get(i, 0) and not (i in since and t0 >= since[i] - 600):
                        D['rs_redo'].append(i)
                        continue
                    cov.add(i)
        D['rs_done'] = sorted(cov)
        D['rs_redo'] = sorted(set(D['rs_redo']) - cov)
        last = {}
        for j in D['jobs']:
            last[j['stage']] = j
        D['last'] = last
        D['active'] = [j for j in D['jobs'] if j['status'] in ('submitted', 'queued', 'held', 'running', 'exiting')]

        def ok_after(evname):   # the event of the newest job
            """The newest event called ``evname`` (by job number), or None."""
            es = sorted((e for e in self.ev if e['ev'] == evname), key=lambda e: jobnum(e['job']))
            return es[-1] if es else None
        # sim jobs that stopped cleanly before the last output (pkdgrav3 iWallRunTime / STOP: rc 0, outputs missing)
        D['sim_clean_stop'] = {e['job'] for e in self.ev if e['ev'] == 'sim_outputs' and int(e.get('rc', 1)) == 0
                               and int(e.get('ndone', 0)) < int(e.get('nout', 0))}
        D['verify'] = ok_after('verify_done')
        D['archive'] = ok_after('archive_done')
        # particles archived in batches while the run is computed: the newest snapshot of a passed batch
        # a batch done before a restart that rewrote some of its outputs counts up to the output before the first of them
        # (the next batch copies them again: archive_run.py skips only files of the same size and time)
        ab = [-1]
        for e in self.ev:
            if e['ev'] == 'archive_batch_done' and int(e.get('ok', 0)) == 1:
                u = int(e['upto'])
                bad = [i for i in range(u + 1) if tcut.get(i, 0) > e['t']]
                ab.append(min(bad) - 1 if bad else u)
        D['archb'] = max(ab)
        D['kept'] = ok_after('kept_sums')
        D['fits'] = ok_after('fits_done')
        D['parquet'] = ok_after('parquet_done')
        D['sim_info'] = self.sim_progress()
        return D

    def rewind_cuts(self, J):
        """Restarts of the simulation: rewinds [(t, step K, job)], job -> K, and per output index the time of the
        newest restart from a checkpoint before that output (tcut)."""
        rew, job_step = [], {}
        for e in self.ev:
            k = None
            if e['ev'] == 'sim_rewind' and str(e.get('step', '')).isdigit():
                k = int(str(e['step']), 10)
            elif e['ev'] == 'sim_info' and e.get('restart'):
                m = re.search(r'\.(\d+)\.chk$', str(e['restart']))
                k = int(m.group(1), 10) if m else None
            if k is None:
                continue
            job_step[e['job']] = k
            t = J.get(e['job'], {}).get('start', e['t'])
            if not any(r[2] == e['job'] for r in rew):
                rew.append((t, k, e['job']))
        tcut = {}
        for i, st in ((o[0], o[1]) for o in self.outs):
            for t, k, _ in rew:
                if int(st, 10) > k and t > tcut.get(i, 0):
                    tcut[i] = t
        return dict(rewinds=rew, job_step=job_step, tcut=tcut)

    def sim_progress(self):
        """Step, redshift and step rate of the running simulation from the tail of its pkdgrav3 log.

        Reads the last 64 kB of the newest ``run.<job>.log``; cached for 60 s (and frozen while the run is marked
        stalled). Returns a dict or None if no step was logged yet.
        """
        t, v = self.simcache
        if now() - t < 60 or getattr(self, 'stalled', None):
            return v
        v = None
        # the newest pkdgrav3 log: run.<job>.log while the job runs (run.log links to it only at the end in older jobs)
        logs = [x for x in glob.glob(os.path.join(self.P['sim'], 'run.*.log')) if re.search(r'run\.\d+\.log$', x)]
        f = max(logs, key=os.path.getmtime) if logs else os.path.join(self.P['sim'], 'run.log')
        try:
            with open(f, 'rb') as fh:
                fh.seek(max(0, os.path.getsize(f) - 65536))
                tail = fh.read().decode('utf-8', 'replace')
            m = re.findall(r'Step:([0-9.]+).*?Redshift:([0-9.eE+-]+)', tail)
            if m:
                step, z = float(m[-1][0]), float(m[-1][1])
                nsteps = int(self.outs[-1][1]) if self.outs else None
                if not self.samples or self.samples[-1][1] != step:
                    self.samples = (self.samples + [(now(), step)])[-50:]
                rate = None
                if len(self.samples) >= 2 and self.samples[-1][1] > self.samples[0][1]:
                    rate = (self.samples[-1][1] - self.samples[0][1]) / (self.samples[-1][0] - self.samples[0][0])
                v = dict(step=step, z=z, nsteps=nsteps, rate=rate,
                         eta=(now() + (nsteps - step) / rate) if rate and nsteps else None, log_mtime=os.path.getmtime(f))
        except (OSError, ValueError, IndexError):
            pass
        self.simcache = (now(), v)
        return v

    def phase(self, D):
        """Phase of the run for the page, first match wins: purged, waiting to restart / scheduled, in trash, failed,
        archived, verified (no archive), post (post-processing), not started, rockstar, simulation. From the manager
        state ``ms`` and the derived status ``D``.
        """
        if self.ms.get('purged'):
            return 'purged'
        if self.ms.get('scheduled') and not self.ms.get('failed'):
            return 'waiting to restart' if self.ms['scheduled'].get('restart') else 'scheduled'
        if self.ms.get('trash'):
            return 'in trash'
        if self.ms.get('failed'):
            return 'failed'
        a = D['archive']
        if a and int(a.get('ok', 0)):
            return 'archived'
        v = D['verify']
        if (not self.E.get('ARCHIVE') and v and int(v.get('ok', 0)) and D['sim_done'] and not D['active']
                and len(D['rs_done']) == len(self.outs)):
            return 'verified (no archive)'
        for st in ('archive', 'verify', 'vprof', 'xcorr', 'parquet', 'fits', 'vpot', 'vahc', 'ctrees'):
            j = D['last'].get(st)
            if j and j['status'] in ('submitted', 'queued', 'held', 'running', 'exiting'):
                return st
        if D['rs_done'] and len(D['rs_done']) == len(self.outs):
            return 'post'
        sim = D['last'].get('sim')
        if sim is None:
            return 'not started'
        if D['sim_done']:
            return 'rockstar'
        return 'simulation'

    # ------------------------------------------------------------------ submission
    def submit(self, stages, extra=(), what=None, cfg=None):
        # archive as a process on this (head) node: always for a remote run (a pull from its data node), and for a
        # local run unless archive_on_head is off (the head node writes /storage at ~430 MB/s against ~25-30 MB/s from
        # a compute node, and no compute node is taken)
        """Submit pipeline ``stages`` (comma separated) of this run; returns the new job ids.

        Archive stages run as a copy process on the head node instead (``start_pull``). A remote run is submitted
        with ``submit_pipeline.sh`` on its site's login node over ssh after a budget check; a local run with
        ``submit_pipeline.sh`` here. ``extra`` are extra arguments of submit_pipeline.sh, ``what`` a description
        for the history.
        """
        if Run.GATE:      # maintenance windows: refused (MaintWait) or the wall time cut to the window's start
            cap = Run.GATE(self, stages, what)
            if cap:
                extra = list(extra) + ['--wall', cap]
                what = dict(what or {}, wall_cap=cap)
        if Run.PLACE and not (stages in ('archive', 'archb')) and '--host' not in extra:
            host = Run.PLACE(self, stages)      # a healthy node when bad ones are usable in PBS (NodeWait if none)
            if host:
                extra = list(extra) + ['--host', host]
                what = dict(what or {}, host=host)
        if stages in ('archive', 'archb') and (self.remote or (Run.CFG or {}).get('archive_on_head', True)):
            return self.start_pull(what, stage=stages)
        once = self.ms.get('queue_once') if 'sim' in stages.split(',') else None
        if once:            # the next simulation job only: another queue/project (Manager.action 'queue_once')
            extra = list(extra) + ['--queue', once['queue'], '--project', once.get('project') or '']
            what = dict(what or {}, queue=once['queue'], project=once.get('project') or '')
        if self.remote:     # qsub on the site's login node, with the run's own run.conf there
            BUDGETS.check(cfg or Run.CFG, self, stages, project=(once.get('project') or '') if once else None)
            sub = os.path.join(self.E['REMOTE_SANDBOX'], 'scripts', 'pkdgrav3_pipeline', 'submit_pipeline.sh')
            line = ' '.join(shlex.quote(x) for x in ['bash', sub, self.E['REMOTE_CONF'], '--stages', stages] + list(extra))
            r = ssh((cfg or Run.CFG)['sites'][self.site]['login'], 'PATH=/opt/pbs/bin:$PATH ' + line)
        else:
            sub = os.path.join(self.E['SANDBOX'], 'scripts', 'pkdgrav3_pipeline', 'submit_pipeline.sh')
            cmd = ['bash', sub, self.conf, '--stages', stages] + list(extra)
            env = dict(os.environ, PATH=PBS + ':' + os.environ.get('PATH', ''))
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
        ids = []
        for line in r.stdout.splitlines():
            m = re.match(r'^(\w+) (\d+)\.\S+ \(', line)
            if m:
                ids.append((m.group(1), m.group(2)))
        if r.returncode != 0 or not ids:
            for st, jid in ids:      # part of a chain went in before qsub refused (e.g. a queue's job limit)
                self.ms['submitted'].append(dict(job=jid, stage=st, t=now(), partial=True))
            if ids:
                self.save()
            raise RuntimeError('submit %s failed (rc %d): %s %s' % (stages, r.returncode, r.stdout[-500:], r.stderr[-500:]))
        for st, jid in ids:
            rec = dict(job=jid, stage=st, t=now())
            if what:
                rec.update(what)
            self.ms['submitted'].append(rec)
        if once:
            self.ms['queue_once'] = None
        self.save()
        self.note('submitted %s' % ', '.join('%s %s' % x for x in ids) + (' (%s)' % what if what else ''))
        if once:
            self.note('one-time queue %s%s used; later simulation jobs go to %s again'
                      % (once['queue'], ' -P %s' % once['project'] if once.get('project') else '', run_queue(self)))
        return ids

    def start_pull(self, what=None, stage='archive'):
        """Archive as a process on this host: job_archive.pbs run with bash (a remote run: archive_run.py pulls from
        PULL_HOST; a local run: copies SCRATCH -> ARCHIVE)."""
        jid = 'L%d' % int(now())
        pipe = os.path.join(REPO, 'scripts', 'pkdgrav3_pipeline')
        env = dict(os.environ, CONF=self.conf, PIPE_ENV=os.path.join(pipe, 'pipeline_env.sh'), STAGE=stage,
                   PBS_JOBID=jid, PATH=PBS + ':' + os.environ.get('PATH', ''))
        if stage == 'archb':     # batch: the kept particle files of snapshots 0..upto (job_archive.pbs ARCHIVE_UPTO)
            env['ARCHIVE_UPTO'] = str(int((what or {})['upto']))
        p = subprocess.Popen(['bash', os.path.join(pipe, 'job_archive.pbs')], env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        rec = dict(job=jid, stage=stage, t=now(), pid=p.pid)
        if what:
            rec.update(what)
        self.ms['submitted'].append(rec)
        self.save()
        self.note('archive%s %s %s started on %s (pid %d)%s' % (
            (' batch (particles of snapshots 0-%s)' % what['upto']) if stage == 'archb' else '',
            'pull' if self.remote else 'copy', jid, socket.gethostname(), p.pid,
            (' from %s' % self.E['PULL_HOST']) if self.remote else ''))
        threading.Thread(target=p.wait, daemon=True).start()     # reap it
        return [(stage, jid)]


# ====================================================================================================== manager
class Manager:
    """The run manager: one process that watches all runs, submits their pipeline stages and serves the page.

    The main loop (:meth:`serve`) handles file events (inotify on the runs' state directories), actions from the
    page or the CLI (over a unix socket), and a heartbeat (``heartbeat_s``) that refreshes qstat, the watchdog
    (:meth:`watch`), the scheduler (:meth:`schedule_site`) and the state snapshot shown by the page. The plan
    file is synchronised in its own thread (:meth:`plan_loop`), remote sites in another (``sync_loop``).
    """
    def __init__(self):
        """Create ``HOME`` (mode 700), load the configuration, copy the plan and queue templates if missing,
        load the runs and set up queues, locks and caches.
        """
        os.makedirs(HOME, mode=0o700, exist_ok=True)
        os.chmod(HOME, 0o700)
        self.cfg = load_cfg()
        for k, t in (('plan_file', 'planned_runs.txt'), ('queues_file', 'queues.txt')):
            p = self.cfg.get(k)
            if p and not os.path.exists(p) and os.path.dirname(p) == HOME:
                shutil.copy(os.path.join(TEMPLATES, t), p)
                log('%s: copied the template %s' % (p, t))
        self.queues, self.queue_errors, self.queues_mtime = {}, [], None
        self.maint_windows, self.maint_errors, self.maint_mtime = {}, [], None
        self.nodes = {}            # this site's compute nodes: host -> newest node_check.sh result (+ pbs state)
        self.node_flags = {}       # host -> dict(t, reasons, job, run): a job's node check failed there (exit 75)
        self.pbsn, self.pbsn_t = {}, 0.0     # pbsnodes: host -> dict(state, ncpus_free, mem_free_gb)
        self.probe_ev = threading.Event()
        self.budget_ev = threading.Event()
        self.budget_next = 0.0
        self.probe_next = 0.0
        self.deleting = {}         # (host, dir) -> time a deletion of a leftover directory started
        self.q = queue.Queue()
        self.runs = {}
        self.qjobs, self.qtime, self.qok = {}, 0.0, None
        self.qsite = {}            # other sites: name -> (jobs, qtime, ok)
        self.site_disk = {}        # other sites: SCRATCH -> dict(total, used, free, ...)
        self.waiting = {}
        self.plan_view = []        # planned runs and their state (page)
        self.plan_errors = {}      # (lineno, line, real) -> message of the last failed set-up (not retried until the line changes)
        self.next_hb = 0.0
        self.recheck_at = None
        self.watch_proc = None
        self.watch_dirs = None
        self.snapshot = {}
        self.disk = {}
        self.lock = threading.Lock()
        self.load_runs()

    # ------------------------------------------------------------------ runs
    def load_runs(self):
        """(Re)load the runs listed in the configuration.

        Runs no longer listed are dropped; a new run whose storage does not answer (``PROBE``) is left in
        ``self.unloaded`` and tried again at a later heartbeat.
        """
        self.cfg = load_cfg()
        Run.CFG = self.cfg
        Run.GATE = self.maint_gate
        Run.PLACE = self.place
        want = {r['conf']: r for r in self.cfg['runs']}
        for conf in list(self.runs):
            if conf not in want:
                del self.runs[conf]
        self.unloaded = []
        for conf, r in want.items():
            if conf not in self.runs:
                if not PROBE.ok('conf ' + os.path.dirname(conf), archive_test(os.path.dirname(conf)),
                                int(self.cfg.get('storage_timeout_s', 30))):
                    self.unloaded.append(conf)      # loaded at a later heartbeat, once its storage answers
                    log('run %s not loaded: its storage is not responding; trying again later' % conf)
                    continue
                try:
                    if os.path.dirname(os.path.dirname(conf)).endswith(tuple(S['mirror'] for S in self.cfg['sites'].values())):
                        refresh_mirror_conf(conf, self.cfg)
                    run = Run(conf, paused=r.get('paused', False))
                    self.runs[conf] = run
                    run.read_events()
                    log('run %s (%s): %d events' % (run.name, conf, len(run.ev)))
                except Exception as e:  # noqa: BLE001
                    self.unloaded.append(conf)      # tried again at every heartbeat
                    log('cannot load %s: %s' % (conf, e))
        self.restart_watch()

    def restart_watch(self):
        """(Re)start ``inotifywait`` on ``HOME`` and every run's state directory when that set changed."""
        dirs = sorted([HOME] + [r.P['state'] for r in self.runs.values() if not getattr(r, 'stalled', None)])
        if self.watch_proc and self.watch_proc.poll() is None and dirs == self.watch_dirs:
            return
        if self.watch_proc and self.watch_proc.poll() is None:
            self.watch_proc.terminate()
        self.watch_dirs = dirs
        self.watch_proc = subprocess.Popen(
            ['inotifywait', '-m', '-r', '-q', '-e', 'close_write', '-e', 'moved_to', '-e', 'create', '--format', '%w%f'] + dirs,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        threading.Thread(target=self.watch_reader, args=(self.watch_proc,), daemon=True).start()
        log('watching %d directories' % len(dirs))

    def watch_reader(self, proc):
        """Thread: forward each changed path from inotifywait to the main queue; report when it ends."""
        for line in proc.stdout:
            self.q.put(('file', line.strip()))
        self.q.put(('watch_ended', proc))

    def derive(self, run):
        """``run.derive`` with the qstat snapshot of the run's own site."""
        if run.remote:
            jobs, qt, _ok = self.qsite.get(run.site, ({}, 0.0, None))
            return run.derive(jobs, qt)
        return run.derive(self.qjobs, self.qtime)

    def sites(self):
        """Names of the remote sites with at least one run."""
        return sorted({r.site for r in self.runs.values() if r.remote})

    def refresh_qstat_site(self, site):
        """qstat of our jobs on another site's login node."""
        S = self.cfg['sites'][site]
        r = ssh(S['login'], 'ids=$(/opt/pbs/bin/qselect -u $USER); if [ -n "$ids" ]; then /opt/pbs/bin/qstat -f -F json $ids; '
                            'else echo "{}"; fi', timeout=180)
        try:
            data = json.loads(r.stdout.replace('\\\'', "'")) if r.stdout.strip() else {}
        except ValueError as e:
            log('qstat %s: %s (%s)' % (site, e, r.stderr.strip()[:200]))
            self.qsite[site] = self.qsite.get(site, ({}, 0.0, None))[:2] + (False,)
            return
        if r.returncode != 0 and not data:
            log('qstat %s failed: %s' % (site, r.stderr.strip()[:200]))
            self.qsite[site] = self.qsite.get(site, ({}, 0.0, None))[:2] + (False,)
            return
        jobs = {}
        for jid, j in (data.get('Jobs') or {}).items():
            jobs[short(jid)] = dict(state=j.get('job_state'), name=j.get('Job_Name'),
                                    node=(j.get('exec_host') or '').split('/')[0] or None, queue_time=j.get('qtime'))
        self.qsite[site] = (jobs, now(), True)

    def refresh_site_disk(self, site):
        """Free space of the site's scratch and the bytes of each of its runs (scratch_ops.py usage over ssh)."""
        runs = [r for r in self.runs.values() if r.remote and r.site == site]
        if not runs:
            return
        S = self.cfg['sites'][site]
        cmd = '; '.join('%s %s usage %s 2>/dev/null || echo "{}"' % (
            shlex.quote(os.path.join(r.E['REMOTE_CODES'], 'venv', 'bin', 'python')),
            shlex.quote(os.path.join(r.E['REMOTE_SANDBOX'], 'scripts', 'pkdgrav3_pipeline', 'scratch_ops.py')),
            shlex.quote(r.E['REMOTE_CONF'])) for r in runs)
        out = ssh(S['login'], cmd, timeout=1800).stdout.splitlines()
        for run, line in zip(runs, out):
            try:
                u = json.loads(line)
            except ValueError:
                continue
            if 'bytes' in u:
                run.scratch_bytes = u['bytes']
                self.site_disk[run.E['REMOTE_SCRATCH']] = dict(total=u['total'], used=u['used'], free=u['free'],
                                                               kind='SCRATCH', site=site, paths=[run.E['REMOTE_SCRATCH']])

    def sync_loop(self):
        """Mirror of the remote runs' small files, every sync_s; the mirror's state dirs are watched (inotify)."""
        while True:
            for site in self.sites():
                runs = [r for r in self.runs.values() if r.remote and r.site == site and not r.ms.get('purged')]
                try:
                    sync_site(self.cfg, site, runs)
                except Exception as e:  # noqa: BLE001
                    log('sync %s: %s' % (site, e))
            self.q.put(('wake',))
            time.sleep(min([int(self.cfg['sites'][x].get('sync_s', 120)) for x in self.sites()] or [120]))

    def ask(self, action, timeout=900):
        """Run an action in the main loop (from another thread) and wait for it."""
        box = dict(ev=threading.Event())
        self.q.put(('action', action, box))
        if not box['ev'].wait(timeout):
            raise RuntimeError('timeout: %s' % action)
        if not box['ok']:
            raise RuntimeError(box['result'])
        return box['result']

    def plan_loop(self):
        """Thread: synchronise the plan file (``plan_sync``) every ``plan_s`` seconds; errors are logged."""
        while True:
            try:
                self.plan_sync()
            except Exception as e:  # noqa: BLE001
                log('plan: %s' % e)
                log(traceback.format_exc())
            time.sleep(int(self.cfg.get('plan_s', 300)))

    def plan_setup(self, P):
        """Set up the run of plan entry P on its site (setup_production.py), unless its run.conf exists; returns the conf
        path (on that site)."""
        o = P['opts']
        remote = P['site'] != LOCAL
        S = self.cfg['sites'][P['site']] if remote else self.cfg
        root = o.get('root') or S['scratch_root']
        conf = os.path.join(root, 'sims', P['stem'], 'r%d' % P['real'], 'run.conf')
        args = ['--site', P['site'], '--queue', P['queue'], '--cosmo', P['cosmo'], '--N', str(P['N']), '--L', '%g' % P['L'],
                '--real', str(P['real']), '--root', root]
        for k, flag in (('archive', '--archive'), ('keep_full_z', '--keep-full-z'), ('keep_pct', '--keep-pct'),
                        ('nodes', '--sim-nodes'), ('rs_batch', '--rs-batch'), ('rs_nodes', '--rs-nodes')):
            if k in o:
                args += [flag, str(o[k])]
        if str(o.get('gpu', '0')).lower() in ('1', 'true', 'yes'):
            args += ['--sim-gpu']          # pegasus GPU nodes (3 nodes by default from 2048^3)
        if remote:
            py, sb = S['python'], S['sandbox']
            setup = os.path.join(sb, 'scripts', 'pkdgrav3_pipeline', 'production', 'setup_production.py')
            cmd = '[ -f %s ] && echo EXISTS || %s' % (shlex.quote(conf), ' '.join(shlex.quote(x) for x in [py, setup] + args))
            r = ssh(S['login'], cmd, timeout=1800)
            out, rc = r.stdout + r.stderr, r.returncode
        else:
            if os.path.isfile(conf):
                return conf, 'exists'
            setup = os.path.join(REPO, 'scripts', 'pkdgrav3_pipeline', 'production', 'setup_production.py')
            r = subprocess.run([self.cfg['local_python'], setup] + args, capture_output=True, text=True, timeout=1800)
            out, rc = r.stdout + r.stderr, r.returncode
        if 'EXISTS' in out.split():
            return conf, 'exists'
        if rc != 0:
            raise RuntimeError('setup_production.py on %s failed (rc %d): %s' % (P['site'], rc, out.strip()[-400:]))
        return conf, 'set up'

    def plan_sync(self):
        """Set up, register and schedule what the plan file lists and the manager does not know yet."""
        path = self.cfg.get('plan_file')
        entries = read_plan(path) if path else []
        view = []
        for P in entries:
            if 'error' in P:
                view.append(dict(line=P['line'], lineno=P['lineno'], state='error', msg=P['error']))
                continue
            key = (P['lineno'], P['line'], P['real'])
            run = next((r for r in self.runs.values() if r.name == P['name']), None)
            v = dict(line=P['line'], lineno=P['lineno'], name=P['name'], site=P['site'], queue=P['queue'], priority=P['priority'],
                     hold=bool(P['opts'].get('hold')))
            retired = (self.cfg.get('plan_retired') or {}).get(P['name'])
            if run is None and retired:
                # removed or deleted from the manager: done with; not set up and scheduled again
                v.update(state='retired', msg='%s %s; not set up again (manager.py unretire %s to run it again)'
                         % (retired.get('why', 'removed'), retired.get('t', ''), P['name']))
            elif run is None and key in self.plan_errors:
                v.update(state='error', msg=self.plan_errors[key])
            elif run is None:
                if P['site'] != LOCAL and P['site'] not in self.cfg['sites']:
                    v.update(state='error', msg='unknown site %s' % P['site'])
                    view.append(v)
                    continue
                try:
                    conf, how = self.plan_setup(P)
                    log('plan line %d: %s %s (%s)' % (P['lineno'], P['name'], how, conf))
                    self.ask(dict(action='add_run', conf=('%s:%s' % (P['site'], conf)) if P['site'] != LOCAL else conf,
                                  paused=bool(P['opts'].get('hold'))))
                    if not P['opts'].get('hold'):
                        self.ask(dict(action='schedule', run=P['name'], priority=P['priority'],
                                      not_before=P['opts'].get('not_before') or ''))
                    v.update(state='registered', msg=how)
                except Exception as e:  # noqa: BLE001
                    self.plan_errors[key] = str(e)
                    log('plan line %d (%s): %s' % (P['lineno'], P['name'], e))
                    v.update(state='error', msg=str(e))
            else:
                v.update(state='registered')
                sc = run.ms.get('scheduled')
                if sc and int(sc.get('priority', 0)) != P['priority']:
                    try:
                        self.ask(dict(action='set_priority', run=run.name, priority=P['priority']))
                    except Exception as e:  # noqa: BLE001
                        log('plan: priority of %s: %s' % (run.name, e))
            view.append(v)
        self.plan_view = view

    def remote_ops(self, run, *args, timeout=3600):
        """scratch_ops.py on the run's site -> its JSON answer."""
        S = self.cfg['sites'][run.site]
        cmd = ' '.join(shlex.quote(x) for x in [os.path.join(run.E['REMOTE_CODES'], 'venv', 'bin', 'python'),
                                                 os.path.join(run.E['REMOTE_SANDBOX'], 'scripts', 'pkdgrav3_pipeline', 'scratch_ops.py'),
                                                 args[0], run.E['REMOTE_CONF']] + [str(x) for x in args[1:]])
        r = ssh(S['login'], cmd, timeout=timeout)
        try:
            out = json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise RuntimeError('scratch_ops %s on %s: rc %d %s %s' % (args[0], run.site, r.returncode, r.stdout[-300:], r.stderr[-300:]))
        if not out.get('ok'):
            raise ValueError(out.get('error', 'scratch_ops %s failed' % args[0]))
        return out

    # ------------------------------------------------------------------ PBS
    def refresh_qstat(self):
        """Read this site's PBS jobs of the current user (``qstat -f -F json``) into ``self.qjobs``.

        Each job: state, name, node, queue time. On failure the previous snapshot is kept and ``qok`` is False.
        """
        r = subprocess.run([PBS + '/qstat', '-f', '-F', 'json'], capture_output=True, text=True, timeout=120)
        jobs = {}
        if r.returncode == 0:
            try:
                data = json.loads(r.stdout.replace('\\\'', "'"))
                for jid, j in (data.get('Jobs') or {}).items():
                    if j.get('Job_Owner', '').split('@')[0] != os.environ.get('USER', ''):
                        continue
                    jobs[short(jid)] = dict(state=j.get('job_state'), name=j.get('Job_Name'),
                                            node=(j.get('exec_host') or '').split('/')[0] or None,
                                            queue_time=j.get('qtime'))
                self.qok = True
            except ValueError as e:
                log('qstat json: %s' % e)
                self.qok = False
                return
        else:
            self.qok = False
            log('qstat failed: %s' % r.stderr.strip()[:200])
            return
        self.qjobs, self.qtime = jobs, now()

    def refresh_disk(self):
        """Free and used space of every scratch and archive file system the runs use (one entry per device),
        plus the scratch bytes of each run, for the page and the scratch checks. Storage that does not answer is
        skipped.
        """
        d, seen = {}, {}           # one entry per file system and kind (several scratch roots may share one)
        for run in self.runs.values():
            if getattr(run, 'stalled', None):
                continue
            for k in (('ARCHIVE',) if run.remote else ('SCRATCH', 'ARCHIVE')):
                p = run.E.get(k)
                if k == 'ARCHIVE' and p and PROBE.down('archive ' + p):
                    continue
                if p and p not in d and os.path.isdir(p):
                    key = os.stat(p).st_dev        # one entry per file system (pawna /storage: archive and a scratch)
                    if key in seen:
                        if p not in seen[key]['paths']:
                            seen[key]['paths'].append(p)
                        if k not in seen[key]['kinds']:
                            seen[key]['kinds'].append(k)
                        continue
                    u = shutil.disk_usage(p)
                    d[p] = seen[key] = dict(total=u.total, used=u.used, free=u.free, kind=k, kinds=[k], paths=[p])
            if run.remote:
                continue
            tot = 0
            for top in (run.P['sim'], run.P['halo']):
                for dp, _ds, fs in os.walk(top):
                    for f in fs:
                        try:
                            tot += os.lstat(os.path.join(dp, f)).st_size
                        except OSError:
                            pass
            run.scratch_bytes = tot
        for site in self.sites():
            try:
                self.refresh_site_disk(site)
            except Exception as e:  # noqa: BLE001
                log('disk %s: %s' % (site, e))
        d.update(self.site_disk)
        self.disk = d

    # ------------------------------------------------------------------ policy
    def tick(self):
        """One heartbeat: derive every run, run the watchdog, count active Rockstar and waiting jobs per site,
        submit what is due (post-processing, archive, scheduled simulations) and build the page snapshot.
        """
        active_rs = {}             # per site
        derived = {}
        self.waiting = {}          # per site: our jobs submitted but not started (PBS queue limits)
        self.load_queues()
        self.load_maint()
        for run in self.runs.values():
            D = self.derive(run)
            derived[run.conf] = D
            try:
                self.watch(run, D)
            except Exception as e:  # noqa: BLE001  (the watchdog never stops the manager)
                log('%s: watchdog error: %s' % (run.name, e))
            active_rs[run.site] = active_rs.get(run.site, 0) + sum(1 for j in D['active'] if j['stage'] == 'rockstar')
            self.waiting[run.site] = self.waiting.get(run.site, 0) + sum(
                1 for j in D['active'] if j['status'] in ('submitted', 'queued', 'held') and not j['job'].startswith('L'))
        try:
            self.maint_actions(derived)
        except Exception as e:  # noqa: BLE001  (never stops the manager)
            log('maintenance actions: %s\n%s' % (e, traceback.format_exc()))
        self.schedule_sims(derived)
        # Rockstar turn order over the runs: the one whose next snapshot has waited longest goes first
        def waiting(run):
            """When the oldest output of ``run`` not yet through Rockstar became ready (inf if none): Rockstar
            jobs are started oldest first across runs.
            """
            D = derived[run.conf]
            todo = set(range(len(run.outs))) - set(D['rs_done'])
            return run.seen_ready.get(min(todo), float('inf')) if todo else float('inf')
        for run in sorted(self.runs.values(), key=waiting):
            D = derived[run.conf]
            try:
                active_rs[run.site] = active_rs.get(run.site, 0) + self.policy(run, D, active_rs.get(run.site, 0))
                run.budget_wait = None
                run.maint_msg = None
            except MaintWait as e:
                if getattr(run, 'maint_msg', None) != str(e):
                    log('%s: %s' % (run.name, e))
                run.maint_msg = str(e)
                continue
            except BudgetWait as e:
                if getattr(run, 'budget_wait', None) != str(e).split(':')[0]:
                    run.note('waiting for core hours: %s' % e)
                run.budget_wait = str(e).split(':')[0]
                run.budget_msg = str(e)
            except Exception as e:  # noqa: BLE001
                run.ms['failed'] = dict(stage='manager', msg=str(e), t=now())
                run.note('manager error, automation stopped: %s' % e)
                log(traceback.format_exc())
        return derived

    # ------------------------------------------------------------------ watchdog
    def watch(self, run, D):
        """Alerts for a run (run.alerts: key -> dict(t, msg)): new ones are logged and noted once, gone ones noted as
        resolved. Only looks; never kills or resubmits. See DEFAULT_CFG (sim_stall_s ...)."""
        found = {}
        if not (run.ms.get('trash') or run.ms.get('purged') or getattr(run, 'stalled', None)):
            sim = D['last'].get('sim')
            if sim and sim['status'] == 'running':
                found.update(self.watch_sim(run, D, sim))
            lv = [e for e in run.ev if e['ev'] == 'stage_leftover']
            for e in lv[-3:]:
                found['stage_leftover:%s' % e.get('dir')] = (
                    'a stage left by an earlier job is on %s:%s (%.0f GB; set aside by job %s, not drained): copy what '
                    'is needed from it, then delete it' % (e.get('node'), e.get('dir'), int(e.get('size') or 0) / 1e9,
                                                           e['job']))
            if D.get('rs_redo'):
                found['rs_redo'] = ('Rockstar output of snapshot(s) %s was made before a restart rewrote them: '
                                    'it is run again when they are ready' % ', '.join(str(i) for i in D['rs_redo']))
        old = getattr(run, 'alerts', {})
        dis = run.ms.get('dismissed', {})
        new = {}
        for k, msg in found.items():
            a = old.get(k) if old.get(k, {}).get('msg') == msg else None
            new[k] = a or dict(t=old.get(k, {}).get('t', now()), msg=msg)
            if k not in old:
                run.note('ALERT %s' % msg)
        for k in old:
            if k not in found:
                run.note('alert resolved: %s' % old[k]['msg'])
        if any(k not in found for k in dis):
            run.ms['dismissed'] = {k: v for k, v in dis.items() if k in found}
            run.save()
        run.alerts = new
        run.alerts_shown = {k: a for k, a in new.items() if dis.get(k) != a['t']}   # dismissed: until it ends and comes back

    def watch_sim(self, run, D, sim):
        """Watchdog for one simulation job: returns alerts for a stalled job and for memory pressure.

        A job is stalled when its own log ``run.<job>.log`` has not changed for ``max(sim_stall_s,
        sim_stall_steps / rate)`` (catches a hang before the first step), and alerts on swap activity or low memory
        from the job's ``node_mem`` events (:func:`swap_alert`).
        """
        out = {}
        t_start = sim.get('start') or sim.get('submitted') or now()
        info = D.get('sim_info') or {}
        rate = info.get('rate')
        lim = max(float(self.cfg.get('sim_stall_s', 7200)),
                  float(self.cfg.get('sim_stall_steps', 5)) / rate if rate else 0.0)
        # this job's own log: sim_info has no log time before the job's first step, and a job that hangs before it
        # (pawna r4 2539111, 24 Nov 2026: stuck 10 h in the first tree build after a restart) must still be caught
        jl = os.path.join(run.P['sim'], 'run.%s.log' % sim['job'])
        try:
            lm = os.path.getmtime(jl)
        except OSError:
            lm = info.get('log_mtime')
        if lm and now() - t_start > lim and now() - lm > lim:
            out['sim_stalled'] = ('simulation job %s: its log has not changed for %.1f h (%s; limit %.1f h): hung? '
                                  '(node memory, NFS)' % (sim['job'], (now() - lm) / 3600,
                                                          'step %s' % info['step'] if info.get('step') is not None
                                                          else 'before its first step', lim / 3600))
        # the staging drain of this job
        dl = os.path.join(run.P['sim'], 'drain.%s.log' % sim['job'])
        if os.path.isfile(dl):
            st = getattr(run, 'drain_cache', None)
            m = os.path.getmtime(dl)
            if not st or st[0] != (dl, m):
                fin, drained, cur = {}, set(), None
                with open(dl, errors='replace') as fh:
                    for line in fh:
                        p = line.split()
                        if len(p) < 3:
                            continue
                        if p[1] == 'finished' and len(p) >= 4:
                            fin.setdefault(p[3], p[0])
                        elif p[1] == 'drained':
                            drained.add(p[2])
                        elif p[1] == 'current' and len(p) >= 4:
                            cur = (p[0], p[3])
                st = ((dl, m), fin, drained, cur)
                run.drain_cache = st
            _, fin, drained, cur = st
            lim = float(self.cfg.get('drain_stall_s', 10800))
            late = []
            for name, ts in fin.items():
                if name in drained:
                    continue
                try:
                    t = datetime.datetime.fromisoformat(ts).timestamp()
                except ValueError:
                    continue
                if now() - t > lim:
                    late.append((t, name))
            if late:
                late.sort()
                out['drain_stalled'] = ('job %s: %d finished output(s)/checkpoint(s) not copied to NFS after %.1f h '
                                        '(oldest %s, since %s): drain hung? they exist only on the node\'s disk'
                                        % (sim['job'], len(late), lim / 3600, late[0][1], iso(late[0][0])))
            if cur and '/var/tmp' not in cur[1] and not cur[1].startswith('/tmp'):
                out['drain_direct'] = ('job %s: node disk full, snapshots written straight to NFS since %s (%s)'
                                       % (sim['job'], cur[0], cur[1]))
        mem = [e for e in run.ev if e['ev'] == 'node_mem' and e['job'] == sim['job']]
        if mem:
            msg = swap_alert(mem, self.cfg)
            if msg:
                out['swap'] = 'job %s on %s: %s' % (sim['job'], mem[-1].get('host'), msg)
        return out

    # ------------------------------------------------------------------ node health (this site's compute nodes)
    def pbs_nodes(self, max_age=60):
        """``pbsnodes -a -F json`` -> {host: dict(state, usable, ncpus_free, mem_free_gb, jobs)}, cached ``max_age`` s."""
        if now() - self.pbsn_t < max_age and self.pbsn:
            return self.pbsn
        try:
            r = subprocess.run([PBS + '/pbsnodes', '-a', '-F', 'json'], capture_output=True, text=True, timeout=60)
            data = json.loads(r.stdout)['nodes']
        except Exception as e:  # noqa: BLE001
            log('pbsnodes: %s' % e)
            return self.pbsn
        out = {}
        for h, n in data.items():
            av, asg = n.get('resources_available', {}), n.get('resources_assigned', {})
            st = n.get('state', '')
            out[h] = dict(state=st, usable=not any(x in st for x in ('down', 'offline', 'unknown', 'unresolvable')),
                          ncpus_free=int(av.get('ncpus', 0)) - int(asg.get('ncpus', 0)),
                          mem_free_gb=mem_gb(av.get('mem')) - mem_gb(asg.get('mem')),
                          jobs=sorted({short(j) for j in n.get('jobs', [])}), comment=n.get('comment', ''))
        self.pbsn, self.pbsn_t = out, now()
        return out

    def probe_loop(self):
        """Thread: probe the nodes every ``node_probe_min`` minutes (0: only on request, the page's "probe now")."""
        time.sleep(20)
        while True:
            iv = float(self.cfg.get('node_probe_min', 30) or 0) * 60
            if self.probe_next == 0 or (iv > 0 and now() >= self.probe_next):
                try:
                    self.probe_nodes()
                except Exception as e:  # noqa: BLE001
                    log('node probe: %s\n%s' % (e, traceback.format_exc()))
                self.probe_next = now() + iv if iv > 0 else -1
            elif iv > 0 and self.probe_next < 0:      # switched on again from the page
                self.probe_next = now()
                continue
            self.probe_ev.wait(timeout=60)
            self.probe_ev.clear()

    def budget_loop(self):
        """Thread: read the core-hour budgets every ``budget_check_min`` minutes (0: only on request)."""
        time.sleep(30)
        while True:
            iv = float(self.cfg.get('budget_check_min', 30) or 0) * 60
            if self.budget_next == 0 or (iv > 0 and now() >= self.budget_next):
                try:
                    BUDGETS.refresh(self.cfg, list(self.runs.values()))
                except Exception as e:  # noqa: BLE001
                    log('budget read: %s\n%s' % (e, traceback.format_exc()))
                self.budget_next = now() + iv if iv > 0 else -1
            elif iv > 0 and self.budget_next < 0:
                self.budget_next = now()
                continue
            self.budget_ev.wait(timeout=60)
            self.budget_ev.clear()

    def probe_one(self, host, args):
        """node_check.sh on ``host`` over ssh (script on stdin); its JSON result, or a 'bad' result if no answer."""
        cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
               'bash -s -- ' + ' '.join(shlex.quote(x) for x in args)]
        try:
            with open(NODE_CHECK) as fh:
                r = subprocess.run(cmd, stdin=fh, capture_output=True, text=True, timeout=180)
            line = [l for l in r.stdout.splitlines() if l.startswith('NODECHECK ')]
            if line:
                return json.loads(line[-1][len('NODECHECK '):])
            why = 'node check gave no result (rc %d): %s' % (r.returncode, (r.stderr or '').strip()[-200:])
        except subprocess.TimeoutExpired:
            why = 'node check did not answer within 180 s (ssh or the node hangs)'
        except Exception as e:  # noqa: BLE001
            why = 'node check failed: %s' % e
        return dict(host=host, t=now(), status='bad', bad=[why], warn=[], dirs=[])

    def probe_nodes(self):
        """Probe every compute node PBS lists (in parallel threads) and keep the results in ``self.nodes``;
        log changes of a node's status and clear a job's bad-node flag once a probe after it passes."""
        pn = self.pbs_nodes(max_age=0)
        u = os.environ.get('USER', '')
        paths = self.cfg.get('node_probe_paths') or [os.path.join(self.cfg['scratch_root'], 'state')] + (
            ['/storage/' + u] if u and os.path.isdir('/storage/' + u) else [])
        args = ['--paths', ' '.join(paths), '--user', u]
        res = {}

        def one(h):
            """Probe node ``h`` into ``res``."""
            res[h] = self.probe_one(h, args)
        th = []
        for h, n in pn.items():
            if 'down' in n['state']:
                res[h] = dict(host=h, t=now(), status='down', bad=[], warn=['PBS: %s %s' % (n['state'], n.get('comment', ''))], dirs=[])
                continue
            t = threading.Thread(target=one, args=(h,), daemon=True)
            t.start()
            th.append(t)
        for t in th:
            t.join(200)
        for h, r in list(res.items()):
            r['pbs_state'] = pn.get(h, {}).get('state')
            old = self.nodes.get(h, {})
            if old.get('status') != r.get('status'):
                log('node %s: %s -> %s%s' % (h, old.get('status', 'not probed'), r.get('status'),
                                            (': ' + '; '.join(r.get('bad', []) + r.get('warn', []))) if r.get('bad') or r.get('warn') else ''))
            f = self.node_flags.get(h)
            if f and r.get('status') in ('ok', 'warn') and r.get('t', 0) > f['t']:
                log('node %s: probe passed after the failed node check of job %s; no longer avoided' % (h, f.get('job')))
                self.node_flags.pop(h, None)
            self.nodes[h] = r
        self.q.put(('wake',))

    def bad_nodes(self):
        """{host: reasons} of this site's nodes to avoid: the newest probe was bad (not older than 3 intervals or
        2 h), or a job's node check failed there within ``node_bad_h`` h (and no probe has passed since)."""
        out, t = {}, now()
        iv = max(float(self.cfg.get('node_probe_min', 30) or 0) * 60 * 3, 7200)
        for h, r in self.nodes.items():
            if r.get('status') == 'bad' and t - r.get('t', 0) < iv:
                out[h] = 'probe: ' + '; '.join(r.get('bad', []))
        for h, f in list(self.node_flags.items()):
            if t - f['t'] > float(self.cfg.get('node_bad_h', 6)) * 3600:
                self.node_flags.pop(h, None)
            else:
                out.setdefault(h, 'job %s: %s' % (f.get('job'), f.get('reasons')))
        return out

    def place(self, run, stages):
        """Node for the job of ``stages`` of a run here when bad nodes are usable in PBS: the healthy node with room
        (cores and memory free now) with the most free cores; None when no bad node is usable (PBS chooses);
        NodeWait when no healthy node has room (submitted at a later pass). Remote runs and multi-node jobs: None."""
        if run.remote or run.site != LOCAL or run.E.get('SIM_SELECT'):
            return None
        bad = self.bad_nodes()
        if not bad:
            return None
        pn = self.pbs_nodes()
        if not any(pn.get(h, {}).get('usable') for h in bad):
            return None
        ncpu, mem = 0, 0.0
        for st in stages.split(','):
            if st in STAGE_RES:
                kc, km, dc, dm = STAGE_RES[st]
                ncpu = max(ncpu, int(run.E.get(kc) or dc))
                mem = max(mem, mem_gb(run.E.get(km) or dm))
        good = sorted((h for h, n in pn.items() if n['usable'] and h not in bad and n['ncpus_free'] >= ncpu
                       and n['mem_free_gb'] >= mem), key=lambda h: -pn[h]['ncpus_free'])
        if good:
            return good[0]
        raise NodeWait('node(s) %s avoided (%s); no healthy node has %d cores and %.0f GB free now: %s waits' % (
            ', '.join(sorted(bad)), '; '.join('%s: %s' % x for x in sorted(bad.items())), ncpu, mem, stages))

    def node_bad_job(self, run, j, D):
        """A job that exited 75 (its node failed the check in pipeline_env.sh): avoid that node and resubmit the job
        (a simulation is rescheduled, other stages retried automatically), at most ``node_retry_max`` times a day per
        run; returns True if the run's automation waits."""
        run.ms.setdefault('acked', []).append(j['job'])
        host = j.get('host') or '?'
        ev = [e for e in run.ev if e['ev'] == 'node_bad' and e['job'] == j['job']]
        why = ev[-1].get('reasons', '') if ev else 'node check failed'
        self.node_flags[host] = dict(t=now(), reasons=why, job=j['job'], run=run.name)
        log('node %s: avoided for %s h: node check of %s job %s failed: %s' % (host, self.cfg.get('node_bad_h', 6), j['stage'], j['job'], why))
        recent = [t for t in run.ms.get('nodebad', []) if now() - t < 86400] + [now()]
        run.ms['nodebad'] = recent
        if len(recent) > int(self.cfg.get('node_retry_max', 3)):
            run.ms['failed'] = dict(stage=j['stage'], job=j['job'], rc=NODE_RC, t=now(), first=j.get('first'), last=j.get('last'),
                                    msg='node check failed %d times in 24 h (last on %s: %s); automation stopped' % (len(recent), host, why))
            run.note('%s job %s: node check failed on %s (%s), %d times in 24 h; automation stopped' % (j['stage'], j['job'], host, why, len(recent)))
            run.save()
            return True
        if j['stage'] == 'sim':
            restart = self.has_checkpoint(run, D)
            run.ms['scheduled'] = dict(t=now(), priority=-1, restart=restart, maint=True, after=j['job'])
            run.note('simulation job %s did not start: node %s failed its check (%s); scheduled again on another node '
                     '(not counted in max_sim_restarts)' % (j['job'], host, why))
            run.save()
            return False
        run.ms['failed'] = dict(stage=j['stage'], job=j['job'], rc=NODE_RC, log=j.get('log'), t=now(), auto_retry=True,
                                first=j.get('first'), last=j.get('last'),
                                msg='node %s failed its check (%s); resubmitted on another node' % (host, why))
        run.note('%s job %s did not start: node %s failed its check (%s); resubmitted on another node' % (j['stage'], j['job'], host, why))
        run.save()
        return True

    def classify_dir(self, host, d):
        """Kind of a directory of ours in a node's /var/tmp and whether a job uses it: (kind, in_use, note).
        Rockstar output of a job still in PBS, or the stage of a run whose simulation runs on that node, is in use;
        the rest are leftovers of ended jobs."""
        base = os.path.basename(d['dir'])
        m = re.match(r'^[\w.-]+?_rockstar_(\d+)$', base)
        if m:
            q = self.qjobs.get(m.group(1))
            return 'rockstar output', bool(q), ('job %s %s' % (m.group(1), q['state']) if q else 'job %s ended' % m.group(1))
        m = re.match(r'^[\w.-]+?_stage_(.+?)(\.left\.(\d+))?$', base)
        if m:
            if m.group(2):
                return 'stage set aside', False, 'set aside by job %s (not drained: may hold the only copy of outputs)' % m.group(3)
            for run in self.runs.values():
                if run.site == LOCAL and '%s_r%s' % (run.E.get('STEM', '').replace('/', '_'), run.E.get('REAL')) == m.group(1):
                    ids = {x['job'] for x in run.ms.get('submitted', []) if x.get('stage') == 'sim'}
                    on = [i for i in ids if (self.qjobs.get(i) or {}).get('node') == host
                          and self.qjobs[i].get('state') in ('R', 'E')]
                    return 'stage', bool(on), ('stage of %s (simulation running here)' % run.name if on else
                                               'stage of %s; no simulation of it here (check it holds nothing undrained)' % run.name)
            return 'stage', False, 'stage of an unknown run'
        return 'other', True, ''

    def nodes_view(self):
        """Nodes for the page: probe result, PBS state, avoided or not (and why), our directories in /var/tmp."""
        bad = self.bad_nodes()
        out = []
        for h in sorted(set(self.nodes) | set(self.pbsn)):
            r = dict(self.nodes.get(h) or dict(host=h, status='not probed', bad=[], warn=[], dirs=[]))
            r['pbs'] = self.pbsn.get(h, {})
            r['avoided'] = bad.get(h)
            dirs = []
            for d in r.get('dirs', []):
                kind, used, note = self.classify_dir(h, d)
                dirs.append(dict(d, kind=kind, in_use=used, note=note, deleting=(h, d['dir']) in self.deleting))
            r['dirs'] = dirs
            out.append(r)
        return out

    def delete_leftover(self, host, d, confirm):
        """Delete a leftover directory of ours in a node's /var/tmp (page button, the name typed as confirmation):
        only a <user>_stage_* / <user>_rockstar_* directory listed by the newest probe and not used by a job.
        ``rm -rf`` over ssh in a thread; the nodes are probed again afterwards."""
        u = os.environ.get('USER', '')
        if not re.match(r'^/var/tmp/%s_(stage|rockstar)_[\w.-]+$' % re.escape(u), d or '') or '..' in d:
            raise ValueError('not a directory the manager deletes: %s' % d)
        if confirm != os.path.basename(d):
            raise ValueError('not deleted: type the directory name (%s) to confirm' % os.path.basename(d))
        r = self.nodes.get(host) or {}
        hit = [x for x in r.get('dirs', []) if x['dir'] == d]
        if not hit:
            raise ValueError('%s:%s is not in the newest probe of %s; probe again' % (host, d, host))
        kind, used, note = self.classify_dir(host, hit[0])
        if used:
            raise ValueError('%s:%s is in use (%s)' % (host, d, note))
        if (host, d) in self.deleting:
            raise ValueError('already being deleted')
        self.deleting[(host, d)] = now()
        log('deleting %s:%s (%s, %.1f GB; %s), confirmed on the page' % (host, d, kind, hit[0].get('gb', -1), note))

        def work():
            """rm -rf over ssh, then probe again."""
            try:
                p = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, 'rm -rf -- %s' % shlex.quote(d)],
                                   capture_output=True, text=True, timeout=3600)
                log('deleted %s:%s (rc %d %s)' % (host, d, p.returncode, p.stderr.strip()[-200:]))
            except Exception as e:  # noqa: BLE001
                log('deleting %s:%s failed: %s' % (host, d, e))
            self.deleting.pop((host, d), None)
            self.probe_next = 0
            self.probe_ev.set()
        threading.Thread(target=work, daemon=True).start()
        return 'deleting %s:%s (%.1f GB)' % (host, d, hit[0].get('gb', -1))

    # ------------------------------------------------------------------ maintenance windows
    def load_maint(self):
        """Re-read maintenance.txt (windows per site) if it changed; log it."""
        p = self.cfg.get('maintenance_file')
        try:
            m = os.path.getmtime(p) if p else None
        except OSError:
            m = None
        if m != self.maint_mtime:
            self.maint_windows, self.maint_errors = read_maint(p)
            self.maint_mtime = m
            log('maintenance: %s%s' % ('; '.join(self.maint_label(st, w) for st, ws in sorted(self.maint_windows.items())
                                                 for w in ws) or 'no windows',
                                       ('; ERRORS: ' + '; '.join(self.maint_errors)) if self.maint_errors else ''))

    @staticmethod
    def maint_label(site, w):
        """``maintenance of <site> <start> - <end>`` for messages."""
        return 'maintenance of %s %s - %s' % (site, time.strftime('%a %d %b %H:%M', time.localtime(w['start'])),
                                              time.strftime('%a %d %b %H:%M', time.localtime(w['end'])) if w['end'] else 'end not known')

    def maint_next(self, site):
        """The window of ``site`` that matters now: the first not yet over (end + maint_settle_min), or None."""
        settle = float(self.cfg.get('maint_settle_min', 30)) * 60
        for w in self.maint_windows.get(site, []):
            if w['end'] is None or now() < w['end'] + settle:
                return w
        return None

    def maint_now(self, site):
        """The window of ``site`` in progress (started, not over including the settle time), or None."""
        w = self.maint_next(site)
        return w if w and now() >= w['start'] else None

    def maint_stop_h(self, site):
        """Hours before a window's start at which running simulations of ``site`` get their STOP file."""
        if site == LOCAL:
            return float(self.cfg.get('maint_stop_h', 6))
        return float(self.cfg['sites'][site].get('maint_stop_h', MAINT_STOP_H_REMOTE))

    def stage_wall_s(self, run, stage):
        """Requested wall time of ``stage`` (run.conf, else submit_pipeline.sh's default) in seconds."""
        key, dflt = STAGE_WALL.get(stage, (None, '24:00:00'))
        return hours((run.E.get(key) if key else None) or dflt) * 3600

    def expected_s(self, run, stage, what=None):
        """Expected run time (s) of one job of ``stage``: 1.25 x the longest done job of that stage in this run,
        else in the runs of the same site and particle number (Rockstar: per snapshot x the batch size); at most
        the requested wall time, which is also the answer without such jobs."""
        req = self.stage_wall_s(run, stage)
        k = 1
        if stage == 'rockstar':
            w = what or {}
            k = (int(w['last']) - int(w['first']) + 1) if w.get('first') is not None and w.get('last') is not None else len(run.outs)
        for group in ([run], [r for r in self.runs.values() if r is not run and r.site == run.site
                              and r.E.get('NPART') == run.E.get('NPART')]):
            d = []
            for r in group:
                for j in self.derive(r)['jobs']:
                    if j['stage'] == stage and j['status'] == 'done' and j.get('start') and j.get('end'):
                        n = 1
                        if stage == 'rockstar' and j.get('first') is not None and j.get('last') is not None:
                            n = int(j['last']) - int(j['first']) + 1
                        d.append((j['end'] - j['start']) / max(1, n))
            if d:
                return min(req, 1.25 * max(d) * k)
        return req

    def maint_gate(self, run, stages, what=None):
        """Called for every submission (Run.GATE): raise MaintWait if a maintenance window of the run's site (for
        archive copies and pulls also of this host) is in progress or the job would not end
        ``maint_guard_min`` before its start; return a shorter wall time (``HH:MM:SS``) when the requested one
        would run past the start, else None. A simulation needs ``maint_min_sim_h`` of computing before its STOP
        (``maint_stop_h`` before the start); it stops cleanly there and restarts after the window."""
        sites = [run.site] + ([LOCAL] if stages in ('archive', 'archb') and run.site != LOCAL else [])
        cap = None
        t = now()
        for site in sites:
            w = self.maint_next(site)
            if not w:
                continue
            lab = self.maint_label(site, w)
            if t >= w['start']:
                raise MaintWait('%s in progress: nothing is submitted until %s' % (
                    lab, '%d min after its end' % self.cfg.get('maint_settle_min', 30) if w['end'] else 'the window is given an end or removed'))
            left = w['start'] - float(self.cfg.get('maint_guard_min', 30)) * 60 - t
            st = stages.split(',')
            if st == ['sim']:
                room = w['start'] - self.maint_stop_h(site) * 3600 - t
                if room < float(self.cfg.get('maint_min_sim_h', 4)) * 3600:
                    raise MaintWait('%s: only %.1f h of computing before the STOP at %s (maint_min_sim_h %s): the '
                                    'simulation waits until after the window' % (
                                        lab, max(0, room) / 3600., time.strftime('%a %H:%M', time.localtime(t + max(0, room))),
                                        self.cfg.get('maint_min_sim_h', 4)))
            else:
                need = sum(self.expected_s(run, x, what) for x in st)
                if need > left:
                    raise MaintWait('%s: %s needs ~%.1f h (earlier jobs, else its wall time), %.1f h left before the '
                                    'start (maint_guard_min %s): waits until after the window' % (
                                        lab, stages, need / 3600., max(0, left) / 3600., self.cfg.get('maint_guard_min', 30)))
            if site == run.site and stages not in ('archive', 'archb') and left < max(self.stage_wall_s(run, x) for x in st):
                cap = hms(left)
        return cap

    def maint_stopped(self, run):
        """Simulation jobs of ``run`` that got a maintenance STOP file."""
        return {k.split('@')[0] for k in run.ms.get('maint_done', []) if k.endswith('@stop')}

    def sim_dir_cmd(self, run):
        """(host or None, simulation directory) of a run: on the site for a remote run, here otherwise."""
        if run.remote:
            return (self.cfg['sites'][run.site]['login'],
                    '%s/sims/%s/r%s' % (run.E['REMOTE_SCRATCH'].rstrip('/'), run.E['STEM'], run.E['REAL']))
        return None, run.P['sim']

    def maint_actions(self, derived):
        """Before a window (from ``maint_stop_h`` before its start): write a STOP file in the run directory of every
        running simulation of the site (pkdgrav3 checkpoints, exits cleanly and removes the file; the restart is
        scheduled for after the window), and take simulation jobs still waiting in PBS out (qdel) and schedule
        them again for after the window (they could start with too little time, or after it with a cut wall
        time). Each job once per window (``maint_done`` in manager.json)."""
        t = now()
        for run in self.runs.values():
            w = self.maint_next(run.site)
            if not w or t >= w['start'] or getattr(run, 'stalled', None) or run.err:
                continue
            tstop = w['start'] - self.maint_stop_h(run.site) * 3600
            if t < tstop:
                self.recheck_at = min(self.recheck_at or 1e30, tstop + 1)
                continue
            D = derived.get(run.conf) or self.derive(run)
            done = run.ms.setdefault('maint_done', [])
            for j in D['active']:
                if j['stage'] != 'sim' or any(k.startswith(j['job'] + '@') for k in done):
                    continue
                host, sdir = self.sim_dir_cmd(run)
                lab = self.maint_label(run.site, w)
                if j['status'] in ('running', 'exiting'):
                    stop = os.path.join(sdir, 'STOP')
                    if host:
                        r = ssh(host, 'touch %s' % shlex.quote(stop), timeout=120)
                    else:
                        r = subprocess.run(['timeout', '120', 'touch', stop], capture_output=True, text=True)
                    if r.returncode != 0:
                        run.note('%s: STOP file for simulation job %s not written (rc %d %s); trying again at the next '
                                 'pass' % (lab, j['job'], r.returncode, (r.stderr or '')[-200:]))
                        continue
                    done.append(j['job'] + '@stop')
                    run.note('%s: STOP file written for simulation job %s (%s): pkdgrav3 writes a checkpoint at the end '
                             'of its current step and exits; restart scheduled after the window' % (lab, j['job'], stop))
                else:
                    q = 'PATH=/opt/pbs/bin:$PATH qdel %s' % j['job']
                    r = ssh(host, q, timeout=120) if host else subprocess.run(
                        ['bash', '-c', q], capture_output=True, text=True, timeout=120)
                    if r.returncode != 0:
                        run.note('%s: qdel of the waiting simulation job %s failed (rc %d %s)' % (
                            lab, j['job'], r.returncode, (r.stderr or '')[-200:]))
                        continue
                    done.append(j['job'] + '@qdel')
                    run.ms.setdefault('acked', []).append(j['job'])
                    restart = self.has_checkpoint(run, D)
                    run.ms['scheduled'] = dict(t=now(), priority=-1, restart=restart, maint=True)
                    run.note('%s: simulation job %s was still waiting in PBS at the STOP time; deleted (qdel) and '
                             'scheduled again for after the window%s' % (lab, j['job'], ' (restart from checkpoint)' if restart else ''))
            run.save()

    def has_checkpoint(self, run, D):
        """Does the run have a checkpoint to restart from? Here: checkpoints/*.chk; a remote run (checkpoints
        not mirrored): an earlier simulation job stopped cleanly or itself restarted from one."""
        if run.remote:
            return bool(D['sim_clean_stop']) or any(e['ev'] == 'sim_info' and e.get('restart') for e in run.ev)
        return bool(glob.glob(os.path.join(run.P['sim'], 'checkpoints', '*.chk')))

    def maint_killed(self, run, j):
        """Did job ``j`` (failed or vanished) end because of a maintenance window: started (or submitted) before a
        window of its site (archive copies and pulls: also of this host) began, and ended no earlier than an hour
        before it (or has no end event)? Windows that ended more than 7 days ago are not considered."""
        sites = {run.site} | ({LOCAL} if j['stage'] in ('archive', 'archb') else set())
        t = now()
        on, off = j.get('start') or j.get('submitted') or 0, j.get('end') or t
        for site in sites:
            for w in self.maint_windows.get(site, []):
                if t < w['start'] or (w['end'] and t - w['end'] > 7 * 86400):
                    continue
                if on < w['start'] and off >= w['start'] - 3600:
                    return site, w
        return None

    def maint_lost(self, run, j, D):
        """A job killed by a maintenance shutdown: a simulation is scheduled to restart after the window; any other
        stage is marked failed with ``maint`` set and resubmitted automatically once the window is over
        (:meth:`maint_retry`). Returns True if the run's automation now waits for that."""
        lab = self.maint_label(*self.maint_killed(run, j))
        run.ms.setdefault('acked', []).append(j['job'])
        if j['stage'] == 'sim':
            restart = self.has_checkpoint(run, D)
            run.ms['scheduled'] = dict(t=now(), priority=-1, restart=restart, maint=True, after=j['job'])
            run.note('simulation job %s %s during the %s; scheduled to %s after the window' % (
                j['job'], 'vanished' if j['status'] == 'gone' else 'ended (rc %s)' % j.get('rc'), lab,
                'restart from the newest checkpoint' if restart else 'start again from the ICs (no checkpoint)'))
            run.save()
            return False
        if j['stage'] == 'archb':
            run.note('archive batch %s %s during the %s; the next batch (or the final archive) copies it again'
                     % (j['job'], j['status'], lab))
            run.save()
            return False
        run.ms['failed'] = dict(stage=j['stage'], job=j['job'], rc=j.get('rc'), log=j.get('log'), t=now(), maint=True,
                                first=j.get('first'), last=j.get('last'),
                                msg='killed by the %s; resubmitted automatically after the window' % lab)
        run.note('%s job %s %s during the %s; resubmitted automatically after the window' % (
            j['stage'], j['job'], 'vanished' if j['status'] == 'gone' else 'ended (rc %s)' % j.get('rc'), lab))
        run.save()
        return True

    def maint_retry(self, run):
        """Resubmit a job killed by a maintenance shutdown (the retry action) once its window is over; on a refusal
        (e.g. another window) the failure stays marked for the next pass."""
        f = dict(run.ms['failed'])
        try:
            msg = self.action(dict(action='retry', run=run.name))
            run.note('after the maintenance: %s' % msg)
        except MaintWait as e:      # (also NodeWait: no healthy node with room yet)
            run.ms['failed'] = f
            run.maint_msg = str(e)
        except Exception as e:  # noqa: BLE001
            run.ms['failed'] = dict(f, maint=False, msg='automatic resubmission after the maintenance failed: %s' % e)
            run.note('automatic resubmission after the maintenance failed: %s' % e)
        run.save()

    def maint_view(self):
        """Windows for the page: site, start, end, note, state (upcoming / stopping / in progress / over)."""
        out, t = [], now()
        for site, ws in sorted(self.maint_windows.items()):
            for w in ws:
                cur = self.maint_next(site)
                if w['end'] and t >= w['end'] + float(self.cfg.get('maint_settle_min', 30)) * 60:
                    state = 'over'
                elif t >= w['start']:
                    state = 'in progress'
                elif cur is w and t >= w['start'] - self.maint_stop_h(site) * 3600:
                    state = 'stopping simulations'
                else:
                    state = 'upcoming'
                out.append(dict(site=site, start=w['start'], end=w['end'], note=w['note'], state=state,
                                stop_at=w['start'] - self.maint_stop_h(site) * 3600))
        return out

    def load_queues(self):
        """Re-read the queues file (open/drain/closed and limits per site and queue) if it changed; log it."""
        p = self.cfg.get('queues_file')
        try:
            m = os.path.getmtime(p) if p else None
        except OSError:
            m = None
        if m != self.queues_mtime:
            self.queues, self.queue_errors = read_queues(p)
            self.queues_mtime = m
            log('queues: %s%s' % (', '.join('%s:%s %s' % (k[0], k[1], v['state']) for k, v in sorted(self.queues.items()))
                                  or 'none listed (all open)', ('; ERRORS: ' + '; '.join(self.queue_errors)) if self.queue_errors else ''))

    def run_ok(self, run):
        """Does the file system with the run's files (its SCRATCH: state, events, logs) answer? Sets run.stalled."""
        key = run.E['SCRATCH']
        if PROBE.ok(key, storage_test(run.P['state']), int(self.cfg.get('storage_timeout_s', 30))) and not os.path.isdir(key):
            if not getattr(run, 'stalled', None):     # unmounted: answers, but SCRATCH is gone
                run.stalled = 'storage: %s does not exist (file system not mounted?)' % key
                log('%s: %s' % (run.name, run.stalled))
                self.restart_watch()
            return False
        if PROBE.ok(key, storage_test(run.P['state']), int(self.cfg.get('storage_timeout_s', 30))):
            if getattr(run, 'stalled', None):
                run.stalled = None
                self.restart_watch()
            return True
        if not getattr(run, 'stalled', None):
            run.stalled = 'storage %s not responding since %s' % (key, iso(PROBE.down(key)))
            self.restart_watch()
        return False

    def archive_wait(self, run):
        """Why the archive of `run` must wait now (queues file 'pawna archive', or ARCHIVE not answering), or None."""
        qi = self.queues.get((LOCAL, 'archive'))
        if qi and qi['state'] != 'open':
            return 'archive paused (queues file: %s archive %s)' % (LOCAL, qi['state'])
        A = run.E.get('ARCHIVE')
        top = '/' + A.strip('/').split('/')[0] if A else None
        if A and PROBE.ok('archive ' + A, archive_test(A), int(self.cfg.get('storage_timeout_s', 30))) and \
                os.path.ismount(top) is False and len(os.listdir(top)) == 0:
            return 'archive storage %s is empty (not mounted?)' % top
        if A and not PROBE.ok('archive ' + A, archive_test(A), int(self.cfg.get('storage_timeout_s', 30))):
            return 'archive storage %s not responding since %s' % (A, iso(PROBE.down('archive ' + A)))
        return None

    def qinfo(self, run):
        """Queue settings of the run's queue (open with no limits if not listed)."""
        return self.queues.get((run.site, run_queue(run))) or dict(state='open', max_running=None, max_waiting=None)

    def policy(self, run, D, active_rs):
        """Submit what is due for this run; returns the number of Rockstar jobs submitted."""
        if getattr(run, 'stalled', None):
            return 0
        f = run.ms.get('failed')
        if f and not (run.busy or run.err or run.ms.get('paused')) and (f.get('auto_retry') or (
                f.get('maint') and not self.maint_now(run.site)
                and not (f.get('stage') in ('archive', 'archb') and self.maint_now(LOCAL)))):
            self.maint_retry(run)
            return 0
        if run.busy or run.err or run.ms.get('paused') or run.ms.get('failed') or run.ms.get('trash') or run.ms.get('purged'):
            return 0
        E = run.E
        SC = site_cfg(self.cfg, run.site)
        qtime = self.qsite.get(run.site, ({}, 0.0))[1] if run.remote else self.qtime
        # failures and vanished jobs
        waiting_gone = False
        for j in D['jobs']:
            if (j['status'] == 'failed' and j['stage'] == 'sim' and j['job'] in D['sim_clean_stop']
                    and not self.acked(run, j)):
                n = int(run.ms.get('restarts', 0))
                run.ms.setdefault('acked', []).append(j['job'])
                if j['job'] in self.maint_stopped(run):
                    run.ms['scheduled'] = dict(t=now(), priority=-1, restart=True, after=j['job'], maint=True)
                    run.note('simulation job %s stopped cleanly on the maintenance STOP file; scheduled to restart from '
                             'its checkpoint after the window (not counted in max_sim_restarts)' % j['job'])
                    continue
                if n >= int(SC.get('max_sim_restarts', 5)):
                    run.ms['failed'] = dict(stage='sim', job=j['job'], rc=j.get('rc'), log=j.get('log'), t=now(),
                                            msg='stopped at its wall time again after %d restarts (max_sim_restarts)' % n)
                    run.note('simulation job %s stopped at its wall time; %d restarts done, not restarting' % (j['job'], n))
                    return 0
                run.ms['scheduled'] = dict(t=now(), priority=-1, restart=True, after=j['job'])
                run.note('simulation job %s stopped cleanly before the last output (wall time); scheduled to restart '
                         'from the newest checkpoint (restart %d of at most %d)' % (j['job'], n + 1, SC.get('max_sim_restarts', 5)))
                continue
            if j['status'] == 'failed' and j.get('rc') == NODE_RC and not self.acked(run, j):
                if self.node_bad_job(run, j, D):
                    return 0
                continue
            if j['status'] in ('failed', 'gone') and not self.acked(run, j) and self.maint_killed(run, j):
                if j['status'] == 'gone':     # the usual grace: it may be an end event still on its way
                    g = run.ms.setdefault('gone', {})
                    if now() - g.setdefault(j['job'], now()) <= self.cfg['lost_grace_s']:
                        waiting_gone = True
                        continue
                if self.maint_lost(run, j, D):
                    return 0
                continue
            if j['stage'] == 'archb' and j['status'] in ('failed', 'gone') and not self.acked(run, j):
                if j['status'] == 'gone':
                    g = run.ms.setdefault('gone', {})
                    if now() - g.setdefault(j['job'], now()) <= self.cfg['lost_grace_s']:
                        continue
                run.ms.setdefault('acked', []).append(j['job'])
                run.ms['archb_hold'] = j['job']
                run.note('archive batch %s %s: no more batches for this run (the final archive copies everything; '
                         '"retry" is not needed)' % (j['job'], j['status']))
                run.save()
                continue
            if j['status'] == 'failed' and not self.acked(run, j):
                run.ms['failed'] = dict(stage=j['stage'], job=j['job'], rc=j.get('rc'), log=j.get('log'), t=now(),
                                        first=j.get('first'), last=j.get('last'))
                run.note('%s job %s failed (rc %s); automation stopped' % (j['stage'], j['job'], j.get('rc')))
                return 0
            if j['status'] == 'gone' and not self.acked(run, j):
                waiting_gone = True      # not in qstat and no end event (yet): submit nothing until it is resolved
                g = run.ms.setdefault('gone', {})
                first_seen = g.setdefault(j['job'], now())
                run.save()
                if now() - first_seen > self.cfg['lost_grace_s'] and (qtime > first_seen or j['job'].startswith('L')):
                    run.ms['failed'] = dict(stage=j['stage'], job=j['job'], rc=None, t=now(),
                                            msg='job vanished from PBS without an end event (killed, node failure?)',
                                            first=j.get('first'), last=j.get('last'))
                    run.note('%s job %s vanished from PBS without an end event; automation stopped' % (j['stage'], j['job']))
                    return 0
                self.recheck_at = min(self.recheck_at or 1e30, first_seen + self.cfg['lost_grace_s'] + 5)
        if waiting_gone:     # e.g. a remote job ended and its end event is not in the mirror yet
            return 0
        try:
            self.archive_batch(run, D)
        except MaintWait as e:      # the batch waits; Rockstar and the rest may still go in
            run.maint_msg = str(e)
        if D['active'] and any(j['stage'] in ('ctrees', 'vahc', 'vpot', 'fits', 'parquet', 'xcorr', 'vprof', 'verify', 'archive')
                               for j in D['active']):
            return 0
        if self.qinfo(run)['state'] == 'closed':     # nothing new into a closed queue (drain: post-processing goes on)
            return 0
        n_out = len(run.outs)
        rs_done = set(D['rs_done'])
        # ---- Rockstar
        if len(rs_done) < n_out:
            if any(j['stage'] == 'rockstar' for j in D['active']):
                return 0
            if E.get('RS_BATCH', '0') == '1':
                first = min(set(range(n_out)) - rs_done)
                ready = set(D['ready'])
                k = 0
                while first + k < n_out and first + k in ready:
                    k += 1
                nmin = int(E.get('RS_BATCH_MIN', '10') or 10)
                if k == 0:
                    return 0
                if k >= nmin or D['sim_done']:          # after the simulation: whatever is left
                    if active_rs >= int(SC.get('max_rockstar', 1)) or not self.pbs_room(run.site):
                        return 0
                    last = first + k - 1
                    run.submit('rockstar', ['--rs-snaps', '%d-%d' % (first, last)], dict(first=first, last=last))
                    self.took(run.site)
                    return 1
                return 0
            if (D['sim_done'] and active_rs < int(SC.get('max_rockstar', 1)) and not D['last'].get('rockstar')
                    and self.pbs_room(run.site)):
                run.submit('rockstar', [], dict(first=0, last=n_out - 1))
                self.took(run.site)
                return 1
            return 0
        # ---- post-processing chain, once Rockstar covers everything
        if not D['sim_done']:
            return 0
        lv = D['last'].get('verify')
        archived = bool(D['archive'] and int(D['archive'].get('ok', 0)))
        post = [s for s in post_of(E) if not (archived and s == 'parquet')]   # an archived run stays as archived
        if site_cfg(self.cfg, run.site).get('max_pbs_queued') is not None:
            # one stage at a time: the first whose newest job is not a success newer than the previous stage's
            prev = None
            for st in post:
                j = D['last'].get(st)
                if j is None or j['status'] != 'done' or (prev and jobnum(j['job']) < jobnum(prev['job'])):
                    if self.pbs_room(run.site):
                        run.submit(st, [], dict(chain=st, sequential=True))
                        self.took(run.site)
                    return 0
                prev = j
        elif not any(D['last'].get(s) for s in post):
            stages = ','.join(post)
            run.submit(stages, [], dict(chain=stages))
            return 0
        elif 'parquet' in post:
            # PARQUET=1 on a run whose FITS were made without it (or again after it): parquet, then verify again,
            # before the archive (which takes the Parquet files as kept files)
            lf, lp = D['last'].get('fits'), D['last'].get('parquet')
            if lf and lf['status'] == 'done' and (lp is None or jtime(lp) < jtime(lf)):
                run.submit('parquet,verify', [], dict(chain='parquet,verify'))
                return 0
        v = D['verify']
        run.archive_msg = None
        if lv and lv['status'] == 'done' and v and int(v.get('ok', 0)) == 1 and E.get('ARCHIVE'):
            la = D['last'].get('archive')   # one archive job per passed verify (PBS ids increase; node clocks differ)
            due = not any(j['stage'] in ('archive', 'archb') for j in D['active']) and (
                (la is None or jtime(la) < jtime(lv)) if not run.remote else
                lv['job'] not in {s.get('after_verify') for s in run.ms['submitted'] if s['stage'] == 'archive'})
            if due:
                run.archive_msg = self.archive_wait(run)
                if run.archive_msg:
                    return 0
            if run.remote:
                # pulled here after the checksums of that verify job (KEPT_FILES.json in the mirror)
                k = D['kept']
                kf = os.path.join(run.P['state'], 'KEPT_FILES.json')
                done_for = {s.get('after_verify') for s in run.ms['submitted'] if s['stage'] == 'archive'}
                if (k and k['job'] == lv['job'] and int(k.get('ok', 0)) == 1 and os.path.isfile(kf)
                        and lv['job'] not in done_for):
                    pulls = sum(1 for r in self.runs.values() for j in self.derive(r)['active']
                                if j['job'].startswith('L'))
                    if pulls < int(self.cfg.get('max_pulls', 1)):
                        run.submit('archive', [], dict(after_verify=lv['job']))
            elif la is None or jtime(la) < jtime(lv):
                if self.cfg.get('archive_on_head', True):   # one at a time with the pulls (max_pulls)
                    pulls = sum(1 for r in self.runs.values() for j in self.derive(r)['active']
                                if j['job'].startswith('L'))
                    if pulls >= int(self.cfg.get('max_pulls', 1)):
                        return 0
                run.submit('archive', [], dict(after_verify=lv['job']))
        return 0

    def archive_batch(self, run, D):
        """While the run is computed: archive the kept particle files up to the newest snapshot kept in full whose
        outputs (and all earlier ones) are written (archive_run.py --particles-upto, a process on this node). The
        final archive after verify does the rest and skips what the batches did. Off with archive_batches: false in
        the config or ARCHIVE_BATCH=0 in run.conf."""
        E = run.E
        if (not E.get('ARCHIVE') or not self.cfg.get('archive_batches', True) or E.get('ARCHIVE_BATCH', '1') == '0'
                or run.ms.get('archb_hold') or not (run.remote or self.cfg.get('archive_on_head', True))):
            return
        v = D['verify']
        if v and int(v.get('ok', 0)) == 1:
            return                                   # the final archive takes over
        if any(j['stage'] in ('archive', 'archb') for j in D['active']):
            return
        ready, k = set(D['ready']), 0
        while k in ready:
            k += 1
        cand = [f for f in (run.plan or {}).get('full', []) if D['archb'] < f <= k - 1]
        if not cand:
            return
        pulls = sum(1 for r in self.runs.values() for j in self.derive(r)['active'] if j['job'].startswith('L'))
        if pulls >= int(self.cfg.get('max_pulls', 1)) or self.archive_wait(run):
            return
        run.submit('archb', [], dict(upto=max(cand)))

    def pbs_room(self, site, n=1):
        """May n more jobs wait in the site's PBS? (max_pbs_queued; no limit when unset)"""
        m = site_cfg(self.cfg, site).get('max_pbs_queued')
        return m is None or self.waiting.get(site, 0) + n <= int(m)

    def took(self, site, n=1):
        """Count ``n`` jobs just submitted on ``site`` as waiting, until the next qstat shows them."""
        self.waiting[site] = self.waiting.get(site, 0) + n

    def sim_counts(self, derived, site=LOCAL, q=None):
        """Number of running and of queued simulation jobs on ``site`` (optionally in queue ``q``)."""
        running = queued = 0
        for run in self.runs.values():
            if run.site != site or (q is not None and run_queue(run) != q):
                continue
            for j in derived[run.conf]['active']:
                if j['stage'] == 'sim':
                    if j['status'] in ('running', 'exiting'):
                        running += 1
                    else:
                        queued += 1
        return running, queued

    def schedule_sims(self, derived):
        """Submit scheduled simulations within each site's max_sim_running / max_sim_queued, by priority then age."""
        self.queue_view = []
        for site in [LOCAL] + self.sites():
            self.schedule_site(derived, site)

    def schedule_site(self, derived, site):
        """Submit scheduled simulations of ``site`` in priority order while the limits allow.

        Candidates are runs with ``ms['scheduled']`` that are not busy, paused, failed or in the trash, sorted by
        ``(priority, time scheduled)`` -- the lower priority number goes first. A run waits (reason shown on the page)
        if one of its simulation jobs is active, before ``not_before``, when the queue is closed, draining (only
        restarts) or at its ``max_running``, or when the site's ``max_sim_running`` / ``max_sim_queued`` is reached;
        once one run is blocked by a site limit, later candidates wait too, so the order is kept.
        """
        running, queued = self.sim_counts(derived, site)
        cand = [r for r in self.runs.values() if r.site == site and r.ms.get('scheduled')
                and not (r.busy or r.err or r.ms.get('paused') or r.ms.get('failed') or r.ms.get('trash'))]
        cand.sort(key=lambda r: (r.ms['scheduled'].get('priority', 0), r.ms['scheduled'].get('t', 0)))
        SC = site_cfg(self.cfg, site)
        mr, mq = int(SC.get('max_sim_running', 3)), int(SC.get('max_sim_queued', 1))
        blocked = None
        for run in cand:
            sc = run.ms['scheduled']
            D = derived[run.conf]
            wait = None
            qn, qi = run_queue(run), self.qinfo(run)
            if D['sim_done']:          # finished meanwhile (e.g. submitted by hand): nothing to submit
                run.ms['scheduled'] = None
                run.note('removed from the simulation queue: the simulation has finished')
                continue
            if getattr(run, 'stalled', None):
                wait = run.stalled
            elif any(j['stage'] == 'sim' for j in D['active']):
                wait = 'a simulation job of this run is still active'
            elif sc.get('not_before') and now() < sc['not_before']:
                wait = 'not before %s' % iso(sc['not_before'])
                self.recheck_at = min(self.recheck_at or 1e30, sc['not_before'] + 1)
            elif blocked:
                wait = blocked
            elif qi['state'] == 'closed':
                wait = 'queue %s is closed (queues file)' % qn
            elif qi['state'] == 'drain' and not sc.get('restart'):
                wait = 'queue %s is draining: only restarts of started runs (queues file)' % qn
            elif qi['max_running'] is not None and sum(self.sim_counts(derived, site, qn)) >= qi['max_running']:
                wait = 'queue %s: %d of its max_running %d' % (qn, sum(self.sim_counts(derived, site, qn)), qi['max_running'])
            elif qi['max_waiting'] is not None and self.sim_counts(derived, site, qn)[1] >= qi['max_waiting']:
                wait = 'queue %s: %d waiting in PBS (max_waiting %d)' % (qn, self.sim_counts(derived, site, qn)[1], qi['max_waiting'])
            elif not self.pbs_room(site):
                blocked = wait = 'site PBS queue limit: %d of our jobs waiting (max_pbs_queued %s)' % (
                    self.waiting.get(site, 0), site_cfg(self.cfg, site).get('max_pbs_queued'))
            elif running + queued >= mr or queued >= mq:
                blocked = wait = ('limits: %d running + %d waiting in PBS (max_sim_running %d, max_sim_queued %d)'
                                  % (running, queued, mr, mq))
            elif not sc.get('restart'):
                ok, msg = self.space_check(run)
                if not ok:
                    wait = msg
            if wait is None:
                try:
                    extra = ['--restart', 'latest'] if sc.get('restart') else []
                    run.submit('sim', extra, dict(restart=bool(sc.get('restart')), scheduled=True))
                    self.took(site)
                    if sc.get('restart') and not sc.get('maint'):     # a maintenance stop is not counted
                        run.ms['restarts'] = int(run.ms.get('restarts', 0)) + 1
                    run.ms['scheduled'] = None
                    run.save()
                    queued += 1
                    continue
                except MaintWait as e:
                    wait = str(e)
                except BudgetWait as e:
                    wait = 'core hours: %s' % e
                except Exception as e:  # noqa: BLE001
                    wait = 'submission failed: %s' % e
                    run.ms['failed'] = dict(stage='sim', msg=str(e), t=now())
                    run.note('scheduled submission failed: %s' % e)
            self.queue_view.append(dict(run=run.name, site=site, queue=qn, priority=sc.get('priority', 0), t=sc.get('t'), restart=sc.get('restart'),
                                        not_before=sc.get('not_before'), wait=wait))

    def acked(self, run, j):
        """True if job ``j`` of ``run`` was acknowledged on the page (its failure no longer alerts)."""
        return j['job'] in run.ms.get('acked', [])

    # ------------------------------------------------------------------ actions from the page
    def action(self, a):
        """Carry out an action from the page or the CLI (dict ``a`` with ``action`` and its arguments).

        Actions: add_run, unretire, refresh, set_queue, set_maint, set_limits, delete_preview, delete_run, remove_run,
        dismiss_alert, pause, resume, reload_conf, schedule, set_priority, unschedule, submit_sim (``restart``),
        retry, postprocess, ack, check_trash, approve_trash, undo_trash, approve_purge. Returns a message; raises
        ValueError for a refused action. Slow actions run in a worker thread (:meth:`slow_action`).
        """
        name, act = a.get('run'), a.get('action')
        if act == 'add_run':
            return self.add_run(a['conf'], a.get('paused', True))
        if act == 'unretire':
            if not (self.cfg.get('plan_retired') or {}).pop(a.get('name'), None):
                raise ValueError('%s is not retired' % a.get('name'))
            save_cfg(self.cfg)
            return 'unretired %s: its plan line is set up and scheduled again at the next plan check' % a['name']
        if act == 'refresh':
            self.next_hb = 0
            return 'refreshing'
        if act == 'set_queue':
            set_queue_state(self.cfg['queues_file'], a['site'], a['queue'], a['state'])
            self.load_queues()
            self.next_hb = 0
            return '%s:%s %s' % (a['site'], a['queue'], a['state'])
        if act == 'set_maint':      # add a window (start given) or remove all windows of the site (no start)
            site = a.get('site')
            if site not in [LOCAL] + list(self.cfg['sites']):
                raise ValueError('unknown site %s' % site)
            set_maint(self.cfg['maintenance_file'], site, a.get('start') or None, a.get('end') or None, a.get('note') or '')
            self.load_maint()
            self.next_hb = 0
            return ('%s: maintenance %s - %s' % (site, a['start'], a.get('end') or 'end not known')) if a.get('start') else (
                '%s: maintenance windows removed' % site)
        if act == 'queue_once':
            run = next((r for r in self.runs.values() if r.name == a.get('run') or r.conf == a.get('run')), None)
            if run is None:
                raise ValueError('no run %s' % a.get('run'))
            q = (a.get('queue') or '').strip()
            if not q:
                run.ms['queue_once'] = None
                run.note('one-time queue cleared')
                return 'cleared'
            if not re.match(r'^[\w.-]+$', q) or not re.match(r'^[\w.-]*$', a.get('project') or ''):
                raise ValueError('bad queue or project name')
            run.ms['queue_once'] = dict(queue=q, project=a.get('project') or '', t=now())
            run.note('next simulation job only: queue %s%s (then %s again)'
                     % (q, ' -P %s' % a['project'] if a.get('project') else '', run_queue(run)))
            return 'next simulation job of %s goes to %s' % (run.name, q)
        if act == 'read_budgets':
            self.budget_next = 0
            self.budget_ev.set()
            return 'reading the budgets (results in about a minute)'
        if act == 'probe_nodes':
            self.probe_next = 0
            self.probe_ev.set()
            return 'probing the nodes (results in about a minute)'
        if act == 'delete_leftover':
            return self.delete_leftover(a.get('host'), a.get('dir'), a.get('confirm'))
        if act == 'set_limits':
            changed = []
            for site, S in self.cfg['sites'].items():
                for k in SITE_LIMIT_KEYS:
                    key = '%s.%s' % (site, k)
                    if a.get(key) not in (None, ''):
                        v = int(float(a[key]))
                        if v < (0 if k in ('max_sim_queued', 'max_sim_restarts', 'scratch_reserve_gb') else 1):
                            raise ValueError('%s must be >= 0/1' % key)
                        if S.get(k) != v:
                            S[k] = v
                            changed.append('%s=%d' % (key, v))
            for k in LIMIT_KEYS:
                if a.get(k) not in (None, ''):
                    v = int(float(a[k]))
                    if v < (0 if k in ZERO_OK else 1):
                        raise ValueError('%s must be >= %d' % (k, 0 if k in ZERO_OK else 1))
                    if self.cfg.get(k) != v:
                        self.cfg[k] = v
                        changed.append('%s=%d' % (k, v))
            save_cfg(self.cfg)
            if any(c.startswith('budget_check_min=') for c in changed):
                self.budget_next = 0
                self.budget_ev.set()
            if any(c.startswith('node_probe_min=') for c in changed):
                self.probe_next = 0
                self.probe_ev.set()      # the probe thread picks up the new interval now
            log('limits: %s' % (', '.join(changed) or 'unchanged'))
            return 'set %s' % (', '.join(changed) or 'nothing changed')
        run = next((r for r in self.runs.values() if r.name == name or r.conf == name), None)
        if run is not None and getattr(run, 'stalled', None) and act != 'remove':
            raise ValueError('%s: %s; try again when it answers' % (run.name, run.stalled))
        if run is None:
            raise ValueError('no run %s' % name)
        if act == 'delete_preview':
            return self.delete_text(run, self.delete_plan(run))
        if act == 'delete_run':
            if a.get('confirm') != run.name:
                raise ValueError('not deleted: type the run name (%s) to confirm' % run.name)
            return self.delete_run(run)
        if act == 'remove_run':
            self.cfg['runs'] = [r for r in self.cfg['runs'] if r['conf'] != run.conf]
            self.retire(run.name, 'removed from the manager')
            save_cfg(self.cfg)
            self.load_runs()
            return 'removed %s from the manager (no files touched)' % run.name
        if act == 'dismiss_alert':
            k = a.get('key')
            al = getattr(run, 'alerts', {}).get(k)
            if not al:
                raise ValueError('no alert %s' % k)
            run.ms.setdefault('dismissed', {})[k] = al['t']
            run.alerts_shown = {x: v for x, v in getattr(run, 'alerts_shown', {}).items() if x != k}
            run.note('alert dismissed: %s' % al['msg'])
            return 'dismissed'
        if act == 'pause':
            run.ms['paused'] = True
            run.note('automation paused')
            return 'paused'
        if act == 'resume':
            run.ms['paused'] = False
            run.note('automation resumed')
            return 'resumed'
        if act == 'reload_conf':
            run.load()
            run.offsets, run.ev = {}, []
            run.read_events()
            return 'reloaded %s' % run.conf
        if act == 'schedule':
            D = self.derive(run)
            if D['sim_done']:
                raise ValueError('the simulation of %s has finished' % run.name)
            if any(j['stage'] == 'sim' for j in D['active']):
                raise ValueError('a simulation job of %s is queued or running; not scheduling another' % run.name)
            nb = a.get('not_before')
            if nb not in (None, ''):
                nb = float(nb) if re.match(r'^[0-9.]+$', str(nb)) else time.mktime(time.strptime(str(nb).replace('T', ' ')[:16], '%Y-%m-%d %H:%M'))
            else:
                nb = None
            if run.remote:     # checkpoints are not mirrored: a simulation that stopped at its wall time
                restart = bool(D['sim_clean_stop'])
            else:
                restart = bool(D['last'].get('sim')) and bool(glob.glob(os.path.join(run.P['sim'], 'checkpoints', '*.chk')))
            run.ms['scheduled'] = dict(t=now(), priority=int(a.get('priority') or 0), not_before=nb,
                                       restart=bool(a.get('restart', restart)))
            run.note('scheduled (priority %s%s%s)' % (run.ms['scheduled']['priority'], ', not before %s' % iso(nb) if nb else '',
                                                     ', restart from checkpoint' if run.ms['scheduled']['restart'] else ''))
            return 'scheduled %s' % run.name
        if act == 'set_priority':
            if not run.ms.get('scheduled'):
                raise ValueError('%s is not in the simulation queue' % run.name)
            run.ms['scheduled']['priority'] = int(a['priority'])
            run.note('priority set to %s (plan file)' % a['priority'])
            return 'priority %s' % a['priority']
        if act == 'unschedule':
            run.ms['scheduled'] = None
            run.note('removed from the simulation queue')
            return 'unscheduled %s' % run.name
        if act == 'submit_sim':
            D = self.derive(run)
            if any(j['stage'] == 'sim' for j in D['active']):
                raise ValueError('a simulation job of this run is already queued or running')
            if not a.get('restart'):
                ok, msg = self.space_check(run)
                if not ok and not a.get('force'):
                    raise ValueError('not submitted: ' + msg)
            extra = ['--restart', 'latest'] if a.get('restart') else []
            ids = run.submit('sim', extra, dict(restart=bool(a.get('restart'))))
            if run.ms.get('scheduled'):     # submitted by hand: the queue entry is served (the page showed it waiting)
                run.ms['scheduled'] = None
                run.note('removed from the simulation queue: simulation submitted by hand')
            return 'submitted %s' % ids
        if act == 'retry':
            f = run.ms.get('failed')
            if not f:
                raise ValueError('nothing failed')
            run.ms.setdefault('acked', [])
            if f.get('job'):
                run.ms['acked'].append(f['job'])
            # the other failed (or vanished) jobs of the same stage are superseded by this retry too: acknowledging
            # only the newest left an older one (e.g. a duplicate submission) to stop the run again right away
            for j in self.derive(run)['jobs']:
                if (j['stage'] == f.get('stage') and j['status'] in ('failed', 'gone')
                        and j['job'] not in run.ms['acked']):
                    run.ms['acked'].append(j['job'])
            run.ms['failed'] = None
            run.note('retry after failure of %s job %s (acknowledged)' % (f.get('stage'), f.get('job')))
            st = f.get('stage')
            if st == 'rockstar' and f.get('first') is not None:
                run.submit('rockstar', ['--rs-snaps', '%s-%s' % (f['first'], f['last'])],
                           dict(first=int(f['first']), last=int(f['last']), retry=f.get('job')))
            elif st in post_of(run.E) and site_cfg(self.cfg, run.site).get('max_pbs_queued') is not None:
                run.submit(st, [], dict(chain=st, sequential=True, retry=f.get('job')))     # the rest follows
            elif st in post_of(run.E):
                rest = post_of(run.E)
                stages = ','.join(rest[rest.index(st):])
                run.submit(stages, [], dict(chain=stages, retry=f.get('job')))
            elif st == 'archive':
                lv = self.derive(run)['last'].get('verify')
                run.submit('archive', [], dict(retry=f.get('job'), after_verify=lv['job'] if lv else None))
            elif st == 'sim':
                return 'acknowledged; use "restart simulation" to continue from the newest checkpoint'
            return 'resubmitted %s' % st
        if act == 'postprocess':
            # optional products (postprocess.sh): vpot -> fits -> verify; xcorr -> verify; all. Before the archive only:
            # they change kept files, and the archive is read-only once checked
            chains = dict(vpot='vpot,fits,verify', xcorr='xcorr,verify', vprof='vprof,verify',
                          all='vpot,fits,xcorr,vprof,verify')
            if run.E.get('PARQUET', '0') == '1':      # FITS written again: the Parquet copy too
                chains = {k: v.replace('fits,', 'fits,parquet,') for k, v in chains.items()}
            what = a.get('what')
            if what not in chains:
                raise ValueError('what must be one of %s' % ', '.join(chains))
            D = self.derive(run)
            if run.ms.get('trash') or run.ms.get('purged') or (D['archive'] and int(D['archive'].get('ok', 0))):
                raise ValueError('%s is archived: post-processing would change kept files of the read-only archive' % run.name)
            if D['active']:
                raise ValueError('jobs of %s are active (%s); try when they have finished'
                                 % (run.name, ', '.join('%s %s' % (j['stage'], j['job']) for j in D['active'])))
            if not D['last'].get('ctrees') or D['last']['ctrees']['status'] != 'done':
                raise ValueError('%s: the halo catalogues (ctrees) are not finished' % run.name)
            if what in ('xcorr', 'all') and not D['last'].get('fits'):
                raise ValueError('%s: no FITS tables yet (stage fits)' % run.name)
            ids = run.submit(chains[what], [], dict(chain=chains[what], postprocess=what))
            return 'submitted %s' % ', '.join('%s %s' % x for x in ids)
        if act == 'ack':
            f = run.ms.get('failed')
            if f and f.get('job'):
                run.ms.setdefault('acked', []).append(f['job'])
            run.ms['failed'] = None
            run.note('failure acknowledged without resubmission')
            return 'acknowledged'
        if act == 'check_trash':
            return self.cleanup_check(run)[1]
        if act == 'approve_trash':
            return self.to_trash(run)
        if act == 'undo_trash':
            return self.undo_trash(run)
        if act == 'approve_purge':
            return self.purge(run)
        raise ValueError('unknown action %s' % act)

    def outstanding(self, run, D=None):
        """Bytes a run still adds to SCRATCH: its estimate minus what it uses now, while its simulation is not done."""
        D = D or self.derive(run)
        if D['sim_done'] or run.ms.get('trash') or run.ms.get('purged') or not D['last'].get('sim'):
            return 0
        used = run.scratch_bytes if run.scratch_bytes is not None else 0
        return max(0, scratch_estimate(run.E, len(run.outs)) - used)

    def space_check(self, run):
        """(ok, message): does a new simulation of `run` fit in SCRATCH next to the running ones and the reserve?"""
        if run.remote:
            root = run.E['REMOTE_SCRATCH']
            if root not in self.site_disk:
                self.refresh_site_disk(run.site)
            if root not in self.site_disk:
                return False, 'free space of %s:%s unknown (scratch_ops.py usage failed)' % (run.site, root)
            free = self.site_disk[root]['free']
            others = sum(self.outstanding(r) for r in self.runs.values()
                         if r is not run and r.remote and r.E['REMOTE_SCRATCH'] == root)
        else:
            root = run.E['SCRATCH']
            free = shutil.disk_usage(root).free
            others = sum(self.outstanding(r) for r in self.runs.values() if r is not run and not r.remote and r.E['SCRATCH'] == root)
        need = scratch_estimate(run.E, len(run.outs))
        reserve = int(float(site_cfg(self.cfg, run.site).get('scratch_reserve_gb', 2000)) * 1e9)
        left = free - need - others - reserve
        msg = ('scratch %s: %s free; this run needs up to %s, running simulations %s more, reserve %s -> %s %s'
               % (root, retention.human(free), retention.human(need), retention.human(others), retention.human(reserve),
                  retention.human(abs(left)), 'left' if left >= 0 else 'SHORT'))
        return left >= 0, msg

    def add_run(self, conf, paused=True):
        """Register a run from ``conf`` (a local path, or ``site:/path`` for a run on another site).

        A remote run.conf is mirrored first. Refuses a missing file and a second run with the same name (names
        address runs on the page, in the CLI and in the plan file). Returns a message.
        """
        site, _, rconf = conf.partition(':') if re.match(r'^[A-Za-z]\w*:/', conf) else ('', '', '')
        if site:
            if site not in self.cfg['sites']:
                raise ValueError('unknown site %s (config sites: %s)' % (site, ', '.join(self.cfg['sites'])))
            conf = mirror_conf(self.cfg, site, rconf)
            E = retention.load_conf(conf)
            sync_site(self.cfg, site, [type('R', (), dict(E=E))()])     # output_steps.txt etc. before Run() reads them
        conf = os.path.abspath(conf)
        if not os.path.isfile(conf):
            raise ValueError('no file %s' % conf)
        new = Run(conf)    # loads without error
        same = [r for r in self.runs.values() if r.name == new.name and r.conf != conf]
        if same:           # run names are how the page, the CLI and the plan file address runs
            raise ValueError('a run named %s is already registered (%s); not adding %s' % (new.name, same[0].conf, conf))
        # the archive path is <ARCHIVE>/{sims,halos,state}/<stem>/r<real>, without the site: one per stem and realisation
        clash = [r for r in self.runs.values() if r.conf != conf and r.E.get('ARCHIVE') and
                 (r.E.get('ARCHIVE'), r.E['STEM'], r.E['REAL']) == (new.E.get('ARCHIVE'), new.E['STEM'], new.E['REAL'])]
        if clash:
            raise ValueError('%s would archive to the same place as %s (%s/.../%s/r%s); not adding it: use another '
                             'realisation number or ARCHIVE' % (new.name, clash[0].name, new.E['ARCHIVE'],
                                                                 new.E['STEM'], new.E['REAL']))
        if not any(r['conf'] == conf for r in self.cfg['runs']):
            self.cfg['runs'].append(dict(conf=conf, paused=paused, added=iso()))
            save_cfg(self.cfg)
        self.load_runs()
        return 'added %s' % conf

    # ------------------------------------------------------------------ deletion (after approval only)
    def cleanup_check(self, run):
        """(ok, message): may the run's scratch data be moved to the trash?"""
        if run.ms.get('purged'):
            return False, 'already purged: %s' % run.ms['purged'].get('dir')
        if run.ms.get('trash'):
            return False, 'already in the trash: %s' % run.ms['trash'].get('dir')
        E, P = run.E, run.P
        D = self.derive(run)
        if D['active']:
            return False, 'jobs of this run are queued or running: %s' % ', '.join(j['job'] for j in D['active'])
        v, a = D['verify'], D['archive']
        vf, mf = os.path.join(P['state'], 'verify.json'), os.path.join(P['state'], 'MANIFEST.json')
        # order by file times (one file server), not by event times (node clocks differ by minutes); a remote run:
        # the manifest must be of the newest verify (its time is in both)
        if run.remote:
            fresh = (os.path.isfile(vf) and os.path.isfile(mf)
                     and json.load(open(mf)).get('verify', {}).get('time') == json.load(open(vf)).get('time'))
        else:
            fresh = os.path.isfile(vf) and os.path.isfile(mf) and os.path.getmtime(mf) > os.path.getmtime(vf)
        if not (v and int(v.get('ok', 0)) and a and int(a.get('ok', 0)) and fresh):
            return False, 'needs a passed verify followed by a passed archive'
        if not E.get('ARCHIVE'):
            return False, 'ARCHIVE not set'
        man = os.path.join(E['ARCHIVE'], P['rel']['state'], 'MANIFEST.json')
        if not os.path.isfile(man):
            return False, 'no manifest %s' % man
        M = json.load(open(man))
        if not M.get('complete'):
            return False, 'manifest not complete'
        rows = {r['path']: r for r in M['files']}
        nbad, msgs = 0, []
        for r in M['files']:
            dst = os.path.join(E['ARCHIVE'], r['path'])
            if not os.path.isfile(dst) or os.path.getsize(dst) != r['size'] or r.get('sha256_dst') != r['sha256']:
                nbad += 1
                msgs.append('archive copy wrong or missing: %s' % r['path'])
        # every file the retention plan keeps now must be in the manifest, unchanged
        # (the cosmology files are shared by the realisations and rewritten by each set-up, with its time in a
        # comment: they count as unchanged when their content without comment lines equals the archived copy's)
        if run.remote:      # listed on the run's site
            st_out = self.remote_ops(run, 'stat')
            now_files = {rel: tuple(v) for rel, v in st_out['files'].items()}
            content = st_out.get('content', {})
        else:
            now_files, content = {}, {}
            for cat, _s, src, rel in retention.kept_files(E):
                if cat in ('log', 'events') or rel.endswith('verify.json'):
                    continue
                st = os.stat(src)
                now_files[rel] = (st.st_size, st.st_mtime)
                if cat == 'cosmo':
                    content[rel] = retention.content_sha(src)
        for rel, (size, mtime) in now_files.items():
            r = rows.get(rel)
            if r is None:
                nbad += 1
                msgs.append('kept file not in the manifest: %s' % rel)
            elif r['size'] != size or abs(r['mtime'] - mtime) > 1e-6:
                arch = os.path.join(E['ARCHIVE'], rel)
                if rel in content and os.path.isfile(arch) and retention.content_sha(arch) == content[rel]:
                    continue
                nbad += 1
                msgs.append('kept file changed after the archive: %s' % rel)
        if nbad:
            return False, '%d problem(s): %s' % (nbad, '; '.join(msgs[:10]))
        size = run.scratch_bytes
        scr = ('%s:%s' % (run.site, E['REMOTE_SCRATCH'])) if run.remote else E['SCRATCH']
        return True, ('OK: %d files in the archive (%s), checked; scratch %s/%s and %s/%s (%s) can go to the trash'
                      % (len(M['files']), retention.human(M['bytes']), scr, P['rel']['sim'], scr,
                         P['rel']['halo'], retention.human(size) if size is not None else '?'))

    def _guard(self, run, path, under):
        """Resolve ``path`` and refuse (ValueError) unless it lies strictly inside ``under`` and contains the
        run's name: a safety net for every move and delete.
        """
        rp, ru = os.path.realpath(path), os.path.realpath(under)
        if not rp.startswith(ru.rstrip('/') + '/') or rp == ru:
            raise ValueError('refusing: %s is not inside %s' % (rp, ru))
        if run.E['STEM'].split('/')[0] not in rp:
            raise ValueError('refusing: %s does not contain the run name' % rp)
        return rp

    def to_trash(self, run):
        """Move the run's scratch data to ``SCRATCH/.trash/<run>_<time>`` after ``cleanup_check`` passed.

        Only a move (undo possible until the purge); a remote run is moved on its site. Returns a message.
        """
        if run.ms.get('trash'):     # e.g. a second click: the run directory is gone, the check below cannot run
            raise ValueError('already in the trash: %s (undo, or approve the purge)' % run.ms['trash']['dir'])
        ok, msg = self.cleanup_check(run)
        if not ok:
            raise ValueError('not moved: ' + msg)
        E, P = run.E, run.P
        if run.remote:
            out = self.remote_ops(run, 'trash', time.strftime('%Y%m%dT%H%M%S'))
            run.ms['trash'] = dict(dir=out['dir'], moved=out['moved'], t=now(), check=msg, site=run.site)
            run.note('moved to trash on %s: %s: %s' % (run.site, out['dir'], ', '.join(m[0] for m in out['moved'])))
            return 'moved to %s:%s (undo possible until purged)' % (run.site, out['dir'])
        tdir = os.path.join(E['SCRATCH'], '.trash', '%s_r%s_%s' % (E['STEM'].replace('/', '_'), E['REAL'],
                                                                  time.strftime('%Y%m%dT%H%M%S')))
        os.makedirs(tdir)
        moved = []
        for k in ('sim', 'halo'):
            src = self._guard(run, P[k], E['SCRATCH'])
            if os.path.isdir(src):
                dst = os.path.join(tdir, k)
                os.rename(src, dst)
                moved.append([src, dst])
        run.ms['trash'] = dict(dir=tdir, moved=moved, t=now(), check=msg)
        run.note('moved to trash %s: %s' % (tdir, ', '.join(m[0] for m in moved)))
        return 'moved to %s (undo possible until purged)' % tdir

    def retire(self, name, why):
        """Remember a run taken out of the manager (removed or deleted): its plan line, if any, is not set up again."""
        self.cfg.setdefault('plan_retired', {})[name] = dict(t=iso(), why=why)

    # ------------------------------------------------------------------ delete a run completely (not its archive)
    def delete_plan(self, run):
        """What `delete_run` removes: on the run's site, its sim, halo and state directories and trash directories, and
        the set-up's shared directories when it is their last realisation (scratch_ops.delete_targets); for a run of
        another site also the mirror here. The archive is only looked at."""
        D = self.derive(run)
        if D['active']:
            raise ValueError('jobs of %s are queued or running (%s): stop them first'
                             % (run.name, ', '.join(j['job'] for j in D['active'])))
        E = run.E
        out = dict(site=run.site or LOCAL, remote=None, local=None, conf=None)
        if run.remote:
            base = dict(SCRATCH=E['REMOTE_SCRATCH'], STEM=E['STEM'], REAL=E['REAL'])
            out['remote'] = self.remote_ops(run, 'delete_plan', json.dumps(base))
        out['local'] = scratch_ops.delete_run(E, plan_only=True)     # the run's SCRATCH here (mirror of a remote run)
        if run.remote and os.path.isfile(run.conf):
            out['conf'] = run.conf                                     # mirror config, written by the manager
        # the archive: shown, never touched
        A = []
        if E.get('ARCHIVE'):
            P = retention.paths(E)
            for k in ('sim', 'halo', 'state'):
                d = P['archive'][k]
                A.append(dict(kind=k, path=d, exists=os.path.isdir(d)))
            man = os.path.join(P['archive']['state'], 'MANIFEST.json')
            if os.path.isfile(man):
                try:
                    M = json.load(open(man))
                    src = M.get('scratch')
                    mine = os.path.normpath(src or '') == os.path.normpath(E.get('REMOTE_SCRATCH') or E['SCRATCH'])
                    out['manifest'] = dict(path=man, files=len(M.get('files', [])), bytes=M.get('bytes'),
                                           complete=bool(M.get('complete')), time=M.get('time'), source=src, mine=mine)
                except ValueError:
                    out['manifest'] = dict(path=man, unreadable=True)
        out['archive'] = A
        return out

    def delete_text(self, run, pl):
        """Human-readable list of what ``delete_run`` would remove (paths and sizes from ``delete_plan``)."""
        hm = retention.human
        lines = ['Delete %s completely: remove it from the manager and delete from scratch:' % run.name]
        tot = 0
        for where, part in (('%s' % pl['site'], pl.get('remote')), ('this host' if pl.get('remote') else pl['site'], pl['local'])):
            if not part:
                continue
            for p, b in part['own']:
                lines.append('  %s:%s  (%s)' % (where, p, hm(b)))
                tot += b
            for p, b in part['shared']:
                lines.append('  %s:%s  (%s; shared set-up files, no other realisation left)' % (where, p, hm(b)))
                tot += b
        if pl.get('conf'):
            lines.append('  mirror config %s' % pl['conf'])
        lines.append('  total %s' % hm(tot))
        lines.append('The archive is NOT touched:')
        M = pl.get('manifest')
        if M and not M.get('unreadable'):
            lines.append('  archived %s: %d files, %s, %s%s' % (
                M['time'], M['files'], hm(M['bytes'] or 0), 'complete' if M['complete'] else 'INCOMPLETE',
                '' if M['mine'] else ' -- from ANOTHER run (%s)' % M['source']))
        else:
            lines.append('  no archive manifest: the data is NOT archived and will be gone')
        for a in pl.get('archive', []):
            lines.append('  %-5s %s%s' % (a['kind'], a['path'], '' if a['exists'] else '  (not there)'))
        return '\n'.join(lines)

    def delete_run(self, run):
        """Delete a run completely: its scratch data (here and on its site), its run.conf copies and its entry
        in the configuration; the run is retired so the plan file does not set it up again.
        """
        pl = self.delete_plan(run)           # checks again: no active job
        text = self.delete_text(run, pl)
        if run.remote:
            base = dict(SCRATCH=run.E['REMOTE_SCRATCH'], STEM=run.E['STEM'], REAL=run.E['REAL'])
            self.remote_ops(run, 'delete', json.dumps(base), timeout=7200)
        scratch_ops.delete_run(run.E, plan_only=False)
        if pl.get('conf') and os.path.isfile(pl['conf']):
            os.unlink(pl['conf'])
        saved = os.path.join(HOME, 'confs', run.conf.strip('/').replace('/', '__'))   # the manager's copy of run.conf
        if os.path.isfile(saved):
            os.unlink(saved)
        self.cfg['runs'] = [r for r in self.cfg['runs'] if r['conf'] != run.conf]
        self.retire(run.name, 'deleted (scratch data)')
        save_cfg(self.cfg)
        self.load_runs()
        log('deleted %s:\n%s' % (run.name, text))
        return 'deleted %s (archive untouched)\n%s' % (run.name, text)

    def undo_trash(self, run):
        """Move the run's data back from the trash (refused if the original path exists again)."""
        T = run.ms.get('trash')
        if not T or run.ms.get('purged'):
            raise ValueError('nothing in the trash')
        if run.remote:
            self.remote_ops(run, 'undo', T['dir'])
            run.ms['trash'] = None
            run.note('trash undone on %s: data back in place' % run.site)
            return 'restored'
        for src, dst in T['moved']:
            if os.path.exists(src):
                raise ValueError('%s exists again; not moving back' % src)
            os.rename(dst, src)
        try:
            os.rmdir(T['dir'])     # now empty
        except OSError:
            pass
        run.ms['trash'] = None
        try:                       # the retention plan reads output_steps.txt, which is back now
            run.plan, run.err = retention.plan(run.E), None
        except ValueError as e:
            run.plan, run.err = None, str(e)
        run.note('trash undone: data back in place')
        return 'restored'

    def purge(self, run):
        """Delete the run's trash directory for good, after re-checking every archive file against the manifest
        (present, same size). Requires the move to the trash first.
        """
        T = run.ms.get('trash')
        if not T:
            raise ValueError('nothing in the trash (approve the move to the trash first)')
        # the archive again, from the manifest (data now in the trash dir: the source check is skipped)
        E, P = run.E, run.P
        man = os.path.join(E['ARCHIVE'], P['rel']['state'], 'MANIFEST.json')
        M = json.load(open(man))
        for r in M['files']:
            dst = os.path.join(E['ARCHIVE'], r['path'])
            if not os.path.isfile(dst) or os.path.getsize(dst) != r['size']:
                raise ValueError('archive copy wrong or missing: %s; not purging' % r['path'])
        t0 = now()
        if run.remote:
            tdir = self.remote_ops(run, 'purge', T['dir'], timeout=7200)['dir']
            run.ms['purged'] = dict(t=now(), dir=tdir, seconds=round(now() - t0, 1), site=run.site)
            run.note('purged %s:%s' % (run.site, tdir))
            return 'purged %s:%s' % (run.site, tdir)
        tdir = self._guard(run, T['dir'], os.path.join(E['SCRATCH'], '.trash'))
        shutil.rmtree(tdir)
        run.ms['purged'] = dict(t=now(), dir=tdir, seconds=round(now() - t0, 1))
        run.note('purged %s' % tdir)
        return 'purged %s' % tdir

    # ------------------------------------------------------------------ state for the page
    def archive_pending(self, run, D):
        """(estimate, bytes still to come) of the run's archive: the estimate minus the particle batches archived
        already (archive_partial.json); nothing once archived, in the trash or purged."""
        est = archive_estimate(run.E, run.plan) if run.E.get('ARCHIVE') else None
        if est is None or (D['archive'] and int(D['archive'].get('ok', 0))) or run.ms.get('trash') or run.ms.get('purged'):
            return est, 0
        done = 0
        f = os.path.join(run.P['state'], 'archive_partial.json')
        try:
            done = sum(int(r.get('size', 0)) for r in json.load(open(f)).values())
        except (OSError, ValueError, AttributeError):
            pass
        return est, max(0, est - done)

    def build_snapshot(self, derived):
        """The state shown by the page (``/api/state``): one dict per run plus queues, disks, budgets, plan,
        alerts and the tail of the manager log.
        """
        runs = []
        for run in self.runs.values():
            D = derived.get(run.conf) or self.derive(run)
            pl = run.plan
            a_est, a_pend = self.archive_pending(run, D)
            runs.append(dict(
                name=run.name, conf=run.conf, site=run.site, remote_conf=run.E.get('REMOTE_CONF'), phase=run.phase(D), paused=run.ms.get('paused'), failed=run.ms.get('failed'), busy=run.busy,
                err=run.err, n_out=D['n_out'], ready=D['ready'], rs_done=D['rs_done'], archived_upto=D['archb'], sim_done=D['sim_done'],
                sim=D['sim_info'], jobs=D['jobs'][-40:], active=D['active'],
                verify=self.verify_summary(run, D), archive=D['archive'], fits=D['fits'],
                trash=run.ms.get('trash'), purged=run.ms.get('purged'), scheduled=run.ms.get('scheduled'),
                restarts=run.ms.get('restarts', 0), actions=run.ms.get('actions', [])[-30:],
                budget_msg=getattr(run, 'budget_msg', None) if getattr(run, 'budget_wait', None) else None,
                maint_msg=getattr(run, 'maint_msg', None), queue_once=run.ms.get('queue_once'),
                stalled=getattr(run, 'stalled', None), archive_msg=getattr(run, 'archive_msg', None),
                alerts=getattr(run, 'alerts_shown', {}), rs_redo=D.get('rs_redo', []), rewinds=D.get('rewinds', []),
                keep=dict(full=pl['full_z'] if pl else [], pct=pl['subsample_pct'] if pl else None,
                          tables=pl['tables'] if pl else []),
                outputs=[[o[0], o[1], o[3]] for o in run.outs], scratch_bytes=run.scratch_bytes,
                scratch_estimate=scratch_estimate(run.E, len(run.outs)), outstanding=self.outstanding(run, D),
                scratch=run.E.get('REMOTE_SCRATCH') or run.E.get('SCRATCH'), archive_root=run.E.get('ARCHIVE'), npart=run.E.get('NPART'),
                archive_estimate=a_est, archive_pending=a_pend,
                rs_batch=run.E.get('RS_BATCH', '0'), rs_batch_min=run.E.get('RS_BATCH_MIN', '10'),
                events=[dict(t=e['t'], stage=e['stage'], job=e['job'], ev=e['ev'],
                             info={k: v for k, v in e.items() if k not in ('t', 'time', 'run', 'stage', 'job', 'host', 'ev', 'conf')})
                        for e in run.ev[-60:]]))
        with self.lock:
            sr, sq = self.sim_counts(derived) if derived else (0, 0)
            site_info = {}
            for site in self.sites():
                jobs, qt, qok = self.qsite.get(site, ({}, 0.0, None))
                r_, q_ = self.sim_counts(derived, site) if derived else (0, 0)
                site_info[site] = dict(limits={k: self.cfg['sites'][site].get(k) for k in SITE_LIMIT_KEYS},
                                       sims_running=r_, sims_queued=q_, qtime=qt, qok=qok,
                                       login=self.cfg['sites'][site].get('login'), data=self.cfg['sites'][site].get('data'))
            self.snapshot = dict(t=now(), runs=runs, disk=self.disk, reserve_gb=self.cfg.get('scratch_reserve_gb'),
                                 limits={k: self.cfg.get(k) for k in LIMIT_KEYS}, sims_running=sr, sims_queued=sq,
                                 sites=site_info, local_site=LOCAL, budgets=BUDGETS.view(self.cfg), budget_check_min=self.cfg.get('budget_check_min', 30),
                                 budget_next=self.budget_next, plan=self.plan_view,
                                 plan_file=self.cfg.get('plan_file'), queues_file=self.cfg.get('queues_file'),
                                 queue_errors=self.queue_errors, queues=self.queues_view(derived),
                                 maintenance=self.maint_view(), maintenance_file=self.cfg.get('maintenance_file'),
                                 maintenance_errors=self.maint_errors, nodes=self.nodes_view(),
                                 node_probe_min=self.cfg.get('node_probe_min', 30), node_probe_next=self.probe_next,
                                 queue=getattr(self, 'queue_view', []), qtime=self.qtime, qok=self.qok,
                                 next_heartbeat=self.next_hb, heartbeat_s=self.cfg['heartbeat_s'],
                                 max_rockstar=self.cfg['max_rockstar'], host=socket.gethostname(), log=self.tail_log())

    def queues_view(self, derived):
        """Rows of the queue table: state, limits and running/waiting simulation counts per (site, queue),
        including the pseudo-queue ``archive`` of this host.
        """
        keys = set(self.queues) | {(r.site, run_queue(r)) for r in self.runs.values()} | {(LOCAL, 'archive')}
        out = []
        for site, qn in sorted(keys):
            qi = self.queues.get((site, qn)) or dict(state='open', max_running=None, max_waiting=None, note='not listed')
            if (site, qn) == (LOCAL, 'archive'):     # not a PBS queue: archive copies and pulls into ARCHIVE
                out.append(dict(site=site, queue=qn, state=qi['state'], max_running=None, max_waiting=None,
                                note=qi.get('note', ''), sims_running=0, sims_waiting=0, archive=True,
                                runs=sum(1 for r in self.runs.values() if getattr(r, 'archive_msg', None)), scheduled=0))
                continue
            r_, w_ = self.sim_counts(derived, site, qn) if derived else (0, 0)
            nrun = sum(1 for r in self.runs.values() if r.site == site and run_queue(r) == qn)
            out.append(dict(site=site, queue=qn, state=qi['state'], max_running=qi['max_running'],
                            max_waiting=qi['max_waiting'], note=qi.get('note', ''), sims_running=r_, sims_waiting=w_,
                            runs=nrun, scheduled=sum(1 for x in getattr(self, 'queue_view', [])
                                                     if x['site'] == site and x.get('queue') == qn)))
        return out

    def verify_summary(self, run, D):
        """Cached summary of the run's ``verify.json`` (kept while the run's storage is stalled)."""
        if getattr(run, 'stalled', None):
            return getattr(run, 'verify_cache', None)
        run.verify_cache = self._verify_summary(run)
        return run.verify_cache

    def _verify_summary(self, run):
        """Summary of ``verify.json``: overall ok, time, per-check ok/errors/warnings (first five) and the P(k)
        check's plot and ranges; None if missing or unreadable.
        """
        f = os.path.join(run.P['state'], 'verify.json')
        if not os.path.isfile(f):
            return None
        try:
            V = json.load(open(f))
        except ValueError:
            return None
        pk = V['checks'].get('pk', {}).get('info', {})
        return dict(ok=V['ok'], time=V['time'], checks={k: dict(ok=c['ok'], errors=c['errors'][:5],
                                                                 warnings=c['warnings'][:5]) for k, c in V['checks'].items()},
                    pk=dict(plot=pk.get('plot'), R_range=pk.get('R_range'), G_range=pk.get('G_range'), klin=pk.get('klin'),
                            ic_check=pk.get('ic_check'),
                            last=(pk.get('snapshots') or [None])[-1]) if pk else None)

    def tail_log(self):
        """The last 40 lines of the manager log."""
        try:
            with open(LOGF) as f:
                return f.readlines()[-40:]
        except OSError:
            return []

    def _disk_bg(self):
        """Thread: refresh the disk usage, then wake the main loop."""
        try:
            self.refresh_disk()
        except Exception as e:  # noqa: BLE001
            log('disk refresh: %s' % e)
        self._disk_busy = False
        self.q.put(('wake',))

    def slow_action(self, a, box):
        """Run a slow action (``SLOW``) in a worker thread so the main loop keeps going; ``box`` receives the
        result and its event is set when done. The run is marked busy meanwhile (a second action is refused).
        """
        run = next((r for r in self.runs.values() if r.name == a.get('run') or r.conf == a.get('run')), None)
        if run is None or run.busy:
            box['result'], box['ok'] = ('no run %s' % a.get('run')) if run is None else ('busy: %s' % run.busy), False
            box['ev'].set()
            return
        run.busy = a['action']

        def work():
            """Worker: run the action, store its result or error in ``box``, clear busy, wake the main loop."""
            try:
                box['result'] = self.action(a)
                box['ok'] = True
            except Exception as e:  # noqa: BLE001
                box['result'], box['ok'] = str(e), False
                log('action %s failed: %s' % (a, e))
            run.busy = None
            box['ev'].set()
            self.q.put(('wake',))
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ main loop
    def serve(self):
        """Main loop: start the unix-socket HTTP server and the sync and plan threads, then process file events,
        actions and heartbeats until SIGTERM / Ctrl-C.

        File events are collected in bursts; a change of ``config.json`` reloads the runs; actions are answered
        immediately (slow ones in a thread); every ``heartbeat_s`` (or at an earlier ``recheck_at``) :meth:`tick`
        runs and the snapshot is rebuilt.
        """
        try:
            os.unlink(SOCK)
        except FileNotFoundError:
            pass
        srv = UnixHTTPServer(SOCK, Handler)
        srv.mgr = self
        os.chmod(SOCK, 0o600)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        threading.Thread(target=self.sync_loop, daemon=True).start()
        threading.Thread(target=self.plan_loop, daemon=True).start()
        threading.Thread(target=self.probe_loop, daemon=True).start()
        threading.Thread(target=self.budget_loop, daemon=True).start()
        log('manager on %s, page on %s, %d runs' % (socket.gethostname(), SOCK, len(self.runs)))
        signal.signal(signal.SIGTERM, _sigterm)
        derived = {}
        try:
            while True:
                t = now()
                wake = min(self.next_hb, self.recheck_at or 1e30)
                try:
                    item = self.q.get(timeout=max(0.5, wake - t))
                except queue.Empty:
                    item = ('timer',)
                items = [item]
                if item[0] == 'file':          # collect a burst of file events
                    time.sleep(1.0)
                    while True:
                        try:
                            items.append(self.q.get_nowait())
                        except queue.Empty:
                            break
                changed_cfg = False
                for it in items:
                    if it[0] == 'file' and it[1] == CONFIG:
                        changed_cfg = True
                    elif it[0] == 'action' and it[1].get('action') in SLOW:
                        self.slow_action(it[1], it[2])
                    elif it[0] == 'action':
                        a, box = it[1], it[2]
                        try:
                            box['result'] = self.action(a)
                            box['ok'] = True
                        except Exception as e:  # noqa: BLE001
                            box['result'], box['ok'] = str(e), False
                            log('action %s failed: %s' % (a, e))
                        box['ev'].set()
                    elif it[0] == 'watch_ended':
                        log('inotifywait ended; restarting it')
                        self.watch_proc = None
                        self.restart_watch()
                if changed_cfg:
                    self.load_runs()
                for run in self.runs.values():
                    if self.run_ok(run):
                        run.read_events()
                t = now()
                if t >= self.next_hb or (self.recheck_at and t >= self.recheck_at):
                    self.recheck_at = None
                    if getattr(self, 'unloaded', None):
                        self.load_runs()
                    self.refresh_qstat()
                    for site in self.sites():
                        try:
                            self.refresh_qstat_site(site)
                        except Exception as e:  # noqa: BLE001
                            log('qstat %s: %s' % (site, e))
                    if not getattr(self, '_disk_busy', False):    # statfs/walks on NFS: in the background
                        self._disk_busy = True
                        threading.Thread(target=self._disk_bg, daemon=True).start()
                    self.next_hb = t + self.cfg['heartbeat_s']
                derived = self.tick()
                self.build_snapshot(derived)
        except KeyboardInterrupt:
            log('stopping')
        finally:
            if self.watch_proc:
                self.watch_proc.terminate()
            srv.shutdown()
            try:
                os.unlink(SOCK)
            except OSError:
                pass


def _sigterm(*_):
    """SIGTERM handler: turn the signal into KeyboardInterrupt so ``serve`` shuts down cleanly."""
    raise KeyboardInterrupt


class UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """Threaded HTTP server on a unix socket (the page is reached through an ssh-forwarded socket)."""
    daemon_threads = True

    def get_request(self):
        """Accept a connection; report a dummy client address (unix sockets have none)."""
        req, _ = super().get_request()
        return req, ('local', 0)


class Handler(http.server.BaseHTTPRequestHandler):
    """HTTP handler of the manager page: ``/`` (ui.html), ``/api/state`` (JSON snapshot), ``/api/file``
    (P(k) plots of registered runs) and POST ``/api/action`` (requires header ``X-Sahyadri: 1``).
    """
    def log_message(self, *a):
        """Silence the default per-request log lines."""
        pass

    def local_path(self, p):
        """A path on another site (events, verify.json) -> its copy in the manager's mirror."""
        for r in self.server.mgr.runs.values():
            rs = r.E.get('REMOTE_SCRATCH')
            if r.remote and rs and p.startswith(rs.rstrip('/') + '/'):
                return r.E['SCRATCH'].rstrip('/') + p[len(rs.rstrip('/')):]
        return p

    def _send(self, code, body, ctype='application/json'):
        """Send a complete response with ``Content-Type`` ``ctype`` and no caching."""
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        """Serve the page, the state snapshot, or a PNG that belongs to a registered run (path checked against
        the runs' scratch and archive roots).
        """
        mgr = self.server.mgr
        if self.path in ('/', '/index.html'):
            self._send(200, open(os.path.join(HERE, 'ui.html'), 'rb').read(), 'text/html; charset=utf-8')
        elif self.path.startswith('/api/state'):
            with mgr.lock:
                self._send(200, json.dumps(mgr.snapshot))
        elif self.path.startswith('/api/file?'):
            # an image of a registered run (the P(k) summary plot)
            from urllib.parse import parse_qs, urlparse
            p = self.local_path(parse_qs(urlparse(self.path).query).get('path', [''])[0])
            rp = os.path.realpath(p)
            okroots = [r.E['SCRATCH'] for r in mgr.runs.values()] + [r.E.get('ARCHIVE') for r in mgr.runs.values()]
            if (not rp.endswith('.png') or not os.path.isfile(rp)
                    or not any(rp.startswith(os.path.realpath(x) + '/') for x in okroots if x)):
                self._send(404, json.dumps(dict(error='not found')))
                return
            self._send(200, open(rp, 'rb').read(), 'image/png')
        elif self.path.startswith('/api/log?'):
            # tail of a job log (only files below a registered run's directories)
            from urllib.parse import parse_qs, urlparse
            p = self.local_path(parse_qs(urlparse(self.path).query).get('path', [''])[0])
            okroots = [r.E['SCRATCH'] for r in mgr.runs.values()] + [os.path.dirname(r.E.get('LOGS', '/x')) for r in mgr.runs.values()]
            rp = os.path.realpath(p)
            if not any(rp.startswith(os.path.realpath(x) + '/') for x in okroots if x) or not os.path.isfile(rp):
                self._send(404, json.dumps(dict(error='not found')))
                return
            with open(rp, 'rb') as f:
                f.seek(max(0, os.path.getsize(rp) - 20000))
                self._send(200, json.dumps(dict(path=rp, text=f.read().decode('utf-8', 'replace'))))
        else:
            self._send(404, json.dumps(dict(error='not found')))

    def do_POST(self):
        """Queue an action for the main loop and wait (up to 600 s) for its result; 403 without the
        ``X-Sahyadri`` header, 504 on timeout.
        """
        if self.path != '/api/action':
            self._send(404, '{}')
            return
        if self.headers.get('X-Sahyadri') != '1':     # only the page's own requests
            self._send(403, json.dumps(dict(ok=False, result='missing header')))
            return
        n = int(self.headers.get('Content-Length', 0))
        a = json.loads(self.rfile.read(n) or b'{}')
        box = dict(ev=threading.Event())
        self.server.mgr.q.put(('action', a, box))
        if not box['ev'].wait(600):
            self._send(504, json.dumps(dict(ok=False, result='timeout')))
            return
        self._send(200, json.dumps(dict(ok=box['ok'], result=box['result'])))


# ====================================================================================================== CLI
def call(action):
    """Send an action to the running manager (its socket), as the page does."""
    import http.client

    class C(http.client.HTTPConnection):
        """HTTPConnection over the manager's unix socket."""
        def connect(self):
            """Connect to ``SOCK`` instead of a TCP host."""
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(SOCK)
    c = C('localhost', timeout=900)
    c.request('POST', '/api/action', json.dumps(action), {'X-Sahyadri': '1', 'Content-Type': 'application/json'})
    r = json.loads(c.getresponse().read())
    print(r['result'])
    return 0 if r['ok'] else 1


def main():
    """Command line: ``serve``, ``add <conf>`` / ``add site:/conf``, and the CLI actions (see the module
    docstring for the full list).
    """
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == 'serve':
        Manager().serve()
    elif cmd == 'add' and re.match(r'^[A-Za-z]\w*:/', sys.argv[2]):
        sys.exit(call(dict(action='add_run', conf=sys.argv[2], paused='--paused' in sys.argv)))
    elif cmd == 'add':
        conf = os.path.abspath(sys.argv[2])
        Run(conf)
        c = load_cfg()
        if not any(r['conf'] == conf for r in c['runs']):
            c['runs'].append(dict(conf=conf, paused='--paused' in sys.argv, added=iso()))
            save_cfg(c)
        print('registered %s%s' % (conf, ' (paused)' if '--paused' in sys.argv else ''))
    elif cmd == 'unretire':
        sys.exit(call(dict(action='unretire', name=sys.argv[2])))
    elif cmd == 'delete':
        # preview; with --yes: delete (the run must be named exactly)
        if '--yes' in sys.argv:
            sys.exit(call(dict(action='delete_run', run=sys.argv[2], confirm=sys.argv[2])))
        rc = call(dict(action='delete_preview', run=sys.argv[2]))
        print('\nto delete: manager.py delete %s --yes' % sys.argv[2])
        sys.exit(rc)
    elif cmd == 'remove':
        c = load_cfg()
        keep = []
        for r in c['runs']:
            try:
                nm = Run(r['conf']).name
            except Exception:  # noqa: BLE001
                nm = r['conf']
            if sys.argv[2] not in (nm, r['conf']):
                keep.append(r)
            else:
                c.setdefault('plan_retired', {})[nm] = dict(t=iso(), why='removed from the manager')
        c['runs'] = keep
        save_cfg(c)
        print('%d runs registered' % len(keep))
    elif cmd in ('schedule', 'unschedule'):
        a = dict(action=cmd, run=sys.argv[2])
        if '--priority' in sys.argv:
            a['priority'] = sys.argv[sys.argv.index('--priority') + 1]
        if '--at' in sys.argv:
            a['not_before'] = sys.argv[sys.argv.index('--at') + 1]
        sys.exit(call(a))
    elif cmd == 'queue':
        if len(sys.argv) != 5:
            sys.exit('usage: manager.py queue <site> <queue> open|drain|closed')
        sys.exit(call(dict(action='set_queue', site=sys.argv[2], queue=sys.argv[3], state=sys.argv[4])))
    elif cmd == 'queue-once':
        if len(sys.argv) not in (3, 4, 5):
            sys.exit('usage: manager.py queue-once <stem/rN> <queue> [project]   |   manager.py queue-once <stem/rN> (clear)')
        sys.exit(call(dict(action='queue_once', run=sys.argv[2], queue=sys.argv[3] if len(sys.argv) > 3 else '',
                           project=sys.argv[4] if len(sys.argv) > 4 else '')))
    elif cmd == 'maint':
        if len(sys.argv) == 2:
            w, errs = read_maint(load_cfg().get('maintenance_file'))
            for site, ws in sorted(w.items()):
                for x in ws:
                    print(Manager.maint_label(site, x) + ('  # ' + x['note'] if x['note'] else ''))
            for e in errs:
                print('ERROR', e)
            sys.exit(0)
        if len(sys.argv) == 4 and sys.argv[3] == 'clear':
            sys.exit(call(dict(action='set_maint', site=sys.argv[2])))
        if len(sys.argv) < 5:
            sys.exit('usage: manager.py maint [<site> <start> <end|-> [note] | <site> clear]   (YYYY-MM-DDTHH:MM, local time)')
        sys.exit(call(dict(action='set_maint', site=sys.argv[2], start=sys.argv[3], end=sys.argv[4], note=' '.join(sys.argv[5:]))))
    elif cmd == 'limits':
        a = dict(action='set_limits')
        a.update(kv.split('=', 1) for kv in sys.argv[2:])
        sys.exit(call(a))
    elif cmd == 'list':
        for r in load_cfg()['runs']:
            print(r['conf'], '(paused at registration)' if r.get('paused') else '')
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
