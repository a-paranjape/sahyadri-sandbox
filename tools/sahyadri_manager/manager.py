#!/usr/bin/env python3
"""sahyadri_manager: runs the Sahyadri pipeline for many runs from the pawna head node, with a web page.

  manager.py serve                 the manager (in `screen`; start_manager.sh); page on ~/.sahyadri_mgr/gui.sock
  manager.py add <run.conf> [--paused]   register a run (the running manager picks it up at once)
  manager.py add <site>:<run.conf>       register a run computed on another site (pegasus): conf path there
  manager.py remove <stem/rN>      forget a run (nothing is deleted)
  manager.py list                  registered runs
  manager.py schedule <stem/rN> [--priority N] [--at "YYYY-MM-DD HH:MM"]   queue the simulation (lower priority first)
  manager.py unschedule <stem/rN>
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
  * then ctrees -> vahc -> fits -> verify (one chain, afterok), then archive (copy, check, read-only) automatically
  * a failed job stops the run's automation (shown in red; retry from the page)
  * deletion of the run's scratch data only after you approve it in the page, in two steps: move to
    <SCRATCH>/.trash/ (instant, can be undone), then purge; both re-check the archive manifest first

Runs on another site (cfg sites.<name>, e.g. pegasus; run.conf written there by setup_production.py --site pegasus):
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
                   max_sim_running=3, max_sim_queued=1, max_sim_restarts=5, max_pulls=1,
                   local_codes='/user/shadab/sahyadri_prod/codes',
                   # planned runs (production/planned_runs.txt): set up, registered and scheduled by the manager
                   plan_file=os.path.join(REPO, 'scripts', 'pkdgrav3_pipeline', 'production', 'planned_runs.txt'),
                   plan_s=300, local_python='/user/shadab/sahyadri_prod/codes/venv/bin/python',
                   scratch_root='/user/shadab/sahyadri_scratch',
                   sites=dict(pegasus=dict(login='calam.shadab@pegasus.iucaa.in', data='calam.shadab@data.iucaa.in',
                                           scratch_root='/data/project/hpc2502016/shadab/sahyadri_scratch',
                                           python='/mnt/home/project/calam.shadab/sahyadri_pipeline/codes/venv/bin/python',
                                           sandbox='/mnt/home/project/calam.shadab/sahyadri_pipeline/sahyadri-sandbox',
                                           mirror='/user/shadab/sahyadri_mirror/pegasus', sync_s=120,
                                           max_sim_running=2, max_sim_queued=1, max_sim_restarts=10, max_rockstar=1,
                                           scratch_reserve_gb=2000, max_pbs_queued=6,
                                           # core-hour budgets (amgr): queue PBS_PROJECT set -> the project's, else
                                           # the user's; a job is submitted only if cores x requested wall time fits
                                           # in the balance minus budget_reserve_coreh
                                           budget=dict(user='calam.shadab', reserve_coreh=2000))))
LIMIT_KEYS = ('max_sim_running', 'max_sim_queued', 'max_sim_restarts', 'max_rockstar', 'scratch_reserve_gb', 'heartbeat_s')
# max_pbs_queued: our jobs waiting in the site's PBS at most (pegasus queued_jobs_threshold: project 6, debug 2 per
# user); with it, the post-processing stages are submitted one at a time, each after the previous one finished
SITE_LIMIT_KEYS = ('max_sim_running', 'max_sim_queued', 'max_sim_restarts', 'max_rockstar', 'scratch_reserve_gb',
                   'max_pbs_queued')
POST = ['ctrees', 'vahc', 'fits', 'verify']
LOCAL = 'pawna'
SSH_OPTS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30', '-o', 'ServerAliveInterval=30',
            '-o', 'ControlMaster=auto', '-o', 'ControlPersist=600']


def scratch_estimate(E, n_out):
    """Bytes a run will occupy in SCRATCH at most (before its deletion is approved): every snapshot in full
    (10.5 B/particle measured with the potential at z <= 0.5: 128^3 and 512^3 runs), SIM_KEEPCHK checkpoints
    (48 B/particle), halo catalogues/trees/FITS (~0.5 of the snapshots at 64^3, a small fraction at large N;
    0.15 used), plus 10 %."""
    n3 = float(E.get('NPART', 0)) ** 3
    snaps = 10.5 * n3 * n_out
    chk = 48.0 * n3 * int(E.get('SIM_KEEPCHK', '2') or 2)
    return int(1.10 * (snaps * 1.15 + chk))
STAGES = ['sim', 'rockstar', 'ctrees', 'vahc', 'fits', 'verify', 'archive']
# actions that read the archive on /storage (NFS, can take minutes when it is busy): run in a worker thread, the
# run marked busy (no automation for it meanwhile), so events of the other runs are still handled
SLOW = ('check_trash', 'approve_trash', 'undo_trash', 'approve_purge')


def now():
    return time.time()


def iso(t=None):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t or now()))


def log(msg):
    line = '[%s] %s' % (iso(), msg)
    print(line, flush=True)
    try:
        with open(LOGF, 'a') as f:
            f.write(line + '\n')
    except OSError:
        pass


def load_cfg():
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
    h, m, sec = (list(map(int, str(wall).split(':'))) + [0, 0])[:3]
    return h + m / 60.0 + sec / 3600.0


def job_coreh(E, stage):
    """Core hours PBS reserves for one job of `stage` (cores x requested wall time), from the run config."""
    n = lambda k, d: int(float(E.get(k) or d))     # noqa: E731
    if stage == 'sim':
        return n('SIM_NODES', 1) * n('SIM_NCPU', 1) * hours(E.get('SIM_WALL', '0'))
    pre = dict(rockstar='RS', ctrees='CT', vahc='PP', vpot='PP', fits='FITS', verify='VF', xcorr='XC', archive='AR')[stage]
    dflt = dict(VF=(8, '12:00:00'), XC=(16, '02:00:00'), AR=(2, '72:00:00')).get(pre, (1, '0'))
    return n(pre + '_NCPU', dflt[0]) * hours(E.get(pre + '_WALL') or dflt[1])


class Budgets:
    """amgr balances of the sites' budgets (core hours), cached; debited locally by what was just submitted."""
    TTL = 300

    def __init__(self):
        self.cache = {}           # (site, kind, name, period) -> [t, balance]
        self.lock = threading.Lock()

    @staticmethod
    def which(cfg, run):
        B = cfg['sites'][run.site].get('budget')
        if not B:
            return None
        proj = run.E.get('PBS_PROJECT') or ''
        t = time.localtime()
        period = '%d.h%d' % (t.tm_year, 1 if t.tm_mon <= 6 else 2)
        return (run.site, 'project', proj, period) if proj else (run.site, 'user', B['user'], '')

    def balance(self, cfg, key, fresh=False):
        with self.lock:
            c = self.cache.get(key)
            if c and not fresh and now() - c[0] < self.TTL:
                return c[1]
        site, kind, name, period = key
        cmd = 'amgr login >/dev/null 2>&1; amgr checkbalance %s -n %s%s' % (kind, shlex.quote(name),
                                                                           ' -p %s' % period if period else '')
        r = ssh(cfg['sites'][site]['login'], cmd, timeout=120)
        m = re.search(r"'cost':\s*([-0-9.eE+]+)", r.stdout)
        if not m:
            raise RuntimeError('amgr balance %s: %s %s' % (key, r.stdout.strip()[-200:], r.stderr.strip()[-200:]))
        with self.lock:
            self.cache[key] = [now(), float(m.group(1))]
        return float(m.group(1))

    def check(self, cfg, run, stages):
        key = self.which(cfg, run)
        if key is None:
            return
        need = sum(job_coreh(run.E, st) for st in stages.split(','))
        reserve = float(cfg['sites'][run.site]['budget'].get('reserve_coreh', 0))
        left = self.balance(cfg, key)
        if left - reserve < need:
            left = self.balance(cfg, key, fresh=True)
        if left - reserve < need:
            raise BudgetWait('%s budget %s%s: %.0f core-h left, reserve %.0f, this submission (%s) reserves %.0f'
                             % (run.site, key[2], ' ' + key[3] if key[3] else '', left, reserve, stages, need))
        with self.lock:
            self.cache[key][1] -= need

    def view(self):
        with self.lock:
            return [dict(site=k[0], kind=k[1], name=k[2], period=k[3], coreh=v[1], t=v[0]) for k, v in self.cache.items()]


BUDGETS = Budgets()


def fmt_l(L):
    L = float(L)
    return ('%d' % L) if L.is_integer() else ('%g' % L).replace('.', 'p')


def reals_of(spec):
    out = []
    for part in str(spec).split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


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
                    'sims/%s/r%s/run.*.log' % (st, r), 'sims/%s/r%s/logs' % (st, r), 'halos/%s/r%s/pk' % (st, r)):
            srcs.append('%s/./%s' % (rs, rel))
    os.makedirs(S['mirror'], exist_ok=True)
    cmd = (['rsync', '-rtR', '--partial', '-e', 'ssh ' + ' '.join(SSH_OPTS + ['-o', 'ControlPath=%s/ssh-%%r@%%h' % HOME])]
           + ['%s:%s' % (S['data'], srcs[0])] + [':%s' % x for x in srcs[1:]] + [S['mirror'].rstrip('/') + '/'])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode not in (0, 23, 24):
        log('sync %s: rsync rc %d: %s' % (site, r.returncode, r.stderr.strip()[-300:]))
    return r.returncode


def save_cfg(c):
    os.makedirs(HOME, mode=0o700, exist_ok=True)
    tmp = CONFIG + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(c, f, indent=1)
    os.replace(tmp, CONFIG)


def atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def jobnum(j):
    """PBS job number; local processes (archive pulls) are 'L<epoch>'."""
    try:
        return int(str(j).split('.')[0].lstrip('L'))
    except ValueError:
        return -1


def short(job):
    return str(job).split('.')[0]


# ====================================================================================================== one run
class Run:
    CFG = None          # the manager's configuration (sites), set by Manager.load_runs

    def __init__(self, conf, paused=False):
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
        atomic_json(self.msfile, self.ms)

    def note(self, msg):
        log('%s: %s' % (self.name, msg))
        self.ms['actions'] = (self.ms.get('actions', []) + [[now(), msg]])[-200:]
        self.save()

    # ------------------------------------------------------------------ events
    def read_events(self):
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
        D = dict(name=self.name, conf=self.conf, n_out=len(self.outs), err=self.err)
        J = self.jobs(qjobs, qtime)
        D['jobs'] = sorted(J.values(), key=lambda j: jobnum(j['job']))
        ready = set()
        for e in self.ev:
            if e['ev'] == 'snap_ready' and e.get('step') in self.step_index:
                ready.add(self.step_index[e['step']])
        D['ready'] = sorted(ready)
        D['sim_done'] = any(e['ev'] == 'sim_outputs' and int(e.get('ndone', -1)) == int(e.get('nout', -2))
                            for e in self.ev)
        cov = set()
        for e in self.ev:
            if e['ev'] == 'rockstar_done' and int(e.get('ok', 0)) == 1:
                cov.update(range(int(e['first']), int(e['last']) + 1))
        D['rs_done'] = sorted(cov)
        last = {}
        for j in D['jobs']:
            last[j['stage']] = j
        D['last'] = last
        D['active'] = [j for j in D['jobs'] if j['status'] in ('submitted', 'queued', 'held', 'running', 'exiting')]

        def ok_after(evname):   # the event of the newest job
            es = sorted((e for e in self.ev if e['ev'] == evname), key=lambda e: jobnum(e['job']))
            return es[-1] if es else None
        # sim jobs that stopped cleanly before the last output (pkdgrav3 iWallRunTime / STOP: rc 0, outputs missing)
        D['sim_clean_stop'] = {e['job'] for e in self.ev if e['ev'] == 'sim_outputs' and int(e.get('rc', 1)) == 0
                               and int(e.get('ndone', 0)) < int(e.get('nout', 0))}
        D['verify'] = ok_after('verify_done')
        D['archive'] = ok_after('archive_done')
        D['kept'] = ok_after('kept_sums')
        D['fits'] = ok_after('fits_done')
        D['sim_info'] = self.sim_progress()
        return D

    def sim_progress(self):
        t, v = self.simcache
        if now() - t < 60:
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
        for st in ('archive', 'verify', 'fits', 'vahc', 'ctrees'):
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
        if self.remote and stages == 'archive':
            return self.start_pull(what)
        if self.remote:     # qsub on the site's login node, with the run's own run.conf there
            BUDGETS.check(cfg or Run.CFG, self, stages)
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
        self.save()
        self.note('submitted %s' % ', '.join('%s %s' % x for x in ids) + (' (%s)' % what if what else ''))
        return ids

    def start_pull(self, what=None):
        """Archive of a remote run: job_archive.pbs as a process on this host (archive_run.py pulls from PULL_HOST)."""
        jid = 'L%d' % int(now())
        pipe = os.path.join(REPO, 'scripts', 'pkdgrav3_pipeline')
        env = dict(os.environ, CONF=self.conf, PIPE_ENV=os.path.join(pipe, 'pipeline_env.sh'), STAGE='archive',
                   PBS_JOBID=jid, PATH=PBS + ':' + os.environ.get('PATH', ''))
        p = subprocess.Popen(['bash', os.path.join(pipe, 'job_archive.pbs')], env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        rec = dict(job=jid, stage='archive', t=now(), pid=p.pid)
        if what:
            rec.update(what)
        self.ms['submitted'].append(rec)
        self.save()
        self.note('archive pull %s started on %s (pid %d) from %s' % (jid, socket.gethostname(), p.pid, self.E['PULL_HOST']))
        threading.Thread(target=p.wait, daemon=True).start()     # reap it
        return [('archive', jid)]


# ====================================================================================================== manager
class Manager:
    def __init__(self):
        os.makedirs(HOME, mode=0o700, exist_ok=True)
        os.chmod(HOME, 0o700)
        self.cfg = load_cfg()
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
        self.cfg = load_cfg()
        Run.CFG = self.cfg
        want = {r['conf']: r for r in self.cfg['runs']}
        for conf in list(self.runs):
            if conf not in want:
                del self.runs[conf]
        for conf, r in want.items():
            if conf not in self.runs:
                try:
                    if os.path.dirname(os.path.dirname(conf)).endswith(tuple(S['mirror'] for S in self.cfg['sites'].values())):
                        refresh_mirror_conf(conf, self.cfg)
                    run = Run(conf, paused=r.get('paused', False))
                    self.runs[conf] = run
                    run.read_events()
                    log('run %s (%s): %d events' % (run.name, conf, len(run.ev)))
                except Exception as e:  # noqa: BLE001
                    log('cannot load %s: %s' % (conf, e))
        self.restart_watch()

    def restart_watch(self):
        dirs = sorted([HOME] + [r.P['state'] for r in self.runs.values()])
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
        for line in proc.stdout:
            self.q.put(('file', line.strip()))
        self.q.put(('watch_ended', proc))

    def derive(self, run):
        if run.remote:
            jobs, qt, _ok = self.qsite.get(run.site, ({}, 0.0, None))
            return run.derive(jobs, qt)
        return run.derive(self.qjobs, self.qtime)

    def sites(self):
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
        for k, flag in (('archive', '--archive'), ('keep_full_z', '--keep-full-z'), ('keep_pct', '--keep-pct')):
            if k in o:
                args += [flag, str(o[k])]
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
            if run is None and key in self.plan_errors:
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
        d, seen = {}, {}           # one entry per file system and kind (several scratch roots may share one)
        for run in self.runs.values():
            for k in (('ARCHIVE',) if run.remote else ('SCRATCH', 'ARCHIVE')):
                p = run.E.get(k)
                if p and p not in d and os.path.isdir(p):
                    key = (os.stat(p).st_dev, k)
                    if key in seen:
                        if p not in seen[key]['paths']:
                            seen[key]['paths'].append(p)
                        continue
                    u = shutil.disk_usage(p)
                    d[p] = seen[key] = dict(total=u.total, used=u.used, free=u.free, kind=k, paths=[p])
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
        active_rs = {}             # per site
        derived = {}
        self.waiting = {}          # per site: our jobs submitted but not started (PBS queue limits)
        for run in self.runs.values():
            D = self.derive(run)
            derived[run.conf] = D
            active_rs[run.site] = active_rs.get(run.site, 0) + sum(1 for j in D['active'] if j['stage'] == 'rockstar')
            self.waiting[run.site] = self.waiting.get(run.site, 0) + sum(
                1 for j in D['active'] if j['status'] in ('submitted', 'queued', 'held') and not j['job'].startswith('L'))
        self.schedule_sims(derived)
        # Rockstar turn order over the runs: the one whose next snapshot has waited longest goes first
        def waiting(run):
            D = derived[run.conf]
            todo = set(range(len(run.outs))) - set(D['rs_done'])
            return run.seen_ready.get(min(todo), float('inf')) if todo else float('inf')
        for run in sorted(self.runs.values(), key=waiting):
            D = derived[run.conf]
            try:
                active_rs[run.site] = active_rs.get(run.site, 0) + self.policy(run, D, active_rs.get(run.site, 0))
                run.budget_wait = None
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

    def policy(self, run, D, active_rs):
        """Submit what is due for this run; returns the number of Rockstar jobs submitted."""
        if run.busy or run.err or run.ms.get('paused') or run.ms.get('failed') or run.ms.get('trash') or run.ms.get('purged'):
            return 0
        E = run.E
        SC = site_cfg(self.cfg, run.site)
        qtime = self.qsite.get(run.site, ({}, 0.0))[1] if run.remote else self.qtime
        # failures and vanished jobs
        for j in D['jobs']:
            if (j['status'] == 'failed' and j['stage'] == 'sim' and j['job'] in D['sim_clean_stop']
                    and not self.acked(run, j)):
                n = int(run.ms.get('restarts', 0))
                run.ms.setdefault('acked', []).append(j['job'])
                if n >= int(SC.get('max_sim_restarts', 5)):
                    run.ms['failed'] = dict(stage='sim', job=j['job'], rc=j.get('rc'), log=j.get('log'), t=now(),
                                            msg='stopped at its wall time again after %d restarts (max_sim_restarts)' % n)
                    run.note('simulation job %s stopped at its wall time; %d restarts done, not restarting' % (j['job'], n))
                    return 0
                run.ms['scheduled'] = dict(t=now(), priority=-1, restart=True, after=j['job'])
                run.note('simulation job %s stopped cleanly before the last output (wall time); scheduled to restart '
                         'from the newest checkpoint (restart %d of at most %d)' % (j['job'], n + 1, SC.get('max_sim_restarts', 5)))
                continue
            if j['status'] == 'failed' and not self.acked(run, j):
                run.ms['failed'] = dict(stage=j['stage'], job=j['job'], rc=j.get('rc'), log=j.get('log'), t=now(),
                                        first=j.get('first'), last=j.get('last'))
                run.note('%s job %s failed (rc %s); automation stopped' % (j['stage'], j['job'], j.get('rc')))
                return 0
            if j['status'] == 'gone' and not self.acked(run, j):
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
        if D['active'] and any(j['stage'] in ('ctrees', 'vahc', 'fits', 'verify', 'archive') for j in D['active']):
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
        if site_cfg(self.cfg, run.site).get('max_pbs_queued') is not None:
            # one stage at a time: the first whose newest job is not a success newer than the previous stage's
            prev = None
            for st in POST:
                j = D['last'].get(st)
                if j is None or j['status'] != 'done' or (prev and jobnum(j['job']) < jobnum(prev['job'])):
                    if self.pbs_room(run.site):
                        run.submit(st, [], dict(chain=st, sequential=True))
                        self.took(run.site)
                    return 0
                prev = j
        elif not any(D['last'].get(s) for s in POST):
            stages = 'ctrees,vahc,fits,verify'
            run.submit(stages, [], dict(chain=stages))
            return 0
        v = D['verify']
        if lv and lv['status'] == 'done' and v and int(v.get('ok', 0)) == 1 and E.get('ARCHIVE'):
            la = D['last'].get('archive')   # one archive job per passed verify (PBS ids increase; node clocks differ)
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
            elif la is None or jobnum(la['job']) < jobnum(lv['job']):
                run.submit('archive', [], dict(after_verify=lv['job']))
        return 0

    def pbs_room(self, site, n=1):
        """May n more jobs wait in the site's PBS? (max_pbs_queued; no limit when unset)"""
        m = site_cfg(self.cfg, site).get('max_pbs_queued')
        return m is None or self.waiting.get(site, 0) + n <= int(m)

    def took(self, site, n=1):
        self.waiting[site] = self.waiting.get(site, 0) + n

    def sim_counts(self, derived, site=LOCAL):
        running = queued = 0
        for run in self.runs.values():
            if run.site != site:
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
            if D['sim_done']:          # finished meanwhile (e.g. submitted by hand): nothing to submit
                run.ms['scheduled'] = None
                run.note('removed from the simulation queue: the simulation has finished')
                continue
            if any(j['stage'] == 'sim' for j in D['active']):
                wait = 'a simulation job of this run is still active'
            elif sc.get('not_before') and now() < sc['not_before']:
                wait = 'not before %s' % iso(sc['not_before'])
                self.recheck_at = min(self.recheck_at or 1e30, sc['not_before'] + 1)
            elif blocked:
                wait = blocked
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
                    if sc.get('restart'):
                        run.ms['restarts'] = int(run.ms.get('restarts', 0)) + 1
                    run.ms['scheduled'] = None
                    run.save()
                    queued += 1
                    continue
                except BudgetWait as e:
                    wait = 'core hours: %s' % e
                except Exception as e:  # noqa: BLE001
                    wait = 'submission failed: %s' % e
                    run.ms['failed'] = dict(stage='sim', msg=str(e), t=now())
                    run.note('scheduled submission failed: %s' % e)
            self.queue_view.append(dict(run=run.name, site=site, priority=sc.get('priority', 0), t=sc.get('t'), restart=sc.get('restart'),
                                        not_before=sc.get('not_before'), wait=wait))

    def acked(self, run, j):
        return j['job'] in run.ms.get('acked', [])

    # ------------------------------------------------------------------ actions from the page
    def action(self, a):
        name, act = a.get('run'), a.get('action')
        if act == 'add_run':
            return self.add_run(a['conf'], a.get('paused', True))
        if act == 'refresh':
            self.next_hb = 0
            return 'refreshing'
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
                    if v < (0 if k in ('max_sim_queued', 'max_sim_restarts', 'scratch_reserve_gb') else 1):
                        raise ValueError('%s must be >= %d' % (k, 0 if k in ('max_sim_queued', 'max_sim_restarts', 'scratch_reserve_gb') else 1))
                    if self.cfg.get(k) != v:
                        self.cfg[k] = v
                        changed.append('%s=%d' % (k, v))
            save_cfg(self.cfg)
            log('limits: %s' % (', '.join(changed) or 'unchanged'))
            return 'set %s' % (', '.join(changed) or 'nothing changed')
        run = next((r for r in self.runs.values() if r.name == name or r.conf == name), None)
        if run is None:
            raise ValueError('no run %s' % name)
        if act == 'remove_run':
            self.cfg['runs'] = [r for r in self.cfg['runs'] if r['conf'] != run.conf]
            save_cfg(self.cfg)
            self.load_runs()
            return 'removed %s from the manager (no files touched)' % run.name
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
            return 'submitted %s' % ids
        if act == 'retry':
            f = run.ms.get('failed')
            if not f:
                raise ValueError('nothing failed')
            run.ms.setdefault('acked', [])
            if f.get('job'):
                run.ms['acked'].append(f['job'])
            run.ms['failed'] = None
            run.note('retry after failure of %s job %s (acknowledged)' % (f.get('stage'), f.get('job')))
            st = f.get('stage')
            if st == 'rockstar' and f.get('first') is not None:
                run.submit('rockstar', ['--rs-snaps', '%s-%s' % (f['first'], f['last'])],
                           dict(first=int(f['first']), last=int(f['last']), retry=f.get('job')))
            elif st in POST and site_cfg(self.cfg, run.site).get('max_pbs_queued') is not None:
                run.submit(st, [], dict(chain=st, sequential=True, retry=f.get('job')))     # the rest follows
            elif st in POST:
                rest = POST
                stages = ','.join(rest[rest.index(st):])
                run.submit(stages, [], dict(chain=stages, retry=f.get('job')))
            elif st == 'archive':
                run.submit('archive', [], dict(retry=f.get('job')))
            elif st == 'sim':
                return 'acknowledged; use "restart simulation" to continue from the newest checkpoint'
            return 'resubmitted %s' % st
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
        Run(conf)    # loads without error
        if not any(r['conf'] == conf for r in self.cfg['runs']):
            self.cfg['runs'].append(dict(conf=conf, paused=paused, added=iso()))
            save_cfg(self.cfg)
        self.load_runs()
        return 'added %s' % conf

    # ------------------------------------------------------------------ deletion (after approval only)
    def cleanup_check(self, run):
        """(ok, message): may the run's scratch data be moved to the trash?"""
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
        if run.remote:      # listed on the run's site
            now_files = {rel: tuple(v) for rel, v in self.remote_ops(run, 'stat')['files'].items()}
        else:
            now_files = {}
            for cat, _s, src, rel in retention.kept_files(E):
                if cat in ('log', 'events') or rel.endswith('verify.json'):
                    continue
                st = os.stat(src)
                now_files[rel] = (st.st_size, st.st_mtime)
        for rel, (size, mtime) in now_files.items():
            r = rows.get(rel)
            if r is None:
                nbad += 1
                msgs.append('kept file not in the manifest: %s' % rel)
            elif r['size'] != size or abs(r['mtime'] - mtime) > 1e-6:
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
        rp, ru = os.path.realpath(path), os.path.realpath(under)
        if not rp.startswith(ru.rstrip('/') + '/') or rp == ru:
            raise ValueError('refusing: %s is not inside %s' % (rp, ru))
        if run.E['STEM'].split('/')[0] not in rp:
            raise ValueError('refusing: %s does not contain the run name' % rp)
        return rp

    def to_trash(self, run):
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

    def undo_trash(self, run):
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
    def build_snapshot(self, derived):
        runs = []
        for run in self.runs.values():
            D = derived.get(run.conf) or self.derive(run)
            pl = run.plan
            runs.append(dict(
                name=run.name, conf=run.conf, site=run.site, remote_conf=run.E.get('REMOTE_CONF'), phase=run.phase(D), paused=run.ms.get('paused'), failed=run.ms.get('failed'), busy=run.busy,
                err=run.err, n_out=D['n_out'], ready=D['ready'], rs_done=D['rs_done'], sim_done=D['sim_done'],
                sim=D['sim_info'], jobs=D['jobs'][-40:], active=D['active'],
                verify=self.verify_summary(run, D), archive=D['archive'], fits=D['fits'],
                trash=run.ms.get('trash'), purged=run.ms.get('purged'), scheduled=run.ms.get('scheduled'),
                restarts=run.ms.get('restarts', 0), actions=run.ms.get('actions', [])[-30:],
                budget_msg=getattr(run, 'budget_msg', None) if getattr(run, 'budget_wait', None) else None,
                keep=dict(full=pl['full_z'] if pl else [], pct=pl['subsample_pct'] if pl else None,
                          tables=pl['tables'] if pl else []),
                outputs=[[o[0], o[1], o[3]] for o in run.outs], scratch_bytes=run.scratch_bytes,
                scratch_estimate=scratch_estimate(run.E, len(run.outs)), outstanding=self.outstanding(run, D),
                scratch=run.E.get('REMOTE_SCRATCH') or run.E.get('SCRATCH'), archive_root=run.E.get('ARCHIVE'), npart=run.E.get('NPART'),
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
                                 sites=site_info, local_site=LOCAL, budgets=BUDGETS.view(), plan=self.plan_view,
                                 plan_file=self.cfg.get('plan_file'),
                                 queue=getattr(self, 'queue_view', []), qtime=self.qtime, qok=self.qok,
                                 next_heartbeat=self.next_hb, heartbeat_s=self.cfg['heartbeat_s'],
                                 max_rockstar=self.cfg['max_rockstar'], host=socket.gethostname(), log=self.tail_log())

    def verify_summary(self, run, D):
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
        try:
            with open(LOGF) as f:
                return f.readlines()[-40:]
        except OSError:
            return []

    def _disk_bg(self):
        try:
            self.refresh_disk()
        except Exception as e:  # noqa: BLE001
            log('disk refresh: %s' % e)
        self._disk_busy = False
        self.q.put(('wake',))

    def slow_action(self, a, box):
        run = next((r for r in self.runs.values() if r.name == a.get('run') or r.conf == a.get('run')), None)
        if run is None or run.busy:
            box['result'], box['ok'] = ('no run %s' % a.get('run')) if run is None else ('busy: %s' % run.busy), False
            box['ev'].set()
            return
        run.busy = a['action']

        def work():
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
                    run.read_events()
                t = now()
                if t >= self.next_hb or (self.recheck_at and t >= self.recheck_at):
                    self.recheck_at = None
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
    raise KeyboardInterrupt


class UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def get_request(self):
        req, _ = super().get_request()
        return req, ('local', 0)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def local_path(self, p):
        """A path on another site (events, verify.json) -> its copy in the manager's mirror."""
        for r in self.server.mgr.runs.values():
            rs = r.E.get('REMOTE_SCRATCH')
            if r.remote and rs and p.startswith(rs.rstrip('/') + '/'):
                return r.E['SCRATCH'].rstrip('/') + p[len(rs.rstrip('/')):]
        return p

    def _send(self, code, body, ctype='application/json'):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
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
        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(SOCK)
    c = C('localhost', timeout=900)
    c.request('POST', '/api/action', json.dumps(action), {'X-Sahyadri': '1', 'Content-Type': 'application/json'})
    r = json.loads(c.getresponse().read())
    print(r['result'])
    return 0 if r['ok'] else 1


def main():
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
