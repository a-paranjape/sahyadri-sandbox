#!/usr/bin/env python3
"""sahyadri_manager: runs the Sahyadri pipeline for many runs from the pawna head node, with a web page.

  manager.py serve                 the manager (in `screen`; start_manager.sh); page on ~/.sahyadri_mgr/gui.sock
  manager.py add <run.conf> [--paused]   register a run (the running manager picks it up at once)
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
  * the simulation is submitted from the page (button), never automatically; a restart from the newest checkpoint
    likewise
  * Rockstar in batches (RS_BATCH=1): when RS_BATCH_MIN consecutive snapshots after the last batch are ready, or the
    simulation has finished, one batch job; at most MAX_ROCKSTAR Rockstar jobs over all runs (default 1)
  * then ctrees -> vahc -> fits -> verify (one chain, afterok), then archive (copy, check, read-only) automatically
  * a failed job stops the run's automation (shown in red; retry from the page)
  * deletion of the run's scratch data only after you approve it in the page, in two steps: move to
    <SCRATCH>/.trash/ (instant, can be undone), then purge; both re-check the archive manifest first
"""
import glob
import http.server
import json
import os
import queue
import re
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
                   max_sim_running=3, max_sim_queued=1, max_sim_restarts=5)
LIMIT_KEYS = ('max_sim_running', 'max_sim_queued', 'max_sim_restarts', 'max_rockstar', 'scratch_reserve_gb', 'heartbeat_s')


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
    c = dict(DEFAULT_CFG)
    if os.path.isfile(CONFIG):
        try:
            c.update(json.load(open(CONFIG)))
        except ValueError as e:
            log('config.json unreadable (%s); using defaults' % e)
    return c


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
    try:
        return int(str(j).split('.')[0])
    except ValueError:
        return -1


def short(job):
    return str(job).split('.')[0]


# ====================================================================================================== one run
class Run:
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
        self.name = '%s/r%s' % (self.E['STEM'], self.E['REAL'])
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
            self.plan, self.err = None, str(e)

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
        for jid, j in J.items():
            q = qjobs.get(jid)
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
        D['fits'] = ok_after('fits_done')
        D['sim_info'] = self.sim_progress()
        return D

    def sim_progress(self):
        t, v = self.simcache
        if now() - t < 60:
            return v
        v = None
        f = os.path.join(self.P['sim'], 'run.log')
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
    def submit(self, stages, extra=(), what=None):
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
            raise RuntimeError('submit %s failed (rc %d): %s %s' % (stages, r.returncode, r.stdout[-500:], r.stderr[-500:]))
        for st, jid in ids:
            rec = dict(job=jid, stage=st, t=now())
            if what:
                rec.update(what)
            self.ms['submitted'].append(rec)
        self.save()
        self.note('submitted %s' % ', '.join('%s %s' % x for x in ids) + (' (%s)' % what if what else ''))
        return ids


# ====================================================================================================== manager
class Manager:
    def __init__(self):
        os.makedirs(HOME, mode=0o700, exist_ok=True)
        os.chmod(HOME, 0o700)
        self.cfg = load_cfg()
        self.q = queue.Queue()
        self.runs = {}
        self.qjobs, self.qtime, self.qok = {}, 0.0, None
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
        want = {r['conf']: r for r in self.cfg['runs']}
        for conf in list(self.runs):
            if conf not in want:
                del self.runs[conf]
        for conf, r in want.items():
            if conf not in self.runs:
                try:
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
        d = {}
        for run in self.runs.values():
            for k in ('SCRATCH', 'ARCHIVE'):
                p = run.E.get(k)
                if p and p not in d and os.path.isdir(p):
                    u = shutil.disk_usage(p)
                    d[p] = dict(total=u.total, used=u.used, free=u.free, kind=k)
            tot = 0
            for top in (run.P['sim'], run.P['halo']):
                for dp, _ds, fs in os.walk(top):
                    for f in fs:
                        try:
                            tot += os.lstat(os.path.join(dp, f)).st_size
                        except OSError:
                            pass
            run.scratch_bytes = tot
        self.disk = d

    # ------------------------------------------------------------------ policy
    def tick(self):
        active_rs = 0
        derived = {}
        for run in self.runs.values():
            D = run.derive(self.qjobs, self.qtime)
            derived[run.conf] = D
            active_rs += sum(1 for j in D['active'] if j['stage'] == 'rockstar')
        self.schedule_sims(derived)
        # Rockstar turn order over the runs: the one whose next snapshot has waited longest goes first
        def waiting(run):
            D = derived[run.conf]
            todo = set(range(len(run.outs))) - set(D['rs_done'])
            return run.seen_ready.get(min(todo), float('inf')) if todo else float('inf')
        for run in sorted(self.runs.values(), key=waiting):
            D = derived[run.conf]
            try:
                active_rs += self.policy(run, D, active_rs)
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
        # failures and vanished jobs
        for j in D['jobs']:
            if (j['status'] == 'failed' and j['stage'] == 'sim' and j['job'] in D['sim_clean_stop']
                    and not self.acked(run, j)):
                n = int(run.ms.get('restarts', 0))
                run.ms.setdefault('acked', []).append(j['job'])
                if n >= int(self.cfg.get('max_sim_restarts', 5)):
                    run.ms['failed'] = dict(stage='sim', job=j['job'], rc=j.get('rc'), log=j.get('log'), t=now(),
                                            msg='stopped at its wall time again after %d restarts (max_sim_restarts)' % n)
                    run.note('simulation job %s stopped at its wall time; %d restarts done, not restarting' % (j['job'], n))
                    return 0
                run.ms['scheduled'] = dict(t=now(), priority=-1, restart=True, after=j['job'])
                run.note('simulation job %s stopped cleanly before the last output (wall time); scheduled to restart '
                         'from the newest checkpoint (restart %d of at most %d)' % (j['job'], n + 1, self.cfg.get('max_sim_restarts', 5)))
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
                if now() - first_seen > self.cfg['lost_grace_s'] and self.qtime > first_seen:
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
                    if active_rs >= self.cfg['max_rockstar']:
                        return 0
                    last = first + k - 1
                    run.submit('rockstar', ['--rs-snaps', '%d-%d' % (first, last)], dict(first=first, last=last))
                    return 1
                return 0
            if D['sim_done'] and active_rs < self.cfg['max_rockstar'] and not D['last'].get('rockstar'):
                run.submit('rockstar', [], dict(first=0, last=n_out - 1))
                return 1
            return 0
        # ---- post-processing chain, once Rockstar covers everything
        if not D['sim_done']:
            return 0
        lv = D['last'].get('verify')
        if not any(D['last'].get(s) for s in ('ctrees', 'vahc', 'fits', 'verify')):
            stages = 'ctrees,vahc,fits,verify'
            run.submit(stages, [], dict(chain=stages))
            return 0
        v = D['verify']
        if lv and lv['status'] == 'done' and v and int(v.get('ok', 0)) == 1 and E.get('ARCHIVE'):
            la = D['last'].get('archive')   # one archive job per passed verify (PBS ids increase; node clocks differ)
            if la is None or jobnum(la['job']) < jobnum(lv['job']):
                run.submit('archive', [], dict(after_verify=lv['job']))
        return 0

    def sim_counts(self, derived):
        running = queued = 0
        for run in self.runs.values():
            for j in derived[run.conf]['active']:
                if j['stage'] == 'sim':
                    if j['status'] in ('running', 'exiting'):
                        running += 1
                    else:
                        queued += 1
        return running, queued

    def schedule_sims(self, derived):
        """Submit scheduled simulations within max_sim_running / max_sim_queued, by priority then age."""
        running, queued = self.sim_counts(derived)
        cand = [r for r in self.runs.values() if r.ms.get('scheduled') and not (r.busy or r.err or r.ms.get('paused')
                                                                               or r.ms.get('failed') or r.ms.get('trash'))]
        cand.sort(key=lambda r: (r.ms['scheduled'].get('priority', 0), r.ms['scheduled'].get('t', 0)))
        mr, mq = int(self.cfg.get('max_sim_running', 3)), int(self.cfg.get('max_sim_queued', 1))
        self.queue_view = []
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
                    if sc.get('restart'):
                        run.ms['restarts'] = int(run.ms.get('restarts', 0)) + 1
                    run.ms['scheduled'] = None
                    run.save()
                    queued += 1
                    continue
                except Exception as e:  # noqa: BLE001
                    wait = 'submission failed: %s' % e
                    run.ms['failed'] = dict(stage='sim', msg=str(e), t=now())
                    run.note('scheduled submission failed: %s' % e)
            self.queue_view.append(dict(run=run.name, priority=sc.get('priority', 0), t=sc.get('t'), restart=sc.get('restart'),
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
            D = run.derive(self.qjobs, self.qtime)
            if D['sim_done']:
                raise ValueError('the simulation of %s has finished' % run.name)
            if any(j['stage'] == 'sim' for j in D['active']):
                raise ValueError('a simulation job of %s is queued or running; not scheduling another' % run.name)
            nb = a.get('not_before')
            if nb not in (None, ''):
                nb = float(nb) if re.match(r'^[0-9.]+$', str(nb)) else time.mktime(time.strptime(str(nb).replace('T', ' ')[:16], '%Y-%m-%d %H:%M'))
            else:
                nb = None
            restart = bool(D['last'].get('sim')) and bool(glob.glob(os.path.join(run.P['sim'], 'checkpoints', '*.chk')))
            run.ms['scheduled'] = dict(t=now(), priority=int(a.get('priority') or 0), not_before=nb,
                                       restart=bool(a.get('restart', restart)))
            run.note('scheduled (priority %s%s%s)' % (run.ms['scheduled']['priority'], ', not before %s' % iso(nb) if nb else '',
                                                     ', restart from checkpoint' if run.ms['scheduled']['restart'] else ''))
            return 'scheduled %s' % run.name
        if act == 'unschedule':
            run.ms['scheduled'] = None
            run.note('removed from the simulation queue')
            return 'unscheduled %s' % run.name
        if act == 'submit_sim':
            D = run.derive(self.qjobs, self.qtime)
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
            elif st in ('ctrees', 'vahc', 'fits', 'verify'):
                rest = ['ctrees', 'vahc', 'fits', 'verify']
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
        D = D or run.derive(self.qjobs, self.qtime)
        if D['sim_done'] or run.ms.get('trash') or run.ms.get('purged') or not D['last'].get('sim'):
            return 0
        used = run.scratch_bytes if run.scratch_bytes is not None else 0
        return max(0, scratch_estimate(run.E, len(run.outs)) - used)

    def space_check(self, run):
        """(ok, message): does a new simulation of `run` fit in SCRATCH next to the running ones and the reserve?"""
        root = run.E['SCRATCH']
        free = shutil.disk_usage(root).free
        need = scratch_estimate(run.E, len(run.outs))
        others = sum(self.outstanding(r) for r in self.runs.values() if r is not run and r.E['SCRATCH'] == root)
        reserve = int(float(self.cfg.get('scratch_reserve_gb', 2000)) * 1e9)
        left = free - need - others - reserve
        msg = ('scratch %s: %s free; this run needs up to %s, running simulations %s more, reserve %s -> %s %s'
               % (root, retention.human(free), retention.human(need), retention.human(others), retention.human(reserve),
                  retention.human(abs(left)), 'left' if left >= 0 else 'SHORT'))
        return left >= 0, msg

    def add_run(self, conf, paused=True):
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
        D = run.derive(self.qjobs, self.qtime)
        if D['active']:
            return False, 'jobs of this run are queued or running: %s' % ', '.join(j['job'] for j in D['active'])
        v, a = D['verify'], D['archive']
        vf, mf = os.path.join(P['state'], 'verify.json'), os.path.join(P['state'], 'MANIFEST.json')
        # order by file times (one file server), not by event times (node clocks differ by minutes)
        if not (v and int(v.get('ok', 0)) and a and int(a.get('ok', 0)) and os.path.isfile(vf) and os.path.isfile(mf)
                and os.path.getmtime(mf) > os.path.getmtime(vf)):
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
        for cat, _s, src, rel in retention.kept_files(E):
            if cat in ('log', 'events') or rel.endswith('verify.json'):
                continue
            r = rows.get(rel)
            st = os.stat(src)
            if r is None:
                nbad += 1
                msgs.append('kept file not in the manifest: %s' % rel)
            elif r['size'] != st.st_size or abs(r['mtime'] - st.st_mtime) > 1e-6:
                nbad += 1
                msgs.append('kept file changed after the archive: %s' % rel)
        if nbad:
            return False, '%d problem(s): %s' % (nbad, '; '.join(msgs[:10]))
        size = run.scratch_bytes
        return True, ('OK: %d files in the archive (%s), checked; scratch %s/%s and %s/%s (%s) can go to the trash'
                      % (len(M['files']), retention.human(M['bytes']), E['SCRATCH'], P['rel']['sim'], E['SCRATCH'],
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
        for src, dst in T['moved']:
            if os.path.exists(src):
                raise ValueError('%s exists again; not moving back' % src)
            os.rename(dst, src)
        try:
            os.rmdir(T['dir'])     # now empty
        except OSError:
            pass
        run.ms['trash'] = None
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
        tdir = self._guard(run, T['dir'], os.path.join(E['SCRATCH'], '.trash'))
        t0 = now()
        shutil.rmtree(tdir)
        run.ms['purged'] = dict(t=now(), dir=tdir, seconds=round(now() - t0, 1))
        run.note('purged %s' % tdir)
        return 'purged %s' % tdir

    # ------------------------------------------------------------------ state for the page
    def build_snapshot(self, derived):
        runs = []
        for run in self.runs.values():
            D = derived.get(run.conf) or run.derive(self.qjobs, self.qtime)
            pl = run.plan
            runs.append(dict(
                name=run.name, conf=run.conf, phase=run.phase(D), paused=run.ms.get('paused'), failed=run.ms.get('failed'), busy=run.busy,
                err=run.err, n_out=D['n_out'], ready=D['ready'], rs_done=D['rs_done'], sim_done=D['sim_done'],
                sim=D['sim_info'], jobs=D['jobs'][-40:], active=D['active'],
                verify=self.verify_summary(run, D), archive=D['archive'], fits=D['fits'],
                trash=run.ms.get('trash'), purged=run.ms.get('purged'), scheduled=run.ms.get('scheduled'),
                restarts=run.ms.get('restarts', 0), actions=run.ms.get('actions', [])[-30:],
                keep=dict(full=pl['full_z'] if pl else [], pct=pl['subsample_pct'] if pl else None,
                          tables=pl['tables'] if pl else []),
                outputs=[[o[0], o[1], o[3]] for o in run.outs], scratch_bytes=run.scratch_bytes,
                scratch_estimate=scratch_estimate(run.E, len(run.outs)), outstanding=self.outstanding(run, D),
                scratch=run.E.get('SCRATCH'), archive_root=run.E.get('ARCHIVE'), npart=run.E.get('NPART'),
                rs_batch=run.E.get('RS_BATCH', '0'), rs_batch_min=run.E.get('RS_BATCH_MIN', '10'),
                events=[dict(t=e['t'], stage=e['stage'], job=e['job'], ev=e['ev'],
                             info={k: v for k, v in e.items() if k not in ('t', 'time', 'run', 'stage', 'job', 'host', 'ev', 'conf')})
                        for e in run.ev[-60:]]))
        with self.lock:
            sr, sq = self.sim_counts(derived) if derived else (0, 0)
            self.snapshot = dict(t=now(), runs=runs, disk=self.disk, reserve_gb=self.cfg.get('scratch_reserve_gb'),
                                 limits={k: self.cfg.get(k) for k in LIMIT_KEYS}, sims_running=sr, sims_queued=sq,
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
            p = parse_qs(urlparse(self.path).query).get('path', [''])[0]
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
            p = parse_qs(urlparse(self.path).query).get('path', [''])[0]
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
