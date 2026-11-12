#!/usr/bin/env python3
"""Check that everything of a run was computed and is consistent, before anything is archived or deleted
(stage `verify`, job_verify.pbs). Writes <state dir>/verify.json; exit 0 only if every check passed.

Checks (failures block the archive; warnings are reported only):
  snapshots   every planned output: all .pkc files complete (header/trailer/particle counts, pkcreader), N^3
              particles, redshift as planned, subsample percents = SUBSAMPLES, the potential where expected;
              the kept full snapshots also with pkc_check (every particle ID exactly once)
  subsample   the kept subsample files hold KEEP_SUBSAMPLE_PCT % of the particles
  rockstar    rockstar_done events (one run or batches) cover every snapshot 0..N-1, each ok
  ctrees      out_<snap>.trees for every snapshot in scales.txt; the snapshots without trees are the first ones
              (too few halos); trees/ tree_*.dat, locations.dat, forests.list (or their FITS) present
  vahc        .vahc (and .tens with PP_TENS=1, .vpot with PP_POT=1 where the .pkc store the potential): as many rows
              as the .trees, the same halo IDs in the same order
  fits        the fits stage ended with 0 check failures after the last vahc; every kept FITS table and the
              tree FITS present and non-empty
  halos       halo counts per snapshot (warning when they jump by more than a factor 1.5 between neighbours)

usage: verify_run.py <run.conf> [--threads N] [--skip-pkc-check]
"""
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'post-process'))
import retention  # noqa: E402


def events(state):
    out = []
    for f in sorted(glob.glob(os.path.join(state, 'events', '*.jsonl'))):
        for line in open(f):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return sorted(out, key=lambda e: e.get('t', 0))


def ids_of(path, col):
    """Integer column `col` of an ASCII catalogue (comment lines '#'), as a numpy array."""
    import pandas as pd
    df = pd.read_csv(path, comment='#', sep=r'\s+', header=None, usecols=[col], dtype={col: 'int64'},
                     engine='c')
    return df[col].to_numpy()


class Report:
    def __init__(self):
        self.checks = {}

    def check(self, name):
        c = self.checks.setdefault(name, dict(ok=True, errors=[], warnings=[], info={}))
        return c

    def fail(self, name, msg):
        c = self.check(name)
        c['ok'] = False
        if len(c['errors']) < 200:
            c['errors'].append(msg)
        c['info']['n_errors'] = c['info'].get('n_errors', 0) + 1

    def warn(self, name, msg):
        c = self.check(name)
        if len(c['warnings']) < 200:
            c['warnings'].append(msg)

    def ok(self):
        return all(c['ok'] for c in self.checks.values())


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    threads = int(sys.argv[sys.argv.index('--threads') + 1]) if '--threads' in sys.argv else os.cpu_count()
    E = retention.load_conf(sys.argv[1])
    pl = retention.plan(E)
    P = pl['paths']
    outs = pl['outputs']
    R = Report()
    t0 = time.time()
    N3 = int(E['NPART']) ** 3
    from pkcreader import PkcSnapshot
    # ------------------------------------------------------------ snapshots
    c = R.check('snapshots')
    want_pcts = retention.subsample_pcts(E)
    pot_zmax = None
    par = E.get('PKD_PAR') or os.path.join(P['sim'], 'p.par')
    if os.path.isfile(par):
        for line in open(par):
            if line.split('#')[0].strip().startswith('dCompPotMaxRedshift'):
                pot_zmax = float(line.split('#')[0].split('=')[1])
    has_pot = {}
    sub_frac = []
    for snap, step, _a, z in outs:
        pre = os.path.join(P['pkc'], 'snap.%s' % step)
        try:
            S = PkcSnapshot(pre, threads=1)
        except Exception as e:  # noqa: BLE001
            R.fail('snapshots', 'snapshot %d (step %s): %s' % (snap, step, e))
            continue
        if S.npart != N3:
            R.fail('snapshots', 'snapshot %d: %d particles, expected %d' % (snap, S.npart, N3))
        if z == z and abs(S.redshift - z) > 1e-3 * (1 + z):
            R.fail('snapshots', 'snapshot %d: redshift %.6f, planned %.6f' % (snap, S.redshift, z))
        got = sorted(round(f.percent, 6) for f in S.files)
        if sorted(set(got)) != sorted(set(round(p, 6) for p in want_pcts)):
            R.fail('snapshots', 'snapshot %d: subsamples %s, expected %s' % (snap, sorted(set(got)), want_pcts))
        has_pot[snap] = bool(S.has_potential)
        if pot_zmax is not None and (S.redshift <= pot_zmax + 1e-6) != S.has_potential:
            R.warn('snapshots', 'snapshot %d (z=%.3f): potential stored = %s (dCompPotMaxRedshift %g)'
                   % (snap, S.redshift, S.has_potential, pot_zmax))
        if pl['subsamples'] and snap not in pl['full']:
            n = sum(f.header['n_particles'] for f in S.files if any(abs(f.percent - p) < 1e-6 for p in pl['subsamples']))
            sub_frac.append((snap, n / float(S.npart)))
    c['info']['n_outputs'] = len(outs)
    c['info']['with_potential'] = sorted(s for s, v in has_pot.items() if v)
    # kept full snapshots: every ID once
    pkc_check = os.path.join(E.get('PKD_IO_ROOT', ''), 'build_snapcomp', 'pkc_check')
    if '--skip-pkc-check' not in sys.argv:
        for snap in pl['full']:
            pre = os.path.join(P['pkc'], 'snap.%s' % outs[snap][1])
            t1 = time.time()
            r = subprocess.run([pkc_check, '--threads', str(threads), pre], capture_output=True, text=True)
            tail = (r.stdout + r.stderr).strip().splitlines()[-2:]
            if r.returncode != 0:
                R.fail('snapshots', 'pkc_check snapshot %d: rc %d: %s' % (snap, r.returncode, ' | '.join(tail)))
            c['info'].setdefault('pkc_check', {})[str(snap)] = dict(rc=r.returncode, seconds=round(time.time() - t1, 1),
                                                                    out=' | '.join(tail))
    # ------------------------------------------------------------ subsample fraction
    c = R.check('subsample')
    c['info']['kept_pct'] = pl['subsample_pct']
    for snap, fr in sub_frac:
        # random subsample: binomial scatter, allow 5 sigma
        p = pl['subsample_pct'] / 100.0
        sig = (p * (1 - p) / N3) ** 0.5
        if abs(fr - p) > 5 * sig + 1e-9:
            R.fail('subsample', 'snapshot %d: kept files hold %.5f%% of the particles, expected %g%%' % (snap, 100 * fr, 100 * p))
    if sub_frac:
        c['info']['fraction_range'] = [min(f for _s, f in sub_frac), max(f for _s, f in sub_frac)]
    # ------------------------------------------------------------ rockstar
    ev = events(P['state'])
    c = R.check('rockstar')
    covered = set()
    for e in ev:
        if e.get('ev') == 'rockstar_done' and int(e.get('ok', 0)) == 1:
            covered.update(range(int(e['first']), int(e['last']) + 1))
    missing = sorted(set(range(len(outs))) - covered)
    if missing:
        R.fail('rockstar', 'no successful rockstar_done event for snapshots %s' % compress(missing))
    c['info']['batches'] = [(e['first'], e['last'], e['job']) for e in ev if e.get('ev') == 'rockstar_done']
    # ------------------------------------------------------------ ctrees
    c = R.check('ctrees')
    scales = os.path.join(os.path.dirname(P['halo']), 'scales.txt')
    tree_snaps = []
    if not os.path.isfile(scales):
        R.fail('ctrees', 'no %s' % scales)
    else:
        tree_snaps = sorted(int(l.split()[0]) for l in open(scales) if l.strip())
    for s in tree_snaps:
        if not os.path.isfile(os.path.join(P['halo'], 'out_%d.trees' % s)):
            R.fail('ctrees', 'no out_%d.trees (snapshot in scales.txt)' % s)
    if tree_snaps:
        without = sorted(set(range(len(outs))) - set(tree_snaps))
        if without and max(without) > min(tree_snaps):
            R.fail('ctrees', 'snapshots without trees after the first one with trees: %s' % compress([s for s in without if s > min(tree_snaps)]))
        c['info']['first_with_trees'] = min(tree_snaps)
        c['info']['n_with_trees'] = len(tree_snaps)
    td = os.path.join(P['halo'], 'trees')
    tf = os.path.join(P['halo'], 'compressed', 'trees')
    for n in ('locations', 'forests'):
        if not (glob.glob(os.path.join(td, n + '*')) or glob.glob(os.path.join(tf, n + '*'))):
            R.fail('ctrees', 'no trees/%s* (ASCII or FITS)' % n)
    if not (glob.glob(os.path.join(td, 'tree_*.dat')) or glob.glob(os.path.join(tf, 'tree_*.fits*'))):
        R.fail('ctrees', 'no tree_* files (ASCII or FITS)')
    # ------------------------------------------------------------ vahc / tens / vpot  and halo counts
    c = R.check('vahc')
    want = ['vahc'] + (['tens'] if E.get('PP_TENS', '1') == '1' else [])
    nh = {}
    for s in tree_snaps:
        tr = os.path.join(P['halo'], 'out_%d.trees' % s)
        if not os.path.isfile(tr):
            continue
        tid = ids_of(tr, 1)
        nh[s] = len(tid)
        kinds = list(want) + (['vpot'] if E.get('PP_POT', '0') == '1' and has_pot.get(s) else [])
        for k in kinds:
            f = os.path.join(P['halo'], 'out_%d.%s' % (s, k))
            if not os.path.isfile(f):
                R.fail('vahc', 'no out_%d.%s' % (s, k))
                continue
            vid = ids_of(f, 0) if len(tid) else []
            if len(vid) != len(tid):
                R.fail('vahc', 'out_%d.%s: %d rows, .trees %d' % (s, k, len(vid), len(tid)))
            elif len(tid) and not (vid == tid).all():
                R.fail('vahc', 'out_%d.%s: halo IDs differ from .trees (order or values)' % (s, k))
    c['info']['kinds'] = want + (['vpot'] if E.get('PP_POT', '0') == '1' else [])
    c = R.check('halos')
    c['info']['counts'] = {str(k): v for k, v in sorted(nh.items())}
    ks = sorted(nh)
    for a, b in zip(ks, ks[1:]):
        if nh[a] > 1000 and nh[b] > 0 and not (1 / 1.5 < nh[b] / float(nh[a]) < 1.5):
            R.warn('halos', 'halo count %d -> %d between snapshots %d and %d' % (nh[a], nh[b], a, b))
    # ------------------------------------------------------------ fits
    c = R.check('fits')
    fe = [e for e in ev if e.get('ev') == 'fits_done']
    ve = [e for e in ev if e.get('ev') == 'vahc_done']
    if not fe:
        R.fail('fits', 'no fits_done event')
    else:
        last = fe[-1]
        if int(last.get('rc', 1)) != 0 or int(last.get('nfail', 1) or 0) != 0:
            R.fail('fits', 'last fits stage (job %s): rc %s, %s check failures (%s)' % (
                last['job'], last.get('rc'), last.get('nfail'), last.get('check_log')))
        if ve and ve[-1]['t'] > last['t']:
            R.fail('fits', 'vahc ran again (job %s) after the last fits stage (job %s)' % (ve[-1]['job'], last['job']))
        c['info']['job'] = last['job']
        c['info']['npass'] = last.get('npass')
    comp = os.path.join(P['halo'], 'compressed')
    for s in tree_snaps:
        for t in pl['tables']:
            f = os.path.join(comp, 'out_%d_%s.fits.gz' % (s, t))
            if not os.path.isfile(f) or os.path.getsize(f) == 0:
                R.fail('fits', 'missing or empty %s' % os.path.relpath(f, P['halo']))
    if pl['trees']:
        for n in ('locations', 'forests', 'tree_'):
            if not glob.glob(os.path.join(comp, 'trees', n + '*.fits*')):
                R.fail('fits', 'no compressed/trees/%s*.fits*' % n)
    # ------------------------------------------------------------ write
    out = dict(run='%s/r%s' % (E['STEM'], E['REAL']), conf=E['CONF'], time=time.strftime('%Y-%m-%dT%H:%M:%S'),
               t=time.time(), seconds=round(time.time() - t0, 1), ok=R.ok(), checks=R.checks,
               retention=dict(full=pl['full_z'], subsamples=pl['subsamples'], tables=pl['tables'], trees=pl['trees'],
                              checkpoint=pl['checkpoint']),
               job=os.environ.get('PBS_JOBID', ''))
    os.makedirs(P['state'], exist_ok=True)
    tmp = os.path.join(P['state'], 'verify.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, os.path.join(P['state'], 'verify.json'))
    for name, c in R.checks.items():
        print('%-10s %s%s%s' % (name, 'PASS' if c['ok'] else 'FAIL',
                                ('  %d error(s): ' % len(c['errors']) + '; '.join(c['errors'][:5])) if c['errors'] else '',
                                ('  warnings: ' + '; '.join(c['warnings'][:3])) if c['warnings'] else ''))
    print('verify %s in %.0f s -> %s' % ('PASSED' if R.ok() else 'FAILED', time.time() - t0,
                                         os.path.join(P['state'], 'verify.json')))
    sys.exit(0 if R.ok() else 1)


def compress(xs):
    """[1,2,3,7,8] -> '1-3,7-8'"""
    xs = sorted(xs)
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append('%d' % xs[i] if i == j else '%d-%d' % (xs[i], xs[j]))
        i = j + 1
    return ','.join(out)


if __name__ == '__main__':
    main()
