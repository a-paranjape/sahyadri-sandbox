#!/usr/bin/env python3
"""Check that everything of a run was computed and is consistent, before anything is archived or deleted
(stage `verify`, job_verify.pbs). Writes <state dir>/verify.json; exit 0 only if every check passed.

Checks (failures block the archive; warnings are reported only):
  snapshots   every planned output: all .pkc files complete (header/trailer/particle counts, pkcreader), N^3
              particles, redshift as planned, subsample percents = SUBSAMPLES, the potential where expected;
              the kept full snapshots also with pkc_check (every particle ID exactly once; VF_PKC_NPROC at a time,
              passes cached by file size/mtime)
  subsample   the kept subsample files hold KEEP_SUBSAMPLE_PCT % of the particles
  rockstar    rockstar_done events (one run or batches) cover every snapshot 0..N-1, each ok
  ctrees      out_<snap>.trees for every snapshot in scales.txt; the snapshots without trees are the first ones
              (too few halos); trees/ tree_*.dat, locations.dat, forests.list (or their FITS) present
  vahc        .vahc (and .tens with PP_TENS=1, .vpot with PP_POT=1 where the .pkc store the potential): as many rows
              as halos (basic FITS table rows; .trees text only without that table), the same halo IDs in the same
              order in all of them (row-by-row equality with the .trees: the fits stage's check, required below)
  fits        the fits stage ended with 0 check failures after the last vahc; every kept FITS table and the
              tree FITS present and non-empty
  parquet     (PARQUET=1) every kept FITS file has its Parquet copy (stage parquet, after the last fits, passed its
              DATASUM and read-back checks), made from the current FITS file, with its rows and columns
  halos       halo counts per snapshot (warning when they jump by more than a factor 1.5 between neighbours)
  pk          (PP_PK=1) every output's matter P(k) against CLASS: absolute and growth ratios, CLASS vs the IC
              table, halo bias (pk_check.py); writes <halo dir>/pk/pk_summary.png

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
    """All events of a run (``state/events/*.jsonl``) sorted by time; unreadable lines are skipped."""
    out = []
    for f in sorted(glob.glob(os.path.join(state, 'events', '*.jsonl'))):
        for line in open(f):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return sorted(out, key=lambda e: e.get('t', 0))


def ids_of(path, col):
    """Integer column `col` of a catalogue (comment lines '#'), as a numpy array; .vahc/.tens/.vpot may be binary
    (halo_table_io)."""
    import pandas as pd
    import halo_table_io
    if halo_table_io.is_binary(path):
        return halo_table_io.column(path, halo_table_io.names(path)[col]).astype('int64')
    df = pd.read_csv(path, comment='#', sep=r'\s+', header=None, usecols=[col], dtype={col: 'int64'},
                     engine='c')
    return df[col].to_numpy()


class Report:
    """Collects the checks of a verification: per check ok flag, errors and warnings (at most 200 each) and
    info values; written to ``verify.json``.
    """
    def __init__(self):
        """No checks yet."""
        self.checks = {}

    def check(self, name):
        """The record of check ``name`` (created ok and empty on first use)."""
        c = self.checks.setdefault(name, dict(ok=True, errors=[], warnings=[], info={}))
        return c

    def fail(self, name, msg):
        """Record an error for check ``name`` and mark it failed."""
        c = self.check(name)
        c['ok'] = False
        if len(c['errors']) < 200:
            c['errors'].append(msg)
        c['info']['n_errors'] = c['info'].get('n_errors', 0) + 1

    def warn(self, name, msg):
        """Record a warning for check ``name`` (does not fail it)."""
        c = self.check(name)
        if len(c['warnings']) < 200:
            c['warnings'].append(msg)

    def ok(self):
        """True if every check passed."""
        return all(c['ok'] for c in self.checks.values())


def main():
    """Command line: ``verify_run.py <run.conf> [--threads N]``.

    Checks a finished run: snapshots (all outputs, sizes, ``pkc_check`` of each compressed output in
    parallel), subsamples, Rockstar, consistent-trees, vahc and FITS products, halo counts and P(k) against
    CLASS; writes ``state/verify.json`` and exits non-zero if a check failed.
    """
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
    # kept full snapshots: every ID once. Several snapshots at a time (VF_PKC_NPROC, default 4; the job's threads
    # shared among them); a snapshot that passed before with the same files (names, sizes, mtimes) is not read again
    # (<state>/pkc_check_cache.json), so a verify rerun after another failure does not repeat hours of reading.
    pkc_check = os.path.join(E.get('PKD_IO_ROOT', ''), 'build_snapcomp', 'pkc_check')
    if '--skip-pkc-check' not in sys.argv and pl['full']:
        cache_f = os.path.join(P['state'], 'pkc_check_cache.json')
        try:
            cache = json.load(open(cache_f))
        except (OSError, ValueError):
            cache = {}
        todo, sig = [], {}
        for snap in pl['full']:
            pre = os.path.join(P['pkc'], 'snap.%s' % outs[snap][1])
            sig[snap] = [[os.path.basename(f), os.path.getsize(f), os.path.getmtime(f)] for f in sorted(glob.glob(pre + '*'))]
            prev = cache.get(str(snap))
            if prev and prev.get('rc') == 0 and prev.get('files') == sig[snap]:
                c['info'].setdefault('pkc_check', {})[str(snap)] = dict(prev, cached=True)
            else:
                todo.append((snap, pre))
        nproc = max(1, min(int(E.get('VF_PKC_NPROC') or 4), len(todo) or 1))
        per = max(1, threads // nproc)

        def run(task):
            """Run ``pkc_check`` on one output; returns (snapshot, completed process, seconds)."""
            snap, pre = task
            t1 = time.time()
            r = subprocess.run([pkc_check, '--threads', str(per), pre], capture_output=True, text=True)
            return snap, r, time.time() - t1
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(nproc) as ex:
            for snap, r, dt in ex.map(run, todo):
                tail = (r.stdout + r.stderr).strip().splitlines()[-2:]
                if r.returncode != 0:
                    R.fail('snapshots', 'pkc_check snapshot %d: rc %d: %s' % (snap, r.returncode, ' | '.join(tail)))
                res = dict(rc=r.returncode, seconds=round(dt, 1), threads=per, out=' | '.join(tail))
                c['info'].setdefault('pkc_check', {})[str(snap)] = res
                if r.returncode == 0:
                    cache[str(snap)] = dict(res, files=sig[snap])
                else:
                    cache.pop(str(snap), None)
        c['info']['pkc_check_parallel'] = [nproc, per]
        os.makedirs(P['state'], exist_ok=True)
        with open(cache_f + '.tmp', 'w') as fh:
            json.dump(cache, fh, indent=1)
        os.replace(cache_f + '.tmp', cache_f)
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
    scales = retention.scales_file(P)
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
    # The halo count of a snapshot is the row count of its basic FITS table (header only; the fits stage checked that
    # table against the .trees row by row, and that every .vahc/.tens/.vpot row is the .trees halo in order). The
    # .trees text is parsed only when that table is missing (2048^3: ~16 GB of text per snapshot). Here: every
    # catalogue present, as many rows as halos, and the same halo IDs in .vahc, .tens and .vpot.
    try:
        import fits_to_parquet as f2p
    except ImportError:          # no pyarrow here: counts from the .trees text
        f2p = None
    c = R.check('vahc')
    want = ['vahc'] + (['tens'] if E.get('PP_TENS', '1') == '1' else [])
    nh, from_text = {}, []
    for s in tree_snaps:
        tr = os.path.join(P['halo'], 'out_%d.trees' % s)
        fb = os.path.join(P['halo'], 'compressed', 'out_%d_basic.fits.gz' % s)
        tid = None
        try:
            nh[s] = int(f2p.table_rows(fb)[0])   # AttributeError without f2p
        except Exception:  # noqa: BLE001  no or unreadable FITS table (the fits check below fails it): the text
            if not os.path.isfile(tr):
                continue
            tid = ids_of(tr, 1)
            nh[s] = len(tid)
            from_text.append(s)
        kinds = list(want) + (['vpot'] if E.get('PP_POT', '0') == '1' and has_pot.get(s) else [])
        ref = None
        for k in kinds:
            f = os.path.join(P['halo'], 'out_%d.%s' % (s, k))
            if not os.path.isfile(f):
                R.fail('vahc', 'no out_%d.%s' % (s, k))
                continue
            vid = ids_of(f, 0) if nh[s] else []
            if len(vid) != nh[s]:
                R.fail('vahc', 'out_%d.%s: %d rows, %d halos' % (s, k, len(vid), nh[s]))
                continue
            if not nh[s]:
                continue
            if tid is not None and not (vid == tid).all():
                R.fail('vahc', 'out_%d.%s: halo IDs differ from .trees (order or values)' % (s, k))
            elif ref is not None and not (vid == ref[1]).all():
                R.fail('vahc', 'out_%d.%s: halo IDs differ from out_%d.%s (order or values)' % (s, k, s, ref[0]))
            ref = ref or (k, vid)
    if from_text:
        c['info']['counts_from_trees_text'] = compress(from_text)
    c['info']['kinds'] = want + (['vpot'] if E.get('PP_POT', '0') == '1' else [])
    c = R.check('halos')
    c['info']['counts'] = {str(k): v for k, v in sorted(nh.items())}
    ks = sorted(nh)
    for a, b in zip(ks, ks[1:]):
        if nh[a] > 1000 and nh[b] > 0 and not (1 / 1.5 < nh[b] / float(nh[a]) < 1.5):
            R.warn('halos', 'halo count %d -> %d between snapshots %d and %d' % (nh[a], nh[b], a, b))
    # ------------------------------------------------------------ fits
    c = R.check('fits')
    fe = sorted([e for e in ev if e.get('ev') == 'fits_done'], key=lambda e: jobnum(e['job']))
    if not fe:
        R.fail('fits', 'no fits_done event')
    else:
        last = fe[-1]
        if int(last.get('rc', 1)) != 0 or int(last.get('nfail', 1) or 0) != 0:
            R.fail('fits', 'last fits stage (job %s): rc %s, %s check failures (%s)' % (
                last['job'], last.get('rc'), last.get('nfail'), last.get('check_log')))
        # a value-added catalogue newer than its FITS table (a vahc rerun that only added .pk files changes nothing
        # here). On Lustre a file's mtime is the writing node's clock, and nodes disagree (pride, 25 Nov 2026: -383 s
        # to +85 s against the login node), so only a difference beyond VF_CLOCK_SLACK_S (default 900 s) counts: a
        # real vahc rerun after fits is later by at least the length of a vahc job
        slack = float(os.environ.get('VF_CLOCK_SLACK_S', '900') or 900)
        comp_ = os.path.join(P['halo'], 'compressed')
        newer = []
        for s in tree_snaps:
            ft = os.path.join(comp_, 'out_%d_vahc.fits.gz' % s)
            for kind in ('vahc', 'tens', 'vpot'):
                f = os.path.join(P['halo'], 'out_%d.%s' % (s, kind))
                if os.path.isfile(f) and os.path.isfile(ft) and os.path.getmtime(f) > os.path.getmtime(ft) + slack:
                    newer.append('out_%d.%s' % (s, kind))
        if newer:
            R.fail('fits', '%d catalogues newer than their FITS tables by more than %.0f s (vahc ran again after fits?): %s'
                   % (len(newer), slack, ', '.join(newer[:5])))
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
    # ------------------------------------------------------------ parquet (PARQUET=1)
    if pl.get('parquet'):
        check_parquet(R, ev, P, pl, comp)
    # ------------------------------------------------------------ power spectra (pk_check.py)
    if E.get('PP_PK', '1') == '1':
        import pk_check
        rows = pk_check.check(E, P, outs, R)
        try:
            png = pk_check.plot(rows, os.path.join(P['halo'], 'pk'), os.path.join(P['halo'], 'pk', 'pk_summary.png'),
                                '%s r%s' % (E['STEM'], E['REAL']))
            R.check('pk')['info']['plot'] = png
        except Exception as e:  # noqa: BLE001  a plot failure is not a data failure
            R.warn('pk', 'plot failed: %s' % e)
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


def check_parquet(R, ev, P, pl, comp):
    """The Parquet copy (stage parquet) of every kept FITS file: made after the last fits stage, by a parquet stage
    that passed its own checks (DATASUM, read-back hashes), from the FITS file as it is now (size and mtime recorded
    in the Parquet), with the FITS table's rows and columns."""
    import pyarrow.parquet as pq
    import fits_to_parquet as f2p
    c = R.check('parquet')
    pe = sorted([e for e in ev if e.get('ev') == 'parquet_done'], key=lambda e: e.get('t', 0))
    fe = sorted([e for e in ev if e.get('ev') == 'fits_done'], key=lambda e: e.get('t', 0))
    if not pe:
        R.fail('parquet', 'no parquet_done event (stage parquet)')
    else:
        last = pe[-1]
        if int(last.get('rc', 1)) != 0 or int(last.get('nfail', 0) or 0) != 0 or last.get('skipped'):
            R.fail('parquet', 'last parquet stage (job %s): rc %s, %s failures%s (%s)' % (
                last['job'], last.get('rc'), last.get('nfail'), ', skipped' if last.get('skipped') else '', last.get('log')))
        if fe and fe[-1].get('t', 0) > last.get('t', 0):
            R.fail('parquet', 'fits stage (job %s) ran after the last parquet stage (job %s)' % (fe[-1]['job'], last['job']))
        c['info']['job'] = last['job']
    pqd = os.path.join(P['halo'], 'parquet')
    srcs = []
    for t in pl['tables']:
        srcs += glob.glob(os.path.join(comp, 'out_*_%s.fits.gz' % t))
    if pl['trees']:
        srcs += glob.glob(os.path.join(comp, 'trees', '*.fits.gz'))
    n_ok, nb = 0, 0
    for src in sorted(srcs):
        rel = os.path.relpath(src, comp)[:-len('.fits.gz')] + '.parquet'
        dst = os.path.join(pqd, rel)
        if not os.path.isfile(dst):
            R.fail('parquet', 'no parquet/%s' % rel)
            continue
        try:
            md = pq.read_metadata(dst)
            m = md.metadata or {}
            st = os.stat(src)
            if (m.get(b'source_size') != str(st.st_size).encode() or m.get(b'source_mtime') != repr(st.st_mtime).encode()):
                R.fail('parquet', 'parquet/%s was not made from the current %s (size/mtime differ)' % (rel, os.path.basename(src)))
                continue
            rows, cols = f2p.table_rows(src)
            if md.num_rows != rows or md.num_columns != cols:
                R.fail('parquet', 'parquet/%s: %d rows x %d columns, FITS %d x %d' % (rel, md.num_rows, md.num_columns, rows, cols))
                continue
            n_ok += 1
            nb += os.path.getsize(dst)
        except Exception as e:  # noqa: BLE001
            R.fail('parquet', 'parquet/%s unreadable: %s' % (rel, e))
    c['info']['files'] = n_ok
    c['info']['bytes'] = nb
    if not srcs:
        R.fail('parquet', 'no FITS files to compare with')


def jobnum(j):
    """Numeric PBS job id (``'2539127.pawna'`` -> 2539127; -1 if not a number)."""
    try:
        return int(str(j).split('.')[0])
    except ValueError:
        return -1


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
