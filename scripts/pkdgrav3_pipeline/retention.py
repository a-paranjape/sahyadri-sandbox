#!/usr/bin/env python3
"""What of a run is kept long term, and where it goes (standard library only: used by the jobs and by
tools/sahyadri_manager on the head node).

Settings (run.conf; defaults in brackets):
  SCRATCH             where the run lives while it is computed (sims/, halos/, state/ below it)
  ARCHIVE             long-term root with the same layout ([empty] = no archive stage)
  KEEP_FULL_Z         full snapshots kept: the planned output nearest each redshift ["0 0.5 1 2 3 5 7"]
  KEEP_FULL_SNAPS     ... or explicit Rockstar snapshot numbers (overrides KEEP_FULL_Z), e.g. "100 93 80"
  KEEP_SUBSAMPLE_PCT  particles kept of every other snapshot, percent [1]; must be a sum of the leading SUBSAMPLES
                      (1,3,5,9,10,12,15,20,25 -> 1, 4, 9, 18, 28, 40, 55, 75, 100); 0 = none
  KEEP_HALO_TABLES    FITS halo tables kept for every snapshot ["basic,extended,tens,vahc"]
  KEEP_TREES          1: merger-tree FITS (whole run) [1]
  KEEP_CHECKPOINT     1: newest complete checkpoint [0]
  ARCHIVE_FILE_MODE   ["0444"]  ARCHIVE_DIR_MODE ["0555"]  (after the archive is checked: read-only for everyone)

usage: retention.py <run.conf> [--list] [--json]     print the resolved plan (and with --list every file)
"""
import glob
import json
import os
import re
import subprocess
import sys

DEFAULTS = dict(KEEP_FULL_Z='0 0.5 1 2 3 5 7', KEEP_FULL_SNAPS='', KEEP_SUBSAMPLE_PCT='1',
                KEEP_HALO_TABLES='basic,extended,tens,vahc', KEEP_TREES='1', KEEP_CHECKPOINT='0',
                ARCHIVE='', ARCHIVE_FILE_MODE='0444', ARCHIVE_DIR_MODE='0555',
                SUBSAMPLES='1,3,5,9,10,12,15,20,25', PKC_SUBDIR='')


def load_conf(conf):
    """The run configuration as the jobs see it (bash `source` in a clean environment), as a dict."""
    base = {k: os.environ[k] for k in ('HOME', 'USER', 'PATH', 'LOGNAME') if k in os.environ}
    out = subprocess.run(['bash', '-c', 'set -a; source "$0" >/dev/null 2>&1; env -0', conf], env=base,
                         capture_output=True, check=True).stdout
    E = dict(DEFAULTS)
    E.update(kv.split('=', 1) for kv in out.decode().split('\0') if '=' in kv)
    E['CONF'] = os.path.abspath(conf)
    return E


def paths(E):
    """Directories of the run (same as pipeline_env.sh), and their archive counterparts."""
    stem, real = E['STEM'], E['REAL']
    rel = dict(sim='sims/%s/r%s' % (stem, real), halo='halos/%s/r%s' % (stem, real),
               state='state/%s/r%s' % (stem, real), cosmo='sims/%s' % stem)
    P = {k: os.path.join(E['SCRATCH'], v) for k, v in rel.items()}
    P['pkc'] = P['sim'] + ('/' + E['PKC_SUBDIR'] if E.get('PKC_SUBDIR') else '')
    P['rel'] = rel
    P['archive'] = {k: os.path.join(E['ARCHIVE'], v) for k, v in rel.items()} if E.get('ARCHIVE') else {}
    return P


def planned_outputs(E, P=None):
    """[(snap, step, a, z)] of the planned outputs (output_steps.txt), else of snapnames.txt (z unknown)."""
    P = P or paths(E)
    f = os.path.join(P['sim'], 'output_steps.txt')
    out = []
    if os.path.isfile(f):
        for line in open(f):
            if line.strip() and not line.startswith('#'):
                s, a, z = line.split()[:3]
                out.append((len(out), s, float(a), float(z)))
        return out
    f = os.path.join(P['sim'], 'snapnames.txt')
    if os.path.isfile(f):
        try:
            zs = [float(z) for z in E.get('OUT_Z', '').split()]
        except ValueError:
            zs = []
        for i, line in enumerate(l for l in open(f) if l.strip()):
            z = zs[i] if i < len(zs) else float('nan')
            out.append((i, line.strip(), 1 / (1 + z) if z == z else float('nan'), z))
    return out


def subsample_pcts(E):
    return [float(x) for x in E['SUBSAMPLES'].split(',') if x.strip()]


def kept_subsamples(E):
    """Leading SUBSAMPLES whose sum is KEEP_SUBSAMPLE_PCT (ValueError otherwise)."""
    want = float(E.get('KEEP_SUBSAMPLE_PCT') or 0)
    if want <= 0:
        return []
    acc, out = 0.0, []
    for p in subsample_pcts(E):
        out.append(p)
        acc += p
        if abs(acc - want) < 1e-6:
            return out
        if acc > want:
            break
    sums, a = [], 0.0
    for p in subsample_pcts(E):
        a += p
        sums.append('%g' % a)
    raise ValueError('KEEP_SUBSAMPLE_PCT=%g is not a sum of the leading SUBSAMPLES (%s): possible %s'
                     % (want, E['SUBSAMPLES'], ', '.join(sums)))


def keep_full_snaps(E, outs):
    """Rockstar snapshot numbers whose full particle data are kept."""
    if E.get('KEEP_FULL_SNAPS', '').strip():
        want = sorted({int(x) for x in E['KEEP_FULL_SNAPS'].replace(',', ' ').split()})
        bad = [s for s in want if s < 0 or s >= len(outs)]
        if bad:
            raise ValueError('KEEP_FULL_SNAPS %s outside 0..%d' % (bad, len(outs) - 1))
        return want
    keep = set()
    for zz in E.get('KEEP_FULL_Z', '').replace(',', ' ').split():
        z = float(zz)
        cand = [(abs(o[3] - z), o[0]) for o in outs if o[3] == o[3]]
        if not cand:
            raise ValueError('no redshifts for the outputs (output_steps.txt missing): use KEEP_FULL_SNAPS')
        keep.add(min(cand)[1])
    return sorted(keep)


def pct_tag(p):
    return 's%02d' % int(round(p))


def plan(E):
    """Resolved retention plan (no file system access beyond output_steps.txt)."""
    P = paths(E)
    outs = planned_outputs(E, P)
    full = keep_full_snaps(E, outs)
    subs = kept_subsamples(E)
    tables = [t.strip() for t in E.get('KEEP_HALO_TABLES', '').split(',') if t.strip()]
    return dict(outputs=outs, full=full, subsamples=subs, subsample_pct=sum(subs), tables=tables,
                trees=E.get('KEEP_TREES', '1') == '1', checkpoint=E.get('KEEP_CHECKPOINT', '0') == '1',
                full_z=[(s, outs[s][3]) for s in full], paths=P)


META_SIM = ['run.conf', 'p.par', 'output_steps.txt', 'snapnames.txt', 'README.txt', 'submit.sh']
META_HALO = ['rockstar_input.cfg', 'README.txt', 'rockstar.cfg', 'rs_input.cfg', 'auto-rockstar.cfg']


def kept_files(E, pl=None):
    """Every file to archive: [(category, snap or None, source path, path relative to SCRATCH)]."""
    pl = pl or plan(E)
    P = pl['paths']
    root = E['SCRATCH'].rstrip('/') + '/'
    out = []

    def add(cat, snap, src, rel=None):
        if os.path.isfile(src) and not os.path.islink(src):
            assert src.startswith(root), src
            out.append((cat, snap, src, rel or src[len(root):]))

    sub_tags = [pct_tag(p) for p in pl['subsamples']]
    full = set(pl['full'])
    for snap, step, _a, _z in pl['outputs']:
        for f in sorted(glob.glob(os.path.join(glob.escape(P['pkc']), 'snap.%s.s*.pkc' % step))):
            tag = os.path.basename(f).split('.')[2]
            if snap in full:
                add('pkc_full', snap, f)
            elif tag in sub_tags:
                add('pkc_sub', snap, f)
    comp = os.path.join(P['halo'], 'compressed')
    for f in sorted(glob.glob(os.path.join(glob.escape(comp), 'out_*_*.fits.gz'))):
        m = re.match(r'out_(\d+)_(\w+)\.fits\.gz$', os.path.basename(f))
        if m and m.group(2) in pl['tables']:
            add('halo_fits', int(m.group(1)), f)
    if pl['trees']:
        for f in sorted(glob.glob(os.path.join(glob.escape(comp), 'trees', '*'))):
            add('trees_fits', None, f)
    for f in sorted(glob.glob(os.path.join(glob.escape(P['halo']), 'xcorr', '*'))):
        add('xcorr', None, f)
    for n in META_SIM:
        add('meta', None, os.path.join(P['sim'], n))
    for f in sorted(glob.glob(os.path.join(glob.escape(P['sim']), '*.log'))):
        add('log', None, f)
    for f in sorted(glob.glob(os.path.join(glob.escape(P['sim']), 'logs', '*'))):
        add('log', None, f)
    for n in META_HALO:
        add('meta', None, os.path.join(P['halo'], n))
    for f in sorted(glob.glob(os.path.join(glob.escape(P['halo']), 'logs', '*'))):
        add('log', None, f)
    for f in sorted(glob.glob(os.path.join(glob.escape(P['halo']), '*.log'))):
        add('log', None, f)
    # cosmology.txt, transfer table, ...: shared by the realisations in SCRATCH (and rewritten by each set-up), so
    # each run's archive gets its own copy in sims/<stem>/r<real>/cosmology/
    for f in sorted(glob.glob(os.path.join(glob.escape(P['cosmo']), '*'))):
        add('cosmo', None, f, os.path.join(P['rel']['sim'], 'cosmology', os.path.basename(f)))
    add('meta', None, scales_file(P))
    for f in sorted(glob.glob(os.path.join(glob.escape(P['state']), 'events', '*.jsonl'))):
        add('events', None, f)
    for n in ('verify.json',):
        add('meta', None, os.path.join(P['state'], n))
    if pl['checkpoint']:
        chk = sorted(glob.glob(os.path.join(glob.escape(P['sim']), 'checkpoints', '*.chk')))
        if chk:
            base = chk[-1]
            for f in sorted(glob.glob(glob.escape(base) + '*')):
                add('checkpoint', None, f)
    return out


def scales_file(P):
    """consistent-trees' snapshot list of this run: <halo dir>/scales.txt (job_ctrees.pbs copies it there), else the
    shared <halo dir>/../scales.txt of older runs (overwritten by every realisation's ctrees stage)."""
    own = os.path.join(P['halo'], 'scales.txt')
    return own if os.path.isfile(own) else os.path.join(os.path.dirname(P['halo']), 'scales.txt')


def human(n):
    for u in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(n) < 1024 or u == 'TB':
            return '%.1f %s' % (n, u) if u != 'B' else '%d B' % n
        n /= 1024.0


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    E = load_conf(sys.argv[1])
    pl = plan(E)
    P = pl['paths']
    files = kept_files(E, pl)
    by = {}
    for cat, _s, src, _r in files:
        n, b = by.get(cat, (0, 0))
        by[cat] = (n + 1, b + os.path.getsize(src))
    if '--json' in sys.argv:
        print(json.dumps(dict(full=pl['full_z'], subsamples=pl['subsamples'], tables=pl['tables'],
                              trees=pl['trees'], checkpoint=pl['checkpoint'], scratch=E['SCRATCH'],
                              archive=E.get('ARCHIVE', ''), totals=by), indent=1))
        return
    print('%s r%s: scratch %s -> archive %s' % (E['STEM'], E['REAL'], E['SCRATCH'], E.get('ARCHIVE') or '(none)'))
    print('full snapshots: %s' % ', '.join('%d (z=%.3f, step %s)' % (s, z, pl['outputs'][s][1]) for s, z in pl['full_z']))
    print('other %d snapshots: %g%% of the particles (%s)' % (len(pl['outputs']) - len(pl['full']), pl['subsample_pct'],
                                                            ','.join(pct_tag(p) for p in pl['subsamples']) or 'none'))
    print('halo tables: %s; trees: %s; checkpoint: %s' % (','.join(pl['tables']), pl['trees'], pl['checkpoint']))
    tot = 0
    for cat, (n, b) in sorted(by.items()):
        print('  %-11s %6d files %12s' % (cat, n, human(b)))
        tot += b
    print('  %-11s %6d files %12s (files present now)' % ('total', len(files), human(tot)))
    if '--list' in sys.argv:
        for cat, s, _src, rel in files:
            print('%-11s %4s %s' % (cat, '' if s is None else s, rel))


if __name__ == '__main__':
    main()
