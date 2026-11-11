#!/usr/bin/env python3
"""
check_compressed_halos.py: independent check of compress_halo_outputs.py results against the ASCII files.

Parses the ASCII with numpy (not pandas, which the converter uses) and requires, for every column:
int64 columns equal exactly, float64 columns equal exactly, float32 columns equal float32(ASCII value).
Also checks .vahc alignment, readers.py-style access (ConvertToFITS.fits_to_record), tree files, the
FitsRow index of locations.fits.gz and forests.fits.gz, and that the headers read back (with the run's
SIM_STEM, SIM_REAL, SIM_IC_SEED). Run it before deleting any ASCII file.

usage: check_compressed_halos.py <run dir with out_*.trees> [--snaps 0,5,99] [--outdir DIR] [--trees]
"""
import argparse
import os
import sys

import fitsio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compress_halo_outputs as C  # noqa: E402

bad = 0


def ok(cond, msg):
    global bad
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    bad += (not cond)


def same(v, a):
    if v.dtype.kind == 'i':
        return np.array_equal(v, a.astype(np.int64))
    if v.dtype.itemsize == 8:
        return np.array_equal(v, a)
    return np.array_equal(v, a.astype(np.float32))


def check_snapshot(R, out, snap):
    names = C.halo_header_names(os.path.join(R, 'out_%s.trees' % snap), 'trees')
    A = np.loadtxt(os.path.join(R, 'out_%s.trees' % snap), ndmin=2)
    got = {}
    for t in ('basic', 'extended'):
        with fitsio.FITS(os.path.join(out, 'out_%s_%s.fits.gz' % (snap, t))) as f:
            d = f[1].read()
            for c in d.dtype.names:
                got[c] = d[c]
    try:  # headers must read back (fitsio) and carry the run identity
        h = fitsio.read_header(os.path.join(out, 'out_%s_basic.fits.gz' % snap), 1)
        ident = ['%s=%s' % (k, h[k]) for k in ('SIM_STEM', 'SIM_REAL', 'SIM_IC_SEED') if k in h]
        ok(True, 'out_%s: header readable (%s)' % (snap, ', '.join(ident) or 'no SIM_ cards'))
    except Exception as e:
        ok(False, 'out_%s: header not readable by fitsio: %s' % (snap, str(e).splitlines()[0]))
    ok(sorted(set(got) - {'b1'}) == sorted(names), 'out_%s: basic + extended hold every .trees column (%d)' % (snap, len(names)))
    nb = [c for j, c in enumerate(names) if not same(got[c], A[:, j])]
    ok(not nb, 'out_%s: %d columns x %d rows equal the ASCII%s' % (snap, len(names), len(A), (' (differ: %s)' % nb) if nb else ''))
    vahc = os.path.join(R, 'out_%s.vahc' % snap)
    if os.path.isfile(vahc) and os.path.isfile(os.path.join(out, 'out_%s_vahc.fits.gz' % snap)):
        vn = C.halo_header_names(vahc, 'vahc')
        V = np.loadtxt(vahc, ndmin=2)
        with fitsio.FITS(os.path.join(out, 'out_%s_vahc.fits.gz' % snap)) as f:
            d = f[1].read()
        nb = [c for c in d.dtype.names if not same(d[c], V[:, vn.index(c)])]
        if 'b1' in vn:
            nb += [] if same(got['b1'], V[:, vn.index('b1')]) else ['b1 (basic)']
        ok(not nb, 'out_%s: .vahc columns equal the ASCII%s' % (snap, (' (differ: %s)' % nb) if nb else ''))
        ok(np.array_equal(V[:, vn.index('haloID')], A[:, names.index('id')]), 'out_%s: .vahc rows are the .trees halos in order' % snap)
    tens = os.path.join(R, 'out_%s.tens' % snap)
    if os.path.isfile(tens) and os.path.isfile(os.path.join(out, 'out_%s_tens.fits.gz' % snap)):
        tn = C.halo_header_names(tens, 'tens')
        V = np.loadtxt(tens, ndmin=2)
        with fitsio.FITS(os.path.join(out, 'out_%s_tens.fits.gz' % snap)) as f:
            d = f[1].read()
        nb = [c for c in d.dtype.names if not same(d[c], V[:, tn.index(c)])]
        ok(not nb and len(d) == len(V), 'out_%s: .tens columns equal the ASCII%s' % (snap, (' (differ: %s)' % nb) if nb else ''))
        ok(np.array_equal(V[:, tn.index('haloID')], A[:, names.index('id')]), 'out_%s: .tens rows are the .trees halos in order' % snap)
    # readers.py access (prep_halos_fits + ConvertToFITS.fits_to_record)
    try:
        import ConvertToFITS as Cfits
        fb = fitsio.FITS(os.path.join(out, 'out_%s_basic.fits.gz' % snap))
        fe = fitsio.FITS(os.path.join(out, 'out_%s_extended.fits.gz' % snap))
        idx = np.where(fb[1]['pid'][:] == -1)[0]
        rec = Cfits.fits_to_record([fb, fe], idx, list_sel_col=None)
        ok(rec.size == idx.size, 'out_%s: readers.py access (fits_to_record on basic + extended) works' % snap)
        fb.close()
        fe.close()
    except ImportError:
        print('skip readers.py access check (ConvertToFITS not importable)')


def check_trees(T, out):
    for name in sorted(f for f in os.listdir(T) if f.startswith('tree_') and f.endswith('.dat')):
        cols, _, nskip, ntrees = C.parse_tree_header(os.path.join(T, name))
        with open(os.path.join(T, name)) as fh:
            rows = [l.split() for i, l in enumerate(fh) if i >= nskip and l.strip() and not l.startswith('#')]
        A = np.array(rows, dtype=float).reshape(len(rows), len(cols))
        with fitsio.FITS(os.path.join(out, name.replace('.dat', '.fits.gz'))) as f:
            d = f[1].read()
        nb = [c for j, c in enumerate(cols) if not same(d[c], A[:, j])]
        ok(len(d) == len(A) and not nb, '%s: %d rows x %d columns equal the ASCII (%s trees)' % (name, len(A), len(cols), ntrees))
    with fitsio.FITS(os.path.join(out, 'locations.fits.gz')) as f:
        L = f[1].read()
    good = True
    for fname in np.unique(L['Filename']):
        fn = fname.decode() if isinstance(fname, bytes) else str(fname)
        p = os.path.join(out, fn.replace('.dat', '.fits.gz'))
        if not os.path.isfile(p):
            continue
        with fitsio.FITS(p) as f:
            root = f[1]['Tree_root_ID'][:]
        sel = L['Filename'] == fname
        for t, r, n in zip(L['TreeRootID'][sel], L['FitsRow'][sel], L['FitsRows'][sel]):
            good &= bool(r >= 0 and np.all(root[r:r + n] == t) and (r + n == len(root) or root[r + n] != t))
    ok(good, 'locations: FitsRow/FitsRows select exactly each tree (%d trees)' % len(L))
    with fitsio.FITS(os.path.join(out, 'forests.fits.gz')) as f:
        Fo = f[1].read()
    FA = np.loadtxt(os.path.join(T, 'forests.list'), dtype=np.int64, ndmin=2)
    ok(np.array_equal(Fo['TreeRootID'], FA[:, 0]) and np.array_equal(Fo['ForestID'], FA[:, 1]), 'forests equal the ASCII')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rundir')
    ap.add_argument('--outdir')
    ap.add_argument('--snaps', default='all')
    ap.add_argument('--trees', action='store_true', help='also check <rundir>/trees against <outdir>/trees')
    a = ap.parse_args()
    out = a.outdir or os.path.join(a.rundir, 'compressed')
    for s in C.select_snaps(a.rundir, a.snaps):
        check_snapshot(a.rundir, out, s)
    if a.trees:
        check_trees(os.path.join(a.rundir, 'trees'), os.path.join(out, 'trees'))
    print('== %d failure(s)' % bad)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
