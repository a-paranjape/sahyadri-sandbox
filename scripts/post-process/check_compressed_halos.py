#!/usr/bin/env python3
"""
check_compressed_halos.py: independent check of compress_halo_outputs.py results against the ASCII files.

Parses the ASCII with numpy (not pandas, which the converter uses) and requires, for every column:
int64 columns equal exactly, float64 columns equal exactly, float32 columns equal float32(ASCII value).
Also checks .vahc alignment, readers.py-style access (ConvertToFITS.fits_to_record), tree files, the
FitsRow index of locations.fits.gz and forests.fits.gz, and that the headers read back (with the run's
SIM_STEM, SIM_REAL, SIM_IC_SEED). Run it before deleting any ASCII file.

usage: check_compressed_halos.py <run dir with out_*.trees> [--snaps 0,5,99] [--outdir DIR] [--trees] [--nproc N]
"""
import argparse
import itertools
import os
import sys

import fitsio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compress_halo_outputs as C  # noqa: E402
import halo_table_io as H  # noqa: E402

bad = 0


def ok(cond, msg):
    global bad
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    bad += (not cond)


def same(v, a):
    if v.dtype.kind == 'i':
        return np.array_equal(v, a.astype(np.int64))
    if v.dtype.itemsize == 8:
        return np.array_equal(v, a, equal_nan=True)
    return np.array_equal(v, a.astype(np.float32), equal_nan=True)


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
        V = H.read(vahc)   # text or binary
        with fitsio.FITS(os.path.join(out, 'out_%s_vahc.fits.gz' % snap)) as f:
            d = f[1].read()
        nb = [c for c in vn if c not in C.BASIC_COLS and (c not in d.dtype.names or not same(d[c], V[:, vn.index(c)]))]  # .vpot cols: below
        if 'b1' in vn:
            nb += [] if same(got['b1'], V[:, vn.index('b1')]) else ['b1 (basic)']
        ok(not nb, 'out_%s: .vahc columns equal the ASCII%s' % (snap, (' (differ: %s)' % nb) if nb else ''))
        ok(np.array_equal(V[:, vn.index('haloID')], A[:, names.index('id')]), 'out_%s: .vahc rows are the .trees halos in order' % snap)
        vpot = os.path.join(R, 'out_%s.vpot' % snap)
        if os.path.isfile(vpot):
            pn = C.halo_header_names(vpot, 'vpot')
            P = H.read(vpot)
            nb = [c for c in pn if c != 'haloID' and (c not in d.dtype.names or not same(d[c], P[:, pn.index(c)]))]
            ok(not nb and np.array_equal(P[:, pn.index('haloID')], A[:, names.index('id')]),
               'out_%s: .vpot columns (%d) in the vahc file equal the ASCII, rows in order%s' % (
                   snap, len(pn) - 1, (' (differ: %s)' % nb) if nb else ''))
    tens = os.path.join(R, 'out_%s.tens' % snap)
    if os.path.isfile(tens) and os.path.isfile(os.path.join(out, 'out_%s_tens.fits.gz' % snap)):
        tn = C.halo_header_names(tens, 'tens')
        V = H.read(tens)
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


def tree_plan(T, out):
    """[(FITS tree file, [(ASCII tree file, first row, rows or None)])]: one tree_X_Y_Z.fits.gz per tree_X_Y_Z.dat,
    or the merged tree_0_0_0.fits.gz (header MERGED, TILEnnnn cards) holding several."""
    dats = sorted(f for f in os.listdir(T) if f.startswith('tree_') and f.endswith('.dat'))
    merged = os.path.join(out, C.MERGED_NAME)
    if len(dats) > 1 and os.path.isfile(merged):
        st = C.StreamTable(merged)
        if st.card('MERGED'):
            tiles = [str(st.card('TILE%04d' % i)).split() for i in range(int(st.card('NTILES') or 0))]
            st.s.close()
            return [(C.MERGED_NAME, [(f, int(a), int(n)) for f, a, n, _t in tiles])], dats
        st.s.close()
    return [(d.replace('.dat', '.fits.gz'), [(d, 0, None)]) for d in dats], dats


def ascii_tree_chunks(path, nskip, ncol, chunk):
    """Rows of a tree_X_Y_Z.dat parsed with numpy (independent of the converter's pandas), chunk lines at a time."""
    with open(path) as fh:
        for _ in range(nskip):
            fh.readline()
        while True:
            lines = list(itertools.islice(fh, chunk))
            if not lines:
                return
            rows = [l for l in lines if l.strip() and not l.startswith('#')]
            if rows:
                yield np.loadtxt(rows, ndmin=2).reshape(len(rows), ncol)


def check_trees(T, out, chunk=1000000):
    """Every tree file against the ASCII, both read a chunk at a time (memory bounded: the tree files of a 2048^3 run
    hold ~1e9 rows); the FITS file through C.StreamTable (its DATASUMs checked too). Then locations (FitsRow,
    FitsRows select exactly each tree: a run of equal Tree_root_ID starting there, of that length) and forests."""
    plan, dats = tree_plan(T, out)
    covered = sorted(d for _f, tl in plan for d, _a, _n in tl)
    ok(covered == dats, 'trees: %d ASCII tree files, all in %d FITS file(s)' % (len(dats), len(plan)))
    runs = {}                                   # FITS file -> (roots, first rows, lengths) of the runs of Tree_root_ID
    for fits, tiles in plan:
        p = os.path.join(out, fits)
        if not os.path.isfile(p):
            ok(False, '%s missing' % fits)
            continue
        st = C.StreamTable(p)
        roots, starts, lens = [], [], []
        for name, first, nrows in tiles:
            cols, _, nskip, ntrees = C.parse_tree_header(os.path.join(T, name))
            if st.nrows - st.left != first:
                ok(False, '%s: starts at row %d of %s, expected %d' % (name, st.nrows - st.left, fits, first))
                break
            n, nb = 0, set()
            for A in ascii_tree_chunks(os.path.join(T, name), nskip, len(cols), chunk):
                d = st.read(len(A))
                if len(d) != len(A):
                    nb.add('(rows)')
                    break
                nb |= {c for j, c in enumerate(cols) if not same(d[c], A[:, j])}
                r = d['Tree_root_ID'].astype(np.int64)
                new = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
                for k, i in enumerate(new):
                    j = new[k + 1] if k + 1 < len(new) else len(r)
                    if lens and roots[-1] == r[i] and starts[-1] + lens[-1] == first + n + i and i == 0:
                        lens[-1] += j - i          # a tree continued from the previous chunk
                    else:
                        roots.append(int(r[i])); starts.append(first + n + i); lens.append(j - i)
                n += len(A)
            if nrows is not None and n != nrows:
                nb.add('(%d rows, TILE card %d)' % (n, nrows))
            ok(not nb, '%s: %d rows x %d columns equal the ASCII (%s trees)%s' % (
                name, n, len(cols), ntrees, (' (differ: %s)' % sorted(nb)) if nb else ''))
        rest = st.left
        bad_sum = st.finish()
        ok(rest == 0 and not bad_sum, '%s: no rows beyond the ASCII (%d), DATASUM of every HDU%s' % (
            fits, rest, (' (differs in HDU %s)' % bad_sum) if bad_sum else ''))
        runs[fits] = (np.array(roots, np.int64), np.array(starts, np.int64), np.array(lens, np.int64))
    with fitsio.FITS(os.path.join(out, 'locations.fits.gz')) as f:
        L = f[1].read()
    ff = (L['FitsFile'] if 'FitsFile' in L.dtype.names else
          np.char.replace(L['Filename'].astype('S'), b'.dat', b'.fits.gz'))
    good, ntree = True, 0
    for fits, (roots, starts, lens) in runs.items():
        sel = np.char.strip(ff.astype('S')) == fits.encode()
        o = np.argsort(starts)
        i = np.searchsorted(starts[o], L['FitsRow'][sel])
        i = np.clip(i, 0, max(len(o) - 1, 0))
        j = o[i] if len(o) else i
        nr = L['FitsRows'][sel]                  # -1 when locations.dat has no Num_Halos: not checked
        good &= bool(len(o) and np.all(starts[j] == L['FitsRow'][sel]) and np.all(roots[j] == L['TreeRootID'][sel])
                     and np.all((nr < 0) | (lens[j] == nr)))
        good &= int(sel.sum()) == len(roots)        # every run of Tree_root_ID is one tree of locations
        ntree += int(sel.sum())
    ok(good and ntree == len(L), 'locations: FitsRow/FitsRows select exactly each tree (%d trees)' % len(L))
    with fitsio.FITS(os.path.join(out, 'forests.fits.gz')) as f:
        Fo = f[1].read()
    FA = np.loadtxt(os.path.join(T, 'forests.list'), dtype=np.int64, ndmin=2)
    ok(np.array_equal(Fo['TreeRootID'], FA[:, 0]) and np.array_equal(Fo['ForestID'], FA[:, 1]), 'forests equal the ASCII')


def _task(task):
    """One snapshot (or 'trees') in a worker: its PASS/FAIL lines and failure count."""
    global bad
    import contextlib
    import io
    kind, rundir, out, snap = task
    bad, buf = 0, io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            if kind == 'trees':
                check_trees(os.path.join(rundir, 'trees'), os.path.join(out, 'trees'))
            else:
                check_snapshot(rundir, out, snap)
        except Exception as e:  # a crash in one snapshot is a failure of that snapshot, not of the others
            ok(False, '%s: check raised %s: %s' % (snap if kind != 'trees' else 'trees', type(e).__name__, e))
    return buf.getvalue(), bad


def main():
    global bad
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rundir')
    ap.add_argument('--outdir')
    ap.add_argument('--snaps', default='all')
    ap.add_argument('--trees', action='store_true', help='also check <rundir>/trees against <outdir>/trees')
    ap.add_argument('--nproc', type=int, default=1,
                    help='snapshots checked in parallel (the tree check runs as one more task); memory per process '
                         'is about one snapshot: its .trees parsed as float64 (2048^3: ~15-30 GB)')
    a = ap.parse_args()
    out = a.outdir or os.path.join(a.rundir, 'compressed')
    tasks = ([('trees', a.rundir, out, None)] if a.trees else []) + \
            [('snap', a.rundir, out, s) for s in C.select_snaps(a.rundir, a.snaps)]
    if a.nproc <= 1 or len(tasks) <= 1:
        res = map(_task, tasks)
    else:
        import multiprocessing as mp
        pool = mp.Pool(min(a.nproc, len(tasks)), maxtasksperchild=1)
        res = pool.imap(_task, tasks)   # in order: the output reads as the serial run
    total, text_trees = 0, ''
    for t, (text, nbad) in zip(tasks, res):
        total += nbad
        if t[0] == 'trees':
            text_trees = text          # printed after the snapshots, as before
        else:
            sys.stdout.write(text)
            sys.stdout.flush()
    sys.stdout.write(text_trees)
    bad = total
    print('== %d failure(s)' % bad)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
