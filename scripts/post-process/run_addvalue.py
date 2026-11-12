#!/usr/bin/env python3
"""
run_addvalue.py: value-added halo catalogs (out_<snap>.vahc) and tidal tensors (out_<snap>.tens).

The same computation as the add_value branch of postprocess.py (production settings: kmax = 0.1,
massdef = m200b, seed 42, halos in .trees order), with arguments instead of hard-coded flags, one
density field per snapshot for both outputs, snapshots in parallel, and restartable (snapshots whose
outputs exist are skipped).

Inputs (readers.py layout; set SAHYADRI_SCRATCH to the directory holding sims/ and halos/):
  particles  $SAHYADRI_SCRATCH/sims/<stem>/r<real>/   Gadget-4 HDF5 snapshot_<snap>.hdf5, or pkdgrav3
             .pkc files with snapnames.txt (line <snap> = pkdgrav3 step of Rockstar snapshot <snap>)
  halos      $SAHYADRI_SCRATCH/halos/<stem>/r<real>/out_<snap>.trees (consistent-trees hlists) and
             $SAHYADRI_SCRATCH/halos/<stem>/scales.txt
Outputs: out_<snap>.vahc / out_<snap>.tens next to out_<snap>.trees.

usage: run_addvalue.py <stem> [--real 1] [--snaps all|0-100|5,7] [--grid 512] [--down-to 512]
                       [--no-vahc] [--tens] [--nproc 4] [--overwrite] [--fast] [--pot]
--fast uses vahc_fast.py (same quantities, much faster; see that file); without it addvalue.AddValue as before.
--pot (with --fast) also writes out_<snap>.vpot for snapshots whose .pkc files store the potential, and for Gadget-4
HDF5 snapshots with a Potential field (vpot_hdf5.py; all particles, --down-to 0).
Memory per process is about 12 x 16 x grid^3 bytes (grid 512: ~26 GB; 768: ~85 GB). With .pkc input and no
downsampling (--down-to 0, the default) the density is built block by block, so the particles add almost
nothing at any resolution; with --down-to N (or Gadget input) the N^3 (all) positions are held at once.
"""
import argparse
import gc
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def snaps_of(args):
    from utilities import Paths
    P = Paths()
    import numpy as np
    own = P.halo_path + args.stem + '/r%d/scales.txt' % args.real     # this run's copy (job_ctrees.pbs)
    sc = np.loadtxt(own if os.path.isfile(own) else P.halo_path + args.stem + '/scales.txt', ndmin=2)
    have = [int(s) for s in sc[:, 0]]
    if getattr(args, 'pk', False):     # P(k) of every output, also those without halo catalogues
        names = P.sim_path + args.stem + '/r%d/snapnames.txt' % args.real
        have = sorted(set(have) | set(range(sum(1 for l in open(names) if l.strip()))))
    if args.snaps == 'all':
        return have
    out = []
    for part in args.snaps.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [s for s in out if s in have]


def streamed_density(pkc, ps, threads):
    """CIC density contrast of all particles of a .pkc snapshot, built block by block.

    Same result, bit for bit, as ps.density_field(pkc.read_block('pos')): the blocks are added in the
    order read() stores them, with the same float32 rounding of the positions, through the same C
    routine. Memory is the grid plus a few decoded blocks (65536 particles each) per thread, independent
    of the number of particles, so a 2048^3 snapshot needs no downsampling.
    """
    from concurrent.futures import ThreadPoolExecutor
    from ctypes import c_long, c_int, c_double
    from numpy.ctypeslib import ndpointer
    import numpy as np
    upd = ps.lib_cic.update_density
    upd.restype = None
    upd.argtypes = [c_long, c_int, ndpointer(c_double, flags="C_CONTIGUOUS"),
                    ndpointer(c_double, flags="C_CONTIGUOUS"), ndpointer(c_double, flags="C_CONTIGUOUS"),
                    ndpointer(c_double, flags="C_CONTIGUOUS"), c_double, c_double, c_double, c_double]
    density = np.zeros(ps.grid**3)
    jobs = [(f, i) for f in pkc.files for i in range(len(f.counts))]

    def decode(job):
        f, i = job
        h = f.header
        X = f.block(i)[0]
        return ((X + 0.5) / h["q"] * h["box_size"]).astype(np.float32)  # as PkcSnapshot.read(units='mpc')

    n = 0
    chunk = 4 * max(1, threads)  # blocks in flight: bounds memory, keeps the order
    with ThreadPoolExecutor(max(1, threads)) as ex:
        for k in range(0, len(jobs), chunk):
            for x in ex.map(decode, jobs[k:k + chunk]):
                upd(c_long(x.shape[0]), c_int(ps.grid), density,
                    np.ascontiguousarray(x[:, 0], np.float64), np.ascontiguousarray(x[:, 1], np.float64),
                    np.ascontiguousarray(x[:, 2], np.float64),
                    c_double(ps.cell_size_x), c_double(ps.cell_size_y), c_double(ps.cell_size_z), c_double(0.0))
                n += x.shape[0]
    if n != pkc.npart:
        raise ValueError('decoded %d particles, header says %d' % (n, pkc.npart))
    density = np.reshape(density, (ps.grid, ps.grid, ps.grid), order='F')
    density = density * ps.grid**3 / (1.0 * n + ps.TINY) - 1.0
    return density, n


def do_snap(snap, args):
    from readers import SnapshotReader, HaloReader
    from correlations import PowerSpectrum
    from addvalue import AddValue
    t0 = time.time()
    hr = HaloReader(sim_stem=args.stem, real=args.real, snap=snap, verbose=False)
    base = hr.halo_path + hr.halocat_stem
    want = [k for k, on in (('vahc', args.vahc), ('tens', args.tens), ('vpot', args.pot), ('pk', args.pk)) if on]
    todo = [k for k in want if args.overwrite or not os.path.isfile(base + '.' + k)]
    if not todo:
        return '%d: skipped (exists)' % snap
    if not os.path.isfile(base + '.trees'):
        if todo == ['pk'] or (args.pk and 'pk' in todo):
            return _pk_only(snap, args, base, t0)
        return '%d: no %s.trees' % (snap, base)
    sr = SnapshotReader(sim_stem=args.stem, real=args.real, snap=snap, verbose=False)
    if 'vpot' in todo and not (getattr(getattr(sr, 'pkc', None), 'has_potential', False) or _hdf5_pot(sr)):
        todo.remove('vpot')     # .vpot only for snapshots that store the potential
        if not todo:
            return '%d: skipped (outputs exist; no potential in this snapshot, no .vpot)' % snap
    ps = PowerSpectrum(grid=args.grid, Lbox=sr.Lbox, verbose=False)
    if args.fast:
        return _fast_snap(snap, args, hr, sr, ps, base, todo, t0)
    if getattr(sr, 'pkc', None) is not None and args.stream and (args.down_to == 0 or args.down_to**3 >= sr.npart):
        density, npos = streamed_density(sr.pkc, ps, args.threads)
    else:
        pos = sr.read_block('pos', down_to=args.down_to, seed=args.seed)
        npos = pos.shape[1]
        density = ps.density_field(pos)
        del pos
    FT = ps.fourier_transform_density(density)
    del density
    gc.collect()
    t1 = time.time()
    hpos, halos = hr.prep_halos(QE=None, Npmin=0.0, keep_subhalos=True, sorthalos=False)
    av = AddValue(sim_stem=args.stem, real=args.real, snap=snap, grid=args.grid, kmax=args.kmax,
                  massdef=args.massdef, ps=ps, density=FT, hpos=hpos, halos=halos, input_is_FTdensity=True,
                  verbose=False)
    # AddValue writes <halo_path><stem>/r<real>/out_<snap>.<k>; point it to a temporary tree on the same
    # file system and rename when complete, so an interrupted run never leaves a partial file behind
    tmproot = os.path.join(os.path.dirname(base), '.addvalue_tmp_%d' % snap, '')
    os.makedirs(os.path.dirname(tmproot + hr.halocat_stem), exist_ok=True)
    av.halo_path = tmproot
    for k in todo:
        part = tmproot + hr.halocat_stem + '.' + k
        if os.path.isfile(part):
            os.remove(part)
        av.add_value(write_vahc=True, only_tensor=(k == 'tens'))
        os.replace(part, base + '.' + k)
        gc.collect()
    import shutil
    shutil.rmtree(tmproot, ignore_errors=True)
    return '%d: %s; %d halos; %d particles (%s); read+density %.0f s, add_value %.0f s' % (
        snap, '+'.join(todo), halos.size, npos, sr.snapshot_file + sr.snapshot_ext, t1 - t0, time.time() - t1)


def _pk_only(snap, args, base, t0):
    """Matter P(k) of a snapshot without a halo catalogue (early outputs): density pass only."""
    from readers import SnapshotReader
    from correlations import PowerSpectrum
    import pk_measure
    sr = SnapshotReader(sim_stem=args.stem, real=args.real, snap=snap, verbose=False)
    ps = PowerSpectrum(grid=args.grid, Lbox=sr.Lbox, verbose=False)
    if getattr(sr, 'pkc', None) is not None and args.stream and (args.down_to == 0 or args.down_to**3 >= sr.npart):
        from vahc_fast import streamed_fields
        density, npos, _ = streamed_fields(sr.pkc, ps, args.threads)
    else:
        pos = sr.read_block('pos', down_to=args.down_to, seed=args.seed)
        npos = pos.shape[1]
        density = ps.density_field(pos)
        del pos
    t1 = time.time()
    gp = pk_measure.GridPower(ps)
    B = gp.transform(density, args.threads)
    del density
    meta = dict(snap=snap, redshift=float(sr.redshift), scale=1.0 / (1.0 + float(sr.redshift)),
                snapshot=sr.snapshot_file + sr.snapshot_ext, mpart=float(sr.mpart), halos='no catalogue')
    m, cols, tab = pk_measure.measure(gp, B, npos, meta=meta, klin=args.pk_klin)
    pk_measure.write(base + '.pk', m, cols, tab)
    return '%d: pk only (no halo catalogue); %d particles; read+density %.0f s, pk %.0f s' % (
        snap, npos, t1 - t0, time.time() - t1)


def _hdf5_pot(sr):
    from vpot_hdf5 import has_potential
    return has_potential(sr)


def _fast_snap(snap, args, hr, sr, ps, base, todo, t0):
    """vahc_fast.FastAddValue: .vahc, .tens (and .vpot) from one pass (see vahc_fast.py); files written to a
    temporary name and renamed when complete."""
    from vahc_fast import FastAddValue, streamed_fields
    hpos, halos = hr.prep_halos(QE=None, Npmin=0.0, keep_subhalos=True, sorthalos=False)
    vpot = None
    if getattr(sr, 'pkc', None) is not None and args.stream and (args.down_to == 0 or args.down_to**3 >= sr.npart):
        density, npos, vpot = streamed_fields(sr.pkc, ps, args.threads, halos=halos if 'vpot' in todo else None,
                                              hpos=hpos, Om=hr.Om)
    elif 'vpot' in todo and args.down_to == 0 and _hdf5_pot(sr):   # Gadget-4 HDF5 with Potential (vpot_hdf5.py)
        from vpot_hdf5 import hdf5_fields
        density, npos, vpot = hdf5_fields(sr, ps, args.threads, halos, hpos, hr.Om)
    else:
        pos = sr.read_block('pos', down_to=args.down_to, seed=args.seed)
        npos = pos.shape[1]
        density = ps.density_field(pos)
        del pos
        if 'vpot' in todo:
            todo.remove('vpot')
    t1 = time.time()
    tm = {}
    av = FastAddValue(density, threads=args.threads, timing=tm, vpot=vpot, sim_stem=args.stem, real=args.real,
                      snap=snap, grid=args.grid, kmax=args.kmax, massdef=args.massdef, ps=ps, hpos=hpos,
                      halos=halos, verbose=False)
    del density
    gc.collect()
    kinds = [k for k in todo if k != 'pk']
    res = av.compute(want=kinds) if kinds else {}
    for k in todo:
        if k == 'pk':
            continue
        part = base + '.' + k + '.part'
        av.write(k, res[k], part)
        os.replace(part, base + '.' + k)
    if 'pk' in todo:   # matter and halo spectra from the same transformed density (pk_measure.py)
        import pk_measure
        t2 = time.time()
        gp = pk_measure.GridPower(ps)
        meta = dict(snap=snap, redshift=float(sr.redshift), scale=1.0 / (1.0 + float(sr.redshift)),
                    snapshot=sr.snapshot_file + sr.snapshot_ext, mpart=float(sr.mpart))
        m, cols, tab = pk_measure.measure(gp, av.B, npos, halos=halos, hpos=hpos, mpart=float(sr.mpart),
                                          mbins=args.pk_mbins, threads=args.threads, klin=args.pk_klin, meta=meta)
        pk_measure.write(base + '.pk', m, cols, tab)
        tm['pk'] = time.time() - t2
    return '%d: %s (fast, %d threads); %d halos; %d particles (%s); read+density %.0f s, add_value %.0f s (%s)' % (
        snap, '+'.join(todo), args.threads, halos.size, npos, sr.snapshot_file + sr.snapshot_ext, t1 - t0,
        time.time() - t1, ', '.join('%s %.0f' % kv for kv in tm.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stem', help='e.g. sahyadri/default2048 (sims/<stem>, halos/<stem>)')
    ap.add_argument('--real', type=int, default=1)
    ap.add_argument('--snaps', default='all')
    ap.add_argument('--grid', type=int, default=512, help='density grid (production: 512; 768 for 2048^3)')
    ap.add_argument('--down-to', type=int, default=0,
                    help='use a random down_to^3 particles (production: 512 when N > 512^3; 0 = all)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--kmax', type=float, default=0.1)
    ap.add_argument('--massdef', default='m200b')
    ap.add_argument('--no-vahc', dest='vahc', action='store_false')
    ap.add_argument('--tens', action='store_true', help='also write .tens')
    ap.add_argument('--nproc', type=int, default=1, help='snapshots in parallel')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--no-stream', dest='stream', action='store_false',
                    help='.pkc without downsampling: load all positions at once instead of building the density '
                         'block by block (same result, ~36 B per particle more memory; for tests)')
    ap.add_argument('--threads', type=int, default=0,
                    help='decoding threads per process (default: $NCPUS or the core count, divided by --nproc)')
    ap.add_argument('--fast', action='store_true',
                    help='vahc_fast.py: one pass for .vahc and .tens, multithreaded real FFTs, no per-halo loops '
                         '(same results to rounding; uses --threads FFT threads per process)')
    ap.add_argument('--pot', action='store_true',
                    help='also write out_<snap>.vpot (mean particle potential within rs, R200b and at the .vahc scales; '
                         'only snapshots that store the potential; implies --fast)')
    ap.add_argument('--pk', action='store_true',
                    help='also write out_<snap>.pk: matter and halo-mass-bin power spectra and biases from the same '
                         'density grid (pk_measure.py), for every output (density-only pass without a halo catalogue)')
    ap.add_argument('--pk-mbins', default=None, help='halo mass bin edges in particles (default %s)' % '100,300,1000,3000,10000')
    ap.add_argument('--pk-klin', type=float, default=0.1, help='k [h/Mpc] up to which the large-scale bias is averaged')
    args = ap.parse_args()
    if args.pk:
        args.fast = True
    if args.pot:
        args.fast = True
    if args.threads <= 0:
        args.threads = max(1, int(os.environ.get('NCPUS', os.cpu_count() or 1)) // max(1, args.nproc))
    if not os.environ.get('SAHYADRI_SCRATCH'):
        print('note: SAHYADRI_SCRATCH not set; using the paths hard-coded in utilities.Paths')
    snaps = snaps_of(args)
    print('snapshots %s, grid %d, down_to %d, vahc %s, tens %s, %d processes' % (
        snaps, args.grid, args.down_to, args.vahc, args.tens, args.nproc), flush=True)
    t0, bad = time.time(), 0
    if args.nproc == 1:
        results = (do_snap(s, args) for s in snaps)
    else:
        pool = mp.get_context('spawn').Pool(args.nproc, maxtasksperchild=1)
        results = pool.imap_unordered(_job, [(s, args) for s in snaps])
    for r in results:
        print(time.strftime('[%H:%M:%S] ') + r, flush=True)
        bad += ': no ' in r
    print('done in %.0f s' % (time.time() - t0))
    return 1 if bad else 0


def _job(a):
    try:
        return do_snap(*a)
    except Exception as e:  # report and go on with the other snapshots
        import traceback
        return '%d: no output, %s\n%s' % (a[0], repr(e), traceback.format_exc())


if __name__ == '__main__':
    sys.exit(main())
