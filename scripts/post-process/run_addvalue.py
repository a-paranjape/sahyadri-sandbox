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
                       [--no-vahc] [--tens] [--nproc 4] [--overwrite]
Memory per process is about 12 x 16 x grid^3 bytes (grid 512: ~26 GB; 768: ~85 GB) plus the particles.
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
    sc = np.loadtxt(P.halo_path + args.stem + '/scales.txt', ndmin=2)
    have = [int(s) for s in sc[:, 0]]
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


def do_snap(snap, args):
    from readers import SnapshotReader, HaloReader
    from correlations import PowerSpectrum
    from addvalue import AddValue
    t0 = time.time()
    hr = HaloReader(sim_stem=args.stem, real=args.real, snap=snap, verbose=False)
    base = hr.halo_path + hr.halocat_stem
    want = [k for k, on in (('vahc', args.vahc), ('tens', args.tens)) if on]
    todo = [k for k in want if args.overwrite or not os.path.isfile(base + '.' + k)]
    if not todo:
        return '%d: skipped (exists)' % snap
    if not os.path.isfile(base + '.trees'):
        return '%d: no %s.trees' % (snap, base)
    sr = SnapshotReader(sim_stem=args.stem, real=args.real, snap=snap, verbose=False)
    ps = PowerSpectrum(grid=args.grid, Lbox=sr.Lbox, verbose=False)
    pos = sr.read_block('pos', down_to=args.down_to, seed=args.seed)
    npos = pos.shape[1]
    FT = ps.fourier_transform_density(ps.density_field(pos))
    del pos
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
    args = ap.parse_args()
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
