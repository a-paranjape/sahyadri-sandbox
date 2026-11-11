#!/usr/bin/env python3
"""Mean gravitational potential as a function of distance from halo centres, stacked in halo-mass bins, with its
scatter. Works on Gadget-4 HDF5 snapshots, pkdgrav3 HDF5 snapshots and pkdgrav3 .pkc snapshots, so the same halos
statistics can be compared between codes.

usage:
  halo_potential_profile.py <kind> <snapshot> <halos> <out.npz> [options]
    kind      gadget   Gadget-4 HDF5: snapshot_<nnn>.hdf5 or the base of snapshot_<nnn>.<i>.hdf5
              pkd      pkdgrav3 HDF5 (snap.<step>)
              pkc      pkdgrav3 compressed snapshot, prefix snap.<step> (needs the potential: dCompPotStep > 0)
    halos     consistent-trees hlist (out_<snap>.trees) or Rockstar out_<snap>.list
  options:
    --mbins 1e13,1e14,1e15,1e16   halo mass bin edges (Msun/h, column mvir by default; --mass-col)
    --rbins 0.05,20,24            log radial bins: rmin, rmax, number
    --r-units rvir|mpc            radii in units of each halo's Rvir (default) or comoving Mpc/h
    --all-halos                   include subhaloes (default: hosts only, pid = -1)
    --max-halos 2000              at most this many halos per mass bin (random, seed 0): bounds the work
    --par p.par                   pkdgrav3 parameter file (dSoft) to remove pkdgrav3's softened self-potential
    --offset X                    subtract X (km/s)^2 from every potential (e.g. offset_median of particles.csv)
    --pkc-subsamples 1,3          .pkc only: use only these subsample percents (unbiased for the mean)
    --chunk 2097152               particles per work unit (HDF5)
    --workers N                   processes (default: all cores)

Potentials are peculiar, physical, (km/s)^2, with the conventions of compare_potential.py:
  Gadget-4 Potential / a; pkdgrav3 Potential * U^2 / a (U = 100 L / sqrt(8 pi/3) km/s), .pkc as stored; with --par
  the pkdgrav3 self-term -(105/64) m / dSoft * U^2 / a is removed.

Algorithm (memory independent of the particle number): the particles are processed in chunks in parallel processes.
For each chunk and each mass bin, all (halo, particle) pairs closer than the bin's largest radius come from a
periodic KD-tree of the chunk queried with the (small) tree of halo centres (cKDTree.sparse_distance_matrix); each
pair is binned in r (or r / Rvir) and np.bincount accumulates count, sum phi and sum phi^2 per (halo, shell). Only
these per-halo arrays (n_halo x n_r) are kept and summed over chunks.

Output (npz), per mass bin b and shell i:
  r_mid, r_edges, m_edges, n_halos[b]
  stack_mean[b,i], stack_std[b,i]    all particles of all halos in the bin together (particle-weighted)
  halo_mean[b,i], halo_std[b,i]      mean and standard deviation over halos of each halo's shell mean
  halo_sem[b,i]                      halo_std / sqrt(number of halos with particles in the shell)
  npart[b,i], nhalo_shell[b,i]
  per halo: halo_shell_mean_<b> (n_halo x n_r, NaN where empty), halo_mass_<b>, halo_rvir_<b>
"""
import argparse
import warnings
import glob
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from scipy.spatial import cKDTree

H0 = 100.0


# ------------------------------------------------------------------ halos
def read_halos(path, mass_col):
    with open(path) as f:
        first = f.readline()
    names = [re.sub(r'\(\d+\)$', '', h) for h in first.lstrip('#').split()]
    d = np.loadtxt(path, comments='#', ndmin=2)
    if d.shape[0] == 0 or d.size == 0:
        return {k: np.zeros(0) for k in names}
    h = {k: d[:, i] for i, k in enumerate(names)}
    if 'x' not in h:          # Rockstar out_*.list: '#ID DescID Mvir Vmax Vrms Rvir Rs Np X Y Z ...'
        low = {k.lower(): v for k, v in h.items()}
        h = dict(x=low['x'], y=low['y'], z=low['z'], rvir=low['rvir'], mvir=low['mvir'], pid=np.full(len(d), -1.0),
                 id=low['id'])
    if mass_col not in h:
        sys.exit('no column %s in %s (have %s)' % (mass_col, path, ', '.join(h)))
    h['mass'] = h[mass_col]
    return h


# ------------------------------------------------------------------ particle sources: list of work units + reader
def gadget_files(path):
    if path.endswith('.hdf5'):
        return [path]
    fl = sorted(glob.glob(path + '.*.hdf5'), key=lambda f: int(f.split('.')[-2]))
    if not fl:
        sys.exit('no %s.hdf5 / %s.<i>.hdf5' % (path, path))
    return fl


def source_info(kind, path, par):
    """-> (box, a, n_total, units dict, list of work units)"""
    if kind in ('gadget', 'pkd'):
        files = gadget_files(path) if kind == 'gadget' else [path]
        with h5py.File(files[0], 'r') as f:
            hd = f['Header'].attrs
            L = float(hd['BoxSize'])
            ntot = int(hd['NumPart_Total'][1])
            a = float(hd['Time']) if kind == 'gadget' else 1 / (1 + float(hd['Redshift']))  # pkdgrav3 Time = code time
        units = {}
        if kind == 'pkd':
            U = H0 * L / math.sqrt(8 * math.pi / 3)
            units['scale'] = U**2 / a
            units['shift'] = -0.5
        else:
            units['scale'] = 1 / a
        units['L'] = L
        return L, a, ntot, units, files
    if kind == 'pkc':
        import pkcreader
        s = pkcreader.PkcSnapshot(path)
        if not s.has_potential:
            sys.exit('%s: no potential stored (dCompPotStep = 0)' % path)
        return s.Lbox, s.scale, s.npart, {'L': s.Lbox}, s
    sys.exit('kind must be gadget, pkd or pkc')


def units_for_self(kind, L, a, ntot, par):
    """pkdgrav3 softened self-potential in (km/s)^2 (0 for Gadget-4 or without --par)."""
    if kind == 'gadget' or not par:
        return 0.0
    d = {}
    for l in open(par):
        m = re.match(r'\s*(dSoft|dOmega0)\s*=\s*(.+?)\s*$', l)
        if m:
            d[m.group(1)] = float(eval(m.group(2), {}, {}))
    U = H0 * L / math.sqrt(8 * math.pi / 3)
    return -(105.0 / 64.0) * d['dOmega0'] / ntot / d['dSoft'] * U**2 / a


# worker globals (set by init)
G = {}


def init(kind, units, bins, rmode, edges, subtract):
    G.update(kind=kind, units=units, bins=bins, rmode=rmode, edges=edges, subtract=subtract,
             trees=[cKDTree(b['c'], boxsize=units['L']) if len(b['c']) else None for b in bins])


def load_unit(u):
    kind, L = G['kind'], G['units']['L']
    if kind in ('gadget', 'pkd'):
        f, s, e = u
        with h5py.File(f, 'r') as h:
            g = h['PartType1']
            x = g['Coordinates'][s:e].astype(np.float64)
            p = g['Potential'][s:e].astype(np.float64)
        if kind == 'pkd':
            x = (x + 0.5) * L
        p *= G['units']['scale']
    else:
        import pkcreader
        path, i = u
        f = pkcreader.PKCFile(path)
        X, V, ids, P = f.block(i, pot=True)
        h = f.header
        x = (X + 0.5) / h['q'] * h['box_size']
        p = P.astype(np.float64) * h['pot_step']
    x = np.mod(x, L)
    x[x >= L] = 0.0
    return x, p - G['subtract']


def work(u):
    x, phi = load_unit(u)
    tp = cKDTree(x, boxsize=G['units']['L'])
    edges = G['edges']
    nr = len(edges) - 1
    out = []
    for b, th in zip(G['bins'], G['trees']):
        nh = len(b['c'])
        acc = np.zeros((3, nh * nr))
        if nh:
            m = th.sparse_distance_matrix(tp, b['rmax'], output_type='ndarray')
            if len(m):
                ih, jp, d = m['i'], m['j'], m['v']
                r = d / b['scale'][ih]
                k = np.searchsorted(edges, r, side='right') - 1
                ok = (k >= 0) & (k < nr)
                flat = ih[ok] * nr + k[ok]
                w = phi[jp[ok]]
                acc[0] = np.bincount(flat, minlength=nh * nr)
                acc[1] = np.bincount(flat, weights=w, minlength=nh * nr)
                acc[2] = np.bincount(flat, weights=w * w, minlength=nh * nr)
        out.append(acc)
    return out, len(phi)


def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)   # empty shells -> NaN
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kind'); ap.add_argument('snapshot'); ap.add_argument('halos'); ap.add_argument('out')
    ap.add_argument('--mbins', default='1e13,1e14,1e15,1e16')
    ap.add_argument('--mass-col', default='mvir')
    ap.add_argument('--rbins', default='0.05,20,24')
    ap.add_argument('--r-units', default='rvir', choices=('rvir', 'mpc'))
    ap.add_argument('--all-halos', action='store_true')
    ap.add_argument('--max-halos', type=int, default=2000)
    ap.add_argument('--par', default='')
    ap.add_argument('--offset', type=float, default=0.0)
    ap.add_argument('--pkc-subsamples', default='')
    ap.add_argument('--chunk', type=int, default=2097152)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    A = ap.parse_args()
    t0 = time.time()

    L, a, ntot, units, src = source_info(A.kind, A.snapshot, A.par)
    selfpot = units_for_self(A.kind, L, a, ntot, A.par)
    subtract = selfpot + A.offset
    if A.kind in ('gadget', 'pkd'):
        uts = []
        for f in src:
            with h5py.File(f, 'r') as h:
                n = h['PartType1/Potential'].shape[0]
            uts += [(f, s, min(n, s + A.chunk)) for s in range(0, n, A.chunk)]
        nused = ntot
    else:
        sel = src._select([float(p) for p in A.pkc_subsamples.split(',')] if A.pkc_subsamples else None)
        uts = [(f.path, i) for f in sel for i in range(len(f.counts))]
        nused = int(sum(f.header['n_particles'] for f in sel))

    h = read_halos(A.halos, A.mass_col)
    keep = np.ones(len(h['mass']), bool) if A.all_halos else (h['pid'] == -1)
    rmin, rmax, nr = A.rbins.split(',')
    edges = np.geomspace(float(rmin), float(rmax), int(nr) + 1)
    medges = np.array([float(v) for v in A.mbins.split(',')])
    rng = np.random.default_rng(0)
    bins = []
    for lo, hi in zip(medges[:-1], medges[1:]):
        idx = np.where(keep & (h['mass'] >= lo) & (h['mass'] < hi))[0]
        if len(idx) > A.max_halos:
            idx = np.sort(rng.choice(idx, A.max_halos, replace=False))
        c = np.mod(np.vstack([h['x'][idx], h['y'][idx], h['z'][idx]]).T, L)
        c[c >= L] = 0.0
        rv = h['rvir'][idx] / 1e3                                    # kpc/h -> Mpc/h comoving
        scale = rv if A.r_units == 'rvir' else np.ones(len(idx))
        bins.append(dict(c=c, scale=scale, rmax=float((edges[-1] * scale).max()) if len(idx) else 0.0,
                         mass=h['mass'][idx], rvir=rv))
    if max(b['rmax'] for b in bins) >= L / 2:
        sys.exit('largest radius %.3g >= L/2: lower --rbins rmax' % max(b['rmax'] for b in bins))
    print('%s %s: a = %.5f, L = %g, %d particles (%d used) in %d work units; self-term %.6g, offset %.6g (km/s)^2'
          % (A.kind, A.snapshot, a, L, ntot, nused, len(uts), selfpot, A.offset))
    print('halos per mass bin: %s' % [len(b['c']) for b in bins], flush=True)

    nb = len(bins)
    acc = [np.zeros((3, len(b['c']) * int(nr))) for b in bins]
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, A.workers), initializer=init,
                             initargs=(A.kind, units, bins, A.r_units, edges, subtract)) as ex:
        for res, n in ex.map(work, uts, chunksize=1):
            for b in range(nb):
                acc[b] += res[b]
            done += n
    assert done == nused, (done, nused)

    nr = int(nr)
    out = dict(r_edges=edges, r_mid=np.sqrt(edges[1:] * edges[:-1]), m_edges=medges, r_units=A.r_units, a=a, z=1 / a - 1,
               box=L, kind=A.kind, snapshot=A.snapshot, halos=A.halos, selfpot=selfpot, offset=A.offset,
               n_halos=np.array([len(b['c']) for b in bins]), particle_fraction=nused / ntot)
    for k in ('stack_mean', 'stack_std', 'halo_mean', 'halo_std', 'halo_sem', 'npart', 'nhalo_shell'):
        out[k] = np.full((nb, nr), np.nan)
    for b in range(nb):
        nh = len(bins[b]['c'])
        out['halo_mass_%d' % b] = bins[b]['mass']
        out['halo_rvir_%d' % b] = bins[b]['rvir']
        if nh == 0:
            out['halo_shell_mean_%d' % b] = np.zeros((0, nr))
            continue
        n, s, s2 = (acc[b][i].reshape(nh, nr) for i in range(3))
        N, S, S2 = n.sum(0), s.sum(0), s2.sum(0)
        with np.errstate(invalid='ignore', divide='ignore'):
            out['stack_mean'][b] = S / N
            out['stack_std'][b] = np.sqrt(np.maximum(S2 / N - (S / N)**2, 0))
            hm = np.where(n > 0, s / n, np.nan)
        out['npart'][b] = N
        nhs = np.sum(n > 0, 0)
        out['nhalo_shell'][b] = nhs
        with np.errstate(invalid='ignore', divide='ignore'):
            out['halo_mean'][b] = np.nanmean(hm, 0)
            out['halo_std'][b] = np.nanstd(hm, 0, ddof=1) if nh > 1 else np.nan
            out['halo_sem'][b] = out['halo_std'][b] / np.sqrt(nhs)
        out['halo_shell_mean_%d' % b] = hm
    np.savez(A.out, **out)
    print('wrote %s in %.1f s' % (A.out, time.time() - t0))


if __name__ == '__main__':
    main()
