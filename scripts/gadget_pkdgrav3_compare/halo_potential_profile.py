#!/usr/bin/env python3
"""Stacked gravitational potential around halos: mean and scatter of phi in shells of comoving distance r (Mpc/h)
from the halo centres, for halos in mass bins. Gadget-4 HDF5, pkdgrav3 HDF5 or pkdgrav3 .pkc snapshots, so the
same statistic can be compared between codes. Only the stacks (per mass bin) are accumulated and stored, never
per-halo profiles: memory is independent of the numbers of halos and particles.

usage:
  halo_potential_profile.py <kind> <snapshot> <halos> <out.npz> [options]
    kind      gadget   Gadget-4 HDF5: snapshot_<nnn>.hdf5 or the base of snapshot_<nnn>.<i>.hdf5
              pkd      pkdgrav3 HDF5 (snap.<step>)
              pkc      pkdgrav3 compressed snapshot, prefix snap.<step> (needs the potential: dCompPotStep > 0)
    halos     consistent-trees hlist (out_<snap>.trees) or Rockstar out_<snap>.list
  options:
    --mbins 3e12,1e13,3e13,1e14,1e15  halo mass bin edges (Msun/h; column --mass-col, default mvir)
    --rbins 0.02,50,36                log bins in comoving r (Mpc/h): rmin, rmax, number
    --all-halos                       include subhaloes (default: hosts only, pid = -1)
    --max-halos 20000                 at most this many halos per mass bin (random, seed 0); 0 = all
    --target 4096                     dilution: particles expected inside a level's outer radius (below)
    --jk 3                            jackknife: jk^3 sub-cubes of the box (by halo centre) for the error
    --par p.par                       pkdgrav3 parameter file (dSoft) to remove pkdgrav3's softened self-potential
    --offset X                        subtract X (km/s)^2 from every potential (e.g. offset_median, particles.csv)
    --pkc-subsamples 1,3              .pkc only: read only these subsample percents
    --chunk 1048576                   particles per work unit (HDF5)
    --workers N                       processes (default: all cores)

Potentials are peculiar, physical, (km/s)^2, with the conventions of compare_potential.py:
  Gadget-4 Potential / a; pkdgrav3 Potential * U^2 / a (U = 100 L / sqrt(8 pi/3) km/s), .pkc as stored; with --par
  the pkdgrav3 self-term -(105/64) m / dSoft * U^2 / a is removed.

Algorithm. Particles are read in chunks by parallel processes. Each chunk is paired with the halo centres of each
mass bin through periodic KD-trees (cKDTree.sparse_distance_matrix of a halo group against the particles), and every
(halo, particle) pair adds 1, phi, phi^2 and r to its (jackknife region, mass bin, shell) - a few small arrays.
Radial dilution keeps the work per halo constant out to large radii: shell i is assigned level l_i, the smallest
l >= 0 with nbar 4/3 pi r_out,i^3 / 8^l <= --target (nbar = particles read / L^3), and only the particles of the
nested random subsample u < 8^-l (u uniform, independent of phi) are paired at that level. Each shell uses one
level, so its mean and scatter are unbiased; the number of pairs per halo is ~ target x (number of levels) instead
of nbar 4/3 pi r_max^3. At 2048^3 / 200 Mpc/h, 50 Mpc/h holds 5.6e8 particles; with target 4096 the outer level
uses 1/8^6 of them. Cost ~ n_halo x levels x target x (local overdensity), plus reading the snapshot once.

Output (npz), mass bin b, shell i:
  r_edges, r_mid (geometric), r_mean[b,i] (pair-weighted), m_edges, n_halos[b], m_mean[b], level[i], frac[i]
  stack_mean[b,i]   mean phi of all (halo, particle) pairs in the shell (particle-weighted stack)
  stack_std[b,i]    standard deviation of phi over those pairs (scatter of the potential in the shell)
  stack_err[b,i]    jackknife error of stack_mean (leave one of jk^3 sub-cubes of halo centres out)
  delta_hm[b,i], delta_hm_err[b,i]   mean matter overdensity in the shell (halo-matter cross-correlation)
  npair[b,i]        pairs used (diluted); npair_eff = npair / frac
  jk_n, jk_s, jk_s2, jk_nh: the per-region sums (region, b, i) / (region, b), to recombine
"""
import argparse
import glob
import math
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from scipy.spatial import cKDTree

H0 = 100.0
HALO_GROUP = 2048          # halo centres per KD-tree: bounds the pairs held at once


# ------------------------------------------------------------------ halos
def read_halos(path, mass_col):
    with open(path) as f:
        first = f.readline()
    names = [re.sub(r'\(\d+\)$', '', h) for h in first.lstrip('#').split()]
    d = np.loadtxt(path, comments='#', ndmin=2)
    if d.shape[0] == 0 or d.size == 0:
        h = {k: np.zeros(0) for k in names}
    else:
        h = {k: d[:, i] for i, k in enumerate(names)}
    for k in list(h):   # lower-case names throughout (hlists: Mvir, Rvir; Rockstar .list: ID, Mvir, Rvir, X, Y, Z)
        h.setdefault(k.lower(), h[k])
    if 'pid' not in h:  # Rockstar out_*.list: no host/subhalo flag, all treated as hosts
        h['pid'] = np.full(len(h['id']), -1.0)
    if mass_col not in h:
        sys.exit('no column %s in %s (have %s)' % (mass_col, path, ', '.join(h)))
    h['mass'] = h[mass_col]
    return h


# ------------------------------------------------------------------ particle sources
def gadget_files(path):
    if path.endswith('.hdf5'):
        return [path]
    fl = sorted(glob.glob(path + '.*.hdf5'), key=lambda f: int(f.split('.')[-2]))
    if not fl:
        sys.exit('no %s.hdf5 / %s.<i>.hdf5' % (path, path))
    return fl


def source_info(kind, path):
    """-> (box, a, n_total, units dict, files or PkcSnapshot)"""
    if kind in ('gadget', 'pkd'):
        files = gadget_files(path) if kind == 'gadget' else [path]
        with h5py.File(files[0], 'r') as f:
            hd = f['Header'].attrs
            L = float(hd['BoxSize'])
            ntot = int(hd['NumPart_Total'][1])
            a = float(hd['Time']) if kind == 'gadget' else 1 / (1 + float(hd['Redshift']))  # pkdgrav3 Time = code time
        if kind == 'pkd':
            U = H0 * L / math.sqrt(8 * math.pi / 3)
            units = dict(scale=U**2 / a)
        else:
            units = dict(scale=1 / a)
        units['L'] = L
        return L, a, ntot, units, files
    if kind == 'pkc':
        import pkcreader
        s = pkcreader.PkcSnapshot(path)
        if not s.has_potential:
            sys.exit('%s: no potential stored (dCompPotStep = 0)' % path)
        return s.Lbox, s.scale, s.npart, {'L': s.Lbox}, s
    sys.exit('kind must be gadget, pkd or pkc')


def pkd_selfpot(kind, L, a, ntot, par):
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


# ------------------------------------------------------------------ workers
G = {}


def init(kind, units, bins, edges, levels, nreg, subtract, seed):
    L = units['L']
    groups = []
    for b in bins:
        gl = []
        for s in range(0, len(b['c']), HALO_GROUP):
            gl.append((s, cKDTree(b['c'][s:s + HALO_GROUP], boxsize=L)))
        groups.append(gl)
    G.update(kind=kind, units=units, bins=bins, edges=edges, levels=levels, nreg=nreg, subtract=subtract,
             seed=seed, groups=groups)


def load_unit(u):
    kind, L = G['kind'], G['units']['L']
    if kind in ('gadget', 'pkd'):
        f, s, e = u[1:]
        with h5py.File(f, 'r') as h:
            g = h['PartType1']
            x = g['Coordinates'][s:e].astype(np.float64)
            p = g['Potential'][s:e].astype(np.float64)
        if kind == 'pkd':
            x = (x + 0.5) * L
        p *= G['units']['scale']
    else:
        import pkcreader
        path, i = u[1:]
        f = pkcreader.PKCFile(path)
        X, V, ids, P = f.block(i, pot=True)
        h = f.header
        x = (X + 0.5) / h['q'] * h['box_size']
        p = P.astype(np.float64) * h['pot_step']
    x = np.mod(x, L)
    x[x >= L] = 0.0
    return x, p - G['subtract']


def work(u):
    """-> per mass bin an array (4, nreg * nr): pair count, sum phi, sum phi^2, sum r"""
    x, phi = load_unit(u)
    L, edges, levels, nreg = G['units']['L'], G['edges'], G['levels'], G['nreg']
    nr = len(edges) - 1
    rng = np.random.default_rng([G['seed'], u[0]])
    ur = rng.random(len(phi))
    out = [np.zeros((4, nreg * nr)) for _ in G['bins']]
    for lev, (k0, k1) in levels.items():          # shells k0..k1-1 use subsample u < 8^-lev
        sel = np.where(ur < 8.0**-lev)[0] if lev else np.arange(len(phi))
        if len(sel) == 0:
            continue
        tp = cKDTree(x[sel], boxsize=L)
        ps = phi[sel]
        rlo, rhi = edges[k0], edges[k1]
        for b, gl, acc in zip(G['bins'], G['groups'], out):
            for s, th in gl:
                m = th.sparse_distance_matrix(tp, rhi, output_type='ndarray')
                if len(m) == 0:
                    continue
                d = m['v']
                ok = d >= rlo
                ih, jp, d = m['i'][ok] + s, m['j'][ok], d[ok]
                k = np.searchsorted(edges, d, side='right') - 1
                ok = (k >= k0) & (k < k1)
                flat = b['reg'][ih[ok]] * nr + k[ok]
                w = ps[jp[ok]]
                n = nreg * nr
                acc[0] += np.bincount(flat, minlength=n)
                acc[1] += np.bincount(flat, weights=w, minlength=n)
                acc[2] += np.bincount(flat, weights=w * w, minlength=n)
                acc[3] += np.bincount(flat, weights=d[ok], minlength=n)
    return out, len(phi)


def jackknife(n, s, nh, f):
    """n, s: (nreg, nr) sums; nh: (nreg,) halos per region. -> error of mean phi and of the overdensity estimator f"""
    N, S = n.sum(0), s.sum(0)
    full = S / N
    reg = [i for i in range(n.shape[0]) if nh[i] > 0]
    if len(reg) < 2:
        return np.full_like(full, np.nan), np.full_like(full, np.nan)
    m = np.array([(S - s[i]) / (N - n[i]) for i in reg])
    dv = np.array([f(N - n[i], nh.sum() - nh[i]) for i in reg])
    k = len(reg)
    e1 = np.sqrt((k - 1) / k * np.nansum((m - np.nanmean(m, 0))**2, 0))
    e2 = np.sqrt((k - 1) / k * np.nansum((dv - np.nanmean(dv, 0))**2, 0))
    return e1, e2


def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)   # empty shells -> NaN
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kind'); ap.add_argument('snapshot'); ap.add_argument('halos'); ap.add_argument('out')
    ap.add_argument('--mbins', default='3e12,1e13,3e13,1e14,1e15')
    ap.add_argument('--mass-col', default='mvir')
    ap.add_argument('--rbins', default='0.02,50,36')
    ap.add_argument('--all-halos', action='store_true')
    ap.add_argument('--max-halos', type=int, default=20000)
    ap.add_argument('--target', type=float, default=4096)
    ap.add_argument('--jk', type=int, default=3)
    ap.add_argument('--par', default='')
    ap.add_argument('--offset', type=float, default=0.0)
    ap.add_argument('--pkc-subsamples', default='')
    ap.add_argument('--chunk', type=int, default=1048576)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    A = ap.parse_args()
    t0 = time.time()

    L, a, ntot, units, src = source_info(A.kind, A.snapshot)
    selfpot = pkd_selfpot(A.kind, L, a, ntot, A.par)
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
    uts = [(i,) + tuple(u) for i, u in enumerate(uts)]           # unit index seeds its dilution random numbers

    rmin, rmax, nr = A.rbins.split(',')
    nr = int(nr)
    edges = np.geomspace(float(rmin), float(rmax), nr + 1)
    if edges[-1] >= L / 2:
        sys.exit('rmax %.3g >= L/2 = %.3g' % (edges[-1], L / 2))
    nbar = nused / L**3
    lev = np.maximum(0, np.ceil(np.log(nbar * 4 / 3 * np.pi * edges[1:]**3 / A.target) / np.log(8))).astype(int)
    lev = np.maximum.accumulate(lev)
    levels = {}
    for l in np.unique(lev):
        k = np.where(lev == l)[0]
        levels[int(l)] = (int(k[0]), int(k[-1]) + 1)
    frac = 8.0**-lev

    h = read_halos(A.halos, A.mass_col)
    keep = np.ones(len(h['mass']), bool) if A.all_halos else (h['pid'] == -1)
    medges = np.array([float(v) for v in A.mbins.split(',')])
    J = A.jk
    nreg = J**3
    rng = np.random.default_rng(0)
    bins = []
    for lo, hi in zip(medges[:-1], medges[1:]):
        idx = np.where(keep & (h['mass'] >= lo) & (h['mass'] < hi))[0]
        if A.max_halos and len(idx) > A.max_halos:
            idx = np.sort(rng.choice(idx, A.max_halos, replace=False))
        c = np.mod(np.vstack([h['x'][idx], h['y'][idx], h['z'][idx]]).T, L) if len(idx) else np.zeros((0, 3))
        c[c >= L] = 0.0
        ij = np.minimum((c / L * J).astype(int), J - 1)
        reg = (ij[:, 0] * J + ij[:, 1]) * J + ij[:, 2]
        bins.append(dict(c=c, reg=reg, mass=h['mass'][idx]))
    print('%s %s: a = %.5f, L = %g, %d particles (%d used) in %d work units; self-term %.6g, offset %.6g (km/s)^2'
          % (A.kind, A.snapshot, a, L, ntot, nused, len(uts), selfpot, A.offset))
    print('dilution levels {level: shells}: %s, fractions %s' % (levels, sorted(set(frac))))
    print('halos per mass bin: %s' % [len(b['c']) for b in bins], flush=True)

    nb = len(bins)
    acc = [np.zeros((4, nreg * nr)) for _ in bins]
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, A.workers), initializer=init,
                             initargs=(A.kind, units, bins, edges, levels, nreg, subtract, A.seed)) as ex:
        for res, n in ex.map(work, uts, chunksize=1):
            for b in range(nb):
                acc[b] += res[b]
            done += n
    assert done == nused, (done, nused)

    shell_vol = 4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    out = dict(r_edges=edges, r_mid=np.sqrt(edges[1:] * edges[:-1]), m_edges=medges, r_units='Mpc/h comoving',
               a=a, z=1 / a - 1, box=L, kind=A.kind, snapshot=A.snapshot, halos=A.halos, selfpot=selfpot,
               offset=A.offset, n_halos=np.array([len(b['c']) for b in bins]), particle_fraction=nused / ntot,
               nbar=nbar, level=lev, frac=frac, target=A.target, jk=J,
               m_mean=np.array([b['mass'].mean() if len(b['mass']) else np.nan for b in bins]))
    for k in ('stack_mean', 'stack_std', 'stack_err', 'r_mean', 'npair', 'delta_hm', 'delta_hm_err'):
        out[k] = np.full((nb, nr), np.nan)
    out['jk_n'], out['jk_s'], out['jk_s2'] = (np.zeros((nreg, nb, nr)) for _ in range(3))
    out['jk_nh'] = np.zeros((nreg, nb))
    for b in range(nb):
        n, s, s2, sr = (acc[b][i].reshape(nreg, nr) for i in range(4))
        nh = np.bincount(bins[b]['reg'], minlength=nreg).astype(float)
        out['jk_n'][:, b], out['jk_s'][:, b], out['jk_s2'][:, b], out['jk_nh'][:, b] = n, s, s2, nh
        if nh.sum() == 0:
            continue
        N, S, S2 = n.sum(0), s.sum(0), s2.sum(0)
        out['npair'][b] = N
        out['stack_mean'][b] = S / N
        out['stack_std'][b] = np.sqrt(np.maximum(S2 / N - (S / N)**2, 0))
        out['r_mean'][b] = sr.sum(0) / N
        dh = lambda nn, hh: nn / frac / (hh * nbar * shell_vol) - 1
        out['delta_hm'][b] = dh(N, nh.sum())
        out['stack_err'][b], out['delta_hm_err'][b] = jackknife(n, s, nh, dh)
    np.savez(A.out, **out)
    print('pairs per mass bin: %s' % [int(np.nansum(v)) for v in out['npair']])
    print('wrote %s in %.1f s' % (A.out, time.time() - t0))


if __name__ == '__main__':
    main()
