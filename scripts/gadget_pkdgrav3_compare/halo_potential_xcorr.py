#!/usr/bin/env python3
"""Halo-halo potential cross-correlation: for host halos in mass bins A and B, statistics of the per-halo potentials
phi (the pipeline's .vpot columns: mean particle potential within rs, R200b, 0.5/1/1.5 Rvir, ...) over all halo
pairs (i in A, j in B) at comoving separation r (Mpc/h), for several phi columns at once.

usage:
  halo_potential_xcorr.py <dir> <snap> <out.npz> [options]
    dir    directory with out_<snap>_basic.fits.gz and out_<snap>_vahc.fits.gz (the FITS stage's compressed/;
           the vahc file must have the .vpot columns, header HAS_VPOT)
  options:
    --phi phi_Rvir,phi_rs,phi_R1p5vir,phi_R0p5vir,phi_R200b   columns of the vahc file (any phi_* column)
    --mbins 2e12,1e13,5e13,2e14,1e16   mass bin edges, Msun/h (as the stacked profiles); --mass-col Mvir
    --pairs all|auto|0-0,0-2,...       bin pairs (A-B, A <= B); all = every A <= B, auto = A = B only
    --rbins log:0.2,50,30 | lin:0,50,0.5    log: rmin, rmax, number of bins; lin: rmin, rmax, width
    --all-halos                        include subhalos (default: hosts, pid = -1)
    --jk 3                             jackknife: jk^3 sub-cubes, by the position of halo i (A)
    --target 2e7                       dilution: at most ~this many pairs per shell (0: no dilution)
    --min-a 2000                       ... but at least this many A halos per shell
    --threads N                        OpenMP threads (default PBS_NCPUS, NCPUS, else all cores)

Statistics, per bin pair, column k and shell (pairs with both phi defined; dphi = phi_i - phi_j, i in A, j in B):
  mean_diff  <phi_i - phi_j>                                  (-> <phi>_A - <phi>_B at large r)
  sf         <(phi_i - phi_j)^2>                              (structure function)
  cov        <phi_i phi_j> - <phi_i><phi_j>;  corr = cov / (sd_i sd_j) with the pair-weighted sd of each side
  each with a jackknife error (*_err). Stored too: the raw sums (S[pair, k, shell, 6] = N, sum phi_i, sum phi_j,
  sum phi_i phi_j, sum phi_i^2, sum phi_j^2; and per jackknife region), npair, r_mean, frac (dilution).
  phi is shifted by the mean of each column over all selected halos before summing (phi_shift; it cancels in every
  statistic and keeps the sums well conditioned).

Algorithm and cost (aimed at 2048^3 catalogues). One pass over the pairs per bin pair and distance range collects
all sums of all columns (halo_pair_sums.c, OpenMP over A, B in a cell grid sized to the range). Dilution: shell i
expects n_A n_B V_i / L^3 pairs; it uses the A halos with u(id) < 2^-l (u: splitmix64 hash of the halo id, nested,
independent of phi), the smallest l with at most --target pairs (and at least --min-a A halos). Shells with the same
l are done in one pass with cells of about a third of the range's outer radius, so a pass tests ~2-3x the pairs it
keeps. Each shell uses one subset of A, so its statistics are unbiased; the dilution only adds noise.
Cost ~ (number of shells) x target x 2.5 pair tests per bin pair; memory: the halo arrays ((4 + K) doubles per
halo, read column by column) and per thread nreg x nbins x (2 + 6K) doubles.
"""
import argparse
import ctypes
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PHI = 'phi_Rvir,phi_rs,phi_R1p5vir,phi_R0p5vir,phi_R200b'


def pair_lib():
    """halo_pair_sums.so next to this file (built if missing or older than the source; $CC, else cc, gcc-NN)."""
    src = os.path.join(HERE, 'halo_pair_sums.c')
    so = os.path.join(HERE, 'halo_pair_sums.so')
    if not (os.path.isfile(so) and os.path.getmtime(so) >= os.path.getmtime(src)):
        ccs = [os.environ['CC']] if os.environ.get('CC') else ['cc'] + ['gcc-%d' % v for v in range(16, 9, -1)]
        for cc in ccs:
            try:
                subprocess.check_call([cc, '-O3', '-fopenmp', '-shared', '-fPIC', '-o', so, src],
                                      stderr=subprocess.DEVNULL)
                break
            except (OSError, subprocess.CalledProcessError):
                continue
        else:
            print('warning: no compiler with OpenMP; building serial', file=sys.stderr)
            subprocess.check_call(['cc', '-O3', '-shared', '-fPIC', '-o', so, src])
    lib = ctypes.CDLL(so)
    D = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
    I = np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS')
    L = np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS')
    lib.pair_sums.restype = ctypes.c_int
    lib.pair_sums.argtypes = [ctypes.c_long, D, D, D, D, I, L, ctypes.c_long, D, D, D, D, L, ctypes.c_int,
                              ctypes.c_double, D, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, D, D, D]
    return lib


def uniform_from_id(ids, seed=0):
    """uniform [0, 1) from integer ids (splitmix64), as halo_potential_profile.py"""
    with np.errstate(over='ignore'):
        x = ids.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15) * np.uint64(seed + 1)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def read_halos(d, snap, phis, mass_col, hosts_only):
    import fitsio
    fb = os.path.join(d, 'out_%s_basic.fits.gz' % snap)
    fv = os.path.join(d, 'out_%s_vahc.fits.gz' % snap)
    with fitsio.FITS(fb) as f:
        h = f[1].read_header()
        L = float(h.get('BOXSIZE', h.get('SIM_BOX_SIZE')))
        a = float(h.get('SCALE', np.nan))
        cols = ['id', 'pid', mass_col, 'x', 'y', 'z']
        B = {c: f[1][c][:] for c in cols}
    with fitsio.FITS(fv) as f:
        if not f[1].read_header().get('HAS_VPOT', False):
            raise SystemExit('%s has no .vpot columns (HAS_VPOT)' % fv)
        vid = f[1]['haloID'][:].astype(np.int64)
        V = {c: f[1][c][:].astype(np.float64) for c in phis}
    sel = B['pid'] == -1 if hosts_only else np.ones(len(B['id']), bool)
    ids = B['id'][sel].astype(np.int64)
    o = np.argsort(vid)
    k = np.searchsorted(vid[o], ids)
    k = np.minimum(k, len(vid) - 1)
    ok = vid[o][k] == ids
    phi = np.full((len(ids), len(phis)), np.nan)
    for j, c in enumerate(phis):
        phi[ok, j] = V[c][o][k[ok]]
    pos = np.stack([B[c][sel].astype(np.float64) % L for c in 'xyz'], axis=1)
    return dict(id=ids, m=B[mass_col][sel].astype(np.float64), pos=pos, phi=phi, L=L, a=a,
                n_missing=int((~ok).sum()))


def parse_rbins(s):
    kind, v = s.split(':', 1)
    v = [float(x) for x in v.split(',')]
    if kind == 'log':
        if v[0] <= 0:
            raise SystemExit('log bins need rmin > 0')
        return kind, np.logspace(np.log10(v[0]), np.log10(v[1]), int(v[2]) + 1)
    if kind == 'lin':
        n = int(round((v[1] - v[0]) / v[2]))
        return kind, v[0] + v[2] * np.arange(n + 1)
    raise SystemExit('--rbins log:rmin,rmax,n or lin:rmin,rmax,width')


def plan_levels(nA, nB, L, edges, target, min_a):
    """dilution level per shell: A fraction 2^-l with expected pairs <= target, A halos >= min_a"""
    vol = 4.0 / 3.0 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    exp_pairs = nA * nB * vol / L**3
    lev = np.zeros(len(vol), int)
    if target > 0:
        lmax = max(0, int(np.floor(np.log2(max(nA / max(min_a, 1), 1)))))
        need = np.log2(np.maximum(exp_pairs / target, 1.0))
        lev = np.minimum(np.ceil(need).astype(int), lmax)
    runs, i = [], 0  # contiguous shell ranges with one level
    while i < len(lev):
        j = i
        while j + 1 < len(lev) and lev[j + 1] == lev[i]:
            j += 1
        runs.append((i, j + 1, int(lev[i])))
        i = j + 1
    return lev, runs, exp_pairs


def stats(S):
    """S[..., 6] -> mean_diff, sf, cov, corr"""
    with np.errstate(invalid='ignore', divide='ignore'):
        N, pi, pj, pij, pi2, pj2 = (S[..., q] for q in range(6))
        mi, mj = pi / N, pj / N
        md = mi - mj
        sf = (pi2 + pj2 - 2 * pij) / N
        cov = pij / N - mi * mj
        vi, vj = pi2 / N - mi**2, pj2 / N - mj**2
        corr = cov / np.sqrt(vi * vj)
    return md, sf, cov, corr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dir'); ap.add_argument('snap'); ap.add_argument('out')
    ap.add_argument('--phi', default=DEFAULT_PHI)
    ap.add_argument('--mbins', default='2e12,1e13,5e13,2e14,1e16')
    ap.add_argument('--mass-col', default='Mvir')
    ap.add_argument('--pairs', default='all')
    ap.add_argument('--rbins', default='log:0.2,50,30')
    ap.add_argument('--all-halos', action='store_true')
    ap.add_argument('--jk', type=int, default=3)
    ap.add_argument('--target', type=float, default=2e7)
    ap.add_argument('--min-a', type=int, default=2000)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    A = ap.parse_args()
    t0 = time.time()
    phis = [c.strip() for c in A.phi.split(',') if c.strip()]
    K = len(phis)
    me = np.array([float(v) for v in A.mbins.split(',')])
    nmb = len(me) - 1
    if A.pairs == 'all':
        pairs = [(a, b) for a in range(nmb) for b in range(a, nmb)]
    elif A.pairs == 'auto':
        pairs = [(a, a) for a in range(nmb)]
    else:
        pairs = [tuple(int(x) for x in p.split('-')) for p in A.pairs.split(',')]
    binning, edges = parse_rbins(A.rbins)
    nbin = len(edges) - 1
    threads = A.threads or int(os.environ.get('PBS_NCPUS', os.environ.get('NCPUS', os.cpu_count() or 1)))

    H = read_halos(A.dir, A.snap, phis, A.mass_col, not A.all_halos)
    L = H['L']
    if edges[-1] > L / 2:
        raise SystemExit('rmax %.3g > L/2 = %.3g' % (edges[-1], L / 2))
    mb = np.digitize(H['m'], me) - 1
    use = (mb >= 0) & (mb < nmb)
    shift = np.array([np.nanmean(H['phi'][use, j]) for j in range(K)])
    phi = np.ascontiguousarray(H['phi'] - shift)
    u = uniform_from_id(H['id'], A.seed)
    nreg = A.jk**3
    reg = np.minimum((H['pos'] / L * A.jk).astype(int), A.jk - 1)
    reg = (reg[:, 0] * A.jk + reg[:, 1]) * A.jk + reg[:, 2]
    n_halos = np.array([(mb == b).sum() for b in range(nmb)])
    print('%s snap %s: L %.1f, a %.4f, %d halos (%d without vahc row), per mass bin %s; %d threads; %s %d bins %.3g-%.3g'
          % (A.dir, A.snap, L, H['a'], len(H['id']), H['n_missing'], n_halos.tolist(), threads, binning, nbin,
             edges[0], edges[-1]), flush=True)

    lib = pair_lib()
    P = len(pairs)
    npair = np.zeros((P, nreg, nbin)); dsum = np.zeros((P, nreg, nbin)); S = np.zeros((P, nreg, nbin, K, 6))
    frac = np.ones((P, nbin)); levels = np.zeros((P, nbin), int); exp_pairs = np.zeros((P, nbin))
    timing = []
    for p, (a, b) in enumerate(pairs):
        ia, ib = np.flatnonzero(mb == a), np.flatnonzero(mb == b)
        if len(ia) == 0 or len(ib) == 0:
            continue
        lev, runs, ep = plan_levels(len(ia), len(ib), L, edges, A.target, A.min_a)
        levels[p], frac[p], exp_pairs[p] = lev, 2.0**-lev, ep
        bx, by, bz = (np.ascontiguousarray(H['pos'][ib, q]) for q in range(3))
        bphi, bid = np.ascontiguousarray(phi[ib]), np.ascontiguousarray(H['id'][ib])
        for i0, i1, l in runs:
            sa = ia[u[ia] < 2.0**-l] if l > 0 else ia
            rhi = edges[i1]
            nc = int(max(1, min(L / max(rhi / 3.0, 1e-3), round((2.0 * len(ib))**(1 / 3.0)) + 1, 512)))
            t = time.time()
            rc = lib.pair_sums(len(sa), *(np.ascontiguousarray(H['pos'][sa, q]) for q in range(3)),
                               np.ascontiguousarray(phi[sa]), np.ascontiguousarray(reg[sa], np.int32),
                               np.ascontiguousarray(H['id'][sa]), len(ib), bx, by, bz, bphi, bid, K, L,
                               np.ascontiguousarray(edges, np.float64), nbin, i0, i1, nc, nreg, threads,
                               npair[p], dsum[p], S[p])
            if rc:
                raise MemoryError('pair_sums: out of memory')
            timing.append(dict(pair=p, i0=i0, i1=i1, level=l, na=len(sa), nb=len(ib), nc=nc, s=time.time() - t))
        print('pair %d-%d: %d x %d halos, %d passes, %.1f s, %.3g pairs'
              % (a, b, len(ia), len(ib), len(runs), sum(x['s'] for x in timing if x['pair'] == p),
                 npair[p].sum()), flush=True)

    St = S.sum(axis=1)                                   # (P, nbin, K, 6)
    md, sf, cov, corr = stats(St)
    jk = St[:, None] - S                                 # leave one region out
    jmd, jsf, jcov, jcorr = stats(jk)

    def jerr(x):
        import warnings
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.sqrt((nreg - 1) / nreg * np.nansum((x - np.nanmean(x, axis=1, keepdims=True))**2, axis=1))

    tot_n = npair.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        r_mean = dsum.sum(axis=1) / tot_n
    r_mid = np.sqrt(edges[1:] * edges[:-1]) if binning == 'log' else 0.5 * (edges[1:] + edges[:-1])
    # (P, nbin, K) -> (P, K, nbin)
    tr = lambda x: np.moveaxis(x, 2, 1)
    np.savez_compressed(A.out, r_edges=edges, r_mid=r_mid, r_mean=r_mean, binning=binning, r_units='Mpc/h comoving',
             m_edges=me, mass_col=A.mass_col, n_halos=n_halos, pairs=np.array(pairs), phi_cols=np.array(phis),
             phi_shift=shift, a=H['a'], z=1 / H['a'] - 1, box=L, hosts_only=not A.all_halos,
             npair=tot_n, frac=frac, level=levels, expected_pairs=exp_pairs, target=A.target, min_a=A.min_a,
             jk=A.jk, mean_diff=tr(md), sf=tr(sf), cov=tr(cov), corr=tr(corr),
             mean_diff_err=tr(jerr(jmd)), sf_err=tr(jerr(jsf)), cov_err=tr(jerr(jcov)), corr_err=tr(jerr(jcorr)),
             S=np.moveaxis(St, 2, 1), S_jk=np.moveaxis(S, 3, 2), npair_jk=npair,
             timing_s=np.array([x['s'] for x in timing]), source=os.path.abspath(A.dir), snap=str(A.snap))
    print('wrote %s in %.1f s' % (A.out, time.time() - t0))


if __name__ == '__main__':
    main()
