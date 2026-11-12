#!/usr/bin/env python3
"""Halo-halo cross-correlation multipoles xi_l(s) (l = 0..4, dipole included) between halo mass bins, in real space,
redshift space (velocities), with the gravitational-redshift shift from the halo potential, and with both, in a
periodic box with a fixed line of sight (plane parallel; default the z axis). Pair counts: Corrfunc theory DDsmu of
the cosmodesi/adematti fork (branch desi), which counts signed mu in (-1, 1).

usage:
  halo_xi_multipoles.py <dir> <snap> <out.npz> [options]
    dir    directory with out_<snap>_basic.fits.gz (x, y, z, vx, vy, vz, Mvir, pid) and, for the potential spaces,
           out_<snap>_vahc.fits.gz with the .vpot columns (HAS_VPOT)
  options:
    --spaces real,rsd,grav,rsd+grav   real: positions; rsd: + v_los / (a H); grav: + v_g / (a H) with
                                      v_g = -phi / c (gravitational redshift of a source at potential phi, observer
                                      at phi = 0); rsd+grav: both. Any subset.
    --phi phi_rs                      potential column(s) for grav; several (comma) give grav[<col>] spaces each
    --phi-fallback phi_R0p5vir        used where the --phi column is undefined (empty rs sphere); '' = drop halo
    --pot-amp 1                       multiply the potential shift (tests of the response; 1 = physical)
    --los z                           line of sight: x, y or z; a list (x,y,z) measures each and stores their mean
    --mbins 2e12,1e13,5e13,2e14,1e16  mass bins (Msun/h; --mass-col Mvir), as the stacked profiles
    --pairs all                       all (A <= B), auto, cross, or a list 0-3,1-2
    --sbins lin:0,100,2               lin:smin,smax,width or log:smin,smax,n (comoving Mpc/h; smax <= L/2)
    --nmu 200                         mu bins in (-1, 1)
    --ells 0,1,2,3,4
    --jk 3                            jackknife: jk^3 sub-cubes by the real-space position of the A halo
    --subsample-a 1                   use the A halos with hash(id) < f (2048^3: fewer pairs, same statistics)
    --threads N                       Corrfunc threads (default PBS_NCPUS, NCPUS, else all cores)
    --all-halos                       include subhalos (default hosts, pid = -1)

Conventions. Pair separation s = x_B - x_A (A = first, B = second bin of a pair); mu = s_los / |s| in (-1, 1), so a
B population shifted by +delta along the line of sight gives xi_1(s) = -delta dxi_0/ds (to first order). Deeper
potentials (more negative phi) get larger v_g, i.e. are pushed away from the observer. For A = B the dipole vanishes
identically (both orderings are counted). Velocities are Rockstar's bulk velocities (peculiar, km/s);
H = 100 E(a) km/s/(Mpc/h), E^2 = Om a^-3 + OL + (1 - Om - OL) a^-2 from the FITS header; displacements wrap
periodically. A constant offset of phi (zero point) moves every halo alike and changes nothing.

Estimator. xi(s, mu) = DD / RR - 1 with the analytic RR of a periodic box, N_A N_B' V_s (dmu / 2) / L^3 (N_B' = N_B,
or N_B - 1 for A = B: ordered pairs without a halo paired with itself; N_A = the A halos used). xi_l(s) = (2l + 1)/2 sum_mu xi(s, mu) int_bin P_l(mu) dmu
(exact for xi constant in each mu bin). Jackknife: DD is counted per region k of the A halo; leave-one-out
DD - DD_k with N_A - N_A,k. All spaces use the same halos and regions, so differences between spaces (the potential
effect: grav - real, rsd+grav - rsd) are stored with their own jackknife errors; they cancel most of the noise.

Output (npz): s_edges, s_mid, mu_edges, ells, spaces, los, pairs, n_halos, ...; DD[space, los, pair, s, mu];
xi_ell[space, los, pair, ell, s] (+ _err) and the mean over the lines of sight xi_ell_mean (+ _err, jackknife of the
mean); xi_ell_jk (per region, of the mean over lines of sight); diff_names, dxi_ell, dxi_ell_err (potential effect);
vg_mean[phi column, mass bin] (mean v_g in km/s, including --pot-amp).
"""
import argparse
import os
import time

import numpy as np

C_KMS = 299792.458


def uniform_from_id(ids, seed=0):
    """uniform [0, 1) from integer ids (splitmix64), as halo_potential_profile.py / halo_potential_xcorr.py"""
    with np.errstate(over='ignore'):
        x = ids.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15) * np.uint64(seed + 1)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def read_halos(d, snap, mass_col, hosts_only, phis):
    import fitsio
    with fitsio.FITS(os.path.join(d, 'out_%s_basic.fits.gz' % snap)) as f:
        h = f[1].read_header()
        L = float(h.get('BOXSIZE', h.get('SIM_BOX_SIZE')))
        a = float(h['SCALE'])
        om = float(h.get('OMEGA_M', h.get('SIM_OMEGA_M')))
        ol = float(h.get('OMEGA_L', h.get('SIM_OMEGA_L', 1 - om)))
        B = {c: f[1][c][:] for c in ['id', 'pid', mass_col, 'x', 'y', 'z', 'vx', 'vy', 'vz']}
    sel = B['pid'] == -1 if hosts_only else np.ones(len(B['id']), bool)
    H = dict(id=B['id'][sel].astype(np.int64), m=B[mass_col][sel].astype(np.float64),
             pos=np.stack([B[c][sel].astype(np.float64) % L for c in 'xyz'], axis=1),
             vel=np.stack([B[c][sel].astype(np.float64) for c in ('vx', 'vy', 'vz')], axis=1),
             L=L, a=a, om=om, ol=ol, phi={})
    if phis:
        with fitsio.FITS(os.path.join(d, 'out_%s_vahc.fits.gz' % snap)) as f:
            if not f[1].read_header().get('HAS_VPOT', False):
                raise SystemExit('out_%s_vahc.fits.gz has no .vpot columns (HAS_VPOT): run vahc with PP_POT=1' % snap)
            vid = f[1]['haloID'][:].astype(np.int64)
            V = {c: f[1][c][:].astype(np.float64) for c in phis}
        o = np.argsort(vid)
        k = np.minimum(np.searchsorted(vid[o], H['id']), len(vid) - 1)
        ok = vid[o][k] == H['id']
        for c in phis:
            p = np.full(len(H['id']), np.nan)
            p[ok] = V[c][o][k[ok]]
            H['phi'][c] = p
    return H


def parse_sbins(s):
    kind, v = s.split(':', 1)
    v = [float(x) for x in v.split(',')]
    if kind == 'lin':
        return v[0] + v[2] * np.arange(int(round((v[1] - v[0]) / v[2])) + 1)
    if kind == 'log':
        return np.logspace(np.log10(v[0]), np.log10(v[1]), int(v[2]) + 1)
    raise SystemExit('--sbins lin:smin,smax,width or log:smin,smax,n')


def legendre_bin_integrals(mu_edges, ells):
    """int_{mu_j}^{mu_j+1} P_l(mu) dmu for each l (rows) and mu bin (columns)"""
    from numpy.polynomial import legendre as Lg
    out = np.zeros((len(ells), len(mu_edges) - 1))
    for i, l in enumerate(ells):
        c = np.zeros(l + 1); c[l] = 1
        F = Lg.legint(c)
        out[i] = Lg.legval(mu_edges[1:], F) - Lg.legval(mu_edges[:-1], F)
    return out


def multipoles(DD, npair, s_edges, mu_edges, L, W, ells):
    """DD[..., s, mu] -> xi_l[..., l, s]; npair = ordered (A, B) pairs possible: N_A N_B, or N_A (N_B - 1) for A = B"""
    vol = 4.0 / 3.0 * np.pi * (s_edges[1:]**3 - s_edges[:-1]**3)
    dmu = np.diff(mu_edges)
    RR = np.asarray(npair, float)[..., None, None] * vol[:, None] * dmu[None, :] / 2.0 / L**3
    with np.errstate(invalid='ignore', divide='ignore'):
        xi = DD / RR - 1.0
    fac = (2 * np.asarray(ells) + 1) / 2.0
    return np.einsum('...sm,lm->...ls', xi, W) * fac[:, None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dir'); ap.add_argument('snap'); ap.add_argument('out')
    ap.add_argument('--spaces', default='real,rsd,grav,rsd+grav')
    ap.add_argument('--phi', default='phi_rs')
    ap.add_argument('--phi-fallback', default='phi_R0p5vir')
    ap.add_argument('--pot-amp', type=float, default=1.0)
    ap.add_argument('--los', default='z')
    ap.add_argument('--mbins', default='2e12,1e13,5e13,2e14,1e16')
    ap.add_argument('--mass-col', default='Mvir')
    ap.add_argument('--pairs', default='all')
    ap.add_argument('--sbins', default='lin:0,100,2')
    ap.add_argument('--nmu', type=int, default=200)
    ap.add_argument('--ells', default='0,1,2,3,4')
    ap.add_argument('--jk', type=int, default=3)
    ap.add_argument('--subsample-a', type=float, default=1.0)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--all-halos', action='store_true')
    A = ap.parse_args()
    try:
        from Corrfunc.theory.DDsmu import DDsmu
    except ImportError:
        raise SystemExit('Corrfunc (fork with signed mu) is needed: bash scripts/pkdgrav3_pipeline/install_corrfunc.sh '
                         '<this python> (docs/pkdgrav3_pipeline_pawna.md section 13)')
    t0 = time.time()
    threads = A.threads or int(os.environ.get('PBS_NCPUS', os.environ.get('NCPUS', os.cpu_count() or 1)))
    spaces_base = [s.strip() for s in A.spaces.split(',') if s.strip()]
    phis = [c.strip() for c in A.phi.split(',') if c.strip()] if any('grav' in s for s in spaces_base) else []
    fallback = A.phi_fallback.strip()
    H = read_halos(A.dir, A.snap, A.mass_col, not A.all_halos, phis + ([fallback] if fallback and phis else []))
    L, a = H['L'], H['a']
    E = np.sqrt(H['om'] * a**-3 + H['ol'] + (1 - H['om'] - H['ol']) * a**-2)
    aH = a * 100.0 * E                                   # km/s per Mpc/h (comoving displacement = v / aH)
    s_edges = parse_sbins(A.sbins)
    s_count = s_edges.copy()                             # Corrfunc counts s >= smin: a halo with itself at s = 0
    s_count[0] = max(s_count[0], 1e-9 * L)               # would land in the first bin for A = B
    if s_edges[-1] > L / 2:
        raise SystemExit('smax %.4g > L/2 = %.4g' % (s_edges[-1], L / 2))
    mu_edges = np.linspace(-1, 1, A.nmu + 1)
    ells = [int(x) for x in A.ells.split(',')]
    W = legendre_bin_integrals(mu_edges, ells)
    me = np.array([float(x) for x in A.mbins.split(',')])
    nmb = len(me) - 1
    if A.pairs in ('all', 'auto', 'cross'):
        pairs = [(i, j) for i in range(nmb) for j in range(i, nmb)
                 if A.pairs == 'all' or (A.pairs == 'auto') == (i == j)]
    else:
        pairs = [tuple(int(x) for x in p.split('-')) for p in A.pairs.split(',')]
    los = [c.strip() for c in A.los.split(',')]
    axis = {'x': 0, 'y': 1, 'z': 2}

    # spaces: (name, uses velocity, potential column or None)
    spaces = []
    for s in spaces_base:
        if s in ('real', 'rsd'):
            spaces.append((s, s == 'rsd', None))
        elif s in ('grav', 'rsd+grav'):
            for c in phis:
                spaces.append((s if len(phis) == 1 else '%s[%s]' % (s, c), s == 'rsd+grav', c))
        else:
            raise SystemExit('unknown space %s' % s)
    vg = {}
    for c in phis:
        p = H['phi'][c].copy()
        if fallback:
            bad = ~np.isfinite(p)
            p[bad] = H['phi'][fallback][bad]
        vg[c] = -A.pot_amp * p / C_KMS                   # km/s
    good = np.ones(len(H['id']), bool)
    for c in phis:
        good &= np.isfinite(vg[c])
    mb = np.digitize(H['m'], me) - 1
    mb[~good] = -1
    u = uniform_from_id(H['id'])
    reg = np.minimum((H['pos'] / L * A.jk).astype(int), A.jk - 1)
    reg = (reg[:, 0] * A.jk + reg[:, 1]) * A.jk + reg[:, 2]
    nreg = A.jk**3
    n_halos = np.array([(mb == b).sum() for b in range(nmb)])
    vg_mean = np.array([[vg[c][mb == b].mean() if (mb == b).any() else np.nan for b in range(nmb)] for c in phis])
    print('%s snap %s: L %.1f, a %.4f, E %.4f, aH %.2f km/s/(Mpc/h); %d hosts (%d dropped: no phi), per bin %s; '
          'spaces %s; los %s; %d s bins %.3g-%.3g, %d mu bins; %d threads'
          % (A.dir, A.snap, L, a, E, aH, len(H['id']), int((~good).sum()), n_halos.tolist(),
             [s[0] for s in spaces], los, len(s_edges) - 1, s_edges[0], s_edges[-1], A.nmu, threads), flush=True)
    if phis:
        for c in phis:
            print('  v_g = -%s/c: median %.2f km/s, 1-99%% %.2f..%.2f (pot-amp %g)'
                  % (c, np.median(vg[c][good]), *np.percentile(vg[c][good], [1, 99]), A.pot_amp))

    ns, nm, P, S, NL = len(s_edges) - 1, A.nmu, len(pairs), len(spaces), len(los)
    DD = np.zeros((S, NL, P, ns, nm))
    DDk = np.zeros((S, NL, P, nreg, ns, nm))
    nA_k = np.zeros((P, nreg)); nB = np.zeros(P); auto = np.array([p[0] == p[1] for p in pairs])
    for il, ax in enumerate(los):
        k = axis[ax]
        perm = [q for q in (0, 1, 2) if q != k] + [k]    # line of sight -> third coordinate
        for isp, (name, use_v, c) in enumerate(spaces):
            pos = H['pos'][:, perm].copy()
            shift = np.zeros(len(pos))
            if use_v:
                shift += H['vel'][:, k]
            if c is not None:
                shift += vg[c]
            pos[:, 2] = (pos[:, 2] + shift / aH) % L
            t = time.time()
            for ip, (ba, bb) in enumerate(pairs):
                ia = np.flatnonzero((mb == ba) & (u < A.subsample_a))
                ib = np.flatnonzero(mb == bb)
                if len(ia) == 0 or len(ib) == 0:
                    continue
                if il == 0 and isp == 0:
                    nA_k[ip] = np.bincount(reg[ia], minlength=nreg)
                    nB[ip] = len(ib)
                X2 = [np.ascontiguousarray(pos[ib, q]) for q in range(3)]
                for r in range(nreg):
                    sa = ia[reg[ia] == r]
                    if len(sa) == 0:
                        continue
                    res = DDsmu(0, threads, s_count, 1.0, nm, *[np.ascontiguousarray(pos[sa, q]) for q in range(3)],
                                X2=X2[0], Y2=X2[1], Z2=X2[2], boxsize=L, periodic=True)
                    DDk[isp, il, ip, r] = res['npairs'].reshape(ns, nm)
            DD[isp, il] = DDk[isp, il].sum(axis=1)
            print('los %s, %-18s %.1f s, %.3g pairs' % (ax, name, time.time() - t, DD[isp, il].sum()), flush=True)

    nA = nA_k.sum(axis=1)
    nBe = nB - auto                                      # partners of an A halo: N_B, or N_B - 1 for A = B
    xi = np.zeros((S, NL, P, len(ells), ns))
    for ip in range(P):
        xi[:, :, ip] = multipoles(DD[:, :, ip], nA[ip] * nBe[ip], s_edges, mu_edges, L, W, ells)
    def xi_loo(d, ip, r):
        """multipoles of DD with region r of the A halos left out"""
        return multipoles(d, (nA[ip] - nA_k[ip, r]) * nBe[ip], s_edges, mu_edges, L, W, ells)

    def jerr(x):  # jackknife error over the region axis (2)
        return np.sqrt((nreg - 1) / nreg * ((x - x.mean(axis=2, keepdims=True))**2).sum(axis=2))

    # mean over lines of sight and its jackknife
    DDm = DD.mean(axis=1)                                # (S, P, s, mu)
    DDkm = DDk.mean(axis=1)                              # (S, P, region, s, mu)
    xim = np.zeros((S, P, len(ells), ns))
    jk = np.zeros((S, P, nreg, len(ells), ns))
    xi_err = np.zeros_like(xi)
    for ip in range(P):
        xim[:, ip] = multipoles(DDm[:, ip], nA[ip] * nBe[ip], s_edges, mu_edges, L, W, ells)
        for r in range(nreg):
            jk[:, ip, r] = xi_loo(DDm[:, ip] - DDkm[:, ip, r], ip, r)
    xim_err = jerr(jk)
    for il in range(NL):                                 # each line of sight on its own
        jl = np.zeros((S, P, nreg, len(ells), ns))
        for ip in range(P):
            for r in range(nreg):
                jl[:, ip, r] = xi_loo(DD[:, il, ip] - DDk[:, il, ip, r], ip, r)
        xi_err[:, il] = jerr(jl)
    # potential effect: grav - real, rsd+grav - rsd (same halos and regions)
    names = [s[0] for s in spaces]
    diffs, dxi, dxi_err = [], [], []
    for i, (name, use_v, c) in enumerate(spaces):
        if c is None:
            continue
        base = 'rsd' if use_v else 'real'
        if base in names:
            j = names.index(base)
            diffs.append('%s - %s' % (name, base))
            dxi.append(xim[i] - xim[j])
            dxi_err.append(jerr(jk[i] - jk[j]))
    s_mid = 0.5 * (s_edges[1:] + s_edges[:-1])
    np.savez_compressed(
        A.out, s_edges=s_edges, s_mid=s_mid, mu_edges=mu_edges, ells=np.array(ells), spaces=np.array(names),
        los=np.array(los), pairs=np.array(pairs), m_edges=me, mass_col=A.mass_col, n_halos=n_halos, n_a=nA, n_b=nB,
        phi_cols=np.array(phis), phi_fallback=fallback, pot_amp=A.pot_amp, vg_mean=vg_mean, a=a, z=1 / a - 1, box=L, aH=aH,
        omega_m=H['om'], omega_l=H['ol'], subsample_a=A.subsample_a, jk=A.jk, DD=DD,
        xi_ell=xi, xi_ell_err=xi_err, xi_ell_mean=xim, xi_ell_mean_err=xim_err, xi_ell_jk=jk,
        diff_names=np.array(diffs), dxi_ell=np.array(dxi), dxi_ell_err=np.array(dxi_err),
        source=os.path.abspath(A.dir), snap=str(A.snap))
    print('wrote %s in %.1f s' % (A.out, time.time() - t0))


if __name__ == '__main__':
    main()
