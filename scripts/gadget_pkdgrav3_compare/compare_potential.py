#!/usr/bin/env python3
"""Compare the gravitational potential of a Gadget-4 and a pkdgrav3 run started from the same ICs, particle by
particle and halo by halo, at every common output.

usage: compare_potential.py <gadget_dir> <pkd_dir> <gadget_halo_dir> <pkd_halo_dir> <out_dir> [--plot-z 3,1,0.5,0]
  gadget_dir   snapshot_<nnn>.hdf5 (with Potential), snapnames.txt         (job_gadget.pbs)
  pkd_dir      snap.<step> (HDF5) + snap.<step>.s*.pkc, snapnames.txt, p.par (pkdgrav3_pipeline job_sim.pbs, ALSO_HDF5=1)
  *_halo_dir   out_<snap>.trees (consistent-trees hlists; job_ctrees.pbs); where there is none, Rockstar's
               out_<snap>.list (all halos, no host/subhalo flag) is used instead
Output in out_dir: particles.csv (one row per output), halos.csv (one row per matched halo and output),
summary.json, snap_<nnn>.npz (particle and halo arrays of the outputs closest to --plot-z, for plot_compare.py).

Conventions (all potentials in peculiar, physical (km/s)^2):
  Gadget-4: Potential attribute scaling a^-1 (snap_io.cc): phi = Potential / a. Self-interaction removed
            (gravity.cc), zero point of the periodic TreePM potential = volume mean 0.
  pkdgrav3: phi = Potential_code * U^2 / a, U = 100 L / sqrt(8 pi / 3) km/s (G = 1, rho_crit = 3/(8 pi), L = 1);
            the same as its .pkc potential (checked here). It keeps each particle's softened self-interaction,
            -(105/64) m / eps (gravity/pp.h at r = 0), a constant for equal masses: phi_pkd_corr = phi - that.
  The two codes can still differ by a constant per snapshot (zero point of the periodic potential, e.g. how a
  particle's own periodic images are counted: on a uniform lattice that is the Madelung term 2.837 m/d). It changes
  no force. particles.csv reports it (offset_median of phi_pkd_corr - phi_gadget); the halo values and snap_*.npz use
  pkdgrav3 potentials aligned by it (phi_pkd_corr - offset_median).
  Particles are matched by ID: Gadget ID - 1 = pkdgrav3 iOrder (gadget4_ic_to_gadget2.py).
  Independent reference: periodic FFT Poisson solution of each snapshot's own particles (CIC on --grid^3,
  deconvolved), lap phi = 1.5 Om H0^2 delta / a, mean 0, interpolated (CIC) to the particles.
Halos: host halos (pid = -1) of each code's out_<snap>.trees with >= --min-np particles within Rvir. For each halo,
  from its own snapshot: particles within Rvir of the centre -> phi_min, <phi>, and <phi> of the 10% most bound.
  Gadget halos are matched to pkdgrav3 halos by mutual nearest centre within 0.5 Rvir_gadget and |log10 M ratio|
  < 0.3. Lagrangian check: the pkdgrav3 potential of the Gadget halo's member particles (same IDs).
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import h5py
import numpy as np
from scipy.spatial import cKDTree

H0 = 100.0  # km/s per Mpc/h


def wrap(x, L):
    x = np.mod(x, L)
    x[x >= L] = 0.0                                  # cKDTree(boxsize=L) needs [0, L)
    return x


def read_par(path):
    """The few scalar values of pkdgrav3's p.par that are needed (dSoft may be an expression)."""
    d = {}
    for l in open(path):
        m = re.match(r'\s*(\w+)\s*=\s*(.+?)\s*$', l)
        if m and m.group(1) in ('dSoft', 'dBoxSize', 'dOmega0', 'h'):
            d[m.group(1)] = float(eval(m.group(2), {}, {}))
    return d


def read_gadget(path):
    with h5py.File(path, 'r') as f:
        a = float(f['Header'].attrs['Time'])
        L = float(f['Header'].attrs['BoxSize'])
        g = f['PartType1']
        ids = g['ParticleIDs'][:].astype(np.int64)
        pos = g['Coordinates'][:].astype(np.float64)
        pot = g['Potential'][:].astype(np.float64)
    o = np.empty(len(ids), np.int64)
    o[ids - 1] = np.arange(len(ids))                  # row of ID k+1 -> index k
    return a, L, wrap(pos[o], L), pot[o] / a


def read_pkd(path, L):
    with h5py.File(path, 'r') as f:
        a = 1.0 / (1.0 + float(f['Header'].attrs['Redshift']))   # pkdgrav3's Header/Time is code time, not a
        assert abs(float(f['Header'].attrs['BoxSize']) - L) < 1e-6 * L, 'pkdgrav3 HDF5 BoxSize (units set?)'
        g = f['PartType1']
        pos = (g['Coordinates'][:].astype(np.float64) + 0.5) * L      # ordered: row = iOrder
        pot = g['Potential'][:].astype(np.float64)
    U = H0 * L / math.sqrt(8 * math.pi / 3)
    return a, wrap(pos, L), pot * U**2 / a, U


def pkc_check(prefix, phi_hdf5):
    """max |phi_pkc - phi_hdf5| (km/s)^2 and the pkc potential step; None if pkcreader is not importable."""
    try:
        import pkcreader
    except ImportError:
        return None, None
    s = pkcreader.PkcSnapshot(prefix)
    r = s.read(fields=('ids', 'pot'), dtype=np.float64)
    p = np.empty(s.npart)
    p[r['ids'].astype(np.int64)] = r['pot']
    return float(np.abs(p - phi_hdf5).max()), float(s.pot_step)


def cic_weights(pos, n, L):
    u = pos / L * n
    i0 = np.floor(u).astype(np.int64)
    f = u - i0
    return i0, f


def fft_potential(pos, n, L, a, Om):
    """Peculiar potential (km/s)^2 at the particles from a CIC density on n^3 (deconvolved twice: assignment and
    interpolation), lap phi = 1.5 Om H0^2 delta / a, k = 0 mode zero."""
    i0, f = cic_weights(pos, n, L)
    rho = np.zeros((n, n, n))
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1 - f[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1 - f[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1 - f[:, 2]
                np.add.at(rho, ((i0[:, 0] + dx) % n, (i0[:, 1] + dy) % n, (i0[:, 2] + dz) % n), wx * wy * wz)
    delta = rho / rho.mean() - 1
    dk = np.fft.rfftn(delta)
    k1 = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    kz = 2 * np.pi * np.fft.rfftfreq(n, d=L / n)
    kx, ky, kzz = np.meshgrid(k1, k1, kz, indexing='ij')
    k2 = kx**2 + ky**2 + kzz**2
    w = (np.sinc(kx * L / n / (2 * np.pi)) * np.sinc(ky * L / n / (2 * np.pi)) * np.sinc(kzz * L / n / (2 * np.pi)))**2
    with np.errstate(divide='ignore', invalid='ignore'):
        pk = -1.5 * Om * H0**2 / a * dk / k2 / w**2
    pk[0, 0, 0] = 0
    phi = np.fft.irfftn(pk, s=(n, n, n))
    out = np.zeros(len(pos))
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1 - f[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1 - f[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1 - f[:, 2]
                out += wx * wy * wz * phi[(i0[:, 0] + dx) % n, (i0[:, 1] + dy) % n, (i0[:, 2] + dz) % n]
    return out


def fit(x, y):
    """slope, offset, correlation of y vs x, and rms of y - (slope x + offset)."""
    A = np.vstack([x, np.ones_like(x)]).T
    (s, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    r = np.corrcoef(x, y)[0, 1]
    return float(s), float(c), float(r), float(np.std(y - s * x - c))


def read_hlist(path):
    with open(path) as f:
        head = f.readline().lstrip('#').split()
    names = [re.sub(r'\(\d+\)$', '', h) for h in head]
    d = np.loadtxt(path, comments='#', ndmin=2)
    if d.size == 0:
        h = {k: np.zeros(0) for k in names}
    else:
        h = {k: d[:, i] for i, k in enumerate(names)}
    if 'x' not in h:   # Rockstar out_<snap>.list (no consistent-trees): ID Mvir Rvir X Y Z ..., no pid column
        low = {k.lower(): v for k, v in h.items()}
        h = dict(id=low['id'], mvir=low['mvir'], rvir=low['rvir'], x=low['x'], y=low['y'], z=low['z'],
                 pid=np.full(len(low['id']), -1.0))
    return h


def halo_file(d, i):
    """consistent-trees catalogue out_<i>.trees, else Rockstar's out_<i>.list (e.g. too few halos for ctrees)."""
    for n in ('out_%d.trees' % i, 'out_%d.list' % i):
        if os.path.exists(os.path.join(d, n)):
            return os.path.join(d, n)
    return None


def halo_props(h, pos, phi, tree, L, mp, min_np):
    """Per host halo: particles within Rvir of the centre in this snapshot."""
    host = h['pid'] == -1
    c = wrap(np.vstack([h['x'][host], h['y'][host], h['z'][host]]).T, L)
    rv = h['rvir'][host] / 1e3                       # kpc/h -> Mpc/h comoving
    out = dict(id=h['id'][host], mvir=h['mvir'][host], rvir=rv, c=c, n=[], phimin=[], phimean=[], phicore=[],
               members=[])
    for ci, r in zip(c, rv):
        m = np.array(tree.query_ball_point(ci, r), dtype=np.int64)
        out['members'].append(m)
        out['n'].append(len(m))
        if len(m):
            p = np.sort(phi[m])
            out['phimin'].append(p[0]); out['phimean'].append(p.mean())
            out['phicore'].append(p[:max(1, len(p) // 10)].mean())
        else:
            out['phimin'].append(np.nan); out['phimean'].append(np.nan); out['phicore'].append(np.nan)
    for k in ('n', 'phimin', 'phimean', 'phicore'):
        out[k] = np.array(out[k])
    keep = out['n'] >= min_np
    for k in ('id', 'mvir', 'rvir', 'c', 'n', 'phimin', 'phimean', 'phicore'):
        out[k] = out[k][keep]
    out['members'] = [m for m, k in zip(out['members'], keep) if k]
    return out


def match(hg, hp, L):
    """Mutual nearest centres within 0.5 Rvir_gadget and |log10 Mg/Mp| < 0.3 -> list of (ig, ip, distance)."""
    if len(hg['id']) == 0 or len(hp['id']) == 0:
        return []
    tg, tp = cKDTree(hg['c'], boxsize=L), cKDTree(hp['c'], boxsize=L)
    dgp, igp = tp.query(hg['c'])
    _, ipg = tg.query(hp['c'])
    out = []
    for ig, (d, ip) in enumerate(zip(dgp, igp)):
        if ipg[ip] == ig and d < 0.5 * hg['rvir'][ig] and abs(math.log10(hg['mvir'][ig] / hp['mvir'][ip])) < 0.3:
            out.append((ig, int(ip), float(d)))
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ('gadget_dir', 'pkd_dir', 'gadget_halo_dir', 'pkd_halo_dir', 'out_dir'):
        ap.add_argument(k)
    ap.add_argument('--plot-z', default='3,1,0.5,0')
    ap.add_argument('--grid', type=int, default=0, help='FFT grid (default 4 x the particle grid)')
    ap.add_argument('--min-np', type=int, default=20)
    ap.add_argument('--snaps', default='', help='comma list of output numbers (default: all)')
    A = ap.parse_args()
    os.makedirs(A.out_dir, exist_ok=True)
    gn = [l.strip() for l in open(os.path.join(A.gadget_dir, 'snapnames.txt')) if l.strip()]
    pn = [l.strip() for l in open(os.path.join(A.pkd_dir, 'snapnames.txt')) if l.strip()]
    assert len(gn) == len(pn), 'different numbers of outputs: %d Gadget, %d pkdgrav3' % (len(gn), len(pn))
    par = read_par(os.path.join(A.pkd_dir, 'p.par'))
    Om = par['dOmega0']
    snaps = [int(s) for s in A.snaps.split(',')] if A.snaps else list(range(len(gn)))
    zs = []
    for i in snaps:
        with h5py.File(os.path.join(A.gadget_dir, 'snapshot_%s.hdf5' % gn[i]), 'r') as f:
            zs.append(max(0.0, 1 / float(f['Header'].attrs['Time']) - 1))
    plot_snaps = sorted({snaps[int(np.argmin([abs(z - zp) for z in zs]))] for zp in map(float, A.plot_z.split(','))})

    prow, hrow = [], []
    for i in snaps:
        ag, L, xg, phig = read_gadget(os.path.join(A.gadget_dir, 'snapshot_%s.hdf5' % gn[i]))
        ap_, xp, phip, U = read_pkd(os.path.join(A.pkd_dir, 'snap.%s' % pn[i]), L)
        N = len(phig)
        assert len(phip) == N
        npart = round(N ** (1 / 3))
        grid = A.grid or 4 * npart
        mcode, eps = Om / N, par['dSoft']
        selfpot = -(105.0 / 64.0) * mcode / eps * U**2 / ap_
        phipc = phip - selfpot
        pkc_err, pkc_step = pkc_check(os.path.join(A.pkd_dir, 'snap.%s' % pn[i]), phip)
        dx = xp - xg
        dx -= L * np.round(dx / L)
        dr = np.sqrt((dx**2).sum(1))
        d = phipc - phig
        s, c, r, rms = fit(phig, phipc)
        fg = fft_potential(xg, grid, L, ag, Om)
        fp = fft_potential(xp, grid, L, ap_, Om)
        sg, cg, rg, rmsg = fit(fg, phig)
        sp, cp, rp, rmsp = fit(fp, phipc)
        sf, cf, rf, rmsf = fit(fg, fp)
        off = float(np.median(d))
        phipa = phipc - off                          # aligned: same zero point as Gadget-4 (see the docstring)
        rel = np.abs(d - off) / np.abs(phig).mean()
        row = dict(out=i, snap_gadget=gn[i], step_pkd=pn[i], z=1 / ag - 1, a_gadget=ag, a_pkd=ap_, N=N,
                   phi_gadget_mean=phig.mean(), phi_gadget_rms=phig.std(), phi_pkd_corr_mean=phipc.mean(),
                   pkd_selfpot=selfpot, offset_median=float(np.median(d)), slope=s, intercept=c, corr=r,
                   rms_resid=rms, rms_diff=float(np.std(d)), rel_diff_p50=float(np.percentile(rel, 50)),
                   rel_diff_p90=float(np.percentile(rel, 90)), rel_diff_p99=float(np.percentile(rel, 99)),
                   dr_median=float(np.median(dr)), dr_p90=float(np.percentile(dr, 90)), dr_max=float(dr.max()),
                   fft_grid=grid, gadget_vs_fft_slope=sg, gadget_vs_fft_corr=rg, gadget_vs_fft_rms=rmsg,
                   pkd_vs_fft_slope=sp, pkd_vs_fft_corr=rp, pkd_vs_fft_rms=rmsp, fft_pkd_vs_fft_gadget_rms=rmsf,
                   pkc_max_err=pkc_err, pkc_step=pkc_step)
        prow.append(row)

        # halos
        hg = hp = None
        fg_, fp_ = halo_file(A.gadget_halo_dir, i), halo_file(A.pkd_halo_dir, i)
        nm = 0
        if fg_ and fp_:
            tg, tp = cKDTree(xg, boxsize=L), cKDTree(xp, boxsize=L)
            hg = halo_props(read_hlist(fg_), xg, phig, tg, L, None, A.min_np)
            hp = halo_props(read_hlist(fp_), xp, phipa, tp, L, None, A.min_np)
            pairs = match(hg, hp, L)
            nm = len(pairs)
            for ig, ip, dist in pairs:
                mg, mp_ = hg['members'][ig], hp['members'][ip]
                shared = len(np.intersect1d(mg, mp_)) / len(mg)
                hrow.append(dict(out=i, z=1 / ag - 1, id_gadget=int(hg['id'][ig]), id_pkd=int(hp['id'][ip]),
                                 mvir_gadget=hg['mvir'][ig], mvir_pkd=hp['mvir'][ip], rvir_gadget=hg['rvir'][ig],
                                 rvir_pkd=hp['rvir'][ip], n_gadget=int(hg['n'][ig]), n_pkd=int(hp['n'][ip]),
                                 centre_offset=dist, shared_members=shared,
                                 phimin_gadget=hg['phimin'][ig], phimin_pkd=hp['phimin'][ip],
                                 phimean_gadget=hg['phimean'][ig], phimean_pkd=hp['phimean'][ip],
                                 phicore_gadget=hg['phicore'][ig], phicore_pkd=hp['phicore'][ip],
                                 phimean_pkd_of_gadget_members=float(phipa[mg].mean()),
                                 phimin_pkd_of_gadget_members=float(phipa[mg].min()), pkd_offset=off))
            row.update(n_halos_gadget=len(hg['id']), n_halos_pkd=len(hp['id']), n_halos_matched=nm,
                       halo_catalogue=os.path.basename(fg_))
        print('out %3d z %7.4f  offset %+.4g (self %+.4g)  slope %.5f corr %.6f rms %.4g  dr_med %.3g  halos %s'
              % (i, row['z'], row['offset_median'], selfpot, s, r, rms, row['dr_median'],
                 '%d/%d matched %d' % (len(hg['id']), len(hp['id']), nm) if hg else '-'), flush=True)
        if i in plot_snaps:
            sel = np.random.default_rng(0).choice(N, min(N, 200000), replace=False)
            np.savez_compressed(os.path.join(A.out_dir, 'snap_%03d.npz' % i), z=row['z'], L=L,
                                phig=phig[sel], phip=phipa[sel], offset=off, fftg=fg[sel], fftp=fp[sel], dr=dr[sel],
                                xg=xg[sel].astype(np.float32), selfpot=selfpot)

    def write_csv(rows, path):
        keys = []
        for r_ in rows:
            keys += [k for k in r_ if k not in keys]
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(rows)
    write_csv(prow, os.path.join(A.out_dir, 'particles.csv'))
    if hrow:
        write_csv(hrow, os.path.join(A.out_dir, 'halos.csv'))
    summ = dict(gadget_dir=A.gadget_dir, pkd_dir=A.pkd_dir, n_outputs=len(prow), plot_outputs=plot_snaps,
                dSoft_pkd=par['dSoft'], Om=Om, last=prow[-1], n_matched_halo_rows=len(hrow))
    json.dump(summ, open(os.path.join(A.out_dir, 'summary.json'), 'w'), indent=1, default=float)
    print('wrote %s/{particles.csv,halos.csv,summary.json,snap_*.npz}' % A.out_dir)


if __name__ == '__main__':
    main()
