#!/usr/bin/env python3
"""Matter and halo power spectra from the density grid that vahc_fast already builds (stage vahc, --pk).

  matter   P_mm(k) of the CIC density contrast on the PP_GRID^3 mesh, CIC window deconvolved, on the k bins of
           correlations.PowerSpectrum (mode counts identical to PowerSpectrum.I_SIZE); shot noise 1/nbar written
           separately (not subtracted); mean k of the modes in each bin
  halos    host halos (pid = -1, not phantoms) in bins of Mvir given in particles (PK_MBINS, default
           100,300,1000,3000,10000 -> 4 bins + the last open one): P_hm (cross, no shot noise) and P_hh (shot noise
           1/n_h written separately); biases b_hm = P_hm / (P_mm - 1/nbar), b_hh = sqrt((P_hh - 1/n_h) / (P_mm -
           1/nbar)), and the large-scale b1 = mode-weighted mean of b_hm over k <= PK_KLIN (default 0.1 h/Mpc)
  output   out_<snap>.pk next to out_<snap>.trees: '# meta: {json}' line, '# columns: ...', then the table
           (one row per k bin with at least one mode)

Aliasing is not corrected (no interlacing): trust k < k_Ny / 2 (k_Ny = pi PP_GRID / Lbox).
"""
import json

import numpy as np
from scipy import fft as sfft

MBINS_DEFAULT = '100,300,1000,3000,10000'


class GridPower:
    """Binned auto/cross spectra of fields given as B = rfftn(delta) / W_CIC^2 (vahc_fast's convention)."""

    def __init__(self, ps):
        self.ps = ps
        N = ps.grid
        self.N = N
        kf = np.arange(N)
        kf[kf >= N / 2] -= N
        kh = np.arange(N // 2 + 1)
        KK = kf[:, None, None]**2 + kf[None, :, None]**2 + kh[None, None, :]**2
        kbin = np.searchsorted(ps.k2_compare, KK, side='right') - 1
        w = np.full(N // 2 + 1, 2.0)           # half grid: modes off the k3 = 0, N/2 planes count twice
        w[0] = 1.0
        if N % 2 == 0:
            w[N // 2] = 1.0
        ok = (kbin >= 0) & (kbin < ps.nbin)
        self.idx = kbin[ok]
        self.w = np.broadcast_to(w[None, None, :], KK.shape)[ok]
        self.ok = ok
        self.nmod = np.bincount(self.idx, weights=self.w, minlength=ps.nbin)[:ps.nbin]
        if not np.array_equal(np.rint(self.nmod).astype(int), ps.I_SIZE):
            raise RuntimeError('mode counts differ from PowerSpectrum.I_SIZE')
        kmag = np.sqrt(KK[ok]) * ps.Delta_k
        self.kmean = np.bincount(self.idx, weights=self.w * kmag, minlength=ps.nbin)[:ps.nbin] / np.maximum(self.nmod, 1)
        self.ngp2 = None

    def transform(self, delta, threads=1):
        """B of a real density contrast on the grid."""
        N = self.N
        if self.ngp2 is None:
            kf = np.arange(N)
            kf[kf >= N / 2] -= N
            kh = np.arange(N // 2 + 1)
            self.ngp2 = (np.sinc(1.0 * kf[:, None, None] / N) * np.sinc(1.0 * kf[None, :, None] / N)
                         * np.sinc(1.0 * kh[None, None, :] / N))**2
        return sfft.rfftn(np.asarray(delta, dtype=np.float64), workers=threads) / self.ngp2

    def power(self, A, B=None):
        """P(k) [(Mpc/h)^3] of A (auto) or Re(A B*) (cross); no shot-noise subtraction."""
        N3 = float(self.N)**3
        x = (np.abs(A[self.ok])**2 if B is None else np.real(A[self.ok] * np.conj(B[self.ok]))) / N3**2
        s = np.bincount(self.idx, weights=self.w * x, minlength=self.ps.nbin)[:self.ps.nbin]
        return s * self.ps.Vbox / np.maximum(self.nmod, 1)


def mass_bins(spec):
    edges = [float(x) for x in (spec or MBINS_DEFAULT).replace(' ', '').split(',') if x]
    return edges + [np.inf]


def measure(gp, Bm, npart, halos=None, hpos=None, mpart=None, mbins=None, threads=1, klin=0.1, meta=None):
    """Spectra of one snapshot -> (meta dict, column names, table). Bm: matter field (B convention)."""
    ps = gp.ps
    V = ps.Vbox
    Pmm = gp.power(Bm)
    shot_m = V / float(npart)
    cols = ['k_center', 'k_mean', 'Nmodes', 'Pmm', 'Pmm_shot']
    data = [ps.ktab, gp.kmean, gp.nmod, Pmm, np.full_like(Pmm, shot_m)]
    m = dict(meta or {})
    m.update(grid=gp.N, Lbox=ps.Lbox, kNy=float(ps.kNy), npart=int(npart), nbar=npart / V, cic_deconvolved=True,
             shot_noise_subtracted=False, interlaced=False, klin=klin, k_reliable_max=float(ps.kNy) / 2)
    Pm = Pmm - shot_m
    lin = (gp.kmean <= klin) & (gp.nmod > 0)
    hbins = []
    if halos is not None and mpart:
        edges = mass_bins(mbins)
        host = halos['pid'] == -1
        if 'phantom' in halos.dtype.names:
            host &= halos['phantom'] == 0
        npm = halos['mvir'] / mpart
        for i in range(len(edges) - 1):
            sel = host & (npm >= edges[i]) & (npm < edges[i + 1])
            nh = int(sel.sum())
            hb = dict(np_lo=edges[i], np_hi=None if np.isinf(edges[i + 1]) else edges[i + 1],
                      m_lo=edges[i] * mpart, m_hi=None if np.isinf(edges[i + 1]) else edges[i + 1] * mpart, n=nh)
            if nh >= 2:
                dh = ps.density_field(np.ascontiguousarray(hpos[sel].T))
                Bh = gp.transform(dh, threads)
                del dh
                Phm, Phh = gp.power(Bh, Bm), gp.power(Bh)
                shot_h = V / nh
                with np.errstate(invalid='ignore', divide='ignore'):
                    bhm = np.where(Pm > 0, Phm / Pm, np.nan)
                    bhh = np.where((Pm > 0) & (Phh - shot_h > 0), np.sqrt((Phh - shot_h) / Pm), np.nan)
                good = lin & np.isfinite(bhm)
                hb['b1'] = float(np.sum(gp.nmod[good] * bhm[good]) / np.sum(gp.nmod[good])) if good.any() else None
                hb['mean_mvir'] = float(np.mean(halos['mvir'][sel]))
                del Bh
            else:
                Phm = Phh = bhm = bhh = np.full_like(Pmm, np.nan)
                shot_h = np.nan
                hb['b1'] = None
            j = len(hbins)
            cols += ['Phm_%d' % j, 'Phh_%d' % j, 'Phh_shot_%d' % j, 'bhm_%d' % j, 'bhh_%d' % j]
            data += [Phm, Phh, np.full_like(Pmm, shot_h), bhm, bhh]
            hbins.append(hb)
    m['halo_bins'] = hbins
    m['halo_selection'] = 'host halos (pid = -1), no phantoms, Mvir in [np_lo, np_hi) particles'
    tab = np.array(data).T
    keep = gp.nmod > 0
    return m, cols, tab[keep]


def write(path, meta, cols, tab):
    tmp = path + '.part'
    with open(tmp, 'w') as f:
        f.write('# Power spectra of the CIC density on the vahc grid (pk_measure.py); units: k h/Mpc, P (Mpc/h)^3\n')
        f.write('# meta: %s\n' % json.dumps(meta))
        f.write('# columns: %s\n' % ' '.join(cols))
        np.savetxt(f, tab, fmt='%.6e')
    import os
    os.replace(tmp, path)


def read(path):
    """-> (meta, dict of columns)."""
    meta, cols = {}, None
    with open(path) as f:
        for line in f:
            if line.startswith('# meta:'):
                meta = json.loads(line[7:])
            elif line.startswith('# columns:'):
                cols = line[10:].split()
            elif not line.startswith('#'):
                break
    tab = np.loadtxt(path, comments='#', ndmin=2)
    return meta, {c: tab[:, i] for i, c in enumerate(cols)}
