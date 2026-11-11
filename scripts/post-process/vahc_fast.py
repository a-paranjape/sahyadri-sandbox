"""
vahc_fast.py: the value-added catalogues of addvalue.AddValue (out_<snap>.vahc and out_<snap>.tens), computed
faster. Same quantities, same halos, same smoothing scales, same file format; the numbers agree with AddValue to
floating-point rounding (the FFT library and the order of the sums differ). addvalue.py and correlations.py are not
changed; run_addvalue.py uses this module with --fast.

What is different from AddValue.add_value (and why the results are the same):
  1. One real-to-complex FFT of the density (scipy.fft.rfftn, multithreaded), reused for every scale and for both
     outputs. AddValue keeps the complex FT = ifftn(density)/NGP^2 and calls add_value once for .vahc and once for
     .tens, recomputing the 33 tidal scales; here every tidal field is computed once and gives both the eigenvalues
     (.vahc) and the components (.tens). For real density d, FT = conj(rfftn(d))/N^3/NGP^2 on the half grid, and
     fftn(F*FT) = irfftn(F*rfftn(d)/NGP^2) for any real filter F with F(-k) = F(k) (all filters here are), so
     each tensor component is one inverse real FFT instead of one complex FFT.
  2. The Gaussian kernel: AddValue samples exp(-x^2/2R^2) on the grid and transforms it in 3-D (one extra full FFT
     per scale). The sampled Gaussian is a product of three 1-D Gaussians, so its transform is the product of three
     1-D transforms (each real because the 1-D samples are even).
  3. Halo loops: the tensor at each halo's cell (nearest grid point, as AddValue) is gathered with array indexing and
     the eigenvalues come from one batched numpy.linalg.eigvalsh; the interpolation to n x R200b is np.interp's
     formula, vectorized.
  4. Linear bias b1: AddValue sums, for every halo, the CIC-weighted phases exp(-i k.x) times delta(k) over all modes
     of the bins kmin..kmax, divides by P(k) and weights the bins. That is a linear filter of delta(k) evaluated at
     the halo, so b1 = sum over the 8 CIC corners of weight x G(corner), with G = Re fftn(c(k) FT(k)) and
     c = Vbox w_bin / ((N_modes,bin + TINY)(P_bin + TINY)) on the modes of the bins: one FFT plus a lookup instead
     of N_halo x N_modes complex exponentials.
"""
import gc
import time

import numpy as np
import scipy.fft as sfft

from addvalue import AddValue


class FastAddValue(AddValue):
    """AddValue with the computation of add_value() replaced (see module docstring).

    density_real: CIC density contrast on the grid (as PowerSpectrum.density_field returns; not the FT).
    threads: FFT threads (scipy.fft workers).
    The other arguments are AddValue's (sim_stem, real, snap, kmax, massdef, hpos, halos, ps, grid, CIC, ...).
    """

    def __init__(self, density_real, threads=1, timing=None, **kw):
        kw['density'] = None
        kw['input_is_FTdensity'] = True
        AddValue.__init__(self, **kw)
        ps = self.ps
        if not (ps.Lx == ps.Ly == ps.Lz):
            raise ValueError('vahc_fast assumes a cubic box')
        self.threads = max(1, int(threads))
        self.timing = {} if timing is None else timing
        N = self.grid
        self.N = N
        t = time.time()
        R = sfft.rfftn(np.asarray(density_real, dtype=np.float64), workers=self.threads)
        # 1-D wave numbers in grid units, as PowerSpectrum's K1, K2, K3 (full axes) and the rfft half axis
        kf = np.arange(N)
        kf[kf >= N / 2] -= N
        kh = np.arange(N // 2 + 1)
        self.kf, self.kh = kf, kh
        a, b, c = kf[:, None, None], kf[None, :, None], kh[None, None, :]
        ngp = np.sinc(1.0 * a / N) * np.sinc(1.0 * b / N) * np.sinc(1.0 * c / N)
        if self.CIC:
            R /= ngp**2       # = N^3 conj(FT) on the half grid
        del ngp
        self.B = R
        self.s1 = np.sin(ps.grid_inv * a)
        self.s2 = np.sin(ps.grid_inv * b)
        self.s3 = np.sin(ps.grid_inv * c)
        self.KK = a**2 + b**2 + c**2          # (N, N, N/2+1) integer
        self.timing['rfft'] = time.time() - t

    # ------------------------------------------------------------------
    def _irfft(self, H):
        return sfft.irfftn(H, s=(self.N,) * 3, workers=self.threads)

    def _kernel(self, Rs):
        """Gaussian smoothing kernel of AddValue (real-space samples, transformed, normalised to 1 at k = 0)."""
        N = self.N
        g1 = np.exp(-0.5 * self.kf**2 / (N * Rs)**2)
        k1 = sfft.ifft(g1).real
        k1 = k1 / k1[0]
        kh1 = k1[:N // 2 + 1]
        return k1[:, None, None] * k1[None, :, None] * kh1[None, None, :]

    def _tensor_at_halos(self, Rs, tensor, idx):
        """Six components (11, 22, 33, 12, 13, 23) of the tidal tensor or density Hessian smoothed at Rs, at the
        grid cells idx (3 arrays). Returns array (len(idx[0]), 6)."""
        BK = self.B * self._kernel(Rs)
        s1, s2, s3 = self.s1, self.s2, self.s3
        if tensor == 'tidal':
            inv = 1.0 / (s1**2 + s2**2 + s3**2 + self.TINY)
            filt = (lambda: s1**2 * inv, lambda: s2**2 * inv, lambda: s3**2 * inv,
                    lambda: s1 * s2 * inv, lambda: s1 * s3 * inv, lambda: s2 * s3 * inv)
        else:
            filt = (lambda: -s1**2, lambda: -s2**2, lambda: -s3**2,
                    lambda: -s1 * s2, lambda: -s1 * s3, lambda: -s2 * s3)
        out = np.empty((idx[0].size, 6))
        for j, f in enumerate(filt):
            field = self._irfft(BK * f())
            out[:, j] = field[idx]
            del field
        del BK
        gc.collect()
        return out

    # ------------------------------------------------------------------
    def calc_b1_fast(self):
        """Halo-by-halo linear bias, as PowerSpectrum.halobyhalo_b1 (via calc_b1) with kmax = self.kmax."""
        ps = self.ps
        N = self.N
        # P(k) as ps.Pk_grid(FT, input_is_FTdensity=1, nbar=inf): |FT|^2 summed over the modes of each bin
        # (full grid = half grid with weight 2 off the k3 = 0 and k3 = N/2 planes), x Vbox / N_modes
        kbin = np.searchsorted(ps.k2_compare, self.KK, side='right') - 1
        w = np.full(N // 2 + 1, 2.0)
        w[0] = 1.0
        if N % 2 == 0:
            w[N // 2] = 1.0
        wfull = np.broadcast_to(w[None, None, :], kbin.shape)
        ok = (kbin >= 0) & (kbin < ps.nbin)
        FT2 = (np.abs(self.B) / float(N)**3)**2
        psum = np.bincount(kbin[ok], weights=(wfull * FT2)[ok], minlength=ps.nbin)[:ps.nbin]
        nmod = np.bincount(kbin[ok], weights=wfull[ok], minlength=ps.nbin)[:ps.nbin]
        del FT2
        if not np.array_equal(np.rint(nmod).astype(int), ps.I_SIZE):
            raise RuntimeError('mode counts differ from PowerSpectrum.I_SIZE')
        Pk = psum * ps.Vbox / (ps.I_SIZE + ps.TINY) - 1 / np.inf
        KMAX = np.min([np.where(ps.ktab <= self.kmax)[0][-1], ps.nbin / 2])
        pos = Pk > 0.0
        ipos = np.where(pos)[0]
        if not ipos.size:
            raise ValueError('Pk = 0 at all k')
        KMIN = ipos[0]
        if KMIN >= KMAX:
            raise ValueError('Incompatible Pk detected in b1. Try again with finer grid.')
        krange = np.arange(KMIN, KMAX, dtype=int)
        i_size = ps.I_SIZE[krange]
        nmodes_k = 1.0 * i_size
        nmodes_k[~pos[krange]] = 0.0
        wts_k = 1.0 * nmodes_k * (Pk[krange] + ps.TINY)
        wts_k = wts_k / np.sum(wts_k)
        cbin = np.zeros(ps.nbin + 1)
        cbin[krange] = ps.Vbox * wts_k / ((i_size + ps.TINY) * (Pk[krange] + ps.TINY))
        cgrid = np.where(ok, cbin[np.clip(kbin, 0, ps.nbin)], 0.0)
        del kbin, ok, wfull
        G = self._irfft(cgrid * self.B)
        del cgrid
        # CIC corners and weights exactly as halobyhalo_b1
        hpos_grid = self.hpos * (self.grid / np.array([ps.Lx, ps.Ly, ps.Lz]))
        hg = np.fmod(hpos_grid, self.grid).astype('int')
        cp = (hg + 1) % self.grid
        d = np.absolute(hpos_grid - np.floor(hpos_grid)).T
        t = 1.0 - d
        hg, cp = hg.T, cp.T
        b1 = (t[0] * t[1] * t[2] * G[hg[0], hg[1], hg[2]]
              + d[0] * t[1] * t[2] * G[cp[0], hg[1], hg[2]]
              + t[0] * d[1] * t[2] * G[hg[0], cp[1], hg[2]]
              + t[0] * t[1] * d[2] * G[hg[0], hg[1], cp[2]]
              + d[0] * d[1] * t[2] * G[cp[0], cp[1], hg[2]]
              + d[0] * t[1] * d[2] * G[cp[0], hg[1], cp[2]]
              + t[0] * d[1] * d[2] * G[hg[0], cp[1], cp[2]]
              + d[0] * d[1] * d[2] * G[cp[0], cp[1], cp[2]])
        del G
        gc.collect()
        return b1

    # ------------------------------------------------------------------
    @staticmethod
    def _interp_rows(x, xp, fp):
        """np.interp(x[i], xp, fp[i]) for every row i (xp increasing), with np.interp's formula."""
        n = xp.size
        j = np.clip(np.searchsorted(xp, x, side='right') - 1, 0, n - 2)
        rows = np.arange(x.size)
        f0, f1 = fp[rows, j], fp[rows, j + 1]
        slope = (f1 - f0) / (xp[j + 1] - xp[j])
        y = slope * (x - xp[j]) + f0
        y = np.where(x < xp[0], fp[:, 0], y)
        y = np.where(x >= xp[-1], fp[:, -1], y)
        return y

    def compute(self, want=('vahc', 'tens')):
        """Returns {kind: structured array} for the kinds in want, as AddValue.add_value(only_tensor=...)."""
        T = self.timing
        allowed = np.where(self.halos[self.massdef] > self.mmin)[0]
        idx = tuple(self.hgrid_round[allowed].T)
        nS = self.N_scales
        t = time.time()
        comp = np.zeros((allowed.size, nS, 6))     # tidal tensor components at every scale
        for s in range(nS):
            comp[:, s] = self._tensor_at_halos(self.smoothing_scales[s], 'tidal', idx)
        T['tidal'] = time.time() - t
        res = {}
        log_rg = np.log10(self.RGtab)
        nRstr = self.N_scalestrings - 3
        lgRG = [np.log10((s + 1) * self.RG_2R200b_NORM
                         * ((self.halos[self.massdef][allowed] / self.M_NORM) * (self.OM_DEF / self.Om))**(1 / 3.))
                for s in range(nRstr)]

        def per_scale(vals):
            """vals (Nallowed, nS, m) -> list over scale_strings of (Nallowed, m): n x R200b by interpolation
            over the RGtab scales, then the 3 fixed scales"""
            out = []
            for s in range(nRstr):
                out.append(np.stack([self._interp_rows(lgRG[s], log_rg, vals[:, :self.N_RG, e])
                                     for e in range(vals.shape[2])], axis=1))
            for s in range(-3, 0):
                out.append(vals[:, s, :])
            return out

        if 'tens' in want:
            t = time.time()
            dt = [('ID', 'i8')]
            for s in self.scale_strings:
                dt.extend([('tens' + c + '_' + s, 'f') for c in ('11', '22', '33', '12', '13', '23')])
            out = np.zeros(self.Nhalo, dtype=dt)
            out['ID'] = self.halos['ID']
            for s, v in zip(self.scale_strings, per_scale(comp)):
                for e, c in enumerate(('11', '22', '33', '12', '13', '23')):
                    col = np.zeros(self.Nhalo)
                    col[allowed] = v[:, e]
                    out['tens' + c + '_' + s] = col
            res['tens'] = out
            T['tens'] = time.time() - t
        if 'vahc' in want:
            t = time.time()
            M = np.empty(comp.shape[:2] + (3, 3))
            M[..., 0, 0], M[..., 1, 1], M[..., 2, 2] = comp[..., 0], comp[..., 1], comp[..., 2]
            M[..., 0, 1] = M[..., 1, 0] = comp[..., 3]
            M[..., 0, 2] = M[..., 2, 0] = comp[..., 4]
            M[..., 1, 2] = M[..., 2, 1] = comp[..., 5]
            lam = np.linalg.eigvalsh(M)          # (Nallowed, nS, 3), ascending as scipy.linalg.eigvalsh
            del M
            out = np.zeros(self.Nhalo, dtype=self.vadtypelist)
            out['ID'] = self.halos['ID']
            for s, v in zip(self.scale_strings, per_scale(lam)):
                for e in range(3):
                    col = np.zeros(self.Nhalo)
                    col[allowed] = v[:, e]
                    out['lam%d_%s' % (e + 1, s)] = col
            del lam
            T['eig'] = time.time() - t
            t = time.time()
            for s, Rs in zip(self.scale_strings[-2:], self.smoothing_scales[-2:]):
                H = self._tensor_at_halos(Rs, 'hessian', idx)
                M = np.empty((allowed.size, 3, 3))
                M[:, 0, 0], M[:, 1, 1], M[:, 2, 2] = H[:, 0], H[:, 1], H[:, 2]
                M[:, 0, 1] = M[:, 1, 0] = H[:, 3]
                M[:, 0, 2] = M[:, 2, 0] = H[:, 4]
                M[:, 1, 2] = M[:, 2, 1] = H[:, 5]
                lamH = np.linalg.eigvalsh(M)
                for e in range(3):
                    col = np.zeros(self.Nhalo)
                    col[allowed] = lamH[:, e]
                    out['lamH%d_%s' % (e + 1, s)] = col
            T['hessian'] = time.time() - t
            t = time.time()
            out['b1'] = self.calc_b1_fast()
            T['b1'] = time.time() - t
            res['vahc'] = out
        del comp
        gc.collect()
        return res

    # ------------------------------------------------------------------
    def header(self, kind):
        h = "# Value added halo catalog. All quantities computed on {0:d}^3 grid\n".format(self.grid)
        h += '# haloID'
        if kind == 'vahc':
            for s in self.scale_strings:
                h += ' lam1_' + s + ' lam2_' + s + ' lam3_' + s
            for s in self.scale_strings[-2:]:
                h += ' lamH1_' + s + ' lamH2_' + s + ' lamH3_' + s
            h += ' b1'
        else:
            for s in self.scale_strings:
                h += ' tens11_' + s + ' tens22_' + s + ' tens33_' + s + ' tens12_' + s + ' tens13_' + s + ' tens23_' + s
        return h + '\n'

    def write(self, kind, arr, path=None):
        """Write as AddValue (header, then one row per halo: str() of every value, space separated)."""
        if path is None:
            path = self.halo_path + self.sim_stem + '/r{0:d}/out_{1:d}.{2}'.format(self.real, self.snap, kind)
        t = time.time()
        cols = [arr[n].astype(str) for n in arr.dtype.names]
        with open(path, 'w') as f:
            f.write(self.header(kind))
            step = 200000
            for i in range(0, arr.size, step):
                f.write(''.join(' '.join(r) + '\n' for r in zip(*(c[i:i + step] for c in cols))))
        self.timing['write_' + kind] = time.time() - t
        return path
