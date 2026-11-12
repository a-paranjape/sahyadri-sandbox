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

.vpot (with run_addvalue.py --pot; only snapshots whose .pkc files store the potential): mean pkdgrav3 particle
potential phi around each halo, (km/s)^2, with pkdgrav3's constant softened self term -(105/64) m/eps (the same for
every particle) removed:
  Np_rs, phi_rs        particles within rs (Rockstar's NFW scale radius) of the halo centre and their mean phi
  Np_R200b, phi_R200b  the same within R200b = (3 M200b / (4 pi 200 rho_m))^(1/3) (comoving), from M200b
  Np_R0p5vir, phi_R0p5vir, Np_Rvir, phi_Rvir, Np_R1p5vir, phi_R1p5vir
                       the same within 0.5, 1 and 1.5 x Rockstar's Rvir
  phi_<scale>          Gaussian-weighted mean phi at the 7 .vahc scales (2, 4, 6, 8 x R200b by interpolation over
                       the same table of smoothing radii as the tidal eigenvalues; 2, 3, 5 Mpc/h), as
                       [W * sum_i phi_i CIC_i] / [W * sum_i CIC_i] at the halo's cell, both grids built in the
                       same pass and smoothed with the .vahc kernel W (same CIC deconvolution)
  Undefined values are nan: no particle within the sphere, or (grid scales) halos below the .vahc mass cut.
  Sphere sums: vpot_sums.c (OpenMP), particles in chunks; every particle is read once (same pass as the density).
"""
import ctypes
import gc
import os
import subprocess
import time

import numpy as np
import scipy.fft as sfft

from addvalue import AddValue


RHO_CRIT = 2.77519737e11   # (Msun/h) / (Mpc/h)^3, as Rockstar (universal_constants.h)
# spheres for .vpot, in column order: (name, radius in Mpc/h from the halo record and Omega_m)
SPHERES = (('rs', lambda h, Om: h['rs'] / 1000.0),                                            # kpc/h -> Mpc/h
           ('R200b', lambda h, Om: np.cbrt(3.0 * np.asarray(h['m200b'], np.float64) / (4.0 * np.pi * 200.0 * Om * RHO_CRIT))),
           ('R0p5vir', lambda h, Om: 0.5 * h['rvir'] / 1000.0),
           ('Rvir', lambda h, Om: h['rvir'] / 1000.0),
           ('R1p5vir', lambda h, Om: 1.5 * h['rvir'] / 1000.0))


def vpot_lib():
    """vpot_sums.so from $SAHYADRI_LIBDIR (setup_pawna.sh builds it), else built next to this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.environ.get('SAHYADRI_LIBDIR', ''), here):
        f = os.path.join(d, 'vpot_sums.so')
        if d and os.path.isfile(f) and os.path.getmtime(f) >= os.path.getmtime(os.path.join(here, 'vpot_sums.c')):
            break
    else:
        f = os.path.join(here, 'vpot_sums.so')
        subprocess.check_call(['cc', '-O3', '-fopenmp', '-shared', '-fPIC', '-o', f, os.path.join(here, 'vpot_sums.c')])
    lib = ctypes.CDLL(f)
    D = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
    L = np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS')
    lib.deposit_weighted.restype = None
    lib.deposit_weighted.argtypes = [ctypes.c_long, ctypes.c_int, D, D, D, D, D, ctypes.c_double, ctypes.c_double,
                                     ctypes.c_double]
    lib.sphere_sums.restype = ctypes.c_int
    lib.sphere_sums.argtypes = [ctypes.c_long, D, D, D, D, ctypes.c_long, D, D, D, ctypes.c_int, D,
                                ctypes.c_double, ctypes.c_int, ctypes.c_int, D, L]
    return lib


def self_term(pkc):
    """pkdgrav3's softened self potential -(105/64) m/eps in (km/s)^2 at this snapshot (constant for all particles):
    code units m = Omega_m / N (box = 1, rho_crit = 1), eps = softening / L; pot_to_kms2 = (100 L / H0_code)^2 / a."""
    import json
    h = pkc.files[0].header
    info = json.loads(h['info']) if isinstance(h.get('info'), str) else h.get('info', {})
    box = info['box']
    if box.get('softening_source') == 'per particle (input file)':
        raise ValueError('per-particle softening: no single self term')
    m = h['omega_m'] / float(h['n_total'])
    eps = box['softening_kpch'] / 1000.0 / h['box_size']
    return -(105.0 / 64.0) * m / eps * h['pot_to_kms2']


def streamed_fields(pkc, ps, threads, halos=None, hpos=None, Om=None, chunk=1 << 25):
    """CIC density contrast of a .pkc snapshot built block by block (bit for bit as run_addvalue.streamed_density),
    and, with halos given and the potential stored, in the same pass: the CIC grid of phi (self term removed) and
    the sphere sums within the SPHERES radii. Returns (density, n, vpot) with vpot None or a dict."""
    from concurrent.futures import ThreadPoolExecutor
    from ctypes import c_long, c_int, c_double
    from numpy.ctypeslib import ndpointer
    upd = ps.lib_cic.update_density
    upd.restype = None
    upd.argtypes = [c_long, c_int, ndpointer(c_double, flags="C_CONTIGUOUS"),
                    ndpointer(c_double, flags="C_CONTIGUOUS"), ndpointer(c_double, flags="C_CONTIGUOUS"),
                    ndpointer(c_double, flags="C_CONTIGUOUS"), c_double, c_double, c_double, c_double]
    G = ps.grid
    density = np.zeros(G**3)
    want_pot = halos is not None and getattr(pkc, 'has_potential', False)
    if want_pot:
        lib = vpot_lib()
        C = self_term(pkc)
        potgrid = np.zeros(G**3)
        L = float(ps.Lbox)
        nh = halos.size
        hx, hy, hz = (np.ascontiguousarray(hpos[:, i] % L, np.float64) for i in range(3))
        nr = len(SPHERES)
        radii = np.ascontiguousarray(np.stack([f(halos, Om) for _, f in SPHERES], axis=1), np.float64)  # (nh, nr)
        ssum = np.zeros((nh, nr))
        scnt = np.zeros((nh, nr), np.int64)
        nc = int(max(1, min(256, L / 0.5)))
        buf, nbuf = [], 0
        tsph = 0.0

        def flush():
            nonlocal buf, nbuf, tsph
            if not nbuf:
                return
            t = time.time()
            X = np.concatenate([b[0] for b in buf])
            P = np.concatenate([b[1] for b in buf])
            buf, nbuf = [], 0
            rc = lib.sphere_sums(X.shape[0], np.ascontiguousarray(X[:, 0]), np.ascontiguousarray(X[:, 1]),
                                 np.ascontiguousarray(X[:, 2]), P, nh, hx, hy, hz, nr, radii, L, nc,
                                 max(1, threads), ssum, scnt)
            if rc:
                raise MemoryError('sphere_sums: out of memory')
            tsph += time.time() - t

    jobs = [(f, i) for f in pkc.files for i in range(len(f.counts))]

    def decode(job):
        f, i = job
        h = f.header
        if want_pot:
            X, _, _, P = f.block(i, pot=True)
            return ((X + 0.5) / h["q"] * h["box_size"]).astype(np.float32), P * h["pot_step"] - C
        X = f.block(i)[0]
        return ((X + 0.5) / h["q"] * h["box_size"]).astype(np.float32), None

    n = 0
    nchunk = 4 * max(1, threads)
    with ThreadPoolExecutor(max(1, threads)) as ex:
        for k in range(0, len(jobs), nchunk):
            for x, phi in ex.map(decode, jobs[k:k + nchunk]):
                xx = np.ascontiguousarray(x[:, 0], np.float64)
                yy = np.ascontiguousarray(x[:, 1], np.float64)
                zz = np.ascontiguousarray(x[:, 2], np.float64)
                upd(c_long(x.shape[0]), c_int(G), density, xx, yy, zz,
                    c_double(ps.cell_size_x), c_double(ps.cell_size_y), c_double(ps.cell_size_z), c_double(0.0))
                if want_pot:
                    phi = np.ascontiguousarray(phi, np.float64)
                    lib.deposit_weighted(x.shape[0], G, potgrid, xx, yy, zz, phi,
                                         ps.cell_size_x, ps.cell_size_y, ps.cell_size_z)
                    buf.append((x.astype(np.float64), phi))
                    nbuf += x.shape[0]
                    if nbuf >= chunk:
                        flush()
                n += x.shape[0]
    if n != pkc.npart:
        raise ValueError('decoded %d particles, header says %d' % (n, pkc.npart))
    density = np.reshape(density, (G, G, G), order='F')
    density = density * G**3 / (1.0 * n + ps.TINY) - 1.0
    vpot = None
    if want_pot:
        flush()
        with np.errstate(invalid='ignore', divide='ignore'):
            vpot = dict(potgrid=np.reshape(potgrid, (G, G, G), order='F'), nbar=n / float(G)**3, self_term=C,
                        n_sph=scnt, phi_sph=np.where(scnt > 0, ssum / scnt, np.nan), sphere_s=tsph, chunk=chunk)
    return density, n, vpot


class FastAddValue(AddValue):
    """AddValue with the computation of add_value() replaced (see module docstring).

    density_real: CIC density contrast on the grid (as PowerSpectrum.density_field returns; not the FT).
    threads: FFT threads (scipy.fft workers).
    The other arguments are AddValue's (sim_stem, real, snap, kmax, massdef, hpos, halos, ps, grid, CIC, ...).
    """

    def __init__(self, density_real, threads=1, timing=None, vpot=None, **kw):
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
        self.vpot = vpot
        if vpot is not None:   # potential grid: transformed and CIC-deconvolved as the density
            Rp = sfft.rfftn(vpot.pop('potgrid'), workers=self.threads)
            if self.CIC:
                Rp /= (np.sinc(1.0 * a / N) * np.sinc(1.0 * b / N) * np.sinc(1.0 * c / N))**2
            self.Bpot = Rp
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
            inv = 1.0 / (s1**2 + s2**2 + s3**2 + self.ps.TINY)
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
        if 'vpot' in want:
            t = time.time()
            res['vpot'] = self._vpot(allowed, idx, per_scale)
            T['vpot'] = time.time() - t
        del comp
        gc.collect()
        return res

    def _vpot(self, allowed, idx, per_scale):
        """.vpot columns (see module docstring)."""
        if self.vpot is None:
            raise ValueError('vpot requested but no potential fields were built')
        V = self.vpot
        nS = self.N_scales
        mean = np.zeros((allowed.size, nS, 1))
        with np.errstate(invalid='ignore', divide='ignore'):
            for s in range(nS):
                K = self._kernel(self.smoothing_scales[s])
                num = self._irfft(self.Bpot * K)[idx]
                den = V['nbar'] * (1.0 + self._irfft(self.B * K)[idx])
                mean[:, s, 0] = np.where(den > 0, num / den, np.nan)
                del K
        dt = [('ID', 'i8')]
        for name, _ in SPHERES:
            dt += [('Np_' + name, 'i8'), ('phi_' + name, 'f')]
        dt += [('phi_' + s, 'f') for s in self.scale_strings]
        out = np.zeros(self.Nhalo, dtype=dt)
        out['ID'] = self.halos['ID']
        for j, (name, _) in enumerate(SPHERES):
            out['Np_' + name], out['phi_' + name] = V['n_sph'][:, j], V['phi_sph'][:, j]
        for s, v in zip(self.scale_strings, per_scale(mean)):
            col = np.full(self.Nhalo, np.nan)
            col[allowed] = v[:, 0]
            out['phi_' + s] = col
        return out

    # ------------------------------------------------------------------
    def header(self, kind):
        if kind == 'vpot':
            V = self.vpot
            h = ("# Mean pkdgrav3 particle potential around halos, (km/s)^2, self term %.6g (km/s)^2 removed; "
                 "Np_*/phi_* within rs, R200b, 0.5/1/1.5 Rvir (spheres), phi_<scale> Gaussian-weighted on %d^3 grid (vahc_fast.py)\n"
                 % (V['self_term'], self.grid))
            h += '# haloID ' + ''.join('Np_%s phi_%s ' % (nm, nm) for nm, _ in SPHERES) + ' '.join('phi_' + s for s in self.scale_strings) + '\n'
            h += '# self_term_kms2 = %.10g ; R200b from M200b with rho_m = Om * %.9g (Msun/h)/(Mpc/h)^3 ; nan = undefined\n' % (
                V['self_term'], RHO_CRIT)
            return h
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
