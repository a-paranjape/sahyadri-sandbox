#!/usr/bin/env python3
"""CLASS matter power spectra at every planned output redshift of a run, for the P(k) checks (stage vahc).

Writes <halo dir>/pk/ (once; --force to redo):
  class_pk_lin.txt   k [h/Mpc], then P_lin(k, z) [(Mpc/h)^3] for z_start and every output (CLASS, linear)
  class_pk_nl.txt    the same with halofit (CLASS non_linear = halofit)
  class_meta.json    CLASS input, redshifts, the simulation's linear growth D(z)/D(z_start), and the check of CLASS
                     at z_start against the table the initial conditions were made from (cosmology PK_FILE)

Reference for the simulation's large scales: P_IC(k) x [D(z)/D(z_start)]^2, with D the linear growth of matter +
curvature + dark energy (w0, wa) WITHOUT radiation, as pkdgrav3 evolves it (setup_production.py chooses dSigma8 so
that the ICs reproduce CLASS at z_start). CLASS at z itself includes radiation (and the baryon-CDM difference), so
P_CLASS(z = 0) is expected to differ from the simulation by a few per cent on large scales; it is reported only.

CLASS input: as config/transfer/class_template_As.ini (pegasus production): H0, Omega_b, Omega_m, Omega_k,
k_pivot = 0.05/Mpc, A_s, n_s; everything else CLASS's defaults.

usage: class_pk.py <run.conf> [--force]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retention  # noqa: E402

KH = np.logspace(-3, np.log10(30.0), 400)   # h/Mpc


def read_kv(path):
    out = {}
    for line in open(path):
        line = line.split('#', 1)[0].strip()
        if '=' in line:
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip().strip('"')
    return out


def cosmology(E, P):
    """Parameters from the run's cosmology.txt (setup_production.py), else the run config."""
    c = {}
    f = os.path.join(P['cosmo'], 'cosmology.txt')
    if os.path.isfile(f):
        c.update(read_kv(f))
    for k in ('OMEGA_M', 'OMEGA_B', 'OMEGA_K', 'HUBBLE', 'NS', 'AS', 'Z_START', 'W0', 'WA'):
        if k not in c and E.get(k):
            c[k] = E[k]
    c.setdefault('OMEGA_K', '0')
    c.setdefault('W0', '-1')
    c.setdefault('WA', '0')
    return c


def growth(a_out, Om, Ok, w0, wa):
    """Linear growth D(a) (D = a deep in matter domination) for matter + curvature + (w0, wa) dark energy, no
    radiation: D'' + (2 + dlnH/dlna) D' = 1.5 Om(a) D, in ln a."""
    from scipy.integrate import solve_ivp
    Ode = 1.0 - Om - Ok

    def E2(a):
        de = a**(-3 * (1 + w0 + wa)) * np.exp(-3 * wa * (1 - a))
        return Om / a**3 + Ok / a**2 + Ode * de

    def dlnE2(a, h=1e-5):
        return (np.log(E2(a * (1 + h))) - np.log(E2(a * (1 - h)))) / (np.log1p(h) - np.log1p(-h))

    def rhs(lna, y):
        a = np.exp(lna)
        D, dD = y
        return [dD, -(2 + 0.5 * dlnE2(a)) * dD + 1.5 * Om / a**3 / E2(a) * D]
    a0 = 1e-4
    sol = solve_ivp(rhs, (np.log(a0), 0.0), [a0, a0], rtol=1e-10, atol=1e-14, dense_output=True)
    return np.array([sol.sol(np.log(a))[0] for a in np.atleast_1d(a_out)])


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    E = retention.load_conf(sys.argv[1])
    P = retention.paths(E)
    outdir = os.path.join(P['halo'], 'pk')
    meta_f = os.path.join(outdir, 'class_meta.json')
    if os.path.isfile(meta_f) and '--force' not in sys.argv:
        print('exists: %s' % meta_f)
        return 0
    os.makedirs(outdir, exist_ok=True)
    c = cosmology(E, P)
    outs = retention.planned_outputs(E, P)
    zs = [o[3] for o in outs]
    zstart = float(c.get('Z_START', 49))
    h = float(c['HUBBLE'])
    Om, Ok, w0, wa = float(c['OMEGA_M']), float(c['OMEGA_K']), float(c['W0']), float(c['WA'])
    from classy import Class
    par = {'output': 'mPk', 'H0': 100 * h, 'Omega_b': float(c['OMEGA_B']), 'Omega_m': Om, 'Omega_k': Ok,
           'k_pivot': 0.05, 'A_s': float(c['AS']), 'n_s': float(c['NS']), 'P_k_max_h/Mpc': 50.0,
           'z_max_pk': max(zs + [zstart]) + 0.5, 'non_linear': 'halofit'}
    if (w0, wa) != (-1.0, 0.0):
        par.update({'Omega_Lambda': 0.0, 'w0_fld': w0, 'wa_fld': wa})
    t0 = time.time()
    M = Class()
    M.set(par)
    M.compute()
    kM = KH * h
    zall = [zstart] + zs
    lin = np.array([[M.pk_lin(k, z) for k in kM] for z in zall]) * h**3
    nl = np.array([[M.pk(k, z) for k in kM] for z in zall]) * h**3
    tclass = time.time() - t0
    a_all = 1.0 / (1.0 + np.array(zall))
    D = growth(a_all, Om, Ok, w0, wa)
    hdr = 'k[h/Mpc] then P(k,z) [(Mpc/h)^3] at z = ' + ' '.join('%.6f' % z for z in zall) + \
          ' (first: z_start; then the outputs, Rockstar snapshot 0, 1, ...)'
    np.savetxt(os.path.join(outdir, 'class_pk_lin.txt'), np.column_stack([KH, lin.T]), fmt='%.6e',
               header='CLASS linear matter power spectrum; ' + hdr)
    np.savetxt(os.path.join(outdir, 'class_pk_nl.txt'), np.column_stack([KH, nl.T]), fmt='%.6e',
               header='CLASS halofit matter power spectrum; ' + hdr)
    # CLASS at z_start against the IC table (cosmology PK_FILE copied as class_pk_z<zstart>.dat)
    icchk = None
    for f in sorted(os.listdir(P['cosmo'])) if os.path.isdir(P['cosmo']) else []:
        if f.startswith('class_pk_z') and f.endswith('.dat'):
            kk, pp = np.loadtxt(os.path.join(P['cosmo'], f), comments='#', unpack=True)
            sel = (kk >= KH[0]) & (kk <= 10.0)
            r = np.interp(np.log(kk[sel]), np.log(KH), lin[0]) / pp[sel]
            icchk = dict(file=f, k_range=[float(kk[sel][0]), float(kk[sel][-1])], max_abs_dev=float(np.max(np.abs(r - 1))),
                         median_ratio=float(np.median(r)))
    meta = dict(class_input=par, z_start=zstart, redshifts=zs, growth_D=dict(z_start=float(D[0]), outputs=[float(x) for x in D[1:]]),
                growth_note='matter + curvature + w0/wa dark energy, no radiation (as pkdgrav3); D = a at early times',
                ic_check=icchk, seconds=round(tclass, 1), time=time.strftime('%Y-%m-%dT%H:%M:%S'))
    json.dump(meta, open(meta_f + '.part', 'w'), indent=1)
    os.replace(meta_f + '.part', meta_f)
    print('CLASS P(k) at %d redshifts in %.0f s -> %s; IC table check: %s' % (len(zall), tclass, outdir, icchk))
    return 0


if __name__ == '__main__':
    sys.exit(main())
