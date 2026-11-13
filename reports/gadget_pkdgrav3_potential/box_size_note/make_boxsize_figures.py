#!/usr/bin/env python3
"""Linear-theory box-size effect on the potential around halos and on potential differences between halos, compared
with the 256^3 / 200 Mpc/h stacks of the Gadget-4 vs pkdgrav3 comparison. Writes figs/*.pdf and numbers.json for
boxsize_note.tex.

usage: make_boxsize_figures.py [--d ../data/pkd256_vs_gadget256_r1/profiles] [--out .]

All quantities at z = 0, peculiar potential in (km/s)^2, comoving r in Mpc/h, Planck-like cosmology of the runs.
P(k): Eisenstein & Hu (1998) no-wiggle transfer function, n_s, sigma_8 = 0.811 (the ratios below barely depend on it).
Poisson: phi_k = -(3/2) Om H0^2 delta_k / (a k^2), H0 = 100 h km/s/Mpc.
A periodic box of side L keeps only the lattice modes k = (2 pi / L) n, n != 0; an integral over k is replaced by
(1/L^3) sum_n, done exactly up to 60 k_f per axis (or k = 1.5 h/Mpc) and with the continuous integral above that
(the lattice and the integral agree there to << 1%).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson

OM, H, OB, NS, S8 = 0.3137721, 0.6736, 0.0493017, 0.9649, 0.811
PREF = -1.5 * OM * 100.0**2          # Poisson prefactor at a = 1, (km/s)^2 (h/Mpc)^2 ... times delta / k^2
BOXES = [200.0, 500.0, 1000.0, 2000.0]


def transfer(k):
    th = 2.7255 / 2.7; omh2 = OM * H * H; fb = OB / OM
    s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1 + 10 * (OB * H * H)**0.75)
    ag = 1 - 0.328 * np.log(431 * omh2) * fb + 0.38 * np.log(22.3 * omh2) * fb**2
    g = OM * H * (ag + (1 - ag) / (1 + (0.43 * k * H * s)**4))
    q = k * th**2 / g
    l0 = np.log(2 * np.e + 1.8 * q); c0 = 14.2 + 731 / (1 + 62.5 * q)
    return l0 / (l0 + c0 * q * q)


KK = np.logspace(-6, 2, 400000)


def _norm():
    w = lambda x: 3 * (np.sin(x) - x * np.cos(x)) / x**3
    p0 = KK**NS * transfer(KK)**2
    return S8**2 / simpson(KK**2 * p0 * w(8 * KK)**2 / (2 * np.pi**2), x=KK)


AMP = _norm()


def pk(k):
    return AMP * k**NS * transfer(k)**2


def j0(x):
    return np.where(x > 0, np.sin(x) / np.where(x > 0, x, 1.0), 1.0)


def kint(f, L=None):
    """int d^3k/(2pi)^3 g(k) for g = f(k) (vectorised in k; may return arrays over r): infinite volume, or the
    lattice sum of a periodic box of side L (k = 0 excluded)"""
    if L is None:
        return simpson(KK[:, None]**2 * f(KK[:, None]), x=KK, axis=0) / (2 * np.pi**2)
    kf = 2 * np.pi / L
    n = int(min(60, 1.5 / kf))
    i = np.arange(-n, n + 1)
    a, b, c = np.meshgrid(i, i, i, indexing='ij')
    m2 = (a**2 + b**2 + c**2).ravel()
    m2, cnt = np.unique(m2[(m2 > 0) & (m2 <= n * n)], return_counts=True)   # sphere |n| <= n (the tail starts
    # at k_f n), grouped in shells of equal |n|: much faster
    kv = kf * np.sqrt(m2)
    lat = (cnt[:, None] * f(kv[:, None])).sum(0) / L**3
    hi = KK > kf * n
    tail = simpson(KK[hi, None]**2 * f(KK[hi, None]), x=KK[hi], axis=0) / (2 * np.pi**2)
    return lat + tail


def profile(r, L=None):
    """xi_phi,m(r): <phi(x + r) delta_m(x)>, the volume-weighted potential around a halo per unit linear bias"""
    return PREF * kint(lambda k: pk(k) / k**2 * j0(k * r[None, :]), L)


def xi_mm(r, L=None):
    return kint(lambda k: pk(k) * j0(k * r[None, :]), L)


def dphi_pair(r, L=None):
    """xi_phi,m(0) - xi_phi,m(r): <phi_B - phi_A> = (b_B - b_A) x this, for halos A, B at separation r"""
    return PREF * kint(lambda k: pk(k) / k**2 * (1 - j0(k * r[None, :])), L)


def xi_phiphi(r, L=None):
    """<phi(x) phi(x + r)> of the matter potential, (km/s)^4: IR divergent for n_s < 1 in an infinite volume (the
    integrand ~ k^(n_s - 2) at small k), so only box values exist"""
    return PREF**2 * kint(lambda k: pk(k) / k**4 * j0(k * r[None, :]), L)


def sig_dphi(r, L=None):
    """rms of phi(x + r) - phi(x) (matter potential at two points), (km/s)^2"""
    return np.sqrt(2 * PREF**2 * kint(lambda k: pk(k) / k**4 * (1 - j0(k * r[None, :])), L))


def cv_sigma(r, L, b, nh):
    """realisation (cosmic-variance) rms of the stacked volume-weighted profile in one periodic box, Gaussian fields:
    Var = (1/L^3) int d^3k/(2pi)^3 [P_phiphi (b^2 P + 1/n_h) + (b P_phim)^2] j0(kr)^2 (lattice sum over the box modes;
    shot noise of the halo centres included, that of the potential's tracer particles neglected)"""
    f = lambda k: (PREF**2 * pk(k) / k**4 * (b * b * pk(k) + 1 / nh) + (b * PREF * pk(k) / k**2)**2) * j0(k * r[None, :])**2
    return np.sqrt(kint(f, L) / L**3)


def sig_phi_band(k1, k2):
    """rms of phi from the modes k1 < k < k2 (infinite volume): the long-wavelength part a box of side 2 pi / k2
    lacks, as seen by a larger box with k1 = 2 pi / L_big"""
    k = np.logspace(np.log10(k1), np.log10(k2), 20000)
    return np.sqrt(PREF**2 * simpson(pk(k) / k**2, x=k) / (2 * np.pi**2))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', default=os.path.join(here, '..', 'data', 'pkd256_vs_gadget256_r1', 'profiles'))
    ap.add_argument('--out', default=here)
    A = ap.parse_args()
    os.makedirs(os.path.join(A.out, 'figs'), exist_ok=True)

    r = np.concatenate([np.linspace(0.5, 10, 20), np.linspace(11, 120, 110)])
    Ls = [None] + BOXES
    prof = {L: profile(r, L) for L in Ls}
    pair = {L: dphi_pair(r, L) for L in Ls}
    sig = {L: sig_dphi(r, L) for L in Ls}
    xim = {L: xi_mm(r, L) for L in (None, 200.0)}
    tab_r = np.array([1, 5, 10, 15, 20, 30, 45, 70, 100.0])
    num = {'r': tab_r.tolist(), 'profile_inf': profile(tab_r).tolist(), 'pair_inf': dphi_pair(tab_r).tolist(),
           'sig_inf': sig_dphi(tab_r).tolist()}
    for L in BOXES:
        num['profile_ratio_%d' % L] = (profile(tab_r, L) / profile(tab_r)).tolist()
        num['pair_ratio_%d' % L] = (dphi_pair(tab_r, L) / dphi_pair(tab_r)).tolist()
        num['sig_ratio_%d' % L] = (sig_dphi(tab_r, L) / sig_dphi(tab_r)).tolist()
        z = r[np.where(np.diff(np.sign(prof[L])) != 0)[0]]
        num['zero_crossing_%d' % L] = float(z[0]) if len(z) else None
    for L in BOXES + [8000.0]:
        num['xi_phiphi_%d' % L] = xi_phiphi(np.concatenate([[0.0], tab_r]), L).tolist()
    num['sig_phi_long_200_2000'] = float(sig_phi_band(2 * np.pi / 2000, 2 * np.pi / 200))
    num['sig_phi_long_200_1000'] = float(sig_phi_band(2 * np.pi / 1000, 2 * np.pi / 200))
    num['sig_phi_long_1000_10000'] = float(sig_phi_band(2 * np.pi / 10000, 2 * np.pi / 1000))

    # measured volume-weighted stacks (Gadget-4, pkdgrav3), bias from the particle-weighted halo-matter
    # cross-correlation delta_hm against the 200 Mpc/h linear xi_mm, 20 - 40 Mpc/h
    meas = {}
    for code in ('gadget', 'pkd'):
        V = np.load(os.path.join(A.d, '%s_out100_vol.npz' % code))
        Pn = np.load(os.path.join(A.d, '%s_out100.npz' % code))
        rm = V['r_mid']
        sel = (Pn['r_mid'] > 20) & (Pn['r_mid'] < 40)
        xl = xi_mm(Pn['r_mid'][sel], 200.0)
        b = [(Pn['delta_hm'][i][sel] * xl).sum() / (xl * xl).sum() for i in range(len(V['n_halos']))]
        nh = V['n_halos'] / float(V['box'])**3
        meas[code] = dict(r=rm, mean=V['stack_mean'], err=V['stack_err'], b=np.array(b), m=V['m_edges'],
                          pred=np.array([bb * profile(rm, 200.0) for bb in b]),
                          cv=np.array([cv_sigma(rm, 200.0, bb, n) for bb, n in zip(b, nh)]))
    num['bias_gadget'] = meas['gadget']['b'].tolist(); num['bias_pkd'] = meas['pkd']['b'].tolist()
    g = meas['gadget']
    k = (g['r'] > 10) & (g['r'] < 46)
    num['meas_over_pred_10_46'] = [[float(v) for v in (g['mean'][i][k] / g['pred'][i][k])] for i in range(len(g['b']))]
    num['meas_over_pred_r'] = g['r'][k].tolist()
    num['meas_minus_pred_over_cv'] = [[float(v) for v in ((g['mean'][i] - g['pred'][i]) / g['cv'][i])[k]]
                                      for i in range(len(g['b']))]
    num['cv_over_pred'] = [[float(v) for v in (g['cv'][i] / np.abs(g['pred'][i]))[k]] for i in range(len(g['b']))]

    col = {None: 'k', 200.0: '#c8553d', 500.0: '#d49a2a', 1000.0: '#6a9f58', 2000.0: '#1f5fa8'}
    lab = lambda L: 'infinite' if L is None else 'L = %d Mpc/h' % L

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for L in Ls:
        ax[0].plot(r, prof[L] / 1e5, color=col[L], lw=1.4 if L is None else 1, label=lab(L))
        if L is not None:
            ax[1].plot(r, prof[L] / prof[None], color=col[L], label='profile, ' + lab(L))
            ax[1].plot(r, pair[L] / pair[None], color=col[L], ls='--')
            ax[2].plot(r, sig[L] / sig[None], color=col[L], label=lab(L))
    ax[0].axhline(0, color='k', lw=0.4); ax[0].set_xscale('log')
    ax[0].set_ylabel(r'$\xi_{\phi m}(r)$  [$10^5$ (km/s)$^2$ per unit bias]'); ax[0].set_xlabel('r  [Mpc/h]')
    ax[0].set_title('volume-weighted potential around a halo (linear, z = 0)', fontsize=9)
    ax[0].legend(fontsize=8)
    ax[1].axhline(1, color='k', lw=0.4); ax[1].axhline(0, color='k', lw=0.4); ax[1].set_ylim(-0.3, 1.1)
    ax[1].set_xlabel('r  [Mpc/h]'); ax[1].set_ylabel('box / infinite')
    ax[1].set_title(r'solid: profile $\xi_{\phi m}(r)$;  dashed: halo-pair difference $\xi_{\phi m}(0)-\xi_{\phi m}(r)$',
                    fontsize=9)
    ax[1].legend(fontsize=7)
    ax[2].axhline(1, color='k', lw=0.4); ax[2].set_ylim(0, 1.1)
    ax[2].set_xlabel('r  [Mpc/h]'); ax[2].set_ylabel('box / infinite')
    ax[2].set_title(r'rms potential difference between two points, $\sigma_{\Delta\phi}(r)$', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(A.out, 'figs', 'boxsize_theory.pdf')); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    cc = ['#1f5fa8', '#c8553d', '#6a9f58', '#333333']
    for i in range(len(g['b'])):
        lo, hi = g['m'][i], g['m'][i + 1]
        for a in ax:
            a.errorbar(g['r'], g['mean'][i] / 1e5, g['err'][i] / 1e5, fmt='o', ms=2.5, color=cc[i],
                       label=r'%.0e-%.0e, $b$ = %.2f' % (lo, hi, g['b'][i]))
            a.plot(meas['pkd']['r'], meas['pkd']['mean'][i] / 1e5, 'x', ms=3, color=cc[i])
            a.plot(r, g['b'][i] * prof[200.0] / 1e5, '-', color=cc[i], lw=1)
            a.fill_between(g['r'], (g['pred'][i] - g['cv'][i]) / 1e5, (g['pred'][i] + g['cv'][i]) / 1e5,
                           color=cc[i], alpha=0.12, lw=0)
            a.plot(r, g['b'][i] * prof[None] / 1e5, ':', color=cc[i], lw=1)
    for a in ax:
        a.axhline(0, color='k', lw=0.4); a.set_xlabel('r  [Mpc/h]')
        a.set_ylabel(r'$\langle\phi\rangle_V$  [$10^5$ (km/s)$^2$]')
    ax[0].set_xscale('log'); ax[0].set_xlim(0.5, 50); ax[0].set_ylim(-12, 1)
    ax[1].set_xlim(5, 120); ax[1].set_ylim(-3.5, 0.5)
    ax[0].legend(fontsize=7, title=r'$M_{\rm vir}$ [M$_\odot$/h], linear bias', title_fontsize=7)
    ax[0].set_title('256$^3$ / 200 Mpc/h, z = 0: Gadget-4 (o), pkdgrav3 (x); '
                    r'$b\,\xi_{\phi m}$: 200 Mpc/h box (solid; band: realisation rms), infinite (dotted)', fontsize=8)
    ax[1].set_title('same, linear r: the linear prediction applies beyond ~10 Mpc/h', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(A.out, 'figs', 'boxsize_measured.pdf')); plt.close(fig)

    json.dump(num, open(os.path.join(A.out, 'numbers.json'), 'w'), indent=1)
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    for kk, v in num.items():
        print(kk, np.array(v) if isinstance(v, list) else v)


if __name__ == '__main__':
    main()
