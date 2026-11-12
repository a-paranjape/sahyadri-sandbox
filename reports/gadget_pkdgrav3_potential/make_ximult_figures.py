#!/usr/bin/env python3
"""Figures of the halo-halo correlation multipoles (scripts/post-process/halo_xi_multipoles.py).

usage: make_ximult_figures.py [--full <dir>] [--d data/pkd256_vs_gadget256_r1/ximult] [--figs figs]
  --full  directory with {gadget256,pkd256}_out100_ximult.npz (lines of sight x, y, z) and pkd256_out100_amp50.npz
          (--pot-amp 50 --spaces real,grav --los x,y,z); reduced copies (no pair counts) are written to --d
  ximult_256.pdf         s^2 xi_l (l = 0, 1, 2, 4) for pairs 0-0, 0-2, 0-3 in the four spaces (Gadget-4 lines with
                         jackknife bands, pkdgrav3 circles), mean over the three lines of sight
  ximult_dipole_256.pdf  the potential's dipole, s^2 [xi_1(rsd+grav) - xi_1(rsd)], for the cross pairs (both codes);
                         and the x50 potential test against -<delta_B - delta_A> dxi_0/ds
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

KEEP = ['s_edges', 's_mid', 'mu_edges', 'ells', 'spaces', 'los', 'pairs', 'm_edges', 'n_halos', 'n_a', 'n_b',
        'phi_cols', 'phi_fallback', 'pot_amp', 'vg_mean', 'z', 'box', 'aH', 'jk', 'xi_ell', 'xi_ell_err', 'xi_ell_mean',
        'xi_ell_mean_err', 'xi_ell_jk', 'diff_names', 'dxi_ell', 'dxi_ell_err']
SPC = {'real': ('#9a9a9a', '-'), 'rsd': ('#1f5fa8', '-'), 'grav': ('#6a9f58', '--'), 'rsd+grav': ('#c8553d', ':')}
COL = ['#1f5fa8', '#c8553d', '#6a9f58', '#333333', '#9a6fb0', '#d49a2a']


def rebin(X, k_space, ip, il, width, minus=None):
    """xi_l of space k_space (minus space `minus`) in wider s bins: volume-weighted mean (exact for DD/RR - 1 in a
    periodic box) and its jackknife error from xi_ell_jk; returns s_mid, value, error"""
    e = X['s_edges']; w0 = e[1] - e[0]; m = max(1, int(round(width / w0)))
    n = (len(e) - 1) // m
    vol = (e[1:]**3 - e[:-1]**3)[:n * m].reshape(n, m)
    def comb(y):  # (..., s) -> (..., n)
        return (y[..., :n * m].reshape(y.shape[:-1] + (n, m)) * vol).sum(-1) / vol.sum(-1)
    y = X['xi_ell_mean'][k_space, ip, il]; jk = X['xi_ell_jk'][k_space, ip, :, il]
    if minus is not None:
        y = y - X['xi_ell_mean'][minus, ip, il]; jk = jk - X['xi_ell_jk'][minus, ip, :, il]
    yb, jb = comb(y), comb(jk)
    nreg = jb.shape[0]
    err = np.sqrt((nreg - 1) / nreg * ((jb - jb.mean(0))**2).sum(0))
    return 0.5 * (e[0:n * m:m] + e[m:n * m + 1:m]), yb, err


def reduce(full, d):
    os.makedirs(d, exist_ok=True)
    for f in ('gadget256_out100_ximult.npz', 'pkd256_out100_ximult.npz', 'pkd256_out100_amp50.npz'):
        if os.path.isfile(os.path.join(full, f)):
            X = np.load(os.path.join(full, f))
            np.savez_compressed(os.path.join(d, f), **{k: (X[k].astype(np.float32) if k == 'xi_ell_jk' else X[k])
                                                      for k in KEEP if k in X.files})


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--full')
    ap.add_argument('--d', default=os.path.join(here, 'data', 'pkd256_vs_gadget256_r1', 'ximult'))
    ap.add_argument('--figs', default=os.path.join(here, 'figs'))
    A = ap.parse_args()
    if A.full:
        reduce(A.full, A.d)
    G = np.load(os.path.join(A.d, 'gadget256_out100_ximult.npz'))
    P = np.load(os.path.join(A.d, 'pkd256_out100_ximult.npz'))
    s = G['s_mid']; s2 = s**2; sp = list(G['spaces']); pairs = [tuple(p) for p in G['pairs']]
    ells = list(G['ells'])
    show_l = [0, 1, 2, 4]
    rows = [(0, 0), (0, 2), (0, 3)]
    fig, ax = plt.subplots(len(rows), len(show_l), figsize=(17, 10), sharex=True)
    for i, pr in enumerate(rows):
        ip = pairs.index(pr)
        for j, l in enumerate(show_l):
            a = ax[i][j]; il = ells.index(l)
            for name in sp:
                c, ls = SPC.get(name, ('k', '-'))
                k = sp.index(name)
                y, e = G['xi_ell_mean'][k, ip, il] * s2, G['xi_ell_mean_err'][k, ip, il] * s2
                a.plot(s, y, color=c, ls=ls, label=name)
                if name in ('real', 'rsd') and not (pr[0] == pr[1] and l % 2):  # odd l of A = A: 0 exactly
                    a.fill_between(s, y - e, y + e, color=c, alpha=0.15, lw=0)
                if name in ('real', 'rsd'):
                    a.plot(s, P['xi_ell_mean'][k, ip, il] * s2, 'o', color=c, ms=2.2, mfc='none')
            a.axhline(0, color='k', lw=0.4)
            if pr[0] == pr[1] and l % 2:                 # identically 0 (both orderings): no rounding-noise axis
                a.set_ylim(-1, 1); a.text(0.5, 0.6, '= 0 (by symmetry)', transform=a.transAxes, ha='center', fontsize=9)
            a.set_title('pair %d-%d, $\\ell = %d$' % (pr + (l,)), fontsize=9)
            if i == len(rows) - 1:
                a.set_xlabel('s  [Mpc/h]')
            if j == 0:
                a.set_ylabel(r'$s^2 \xi_\ell$  [(Mpc/h)$^2$]')
    ax[0][0].legend(fontsize=7, title='Gadget-4 (lines); pkdgrav3: circles', title_fontsize=7)
    fig.suptitle(r'256$^3$, z = 0: halo cross-correlation multipoles, mean over lines of sight x, y, z; '
                 r'mass bins 0: $2\times10^{12}$-$10^{13}$, 2: $5\times10^{13}$-$2\times10^{14}$, '
                 r'3: $>2\times10^{14}$ M$_\odot$/h; potential shift from $\phi_{r_s}$', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(A.figs, 'ximult_256.pdf')); plt.close(fig)

    # potential dipole: rsd+grav - rsd, 10 Mpc/h bins
    cross = [p for p in pairs if p[0] != p[1]]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    for n, pr in enumerate(cross):
        ip = pairs.index(pr)
        for a, X, lab in ((ax[0], G, 'Gadget-4'), (ax[1], P, 'pkdgrav3')):
            sb, y, e = rebin(X, sp.index('rsd+grav'), ip, ells.index(1), 10.0, minus=sp.index('rsd'))
            a.errorbar(sb + 0.8 * (n - 2.5), y * sb**2, e * sb**2, fmt='o-', ms=3, lw=0.8, color=COL[n],
                       label='%d-%d' % pr)
            a.set_title(lab + r': $s^2[\xi_1({\rm rsd+grav}) - \xi_1({\rm rsd})]$, physical $\phi_{r_s}$, 10 Mpc/h bins',
                        fontsize=9)
    for a in ax[:2]:
        a.axhline(0, color='k', lw=0.5); a.set_xlabel('s  [Mpc/h]'); a.set_ylabel(r'$s^2\Delta\xi_1$  [(Mpc/h)$^2$]')
    ax[0].legend(fontsize=7, ncol=2, title='mass bins A-B', title_fontsize=7)
    f50 = os.path.join(A.d, 'pkd256_out100_amp50.npz')
    if os.path.isfile(f50):
        X = np.load(f50); p50 = [tuple(p) for p in X['pairs']]; sp50 = list(X['spaces'])
        for n, pr in enumerate(p50):
            ip = p50.index(pr)
            sb, y, e = rebin(X, sp50.index('grav'), ip, 1, 4.0, minus=sp50.index('real'))
            ax[2].errorbar(sb, y * sb**2, e * sb**2, fmt='o', ms=3, color=COL[n], label='%d-%d measured' % pr)
            _, x0, _ = rebin(X, sp50.index('real'), ip, 0, 4.0)
            dd = (X['vg_mean'][0][pr[1]] - X['vg_mean'][0][pr[0]]) / float(X['aH'])
            ax[2].plot(sb, -dd * np.gradient(x0, sb) * sb**2, '-', color=COL[n], lw=1,
                       label=r'%d-%d: $-(\langle\delta_B\rangle-\langle\delta_A\rangle)\,d\xi_0/ds$' % pr)
        ax[2].axhline(0, color='k', lw=0.5); ax[2].set_xlabel('s  [Mpc/h]'); ax[2].set_xlim(0, 60)
        ax[2].legend(fontsize=7, ncol=2)
        ax[2].set_title(r'pkdgrav3, potential shift $\times 50$: $s^2[\xi_1({\rm grav}) - \xi_1({\rm real})]$, 4 Mpc/h bins',
                        fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(A.figs, 'ximult_dipole_256.pdf')); plt.close(fig)
    print('figures in', A.figs)


if __name__ == '__main__':
    main()
