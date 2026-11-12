#!/usr/bin/env python3
"""Figures of the halo-halo potential cross-correlation (scripts/post-process/halo_potential_xcorr.py).

usage: make_xcorr_figures.py [--full <dir>] [--d data/pkd256_vs_gadget256_r1/xcorr] [--figs figs]
  --full  directory with the tool's outputs {gadget256,pkd256}_out100_xc_{log,lin}.npz (pawna:
          <data>/compare/pkd256_vs_gadget256_r1/xcorr); reduced copies (statistics only, no per-region sums) are
          written to --d, from which the figures are made
  xcorr_256.pdf       phi_Rvir, log bins: mean difference (A != B), sqrt structure function and correlation (A = B);
                      bottom: pkdgrav3 - Gadget-4 over the two jackknife errors in quadrature
  xcorr_lin_256.pdf   the same top row in 0.2 Mpc/h linear bins
  xcorr_cols_256.pdf  Gadget-4, the five phi columns: correlation (0-0, 0-3) and mean difference (0-3)
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

KEEP = ['r_edges', 'r_mid', 'r_mean', 'binning', 'm_edges', 'n_halos', 'pairs', 'phi_cols', 'phi_shift', 'z', 'box',
        'npair', 'frac', 'level', 'target', 'jk', 'mean_diff', 'sf', 'cov', 'corr', 'mean_diff_err', 'sf_err',
        'cov_err', 'corr_err']
MLAB = [r'$2\times10^{12}$-$10^{13}$', r'$10^{13}$-$5\times10^{13}$', r'$5\times10^{13}$-$2\times10^{14}$',
        r'$>2\times10^{14}$']
COL = ['#1f5fa8', '#c8553d', '#6a9f58', '#333333', '#9a6fb0', '#d49a2a']
CLAB = {'phi_Rvir': r'$R_{\rm vir}$', 'phi_rs': r'$r_s$', 'phi_R1p5vir': r'$1.5R_{\rm vir}$',
        'phi_R0p5vir': r'$0.5R_{\rm vir}$', 'phi_R200b': r'$R_{200b}$'}


def reduce(full, d):
    os.makedirs(d, exist_ok=True)
    for c in ('gadget256', 'pkd256'):
        for b in ('log', 'lin'):
            f = os.path.join(full, '%s_out100_xc_%s.npz' % (c, b))
            if os.path.isfile(f):
                X = np.load(f)
                np.savez_compressed(os.path.join(d, '%s_out100_xc_%s.npz' % (c, b)), **{k: X[k] for k in KEEP})


def load(d, b):
    return [np.load(os.path.join(d, '%s_out100_xc_%s.npz' % (c, b))) for c in ('gadget256', 'pkd256')]


def panel_set(G, P, ax, k, logx, bottom=None):
    pairs = [tuple(p) for p in G['pairs']]
    r = G['r_mid']
    cross = [p for p in pairs if p[0] != p[1]]
    auto = [p for p in pairs if p[0] == p[1]]
    for j, (q, sel, scale, lab) in enumerate((
            ('mean_diff', cross, 1e-6, r'$\langle\phi_i-\phi_j\rangle$  [$10^6$ (km/s)$^2$]'),
            ('sf', auto, None, r'$\langle(\phi_i-\phi_j)^2\rangle^{1/2}$  [$10^6$ (km/s)$^2$]'),
            ('corr', auto, 1.0, r'corr$(\phi_i,\phi_j)$'))):
        a = ax[0][j] if bottom is not None else ax[j]
        for n, pr in enumerate(sel):
            p = pairs.index(pr)
            g, pk = G[q][p, k], P[q][p, k]
            ge, pe = G[q + '_err'][p, k], P[q + '_err'][p, k]
            ok = G['npair'][p] >= 20
            if q == 'sf':
                y, yp = np.sqrt(g) * 1e-6, np.sqrt(pk) * 1e-6
                ye = 0.5 * ge / np.sqrt(g) * 1e-6
            else:
                y, yp, ye = g * scale, pk * scale, ge * scale
            c = COL[n % len(COL)]
            lab_ = '%d-%d' % pr
            a.plot(r[ok], y[ok], color=c, label=lab_)
            a.fill_between(r[ok], (y - ye)[ok], (y + ye)[ok], color=c, alpha=0.15, lw=0)
            a.plot(r[ok], yp[ok], 'o', color=c, ms=2.5, mfc='none')
            if bottom is not None:
                with np.errstate(invalid='ignore', divide='ignore'):
                    zz = (pk - g) / np.hypot(ge, pe)
                ax[1][j].plot(r[ok], zz[ok], color=c)
        a.set_ylabel(lab, fontsize=9)
        if logx:
            a.set_xscale('log')
        a.legend(fontsize=7, ncol=2, title='mass bins A-B', title_fontsize=7)
        if bottom is not None:
            b = ax[1][j]
            b.axhline(0, color='k', lw=0.5)
            for v in (-2, 2):
                b.axhline(v, color='k', lw=0.4, ls=':')
            b.set_ylim(-4, 4); b.set_xlabel('r  [Mpc/h]')
            if logx:
                b.set_xscale('log')
            b.set_ylabel('(pkd - Gadget) / err', fontsize=9)
        else:
            a.set_xlabel('r  [Mpc/h]')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--full')
    ap.add_argument('--d', default=os.path.join(here, 'data', 'pkd256_vs_gadget256_r1', 'xcorr'))
    ap.add_argument('--figs', default=os.path.join(here, 'figs'))
    A = ap.parse_args()
    if A.full:
        reduce(A.full, A.d)
    G, P = load(A.d, 'log')
    k = list(G['phi_cols']).index('phi_Rvir')
    fig, ax = plt.subplots(2, 3, figsize=(16, 7), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    panel_set(G, P, ax, k, True, bottom=True)
    fig.suptitle(r'256$^3$, z = 0: halo-halo correlation of $\phi_{R_{\rm vir}}$ (lines and bands: Gadget-4 with '
                 r'jackknife errors; circles: pkdgrav3); A-B = mass bins 0-3 (%s M$_\odot$/h)' % ', '.join(MLAB),
                 fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(A.figs, 'xcorr_256.pdf')); plt.close(fig)

    G, P = load(A.d, 'lin')
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    panel_set(G, P, ax, k, False)
    fig.suptitle(r'As above, 250 linear bins of 0.2 Mpc/h', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(A.figs, 'xcorr_lin_256.pdf')); plt.close(fig)

    G, _ = load(A.d, 'log')
    pairs = [tuple(p) for p in G['pairs']]
    r = G['r_mid']
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for j, (pr, q) in enumerate((((0, 0), 'corr'), ((0, 3), 'corr'), ((0, 3), 'mean_diff'))):
        p = pairs.index(pr)
        ok = G['npair'][p] >= 20
        for n, c in enumerate(G['phi_cols']):
            y = G[q][p, n] * (1e-6 if q == 'mean_diff' else 1)
            ax[j].plot(r[ok], y[ok], color=COL[n], label=CLAB.get(str(c), str(c)))
        ax[j].set_xscale('log'); ax[j].set_xlabel('r  [Mpc/h]')
        ax[j].set_title('pair %d-%d' % pr, fontsize=9)
        ax[j].set_ylabel(r'corr$(\phi_i,\phi_j)$' if q == 'corr' else
                         r'$\langle\phi_i-\phi_j\rangle$  [$10^6$ (km/s)$^2$]', fontsize=9)
    ax[0].legend(fontsize=8, title=r'$\phi$ within', title_fontsize=8)
    fig.suptitle('256$^3$ Gadget-4, z = 0: the five .vpot columns', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(A.figs, 'xcorr_cols_256.pdf')); plt.close(fig)
    print('figures in', A.figs)


if __name__ == '__main__':
    main()
