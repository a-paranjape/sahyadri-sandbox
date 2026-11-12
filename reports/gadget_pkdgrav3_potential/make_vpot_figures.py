#!/usr/bin/env python3
"""Figures of the per-halo pipeline potential (.vpot columns of out_<snap>_vahc.fits.gz), from vpot_match.csv of
scripts/gadget_pkdgrav3_compare/compare_vpot.py.

usage: make_vpot_figures.py [--d data/pkd256_vs_gadget256_r1] [--figs figs]
  vpot_ratio_256.pdf  median and 16-84% of phi_pkd / phi_gadget - 1 against radius, mass bins, z = 3, 1, 0.5, 0
  vpot_vs_z_256.pdf   median (ratio - 1) against z for four radii; delta phi at 5 Mpc/h against z with the
                      particle mean and median of phi_pkd - phi_gadget (particles.csv)
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RADII = ['rs', 'R0p5vir', 'R200b', 'Rvir', 'R1p5vir', 'R2R200b', 'R4R200b', 'R6R200b', 'R8R200b',
         'R2Mpch', 'R3Mpch', 'R5Mpch']
LAB = {'rs': '$r_s$', 'R0p5vir': r'$0.5R_{\rm vir}$', 'R200b': '$R_{200b}$', 'Rvir': r'$R_{\rm vir}$',
       'R1p5vir': r'$1.5R_{\rm vir}$', 'R2R200b': '$2R_{200b}$', 'R4R200b': '$4R_{200b}$', 'R6R200b': '$6R_{200b}$',
       'R8R200b': '$8R_{200b}$', 'R2Mpch': '2 Mpc/h', 'R3Mpch': '3 Mpc/h', 'R5Mpch': '5 Mpc/h'}
MB = {1e11: ('#9a9a9a', r'$10^{11}$-$10^{12}$'), 1e12: ('#1f5fa8', r'$10^{12}$-$10^{13}$'),
      1e13: ('#c8553d', r'$10^{13}$-$10^{14}$'), 1e14: ('#333333', r'$>10^{14}$')}


def ratio_vs_radius(V, out):
    zs = np.unique(V['z'])
    pick = [zs[np.argmin(abs(zs - zp))] for zp in (3, 1, 0.5, 0)]
    x = np.arange(len(RADII))
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for a, z in zip(ax, pick):
        v = V[V['z'] == z]
        for k, (col, lab) in MB.items():
            s = v[v['lo'] == k].set_index('radius').reindex(RADII)
            if s['n'].max() != s['n'].max() or s['n'].max() < 20:
                continue
            a.plot(x, 100 * (s['ratio_med'] - 1), 'o-', color=col, ms=3, label='%s (%d)' % (lab, s['n'].max()))
            a.fill_between(x, 100 * (s['ratio_p16'] - 1), 100 * (s['ratio_p84'] - 1), color=col, alpha=0.12, lw=0)
        a.axhline(0, color='k', lw=0.5)
        a.axvline(4.5, color='k', lw=0.4, ls=':')
        a.set_xticks(x); a.set_xticklabels([LAB[r] for r in RADII], rotation=60, fontsize=8)
        a.set_title('z = %.2f' % z, fontsize=10)
    ax[0].set_ylabel(r'$\phi_{\rm pkd}/\phi_{\rm Gadget} - 1$  [%]'); ax[0].set_ylim(-6, 14)
    ax[0].legend(fontsize=7, title=r'$M_{\rm vir}$ Gadget-4 [M$_\odot$/h] (N)', title_fontsize=7)
    fig.suptitle(r'256$^3$: pipeline .vpot mean particle potential of matched host halos (left of the dotted line: '
                 r'spheres; right: Gaussian-weighted on the VAHC grid); median, 16-84%', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'vpot_ratio_256.pdf')); plt.close(fig)


def vs_z(V, P, out):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, k in zip(ax[:2], (1e12, 1e13)):
        for r, col in zip(('rs', 'Rvir', 'R4R200b', 'R5Mpch'), ('#c8553d', '#1f5fa8', '#6a9f58', '#333333')):
            s = V[(V['lo'] == k) & (V['radius'] == r) & (V['n'] >= 20)].sort_values('z')
            a.plot(s['z'], 100 * (s['ratio_med'] - 1), color=col, label=LAB[r])
        a.axhline(0, color='k', lw=0.5); a.set_xlabel('z'); a.set_xlim(0, 4)
        a.set_title(r'$M_{\rm vir}$ Gadget-4 %s M$_\odot$/h' % MB[k][1], fontsize=10)
    ax[0].set_ylabel(r'median $\phi_{\rm pkd}/\phi_{\rm Gadget} - 1$  [%]'); ax[0].legend(fontsize=8)
    for k, (col, lab) in MB.items():
        s = V[(V['lo'] == k) & (V['radius'] == 'R5Mpch') & (V['n'] >= 20)].sort_values('z')
        ax[2].plot(s['z'], s['diff_med'] / 1e3, color=col, label='halos ' + lab)
    P = P.sort_values('z')
    ax[2].plot(P['z'], (P['phi_pkd_corr_mean'] - P['phi_gadget_mean']) / 1e3, 'k--', label='all particles, mean')
    ax[2].plot(P['z'], P['offset_median'] / 1e3, 'k:', label='all particles, median')
    ax[2].axhline(0, color='k', lw=0.5); ax[2].set_xlabel('z'); ax[2].set_xlim(0, 4)
    ax[2].set_ylabel(r'$\phi_{\rm pkd} - \phi_{\rm Gadget}$  [$10^3$ (km/s)$^2$]')
    ax[2].set_title('5 Mpc/h (median over halos) and all particles', fontsize=10); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'vpot_vs_z_256.pdf')); plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', default=os.path.join(here, 'data', 'pkd256_vs_gadget256_r1'))
    ap.add_argument('--figs', default=os.path.join(here, 'figs'))
    A = ap.parse_args()
    V = pd.read_csv(os.path.join(A.d, 'vpot', 'vpot_match.csv'))
    P = pd.read_csv(os.path.join(A.d, 'particles.csv'))
    P = P[P['z'] < 5]
    ratio_vs_radius(V, A.figs)
    vs_z(V, P, A.figs)
    print('figures in', A.figs)


if __name__ == '__main__':
    main()
