#!/usr/bin/env python3
"""Figures from compare_potential.py output.  usage: plot_compare.py <cmp_dir> [<fig_dir>]"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

d = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, 'figs')
os.makedirs(out, exist_ok=True)


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] not in ('', 'None') else np.nan for r in rows]) for k in rows[0]}


P = load(os.path.join(d, 'particles.csv'))
H = load(os.path.join(d, 'halos.csv')) if os.path.exists(os.path.join(d, 'halos.csv')) else None
C = dict(g='#1f5fa8', p='#c8553d', k='#333333')

# 1. particle statistics against redshift
fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
x = 1 + P['z']
ax[0, 0].plot(x, P['rms_diff'] / P['phi_gadget_rms'], color=C['k'], label='pkdgrav3 - Gadget-4')
ax[0, 0].plot(x, P['fft_pkd_vs_fft_gadget_rms'] / P['phi_gadget_rms'], '--', color=C['k'],
              label='FFT(pkd positions) - FFT(Gadget positions)')
ax[0, 0].set_ylabel('rms difference / rms phi (Gadget)'); ax[0, 0].legend(fontsize=8)
ax[0, 1].plot(x, P['offset_median'], color=C['k'], label='median offset (self-term removed)')
ax[0, 1].plot(x, P['pkd_selfpot'], ':', color=C['p'], label='pkdgrav3 self-term (removed)')
ax[0, 1].set_ylabel('(km/s)$^2$'); ax[0, 1].legend(fontsize=8); ax[0, 1].set_yscale('symlog', linthresh=1e2)
ax[1, 0].plot(x, 1 - P['corr'], color=C['k'], label='pkd vs Gadget')
ax[1, 0].plot(x, 1 - P['gadget_vs_fft_corr'], color=C['g'], label='Gadget vs FFT')
ax[1, 0].plot(x, 1 - P['pkd_vs_fft_corr'], color=C['p'], label='pkdgrav3 vs FFT')
ax[1, 0].set_yscale('log'); ax[1, 0].set_ylabel('1 - correlation'); ax[1, 0].legend(fontsize=8)
ax[1, 1].plot(x, P['dr_median'], color=C['k'], label='median')
ax[1, 1].plot(x, P['dr_p90'], '--', color=C['k'], label='90%')
ax[1, 1].set_yscale('log'); ax[1, 1].set_ylabel('|x_pkd - x_Gadget| (Mpc/h)'); ax[1, 1].legend(fontsize=8)
for a in ax[1]:
    a.set_xscale('log'); a.set_xlabel('1 + z')
fig.tight_layout(); fig.savefig(os.path.join(out, 'particles_vs_z.png'), dpi=130); plt.close(fig)

# 2. particle scatter at the plot outputs
for f in sorted(glob.glob(os.path.join(d, 'snap_*.npz'))):
    s = np.load(f)
    z = float(s['z'])
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    lo, hi = np.percentile(np.r_[s['phig'], s['phip']], [0.05, 100])
    ax[0].hexbin(s['phig'], s['phip'], gridsize=120, bins='log', extent=(lo, hi, lo, hi), cmap='Greys')
    ax[0].plot([lo, hi], [lo, hi], color=C['p'], lw=0.8)
    ax[0].set_xlabel('phi Gadget-4 (km/s)$^2$'); ax[0].set_ylabel('phi pkdgrav3, aligned (km/s)$^2$')
    dphi = s['phip'] - s['phig']
    ax[1].hexbin(s['phig'], dphi, gridsize=120, bins='log', cmap='Greys')
    ax[1].axhline(0, color=C['p'], lw=0.8)
    ax[1].set_xlabel('phi Gadget-4 (km/s)$^2$'); ax[1].set_ylabel('pkdgrav3 - Gadget-4 (km/s)$^2$')
    ax[2].hexbin(np.log10(np.maximum(s['dr'], 1e-5)), dphi, gridsize=100, bins='log', cmap='Greys')
    ax[2].set_xlabel('log10 |x_pkd - x_Gadget| (Mpc/h)'); ax[2].set_ylabel('pkdgrav3 - Gadget-4 (km/s)$^2$')
    fig.suptitle('z = %.3f, particles (self-term removed, offset %.3g aligned)' % (z, float(s['offset'])))
    fig.tight_layout(); fig.savefig(os.path.join(out, 'particles_z%.2f.png' % z), dpi=130); plt.close(fig)

# 3. halos
if H is not None:
    zs = np.unique(H['z'])
    pick = sorted({zs[np.argmin(abs(zs - zp))] for zp in (3, 1, 0.5, 0)})
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    for z, col in zip(pick, plt.cm.viridis(np.linspace(0, 0.9, len(pick)))):
        m = H['z'] == z
        ax[0].loglog(H['mvir_gadget'][m], H['mvir_pkd'][m] / H['mvir_gadget'][m], 'o', ms=3, color=col, label='z=%.2f' % z)
        ax[1].semilogx(H['mvir_gadget'][m], H['phimin_pkd'][m] / H['phimin_gadget'][m] - 1, 'o', ms=3, color=col)
        ax[2].semilogx(H['mvir_gadget'][m], H['phimean_pkd'][m] / H['phimean_gadget'][m] - 1, 'o', ms=3, color=col)
    ax[0].set_ylabel('Mvir pkd / Mvir Gadget'); ax[0].legend(fontsize=8)
    ax[1].set_ylabel('phi_min pkd / Gadget - 1'); ax[2].set_ylabel('<phi>(<Rvir) pkd / Gadget - 1')
    for a in ax:
        a.set_xlabel('Mvir Gadget (Msun/h)'); a.axhline(0 if a is not ax[0] else 1, color='k', lw=0.6)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'halos.png'), dpi=130); plt.close(fig)
print('figures in', out)
