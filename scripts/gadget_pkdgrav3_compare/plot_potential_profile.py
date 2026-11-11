#!/usr/bin/env python3
"""Plot stacked potential profiles from halo_potential_profile.py; the first file is the reference.

usage: plot_potential_profile.py <out.png> <a.npz>[:label] <b.npz>[:label] ...
Top row (one panel per mass bin): mean over halos of the shell-mean potential, band = +-1 standard deviation over
halos (dotted: +-1 particle standard deviation of the stacked shell). Bottom row: (mean - mean_ref) / |mean_ref|,
with the standard error of the mean of each curve as a band.
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1]
runs = []
for arg in sys.argv[2:]:
    p, _, lab = arg.partition(':')
    runs.append((np.load(p, allow_pickle=True), lab or p))
ref = runs[0][0]
nb = len(ref['m_edges']) - 1
cols = ['#1f5fa8', '#c8553d', '#2a9d5c', '#7a4fa0']
fig, ax = plt.subplots(2, nb, figsize=(4.2 * nb, 6.4), sharex=True, squeeze=False,
                       gridspec_kw=dict(height_ratios=[2.2, 1]))
xl = 'r / Rvir' if str(ref['r_units']) == 'rvir' else 'r (Mpc/h)'
for b in range(nb):
    a0, a1 = ax[0, b], ax[1, b]
    for (d, lab), c in zip(runs, cols):
        r, m, s = d['r_mid'], d['halo_mean'][b], d['halo_std'][b]
        a0.plot(r, m, color=c, label='%s (%d halos)' % (lab, d['n_halos'][b]))
        a0.fill_between(r, m - s, m + s, color=c, alpha=0.18, lw=0)
        a0.plot(r, d['stack_mean'][b] - d['stack_std'][b], ':', color=c, lw=0.8)
        a0.plot(r, d['stack_mean'][b] + d['stack_std'][b], ':', color=c, lw=0.8)
        with np.errstate(invalid='ignore', divide='ignore'):
            rel = (m - ref['halo_mean'][b]) / np.abs(ref['halo_mean'][b])
            sem = d['halo_sem'][b] / np.abs(ref['halo_mean'][b])
        a1.plot(r, rel, color=c)
        a1.fill_between(r, rel - sem, rel + sem, color=c, alpha=0.18, lw=0)
    a0.set_xscale('log')
    a0.set_title('%.2g <= M < %.2g Msun/h, z = %.2f' % (ref['m_edges'][b], ref['m_edges'][b + 1], float(ref['z'])),
                 fontsize=9)
    a0.legend(fontsize=7)
    a1.axhline(0, color='k', lw=0.6)
    a1.set_xlabel(xl)
ax[0, 0].set_ylabel('<phi>(r) (km/s)$^2$')
ax[1, 0].set_ylabel('relative to %s' % runs[0][1])
fig.tight_layout()
fig.savefig(out, dpi=130)
print('wrote', out)
