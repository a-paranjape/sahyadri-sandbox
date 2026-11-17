#!/usr/bin/env python3
"""Rockstar over several nodes (job_rockstar.pbs, RS_NODES) against one node: halo catalogues of the 128^3 pegasus test
run (Sahyadri_L100N128/default128/r1, 101 outputs) from three Rockstar runs with the same 64 writers, pegasus debug
queue 17 Nov 2026: 1n (1 node x 64 writers, job 37896), 1nb (the same again, job 37897), 2n (2 nodes x 32, job 37895).

usage: make_rs_multinode_figure.py [--d data/rs_multinode_halos.npz] [--figs figs] [--table rs_multinode_table.tex]
  data: Mvir, X, Y, Z, Vmax of out_<snap>.list, snapshots 30, 50, 80, 100 (rs_multinode_halos.npz)
  rs_multinode.pdf        left: N(>Mvir) relative to 1n at z = 0 and snapshot 50; right: share of halos matched within
                          50 kpc/h with the same Mvir (identical, and within 1 %) per mass bin, z = 0
  rs_multinode_table.tex  per snapshot and pair of runs: halo counts, matched share, identical Mvir, |dM/M| > 1 %
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPS = [30, 50, 80, 100]
PAIRS = [('1n', '1nb'), ('1n', '2n'), ('1nb', '2n')]
LAB = {'1n': '1 node', '1nb': '1 node (repeat)', '2n': '2 nodes'}
COL = {'1nb': '#2a78d6', '2n': '#eb6834'}


def match(A, B, L):
    t = cKDTree(B[:, 1:4] % L, boxsize=L)
    d, j = t.query(A[:, 1:4] % L)
    m = d < 0.05
    r = np.full(len(A), np.nan)
    r[m] = B[j[m], 0] / A[m, 0]
    return m, r


def style(ax):
    ax.grid(True, color='#e6e5e0', lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors='#52514e', labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', default=os.path.join(HERE, 'data', 'rs_multinode_halos.npz'))
    ap.add_argument('--figs', default=os.path.join(HERE, 'figs'))
    ap.add_argument('--table', default=os.path.join(HERE, 'rs_multinode_table.tex'))
    A = ap.parse_args()
    D = np.load(A.d); L = float(D['box'])
    H = {k: D[k] for k in D.files if '_' in k and k[0] == '1' or k.startswith('2n')}

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax = axs[0]; style(ax)
    mg = np.logspace(11, 14.3, 25)
    for snap, ls in ((100, '-'), (50, '--')):
        ref = H['1n_%d' % snap][:, 0]
        nr = np.array([(ref > m).sum() for m in mg])
        for run in ('1nb', '2n'):
            n = np.array([(H['%s_%d' % (run, snap)][:, 0] > m).sum() for m in mg])
            ok = nr >= 10
            ax.plot(mg[ok], n[ok] / nr[ok], color=COL[run], ls=ls, lw=1.6,
                    label='%s, snapshot %d' % (LAB[run], snap))
    ax.axhline(1, color='#9b9a95', lw=0.8)
    ax.set_xscale("log"); ax.set_ylim(0.97, 1.03)
    ax.set_xlabel(r'$M_{\rm vir}$ [$h^{-1}M_\odot$]', fontsize=9)
    ax.set_ylabel(r'$N(>M_{\rm vir})$ / that of 1 node', fontsize=9)
    ax.set_title('Cumulative halo counts relative to the first 1-node run (bins with >= 10 halos)', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax = axs[1]; style(ax)
    be = np.logspace(11, 14.5, 8); bc = np.sqrt(be[1:] * be[:-1])
    A0 = H['1n_100']
    for run in ('1nb', '2n'):
        m, r = match(A0, H['%s_100' % run], L)
        idx = np.digitize(A0[:, 0], be) - 1
        same = [np.mean(m[idx == k] & (r[idx == k] == 1)) if (idx == k).sum() >= 10 else np.nan for k in range(len(bc))]
        w1 = [np.mean(m[idx == k] & (np.abs(r[idx == k] - 1) <= 0.01)) if (idx == k).sum() >= 10 else np.nan
              for k in range(len(bc))]
        ax.plot(bc, same, color=COL[run], marker='o', ms=4, lw=1.6, label='%s: identical $M_{\\rm vir}$' % LAB[run])
        ax.plot(bc, w1, color=COL[run], marker='s', ms=4, lw=1.2, ls='--', label='%s: within 1 %%' % LAB[run])
    ax.set_xscale('log'); ax.set_ylim(0.5, 1.02)
    ax.set_xlabel(r'$M_{\rm vir}$ (1-node run) [$h^{-1}M_\odot$]', fontsize=9)
    ax.set_ylabel('share of the 1-node halos', fontsize=9)
    ax.set_title('Halos found again within 50 kpc/h with the same mass, $z=0$', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle('Rockstar on 2 pegasus nodes (2 x 32 writers) vs 1 node (64 writers), 128$^3$ L100 test run', fontsize=10)
    fig.savefig(os.path.join(A.figs, 'rs_multinode.pdf'))

    rows = []
    for snap in SNAPS:
        for a, b in PAIRS:
            X, Y = H['%s_%d' % (a, snap)], H['%s_%d' % (b, snap)]
            m, r = match(X, Y, L)
            rr = r[m]
            rows.append(r'%d & %s vs %s & %d, %d & %.1f & %.1f & %.1f \\' % (
                snap, LAB[a], LAB[b], len(X), len(Y), 100 * m.mean(), 100 * np.mean(rr == 1),
                100 * np.mean(np.abs(rr - 1) > 0.01)))
        rows.append(r'\midrule')
    with open(A.table, 'w') as f:
        f.write('% written by make_rs_multinode_figure.py\n')
        f.write(r'\begin{tabular}{cccccc}' + '\n' + r'\toprule' + '\n')
        f.write(r'snapshot & runs & halos & matched [\%] & identical $M_{\rm vir}$ [\%] & $|\Delta M/M|>1\%$ [\%] \\'
                + '\n' + r'\midrule' + '\n')
        f.write('\n'.join(rows[:-1]) + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n')


if __name__ == '__main__':
    main()
