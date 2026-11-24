#!/usr/bin/env python3
"""Halo correlation multipoles of the three matched-seed 512^3 boxes (L = 200, 500, 1000 Mpc/h) at z = 0, from
scripts/post-process/halo_xi_multipoles.py (pipeline stage ximult, --los x,y,z): even multipoles in real and redshift
space, and the dipole from the gravitational redshift of the halo potential.

usage: make_ximult_box_figures.py [--d data/box512_L200_L500_L1000] [--figs figs] [--table ximult512_table.tex]
  data: L<box>_out100_ximult.npz (pair counts DD and per-region jackknife values removed), copied from
        pawna:/user/shadab/sahyadri_scratch/halos/Sahyadri_L<box>N512/default512/r1/xcorr/out_100_ximult.npz
  ximult512_even.pdf    s^2 xi_0, s^2 xi_2, s^2 xi_4 (rows) for the three upper same-bin pairs (columns); real space
                        solid, redshift space dashed; mean over the lines of sight x, y, z; jackknife bands
  ximult512_dipole.pdf  potential effect on the dipole, xi_1(grav) - xi_1(real) (top row) and
                        xi_1(rsd+grav) - xi_1(rsd) (bottom row), for the cross pairs 1-2, 1-3, 2-3; mean over the lines
                        of sight with jackknife band, each line of sight thin; dotted: mean-shift estimate
                        -(<v_g>_B - <v_g>_A)/(aH) dxi_0^real/ds
  ximult512_table.tex   dipole (grav - real) averaged over 10-40 Mpc/h: measured, per line of sight, mean-shift estimate
Convention (halo_xi_multipoles.py): s = x_B - x_A, mu = s_los/s; a B population shifted by +delta along the line of
sight gives xi_1 = -delta dxi_0/ds; v_g = -phi/c, so halos in deeper potentials have a larger (redshift) v_g.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BOXES = [200, 500, 1000]
COL = {200: '#2a78d6', 500: '#eb6834', 1000: '#1baf7a'}
MK = {200: 'o', 500: 's', 1000: '^'}
MLAB = {0: r'$2\times10^{12}$–$10^{13}$', 1: r'$10^{13}$–$5\times10^{13}$', 2: r'$5\times10^{13}$–$2\times10^{14}$',
        3: r'$2\times10^{14}$–$10^{16}$'}
AUTO = [(1, 1), (2, 2), (3, 3)]
CROSS = [(1, 2), (1, 3), (2, 3)]
LOS_LS = {'x': (0, (1, 1)), 'y': (0, (3, 1)), 'z': (0, (5, 2))}


def load(d):
    """Load ``L<L>_out100_ximult.npz`` of every box into a dict, with ``pairs_l`` as a list of tuples."""
    D = {}
    for L in BOXES:
        X = np.load(os.path.join(d, 'L%d_out100_ximult.npz' % L))
        D[L] = {k: X[k] for k in X.files}
        D[L]['pairs_l'] = [tuple(int(v) for v in p) for p in X['pairs']]
    return D


def style(ax):
    """Common axis style of the report figures."""
    ax.grid(True, color='#e6e5e0', lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#9b9a95')
    ax.tick_params(colors='#52514e', labelsize=8)


def pidx(X, p):
    """Index of mass-bin pair ``p`` in result ``X``."""
    return X['pairs_l'].index(p)


def fig_even(D, out):
    """Figure: monopole, quadrupole and hexadecapole of the auto pairs per box, real and redshift space."""
    ells = [0, 2, 4]
    fig, axs = plt.subplots(3, 3, figsize=(13, 10), sharex=True, constrained_layout=True)
    for j, p in enumerate(AUTO):
        for i, ell in enumerate(ells):
            ax = axs[i, j]; style(ax)
            for L in BOXES:
                X = D[L]; s = X['s_mid']; ip = pidx(X, p); il = list(X['ells']).index(ell)
                for sp, ls in (('real', '-'), ('rsd', '--')):
                    k = list(X['spaces']).index(sp)
                    y = X['xi_ell_mean'][k, ip, il] * s**2; e = X['xi_ell_mean_err'][k, ip, il] * s**2
                    ax.fill_between(s, y - e, y + e, color=COL[L], alpha=0.12, lw=0)
                    ax.plot(s, y, color=COL[L], ls=ls, lw=1.6 if sp == 'real' else 1.3,
                            marker=MK[L] if sp == 'real' else None, ms=3.5, markevery=5,
                            label=(r'$L=%d$ %s' % (L, sp)) if (i == 0 and j == 0) else None)
            if i == 0:
                ax.set_title(r'$M_{\rm vir}$ ' + MLAB[p[0]] + r' $h^{-1}M_\odot$ (same bin)', fontsize=10)
            if j == 0:
                ax.set_ylabel(r'$s^2\xi_%d$ [$h^{-2}$Mpc$^2$]' % ell, fontsize=9)
            if i == 2:
                ax.set_xlabel(r'$s$ [$h^{-1}$Mpc]', fontsize=9)
    axs[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle('Halo correlation multipoles, 512$^3$ boxes, $z=0$: real space (solid) and redshift space (dashed); '
                 'mean over the lines of sight $x,y,z$, band: jackknife (27 sub-cubes)', fontsize=10)
    fig.savefig(out)
    plt.close(fig)


def shift_estimate(X, p, kphi=0):
    """-(<v_g>_B - <v_g>_A)/(aH) dxi_0^real/ds (mean shift of the B population relative to A), and delta"""
    s = X['s_mid']; ip = pidx(X, p); k = list(X['spaces']).index('real'); i0 = list(X['ells']).index(0)
    xi0 = X['xi_ell_mean'][k, ip, i0]
    delta = (X['vg_mean'][kphi, p[1]] - X['vg_mean'][kphi, p[0]]) / float(X['aH'])
    return -delta * np.gradient(xi0, s), delta


REBIN = 4   # dipole figure: 4 adjacent 2 Mpc/h bins averaged (8 Mpc/h); errors combined as independent


def rebin(x, n=REBIN):
    """Average the last axis of ``x`` over groups of ``n`` bins (a remainder is dropped)."""
    m = (x.shape[-1] // n) * n
    return x[..., :m].reshape(x.shape[:-1] + (m // n, n)).mean(-1)


def rebin_err(e, n=REBIN):
    """Error of :func:`rebin` for independent errors ``e``: sqrt(sum e^2) / n."""
    m = (e.shape[-1] // n) * n
    return np.sqrt((e[..., :m].reshape(e.shape[:-1] + (m // n, n))**2).sum(-1)) / n


def fig_dipole(D, out):
    """Figure: the gravitational-redshift dipole (grav - real, rsd+grav - rsd) of the cross pairs per box."""
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, constrained_layout=True)
    for i in range(2):
        for j, p in enumerate(CROSS):
            ax = axs[i, j]; style(ax); ax.axhline(0, color='#9b9a95', lw=0.8)
            for L in BOXES:
                X = D[L]; ip = pidx(X, p); i1 = list(X['ells']).index(1)
                s = rebin(X['s_mid'])
                y = rebin(X['dxi_ell'][i, ip, i1]) * 1e3; e = rebin_err(X['dxi_ell_err'][i, ip, i1]) * 1e3
                off = {200: -0.8, 500: 0.0, 1000: 0.8}[L]
                ax.errorbar(s + off, y, e, color=COL[L], lw=1.6, marker=MK[L], ms=4, capsize=2,
                            label=r'$L=%d$' % L if (i == 0 and j == 0) else None)
                for il, ax_name in enumerate(X['los']):
                    ax.plot(s + off, rebin(X['dxi_ell_los'][i, il, ip, i1]) * 1e3, color=COL[L], lw=0.7, alpha=0.45,
                            ls=LOS_LS[str(ax_name)])
                est, _ = shift_estimate(X, p)
                ax.plot(s, rebin(est) * 1e3, color=COL[L], lw=1.4, ls=':')
            if i == 0:
                ax.set_title(r'pair %d–%d: A = %s, B = %s' % (p[0], p[1], MLAB[p[0]], MLAB[p[1]]), fontsize=9)
            if j == 0:
                ax.set_ylabel((r'$\xi_1^{\rm grav}-\xi_1^{\rm real}$' if i == 0 else
                               r'$\xi_1^{\rm rsd+grav}-\xi_1^{\rm rsd}$') + r' [$10^{-3}$]', fontsize=9)
            if i == 1:
                ax.set_xlabel(r'$s$ [$h^{-1}$Mpc]', fontsize=9)
            ax.set_ylim(-6, 10); ax.set_xlim(0, 100)
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle('Dipole from the gravitational redshift of the halo potential ($\\phi_{r_s}$), 512$^3$ boxes, $z=0$.'
                 '\nPoints: mean over $x,y,z$ in 8 $h^{-1}$Mpc bins (jackknife errors, bins treated as independent); thin: each line '
                 'of sight;\ndotted: mean-shift estimate '
                 r'$-(\langle v_g\rangle_B-\langle v_g\rangle_A)/(aH)\,d\xi_0^{\rm real}/ds$', fontsize=10)
    fig.savefig(out)
    plt.close(fig)


def table(D, out):
    """LaTeX table of the 10-40 Mpc/h dipole averages per cross pair and box."""
    rows = []
    for p in CROSS:
        for L in BOXES:
            X = D[L]; s = X['s_mid']; ip = pidx(X, p); i1 = list(X['ells']).index(1)
            q = (s > 10) & (s < 40)
            y = X['dxi_ell'][0, ip, i1][q]; e = X['dxi_ell_err'][0, ip, i1][q]
            w = 1 / e**2; m = (y * w).sum() / w.sum(); me = w.sum()**-0.5
            per = []
            for il in range(len(X['los'])):
                yl = X['dxi_ell_los'][0, il, ip, i1][q]; el = X['dxi_ell_los_err'][0, il, ip, i1][q]
                wl = 1 / el**2; per.append((yl * wl).sum() / wl.sum())
            est, delta = shift_estimate(X, p)
            em = (est[q] * w).sum() / w.sum()
            rows.append(r'%d–%d & %d & %d, %d & %.2f & $%+.2f \pm %.2f$ & %s & $%+.2f$ \\' % (
                p[0], p[1], L, X['n_halos'][p[0]], X['n_halos'][p[1]], delta * 1e3, m * 1e3, me * 1e3,
                ', '.join('%+.2f' % (v * 1e3) for v in per), em * 1e3))
        rows.append(r'\midrule')
    with open(out, 'w') as f:
        f.write('% written by make_ximult_box_figures.py\n')
        f.write(r'\begin{tabular}{ccccccc}' + '\n' + r'\toprule' + '\n')
        f.write(r'pair & $L$ & $N_A$, $N_B$ & $\Delta\delta$ [$10^{-3}\,h^{-1}$Mpc] & '
                r'$\xi_1^{\rm grav}-\xi_1^{\rm real}$ & $x$, $y$, $z$ & mean-shift est. \\' + '\n' + r'\midrule' + '\n')
        f.write('\n'.join(rows[:-1]) + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n')


def main():
    """Command line: the 512^3 multipole figures and table (``--d``, ``--figs``, ``--table``)."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', default=os.path.join(HERE, 'data', 'box512_L200_L500_L1000'))
    ap.add_argument('--figs', default=os.path.join(HERE, 'figs'))
    ap.add_argument('--table', default=os.path.join(HERE, 'ximult512_table.tex'))
    A = ap.parse_args()
    D = load(A.d)
    fig_even(D, os.path.join(A.figs, 'ximult512_even.pdf'))
    fig_dipole(D, os.path.join(A.figs, 'ximult512_dipole.pdf'))
    table(D, A.table)
    for L in BOXES:
        X = D[L]
        print('L%d: halos per bin %s; <v_g> (phi_rs) km/s %s; aH %.1f' % (
            L, X['n_halos'].tolist(), np.round(X['vg_mean'][0], 2).tolist(), float(X['aH'])))


if __name__ == '__main__':
    main()
