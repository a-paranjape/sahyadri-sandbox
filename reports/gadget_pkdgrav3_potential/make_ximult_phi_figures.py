#!/usr/bin/env python3
"""Gravitational-redshift dipole of the three matched-seed 512^3 boxes (L = 200, 500, 1000 Mpc/h, z = 0) for every
halo potential column of the vahc catalogue (.vpot), from scripts/post-process/halo_xi_multipoles.py with
--phi <12 columns> --los x,y,z (pawna job 2539117, 24 Nov 2026; /user/shadab/ximult_phi/job.pbs).

usage: make_ximult_phi_figures.py [--d data/box512_L200_L500_L1000] [--figs figs] [--table ximult512_phi_table.tex]
  data: L<box>_out100_ximult_phiall.npz, reduced copies (no DD, no jackknife regions, real-space xi of the even
        multipoles only) of the archived
        /storage/shadab/sahyadri_sims/halos/Sahyadri_L<box>N512/default512/r1/xcorr/out_100_ximult_phiall.npz
  figs/ximult512_phi_vg.pdf      mean v_g = -phi/c per mass bin for each potential column, the three boxes
  figs/ximult512_phi_dipole.pdf  dipole averaged over s = 10-40 Mpc/h for each column: grav - real and
                                 rsd+grav - rsd, four cross pairs, with the mean-shift estimate
  figs/ximult512_phi_s.pdf       L1000 pair 1-3: dipole against s for five columns, with their mean-shift estimates
  ximult512_phi_table.tex        the 10-40 Mpc/h dipole of the best-measured pairs for each column
  log-s bins (100 from 0.01 to 100 Mpc/h, --sbins log:0.01,100,100, pawna job 2539125; L<box>_..._phiall_log.npz,
  archived as .../xcorr/out_100_ximult_phiall_log.npz):
  figs/ximult512_phi_logs.pdf    s^2 x dipole difference against log s, five columns, L1000 1-3 and L500 0-2
  ximult512_phi_log_table.tex    dipole (grav - real) averaged over 1-3, 3-10, 10-40 Mpc/h for each column
Averages over 10-40 Mpc/h as make_ximult_box_figures.py: inverse-variance weights of the 2 Mpc/h bins (jackknife
errors, bins treated as independent); the mean-shift estimate -(<v_g>_B - <v_g>_A)/(aH) dxi_0^real/ds averaged with
the same weights.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BOXES = [200, 500, 1000]
COL = {200: '#2a78d6', 500: '#eb6834', 1000: '#1baf7a'}      # as make_ximult_box_figures.py
MK = {200: 'o', 500: 's', 1000: '^'}
MBIN_COL = ['#9ec5ef', '#5b9be0', '#2a6fc0', '#123f80']      # mass bins 0-3: one hue, light (low) to dark (high)
MLAB = [r'$2\times10^{12}$–$10^{13}$', r'$10^{13}$–$5\times10^{13}$', r'$5\times10^{13}$–$2\times10^{14}$',
        r'$2\times10^{14}$–$10^{16}$']
LAB = {'phi_rs': r'$r_s$', 'phi_R0p5vir': r'$0.5R_{\rm vir}$', 'phi_Rvir': r'$R_{\rm vir}$',
       'phi_R1p5vir': r'$1.5R_{\rm vir}$', 'phi_R200b': r'$R_{200b}$', 'phi_R2R200b': r'$2R_{200b}$',
       'phi_R4R200b': r'$4R_{200b}$', 'phi_R6R200b': r'$6R_{200b}$', 'phi_R8R200b': r'$8R_{200b}$',
       'phi_R2Mpch': r'2 Mpc', 'phi_R3Mpch': r'3 Mpc', 'phi_R5Mpch': r'5 Mpc'}
NSPHERE = 5          # the first 5 columns: mean within a sphere; the other 7: Gaussian-smoothed grid
PAIRS = [(0, 3), (1, 2), (1, 3), (2, 3)]
DIFFS = [('grav[%s] - real', r'$\xi_1^{\rm grav}-\xi_1^{\rm real}$'),
         ('rsd+grav[%s] - rsd', r'$\xi_1^{\rm rsd+grav}-\xi_1^{\rm rsd}$')]
S_COLS = ['phi_rs', 'phi_Rvir', 'phi_R4R200b', 'phi_R8R200b', 'phi_R5Mpch']
S_COLORS = ['#123f80', '#2a78d6', '#1baf7a', '#c9a227', '#eb6834']


def load(d, suffix=''):
    D = {}
    for L in BOXES:
        X = np.load(os.path.join(d, 'L%d_out100_ximult_phiall%s.npz' % (L, suffix)))
        D[L] = {k: X[k] for k in X.files}
        D[L]['pairs_l'] = [tuple(int(v) for v in p) for p in X['pairs']]
        D[L]['cols'] = [str(c) for c in X['phi_cols']]
        D[L]['dn'] = [str(c) for c in X['diff_names']]
    return D


def style(ax):
    ax.grid(True, color='#e6e5e0', lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#9b9a95')
    ax.tick_params(colors='#52514e', labelsize=8)


def estimate(X, p, col):
    """mean-shift estimate -(<v_g>_B - <v_g>_A)/(aH) dxi_0^real/ds against s"""
    s = X['s_mid']; ip = X['pairs_l'].index(p); i0 = list(X['ells']).index(0); kc = X['cols'].index(col)
    delta = (X['vg_mean'][kc, p[1]] - X['vg_mean'][kc, p[0]]) / float(X['aH'])
    return -delta * np.gradient(X['xi_ell_mean_real'][ip, i0], s)


def dipole(X, p, col, kd):
    """dipole difference (kd: 0 grav - real, 1 rsd+grav - rsd) against s, its error"""
    ip = X['pairs_l'].index(p); i1 = list(X['ells']).index(1); k = X['dn'].index(DIFFS[kd][0] % col)
    return X['dxi_ell'][k, ip, i1], X['dxi_ell_err'][k, ip, i1]


def avg(X, p, col, kd, smin=10, smax=40):
    """(mean, error, mean-shift estimate) over smin < s < smax, inverse-variance weights"""
    s = X['s_mid']; q = (s > smin) & (s < smax)
    y, e = dipole(X, p, col, kd)
    w = 1 / e[q]**2
    return (y[q] * w).sum() / w.sum(), w.sum()**-0.5, (estimate(X, p, col)[q] * w).sum() / w.sum()


def skip(L, p):
    return L == 200 and 3 in p      # L200's top bin: 76 halos, jackknife errors unreliable


def sep(ax):
    ax.axvline(NSPHERE - 0.5, color='#9b9a95', lw=0.8, ls='--')


def fig_vg(D, out):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True, constrained_layout=True)
    for ax, L in zip(axs, BOXES):
        X = D[L]; style(ax); sep(ax); x = np.arange(len(X['cols']))
        for b in range(4):
            ax.plot(x, X['vg_mean'][:, b], color=MBIN_COL[b], marker='o', ms=4, lw=1.6,
                    label=MLAB[b] + r' ($N=%d$)' % X['n_halos'][b])
        ax.set_xticks(x); ax.set_xticklabels([LAB[c] for c in X['cols']], rotation=55, fontsize=8)
        ax.set_title(r'$L = %d\,h^{-1}$Mpc' % L, fontsize=10)
        ax.legend(fontsize=7, frameon=False, title=r'$M_{\rm vir}$ [$h^{-1}M_\odot$]', title_fontsize=7)
    axs[0].set_ylabel(r'$\langle v_g\rangle = -\langle\phi\rangle/c$ [km s$^{-1}$]', fontsize=9)
    fig.suptitle('Mean gravitational-redshift velocity per mass bin for each potential column (left of the dashed '
                 'line: mean within a sphere; right: Gaussian-smoothed grid), 512$^3$ boxes, $z = 0$', fontsize=10)
    fig.savefig(out)
    plt.close(fig)


def fig_dipole(D, out):
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.2), sharex=True, constrained_layout=True)
    cols = D[1000]['cols']; x = np.arange(len(cols))
    for kd in range(2):
        for j, p in enumerate(PAIRS):
            ax = axs[kd, j]; style(ax); sep(ax); ax.axhline(0, color='#9b9a95', lw=0.8)
            for L in BOXES:
                if skip(L, p):
                    continue
                r = np.array([avg(D[L], p, c, kd) for c in cols]) * 1e3
                off = {200: -0.2, 500: 0.0, 1000: 0.2}[L]
                ax.errorbar(x + off, r[:, 0], r[:, 1], color=COL[L], marker=MK[L], ms=4.5, lw=0, elinewidth=1.2,
                            capsize=2, )
                ax.plot(x + off, r[:, 2], color=COL[L], lw=1.0, ls=':', marker=MK[L], ms=4, mfc='none',
                        )
            if kd == 0:
                ax.set_title('pair %d–%d' % p, fontsize=10)
            if j == 0:
                ax.set_ylabel(DIFFS[kd][1] + r', $s=10$–$40\,h^{-1}$Mpc [$10^{-3}$]', fontsize=9)
            ax.set_xticks(x); ax.set_xticklabels([LAB[c] for c in cols], rotation=55, fontsize=8)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL[L], marker=MK[L], lw=0, ms=5, label=r'$L=%d$' % L) for L in BOXES]
    h += [Line2D([], [], color='#52514e', marker='o', mfc='none', ls=':', lw=1, ms=5, label='mean-shift estimate')]
    axs[0, 1].legend(handles=h, fontsize=7, frameon=False, loc='lower left')
    fig.suptitle('Gravitational-redshift dipole for each potential column, mean over the lines of sight $x,y,z$ '
                 'averaged over 10–40 $h^{-1}$Mpc (filled, jackknife errors);\nopen/dotted: mean-shift estimate. '
                 'L200 is left out of the pairs with its top bin (76 halos).', fontsize=10)
    fig.savefig(out)
    plt.close(fig)


REBIN = 4


def rebin(x, n=REBIN):
    m = (x.shape[-1] // n) * n
    return x[..., :m].reshape(x.shape[:-1] + (m // n, n)).mean(-1)


def rebin_err(e, n=REBIN):
    m = (e.shape[-1] // n) * n
    return np.sqrt((e[..., :m].reshape(e.shape[:-1] + (m // n, n))**2).sum(-1)) / n


def fig_s(D, out):
    X = D[1000]; p = (1, 3)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, constrained_layout=True)
    s = rebin(X['s_mid'])
    for kd, ax in enumerate(axs):
        style(ax); ax.axhline(0, color='#9b9a95', lw=0.8)
        for k, (c, cc) in enumerate(zip(S_COLS, S_COLORS)):
            y, e = dipole(X, p, c, kd)
            off = (k - 2) * 0.5
            ax.errorbar(s + off, rebin(y) * 1e3, rebin_err(e) * 1e3, color=cc, marker='o', ms=3.5, lw=1.4, capsize=2,
                        label=LAB[c])
            ax.plot(s, rebin(estimate(X, p, c)) * 1e3, color=cc, lw=1.2, ls=':')
        ax.set_xlim(8, 100); ax.set_ylim(-1.5, 9)   # below ~8 Mpc/h halo exclusion dominates
        ax.set_xlabel(r'$s$ [$h^{-1}$Mpc]', fontsize=9)
        ax.set_title(DIFFS[kd][1], fontsize=10)
    axs[0].set_ylabel(r'dipole [$10^{-3}$]', fontsize=9)
    axs[0].legend(fontsize=8, frameon=False, title='potential within / smoothed at', title_fontsize=8)
    fig.suptitle(r'$L = 1000\,h^{-1}$Mpc, pair 1–3 ($10^{13}$–$5\times10^{13}$ with $2\times10^{14}$–$10^{16}$): '
                 'mean over $x,y,z$ in 8 $h^{-1}$Mpc bins (errors: bins treated as independent); dotted: mean-shift '
                 'estimate', fontsize=10)
    fig.savefig(out)
    plt.close(fig)


LOG_PANELS = [(1000, (1, 3), 0), (1000, (1, 3), 1), (500, (0, 2), 0)]
LOG_YLIM = [(-7, 7), (-3, 7), (-1.2, 1.5)]
LOG_REBIN = 2          # 2 log bins (0.08 dex) per point
EST_SMIN = 3.0         # the mean-shift estimate needs a smooth xi_0: not across the exclusion edge (~1-2 Mpc/h)


def wavg(y, e, m):
    """inverse-variance mean of y[m] over bins with pairs (e > 0); (nan, nan) if none"""
    ok = m & (e > 0) & np.isfinite(y)
    if not ok.any():
        return np.nan, np.nan
    w = 1 / e[ok]**2
    return (y[ok] * w).sum() / w.sum(), w.sum()**-0.5


def log_rebin(s, y, e, n=LOG_REBIN):
    """groups of n log bins: geometric-mean s, inverse-variance mean of the bins with pairs"""
    m = (len(s) // n) * n
    S, Y, E = [], [], []
    for i in range(0, m, n):
        sl = slice(i, i + n)
        a, b = wavg(y[sl], e[sl], np.ones(n, bool))
        S.append(np.exp(np.log(s[sl]).mean())); Y.append(a); E.append(b)
    return np.array(S), np.array(Y), np.array(E)


def fig_logs(D, out):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, (L, p, kd), yl in zip(axs, LOG_PANELS, LOG_YLIM):
        X = D[L]; s = X['s_mid']; style(ax); ax.set_ylim(*yl); ax.axhline(0, color='#9b9a95', lw=0.8)
        for k, (c, cc) in enumerate(zip(S_COLS, S_COLORS)):
            y, e = dipole(X, p, c, kd)
            S, Y, E = log_rebin(s, y, e)
            ok = np.isfinite(Y)
            fac = 10**((k - 2) * 0.008)        # small horizontal offsets
            ax.errorbar(S[ok] * fac, S[ok]**2 * Y[ok], S[ok]**2 * E[ok], color=cc, marker='o', ms=3, lw=1.2,
                        capsize=1.5, label=LAB[c])
            est = estimate(X, p, c); q = s > EST_SMIN
            ax.plot(s[q], s[q]**2 * est[q], color=cc, lw=1.1, ls=':')
        ax.set_xscale('log'); ax.set_xlim(1, 100)
        ax.set_xlabel(r'$s$ [$h^{-1}$Mpc]', fontsize=9)
        ax.set_title(r'$L=%d$, pair %d–%d: %s' % (L, p[0], p[1], DIFFS[kd][1]), fontsize=10)
    axs[0].set_ylabel(r'$s^2\times$ dipole difference [$h^{-2}$Mpc$^2$]', fontsize=9)
    axs[0].legend(fontsize=8, frameon=False, title='potential within / smoothed at', title_fontsize=8)
    fig.suptitle(r'Log-$s$ bins (100 from 0.01 to 100 $h^{-1}$Mpc, shown in pairs of bins from 1 $h^{-1}$Mpc; no '
                 r'halo pairs below 0.29 $h^{-1}$Mpc): mean over $x,y,z$, jackknife errors; dotted: mean-shift estimate '
                 r'for $s>3\,h^{-1}$Mpc', fontsize=10)
    fig.savefig(out)
    plt.close(fig)


LOG_RANGES = [(1, 3), (3, 10), (10, 40)]


def table_log(D, out):
    """dipole (grav - real) averaged over 1-3, 3-10 and 10-40 Mpc/h in the log bins, L1000 1-3 and L500 0-2"""
    rows = []
    cols = D[1000]['cols']
    for kc, c in enumerate(cols):
        cells = []
        for L, p in ((1000, (1, 3)), (500, (0, 2))):
            X = D[L]; s = X['s_mid']; y, e = dipole(X, p, c, 0)
            dv = X['vg_mean'][kc, p[1]] - X['vg_mean'][kc, p[0]]
            cells.append('%.1f' % dv)
            for a, b in LOG_RANGES:
                m, err = wavg(y, e, (s > a) & (s < b))
                cells.append(r'$%+.1f\pm%.1f$' % (m * 1e3, err * 1e3))
        rows.append(r'%s & %s \\' % (LAB[c], ' & '.join(cells)))
        if kc == NSPHERE - 1:
            rows.append(r'\midrule')
    with open(out, 'w') as f:
        f.write('% written by make_ximult_phi_figures.py (log-s bins)\n')
        f.write(r'\begin{tabular}{lcccccccc}' + '\n' + r'\toprule' + '\n')
        f.write(r' & \multicolumn{4}{c}{L1000, pair 1–3} & \multicolumn{4}{c}{L500, pair 0–2} \\' + '\n')
        f.write(r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}' + '\n')
        f.write(r'potential & $\Delta v_g$ & 1–3 & 3–10 & 10–40 & $\Delta v_g$ & 1–3 & 3–10 & 10–40 \\' + '\n')
        f.write(r'\midrule' + '\n' + '\n'.join(rows) + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n')


def table(D, out):
    rows = []
    X = D[1000]; cols = X['cols']
    for c in cols:
        kc = cols.index(c)
        dv = X['vg_mean'][kc, 3] - X['vg_mean'][kc, 1]
        cells = []
        for (L, p, kd) in ((1000, (1, 3), 0), (1000, (2, 3), 0), (500, (0, 3), 0), (1000, (1, 3), 1)):
            m, e, est = avg(D[L], p, c, kd)
            cells.append(r'$%+.2f\pm%.2f$ (%+.2f)' % (m * 1e3, e * 1e3, est * 1e3))
        m, e, est = avg(X, (1, 3), c, 0)
        rows.append(r'%s & %.1f & %s & %.2f \\' % (LAB[c], dv, ' & '.join(cells), m / est))
        if kc == NSPHERE - 1:
            rows.append(r'\midrule')
    with open(out, 'w') as f:
        f.write('% written by make_ximult_phi_figures.py\n')
        f.write(r'\begin{tabular}{lcccccc}' + '\n' + r'\toprule' + '\n')
        f.write(r'potential & $\Delta v_g$ & \multicolumn{3}{c}{$\xi_1^{\rm grav}-\xi_1^{\rm real}$} & '
                r'$\xi_1^{\rm rsd+grav}-\xi_1^{\rm rsd}$ & ratio \\' + '\n')
        f.write(r' & km\,s$^{-1}$ & L1000, 1–3 & L1000, 2–3 & L500, 0–3 & L1000, 1–3 & L1000, 1–3 \\' + '\n')
        f.write(r'\midrule' + '\n' + '\n'.join(rows) + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', default=os.path.join(HERE, 'data', 'box512_L200_L500_L1000'))
    ap.add_argument('--figs', default=os.path.join(HERE, 'figs'))
    ap.add_argument('--table', default=os.path.join(HERE, 'ximult512_phi_table.tex'))
    A = ap.parse_args()
    D = load(A.d)
    fig_vg(D, os.path.join(A.figs, 'ximult512_phi_vg.pdf'))
    fig_dipole(D, os.path.join(A.figs, 'ximult512_phi_dipole.pdf'))
    fig_s(D, os.path.join(A.figs, 'ximult512_phi_s.pdf'))
    table(D, A.table)
    DL = load(A.d, '_log')
    fig_logs(DL, os.path.join(A.figs, 'ximult512_phi_logs.pdf'))
    table_log(DL, os.path.join(HERE, 'ximult512_phi_log_table.tex'))
    X = D[1000]
    for kd in range(2):
        print(DIFFS[kd][0] % '*', 'L1000 1-3 measured/estimate:',
              ' '.join('%s %.2f' % (c[4:], avg(X, (1, 3), c, kd)[0] / avg(X, (1, 3), c, kd)[2]) for c in X['cols']))


if __name__ == '__main__':
    main()
