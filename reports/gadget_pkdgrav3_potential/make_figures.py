#!/usr/bin/env python3
"""Figures of the report other than the stacked potentials (those: stack_potential.ipynb).

usage: make_figures.py <run>=<compare dir> [<run>=<compare dir> ...] [--figs figs]
  compare dir: output of job_compare.pbs (particles.csv, snap_*.npz, halos/halo_mf.csv, halos/halo_match*.{csv,npz})
  e.g. make_figures.py 32=data/pkd32_vs_gadget32_r1 256=../../tmp.nosync/cmp256
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

C = dict(g='#1f5fa8', p='#c8553d', k='#333333')
RUNC = {'32': '#9a9a9a', '256': '#1b1b1b'}


def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    return {k: np.array([num(r[k]) for r in rows]) for k in rows[0]}, rows


def particles_vs_z(R, out):
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for run, d in R.items():
        P = d['P']
        x = 1 + P['z']
        c = RUNC.get(run, 'k')
        lab = run + '$^3$'
        ax[0, 0].plot(x, P['rms_diff'] / P['phi_gadget_rms'], color=c, label=lab + ' pkdgrav3 $-$ Gadget-4')
        ax[0, 0].plot(x, P['fft_pkd_vs_fft_gadget_rms'] / P['phi_gadget_rms'], '--', color=c,
                      label=lab + ' FFT(pkd x) $-$ FFT(Gadget x)')
        ax[0, 1].plot(x, 1 - P['corr'], color=c, label=lab + ' pkd vs Gadget')
        ax[0, 1].plot(x, 1 - P['gadget_vs_fft_corr'], ':', color=c, label=lab + ' Gadget vs FFT')
        ax[0, 2].plot(x, np.abs(P['offset_median']), color=c, label=lab + ' |zero-point offset|')
        ax[0, 2].plot(x, -P['pkd_selfpot'], ':', color=c, label=lab + ' pkdgrav3 self-term (removed)')
        ax[1, 0].plot(x, P['dr_median'], color=c, label=lab + ' median')
        ax[1, 0].plot(x, P['dr_p90'], '--', color=c, label=lab + ' 90%')
        if 'dv_median' in P:
            ax[1, 1].plot(x, P['dv_median'], color=c, label=lab + ' median')
            ax[1, 1].plot(x, P['dv_p90'], '--', color=c, label=lab + ' 90%')
            ax[1, 1].plot(x, P['dv_rms'], ':', color=c, label=lab + ' rms')
            ax[1, 2].plot(x, 1 - P['v_corr'], color=c, label=lab + ' 1 $-$ corr(v)')
            ax[1, 2].plot(x, np.abs(P['v_rms_pkd'] / P['v_rms_gadget'] - 1), '--', color=c,
                          label=lab + r' |v$_{\rm rms}$ ratio $-$ 1|')
    ax[0, 0].set_ylabel(r'rms($\Delta\phi$) / rms($\phi_{\rm Gadget}$)')
    ax[0, 1].set_ylabel('1 $-$ correlation of $\\phi$')
    ax[0, 2].set_ylabel('(km/s)$^2$')
    ax[1, 0].set_ylabel(r'$|x_{\rm pkd} - x_{\rm Gadget}|$  [Mpc/h]')
    ax[1, 1].set_ylabel(r'$|v_{\rm pkd} - v_{\rm Gadget}|$  [km/s]')
    ax[1, 2].set_ylabel('velocity statistics')
    for a in ax.ravel():
        a.set_yscale('log'); a.set_xscale('log'); a.legend(fontsize=7)
    for a in ax[1]:
        a.set_xlabel('1 + z')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'particles_vs_z.pdf')); plt.close(fig)


def snap_scatter(run, d, out):
    fl = sorted(glob.glob(os.path.join(d, 'snap_*.npz')))
    if not fl:
        return
    s = np.load(fl[-1])
    z = float(s['z'])
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    lo, hi = np.percentile(np.r_[s['phig'], s['phip']], [0.05, 100])
    ax[0].hexbin(s['phig'], s['phip'], gridsize=120, bins='log', extent=(lo, hi, lo, hi), cmap='Greys')
    ax[0].plot([lo, hi], [lo, hi], color=C['p'], lw=0.8)
    ax[0].set_xlabel(r'$\phi$ Gadget-4  [(km/s)$^2$]'); ax[0].set_ylabel(r'$\phi$ pkdgrav3 (aligned)  [(km/s)$^2$]')
    dphi = s['phip'] - s['phig']
    ax[1].hexbin(np.log10(np.maximum(s['dr'], 1e-5)), dphi, gridsize=100, bins='log', cmap='Greys')
    ax[1].set_xlabel(r'log$_{10}$ $|\Delta x|$  [Mpc/h]'); ax[1].set_ylabel(r'$\Delta\phi$  [(km/s)$^2$]')
    if 'vg' in s:
        vg, vp = s['vg'][:, 0], s['vp'][:, 0]
        m = np.percentile(np.abs(np.r_[vg, vp]), 99.9)
        ax[2].hexbin(vg, vp, gridsize=120, bins='log', extent=(-m, m, -m, m), cmap='Greys')
        ax[2].plot([-m, m], [-m, m], color=C['p'], lw=0.8)
        ax[2].set_xlabel('$v_x$ Gadget-4  [km/s]'); ax[2].set_ylabel('$v_x$ pkdgrav3  [km/s]')
        ax[3].hist(np.log10(np.maximum(s['dv'], 1e-2)), 100, histtype='step', color=C['k'], label=r'$|\Delta v|$ [km/s]')
        ax[3].hist(np.log10(np.maximum(s['dr'], 1e-5)) + 3, 100, histtype='step', color=C['p'],
                   label=r'$|\Delta x|$ [kpc/h]')
        ax[3].set_xlabel('log$_{10}$'); ax[3].set_ylabel('particles (subsample)'); ax[3].legend(fontsize=8)
    fig.suptitle('%s$^3$, z = %.2f: particle by particle (random subsample of %d)' % (run, z, len(dphi)))
    fig.tight_layout(); fig.savefig(os.path.join(out, 'particles_%s_z0.pdf' % run)); plt.close(fig)


def halo_mf(run, d, out):
    p = os.path.join(d, 'halos', 'halo_mf.csv')
    if not os.path.exists(p):
        return
    H, _ = load_csv(p)
    _, rows = load_csv(p)
    q = np.array([r['quantity'] for r in rows])
    zs = np.unique(H['z'][np.isfinite(H['z'])])
    pick = sorted({zs[np.argmin(abs(zs - zp))] for zp in (3, 1, 0.5, 0)})
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex='col', gridspec_kw=dict(height_ratios=[2, 1]))
    for j, (qq, xl) in enumerate((('mvir', r'$M_{\rm vir}$  [M$_\odot$/h]'), ('vmax', r'$V_{\rm max}$  [km/s]'))):
        for z, col in zip(pick, plt.cm.viridis(np.linspace(0, 0.85, len(pick)))):
            m = (q == qq) & (H['z'] == z) & (H['n_gadget'] > 0)
            x = np.sqrt(H['lo'][m] * H['hi'][m])
            ax[0, j].errorbar(x, H['dn_gadget'][m], H['err_gadget'][m], color=col, label='z = %.2f Gadget-4' % z)
            ax[0, j].plot(x, H['dn_pkd'][m], 'o', mfc='none', color=col, label='pkdgrav3')
            ax[1, j].errorbar(x * (1 + 0.03 * pick.index(z)), H['ratio'][m], H['ratio_err'][m], color=col, fmt='o-', ms=3)
        ax[0, j].set_xscale('log'); ax[0, j].set_yscale('log')
        ax[0, j].set_ylabel(r'dn/dlog$_{10}$ (host halos)  [h$^3$ Mpc$^{-3}$]')
        ax[1, j].axhline(1, color='k', lw=0.6); ax[1, j].set_ylabel('pkdgrav3 / Gadget-4'); ax[1, j].set_xlabel(xl)
        ax[1, j].set_ylim(0.5, 1.5)
    ax[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle('%s$^3$: host-halo mass and V$_{\\rm max}$ functions (consistent-trees)' % run)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'halo_mf_%s.pdf' % run)); plt.close(fig)


def halo_match(run, d, out):
    fl = sorted(glob.glob(os.path.join(d, 'halos', 'halo_match_out*.npz')), key=lambda f: int(f.split('out')[-1][:-4]))
    if not fl:
        return
    S = [dict(np.load(f)) for f in fl]
    S = [s for s in S if len(s.get('mvir_gadget', [])) > 0]
    if not S:
        return
    zs = np.array([float(s['z']) for s in S])
    pick = sorted({int(np.argmin(abs(zs - zp))) for zp in (3, 1, 0.5, 0)})
    qs = [('mvir', r'$M_{\rm vir}$'), ('vmax', r'$V_{\rm max}$'), ('conc', r'$c = R_{\rm vir}/r_s$'),
          ('Spin_Bullock', r'$\lambda_{\rm B}$'), ('c_to_a', 'c/a'), ('b1', r'$b_1$ (VAHC)')]
    fig, ax = plt.subplots(2, 4, figsize=(18, 7.5))
    ax = ax.ravel()
    me = np.logspace(11.5, 15.5, 17)
    for k, col in zip(pick, plt.cm.viridis(np.linspace(0, 0.85, len(pick)))):
        s = S[k]
        mg = s['mvir_gadget']
        allm = s['mvir_all_gadget']
        nall = np.histogram(allm, me)[0]
        nmat = np.histogram(mg, me)[0]
        x = np.sqrt(me[1:] * me[:-1])
        ok = nall > 0
        ax[0].plot(x[ok], nmat[ok] / nall[ok], 'o-', ms=3, color=col, label='z = %.2f' % zs[k])
        for j, (q, lab) in enumerate(qs):
            a = ax[j + 1]
            if q + '_gadget' not in s:
                continue
            yg, yp = s[q + '_gadget'], s[q + '_pkd']
            y = (yp - yg) if q == 'b1' else yp / yg
            idx = np.digitize(mg, me) - 1
            med, lo_, hi_ = (np.full(len(x), np.nan) for _ in range(3))
            for b in range(len(x)):
                v = y[(idx == b) & np.isfinite(y)]
                if len(v) >= (5 if len(mg) > 100 else 1):
                    med[b], lo_[b], hi_[b] = np.percentile(v, [50, 16, 84])
            a.plot(x, med, color=col)
            a.fill_between(x, lo_, hi_, color=col, alpha=0.15, lw=0)
            a.set_ylabel(lab + (r' pkd $-$ Gadget' if q == 'b1' else ' pkd / Gadget'))
        ax[7].hist(np.log10(np.maximum(s['d_over_rvir'], 1e-4)), 50, histtype='step', color=col,
                   label='z = %.2f' % zs[k])
    ax[0].set_ylabel('matched fraction (Gadget-4 hosts)'); ax[0].legend(fontsize=7); ax[0].set_ylim(0, 1.05)
    for a in ax[:7]:
        a.set_xscale('log'); a.set_xlabel(r'$M_{\rm vir}$ Gadget-4  [M$_\odot$/h]')
        a.set_xlim(me[0], me[-1])
        a.axhline(0 if 'pkd $-$' in a.get_ylabel() else (1 if a is not ax[0] else 1), color='k', lw=0.5)
    ax[7].set_xlabel(r'log$_{10}$ centre offset / $R_{\rm vir}$'); ax[7].legend(fontsize=7)
    fig.suptitle('%s$^3$: matched host halos (median and 16-84%% range per mass bin)' % run)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'halo_match_%s.pdf' % run)); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--figs', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs'))
    A = ap.parse_args()
    os.makedirs(A.figs, exist_ok=True)
    R = {}
    for r in A.runs:
        run, d = r.split('=', 1)
        R[run] = dict(d=d, P=load_csv(os.path.join(d, 'particles.csv'))[0])
    particles_vs_z(R, A.figs)
    for run, v in R.items():
        snap_scatter(run, v['d'], A.figs)
        halo_mf(run, v['d'], A.figs)
        halo_match(run, v['d'], A.figs)
    print('figures in', A.figs)


if __name__ == '__main__':
    main()
