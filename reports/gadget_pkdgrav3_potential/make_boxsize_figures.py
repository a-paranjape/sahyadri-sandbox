#!/usr/bin/env python3
"""Finite-box effect on the halo potential statistics: the three matched-seed 512^3 production boxes
(L = 200, 500, 1000 Mpc/h, iSeed 12345678), top three mass bins x z = 0, 0.2, 0.5, compared with a linear-theory
estimate of the shift caused by the periodic box. Derivations: boxsize.tex.

usage: make_boxsize_figures.py [--d data/box512_L200_L500_L1000] [--figs figs] [--table boxsize_table.tex]
  data: L<box>_out<snap>_vprof_volume.npz (halo_potential_profile.py, volume weighting, pipeline stage vprof) and
        L<box>_out<snap>_xcorr_log.npz (halo_potential_xcorr.py, stage xcorr; per-region sums removed), copied from
        pawna:/user/shadab/sahyadri_scratch/halos/Sahyadri_L<box>N512/default512/r1/{vprof,xcorr}/
  boxsize_vprof.pdf      <phi>(r): measured (solid, jackknife band) and measured minus the shift (dashed)
  boxsize_vprof_dphi.pdf <phi>_L - <phi>_L1000 measured, and the predicted difference of the shifts
                         (Fourier mode sum, dashed; real-space integral-constraint estimate, dotted)
  boxsize_xcorr.pdf      structure function <(phi_i - phi_j)^2> of phi_Rvir, auto pairs, measured and corrected
  boxsize_xcorr_dsf.pdf  sf_L - sf_L1000 measured and predicted (Fourier, dashed; real-space bulk pull, dotted)
  boxsize_table.tex      shifts per box, bin and z (Fourier and real-space) and the measured box differences

Shift model: linear theory, P(k) = the CLASS P(k) of the ICs at z = 49 (scripts/pkdgrav3_pipeline/production/
cosmologies/default_class_pk_z49.dat) times the LCDM growth factor squared; halo bias from Tinker et al. (2010) at
the bin's mean mass (Delta = Bryan & Norman virial relative to the mean density); potential = peculiar, physical,
(km/s)^2, phi_k = -C delta_k / k^2 with C = 1.5 Om (100 km/s)^2 (1 + z) and k in h/Mpc.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import quad  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKFILE = os.path.join(HERE, '..', '..', 'scripts', 'pkdgrav3_pipeline', 'production', 'cosmologies',
                      'default_class_pk_z49.dat')
OM = 0.3137721
BOXES = [200, 500, 1000]
SNAPS = [(100, 0.0), (82, 0.2), (64, 0.5)]
BINS = [1, 2, 3]
COL = {200: '#2a78d6', 500: '#eb6834', 1000: '#1baf7a'}
MK = {200: 'o', 500: 's', 1000: '^'}
NMAX = 48                   # mode sums and integrals cut at |k| < NMAX k_f (beyond, sum = integral)
MLAB = {1: r'$10^{13}$–$5\times10^{13}$', 2: r'$5\times10^{13}$–$2\times10^{14}$', 3: r'$2\times10^{14}$–$10^{16}$'}

kt, pt = np.loadtxt(PKFILE, unpack=True)
lk, lp = np.log(kt), np.log(pt)


def growth(a):
    """Linear growth factor D(a) of flat LCDM (Heath integral; D = a during matter domination)."""
    E = lambda x: np.sqrt(OM / x**3 + 1 - OM)  # noqa: E731
    return 2.5 * OM * E(a) * quad(lambda x: 1 / (x * E(x))**3, 0, a)[0]


D49 = growth(1 / 50.)


def pk(k, z):
    """Linear P(k) at redshift ``z``: the tabulated z = 49 spectrum scaled by (D(z) / D(49))^2."""
    return np.exp(np.interp(np.log(k), lk, lp)) * (growth(1 / (1 + z)) / D49)**2


def cpot(z):
    """4 pi G rho_m a^2 for the physical peculiar potential, (km/s)^2 per (Mpc/h)^2"""
    return 1.5 * OM * 1e4 * (1 + z)


def sigma(M, z):
    """rms linear density contrast in a top-hat sphere of mass ``M`` (Msun/h) at redshift ``z``."""
    R = (3 * M / (4 * np.pi * 2.775e11 * OM))**(1 / 3)
    k = np.geomspace(1e-4, 1e2, 4000)
    x = k * R
    W = 3 * (np.sin(x) - x * np.cos(x)) / x**3
    return np.sqrt(np.trapz(k**3 * pk(k, z) * W**2 / (2 * np.pi**2), np.log(k)))


def tinker_bias(M, z):
    """Linear halo bias of Tinker et al. (2010) for mass ``M`` (Msun/h) at redshift ``z``, with the halo
    overdensity Delta = Delta_vir(z) / Omega_m(z) relative to the mean density (Bryan & Norman Delta_vir).
    """
    a = 1 / (1 + z)
    Om_z = OM / a**3 / (OM / a**3 + 1 - OM)
    x = Om_z - 1
    y = np.log10((18 * np.pi**2 + 82 * x - 39 * x**2) / Om_z)
    A = 1 + 0.24 * y * np.exp(-(4 / y)**4); aa = 0.44 * y - 0.88
    B, b, c = 0.183, 1.5, 2.4
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y)**4)
    nu = 1.686 / sigma(M, z)
    return 1 - A * nu**aa / (nu**aa + 1.686**aa) + B * nu**b + C * nu**c


def j0(x):
    """Spherical Bessel function j0(x) = sin(x) / x."""
    return np.sinc(x / np.pi)


def fourier_shift(L, z, r, power):
    """[box mode sum - infinite integral] of int d^3k/(2pi)^3 P(k) k^-power j0(kr), at the radii r"""
    kf = 2 * np.pi / L
    n = np.arange(-NMAX, NMAX + 1)
    nx, ny, nz = np.meshgrid(n, n, n, indexing='ij')
    nn = np.sqrt(nx**2 + ny**2 + nz**2).ravel()
    nn = nn[(nn > 0) & (nn < NMAX)]
    nu, cnt = np.unique(np.round(nn, 6), return_counts=True)
    k = nu * kf
    box = (cnt * pk(k, z) * k**-power * j0(np.outer(r, k))).sum(1) / L**3
    kk = np.geomspace(1e-5, NMAX * kf, 20000)
    inf = np.trapz(kk**3 * pk(kk, z) * kk**-power / (2 * np.pi**2) * j0(np.outer(r, kk)), np.log(kk), axis=1)
    return box - inf


_XI = {}


def xi_lin(z):
    """linear xi(s) on s = 0.01..3000 Mpc/h"""
    if z not in _XI:
        k = np.geomspace(1e-5, 50, 100000)
        P = pk(k, z) * np.exp(-(k / 10.)**2)
        sg = np.linspace(0.01, 3000, 3000)
        xg = np.array([np.trapz(k**3 * P * j0(k * s) / (2 * np.pi**2), np.log(k)) for s in sg])
        s = np.linspace(0.01, 3000, 300000)
        _XI[z] = (s, np.interp(s, sg, xg))
    return _XI[z]


def realspace_vprof(L, z):
    """real-space estimate of the mean-potential shift (b = 1): shell theorem + integral constraint
    shift = C [ xbar R_L^2 / 2 + int_{R_L}^inf xi(s) s ds ],  xbar = (3/R_L^3) int_0^R_L xi s^2 ds"""
    s, xi = xi_lin(z)
    RL = (3 / (4 * np.pi))**(1 / 3) * L
    m = s < RL
    xbar = 3 * np.trapz(xi[m] * s[m]**2, s[m]) / RL**3
    return cpot(z) * (xbar * RL**2 / 2 + np.trapz(xi[~m] * s[~m], s[~m])), xbar


def bulk_pull(L, z):
    """<|g_bar|^2> of a sphere with the box volume in an infinite universe: the mean acceleration of the region,
    which (Newton's third law) is the pull of the matter outside it; the periodic box forces it to zero.
    <g_bar^2> = C^2 int d^3k/(2pi)^3 P(k) W^2(kR) / k^2. Missing structure function ~ <g_bar^2> r^2 / 3."""
    R = (3 / (4 * np.pi))**(1 / 3) * L
    k = np.geomspace(1e-5, 10, 20000)
    x = k * R
    W = 3 * (np.sin(x) - x * np.cos(x)) / x**3
    return cpot(z)**2 * np.trapz(k**3 * pk(k, z) * W**2 / k**2 / (2 * np.pi**2), np.log(k))


def vbulk(g, z):
    """linear theory: peculiar velocity from the peculiar acceleration, v = f a H g / C (km/s)"""
    a = 1 / (1 + z)
    E2 = OM / a**3 + 1 - OM
    f = (OM / a**3 / E2)**0.55
    return f * a * 100 * np.sqrt(E2) * g / cpot(z)


def style(ax):
    """Common axis style of the report figures (white background, light grid, no top/right spines)."""
    ax.set_facecolor('white')
    ax.grid(True, color='#e6e5e0', lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#9b9a95')
    ax.tick_params(colors='#52514e', labelsize=8)
    ax.set_xscale('log')


def load_all(d):
    """Load the stacked potential (vprof) and potential cross-correlation (xcorr) results of every box and
    snapshot from ``d``; returns a dict keyed by (L, z, mass bin) with r, values, errors and predictions.
    """
    out = {}
    for L in BOXES:
        for s, z in SNAPS:
            v = np.load(os.path.join(d, 'L%d_out%d_vprof_volume.npz' % (L, s)))
            x = np.load(os.path.join(d, 'L%d_out%d_xcorr_log.npz' % (L, s)))
            for b in BINS:
                bias = tinker_bias(v['m_mean'][b], z)
                rv = v['r_mid']
                e = dict(z=z, bias=bias, rv=rv, phi=v['stack_mean'][b], phi_err=v['stack_err'][b],
                         nh_v=int(v['n_halos'][b]))
                e['dphi'] = -bias * cpot(z) * fourier_shift(L, z, rv, 2)          # predicted shift of <phi>
                rs_shift, xbar = realspace_vprof(L, z)
                e['dphi_rs'] = bias * rs_shift; e['xbar'] = xbar
                p = [q for q, (A, B) in enumerate(x['pairs']) if A == b and B == b][0]
                k = list(x['phi_cols']).index('phi_Rvir')
                ok = x['npair'][p] > 0
                r = np.where(ok & np.isfinite(x['r_mean'][p]), x['r_mean'][p], x['r_mid'])
                K = fourier_shift(L, z, np.concatenate([[0.0], r]), 4) * cpot(z)**2
                e.update(rx=r, sf=np.where(ok, x['sf'][p, k], np.nan), sf_err=np.where(ok, x['sf_err'][p, k], np.nan),
                         dsf=2 * (K[0] - K[1:]), dsf_rs=-bulk_pull(L, z) * r**2 / 3, nh_x=int(x['pair_n_halos'][p][0]),
                         gbar=np.sqrt(bulk_pull(L, z)))
                out[L, z, b] = e
    return out


def grid(title):
    """3 x 3 figure (rows: snapshots, columns: mass bins) with shared x axes and a title."""
    fig, axs = plt.subplots(3, 3, figsize=(13, 11), sharex=True, constrained_layout=True)
    fig.suptitle(title, fontsize=10)
    for i, (s, z) in enumerate(SNAPS):
        for j, b in enumerate(BINS):
            style(axs[i, j])
            if i == 0:
                axs[i, j].set_title(r'$M_{\rm vir}$ ' + MLAB[b] + r' $h^{-1}M_\odot$', fontsize=10)
            if i == 2:
                axs[i, j].set_xlabel(r'$r$ [$h^{-1}$Mpc, comoving]', fontsize=9)
    return fig, axs


def fig_profiles(D, what, out):
    """Figure of the measured profiles per box (``what`` = 'vprof': stacked potential; 'xcorr': structure
    function) with the measured-minus-predicted box-size shift; saved to ``out``.
    """
    sc = 1e5 if what == 'vprof' else 1e10
    ttl = ('Stacked volume-weighted potential around host halos, 512$^3$ boxes with matched seeds'
           if what == 'vprof' else
           r'Halo–halo potential structure function, $\phi_{\rm Rvir}$, pairs within one mass bin, 512$^3$ boxes')
    fig, axs = grid(ttl + '\nsolid: measured (band: jackknife, 27 sub-cubes); dashed: measured minus the '
                    'finite-box shift (linear theory, Fourier mode sum)')
    for i, (s, z) in enumerate(SNAPS):
        for j, b in enumerate(BINS):
            ax = axs[i, j]
            for L in BOXES:
                e = D[L, z, b]
                if what == 'vprof':
                    r, y, err, dy, nh = e['rv'], e['phi'], e['phi_err'], e['dphi'], e['nh_v']
                else:
                    r, y, err, dy, nh = e['rx'], e['sf'], e['sf_err'], e['dsf'], e['nh_x']
                ax.fill_between(r, (y - err) / sc, (y + err) / sc, color=COL[L], alpha=0.12, lw=0)
                ax.plot(r, y / sc, color=COL[L], lw=2, marker=MK[L], ms=4, markevery=3,
                        label=r'$L=%d$ (%d halos)' % (L, nh))
                ax.plot(r, (y - dy) / sc, color=COL[L], lw=1.3, ls='--')
            if j == 0:
                ax.set_ylabel('$z=%.1f$\n' % z + (r'$\langle\phi\rangle(r)$ [$10^5$ (km/s)$^2$]' if what == 'vprof'
                              else r'$\langle(\phi_i-\phi_j)^2\rangle$ [$10^{10}$ (km/s)$^4$]'), fontsize=9)
            ax.legend(fontsize=7, frameon=False)
    fig.savefig(out)
    plt.close(fig)


def fig_diff(D, what, out):
    """Figure of the box-size difference relative to L = 1000 against the predictions (Fourier mode sum and
    real-space integral constraint); saved to ``out``.
    """
    sc = 1e5 if what == 'vprof' else 1e10
    if what == 'vprof':
        ttl = (r'Box-size difference of the stacked potential, $\langle\phi\rangle_L-\langle\phi\rangle_{1000}$'
               '\nsolid: measured (band: jackknife errors in quadrature); dashed: predicted, Fourier mode sum; '
               'dotted: predicted, real-space integral constraint')
    else:
        ttl = (r'Box-size difference of the structure function, $\langle(\Delta\phi)^2\rangle_L-'
               r'\langle(\Delta\phi)^2\rangle_{1000}$, $\phi_{\rm Rvir}$' '\nsolid: measured; dashed: predicted, '
               r'Fourier mode sum; dotted: predicted, real-space missing bulk pull $-\langle\bar g^2\rangle r^2/3$')
    fig, axs = grid(ttl)
    for i, (s, z) in enumerate(SNAPS):
        for j, b in enumerate(BINS):
            ax = axs[i, j]
            ax.axhline(0, color='#9b9a95', lw=0.8)
            ref = D[1000, z, b]
            for L in (200, 500):
                e = D[L, z, b]
                if what == 'vprof':
                    r = e['rv']; y = e['phi'] - ref['phi']; err = np.hypot(e['phi_err'], ref['phi_err'])
                    pf = e['dphi'] - ref['dphi']; pr = np.full_like(r, e['dphi_rs'] - ref['dphi_rs'])
                else:
                    r = e['rx']; y = e['sf'] - np.interp(r, ref['rx'], ref['sf'])
                    err = np.hypot(e['sf_err'], np.interp(r, ref['rx'], ref['sf_err']))
                    pf = e['dsf'] - np.interp(r, ref['rx'], ref['dsf'])
                    pr = -(e['gbar']**2 - ref['gbar']**2) * r**2 / 3
                ax.fill_between(r, (y - err) / sc, (y + err) / sc, color=COL[L], alpha=0.12, lw=0)
                ax.plot(r, y / sc, color=COL[L], lw=2, marker=MK[L], ms=4, markevery=3, label=r'$L=%d$ measured' % L)
                ax.plot(r, pf / sc, color=COL[L], lw=1.5, ls='--', label=r'$L=%d$ Fourier' % L)
                ax.plot(r, pr / sc, color=COL[L], lw=1.5, ls=':', label=r'$L=%d$ real space' % L)
            if what == 'vprof':
                ax.set_ylim(-3, 8)
            else:
                ax.set_ylim(-80, 30)
            if j == 0:
                ax.set_ylabel('$z=%.1f$\n' % z + (r'$\Delta\langle\phi\rangle$ [$10^5$ (km/s)$^2$]' if what == 'vprof'
                              else r'$\Delta\langle(\phi_i-\phi_j)^2\rangle$ [$10^{10}$ (km/s)$^4$]'), fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.savefig(out)
    plt.close(fig)


def table(D, out):
    """LaTeX table of the box-size shifts, measured against predicted, per snapshot, mass bin and box."""
    rows = []
    for s, z in SNAPS:
        for b in BINS:
            ref = D[1000, z, b]
            for L in BOXES:
                e = D[L, z, b]
                q = (e['rv'] > 1) & (e['rv'] < 20)
                w = 1 / np.maximum(e['phi_err'][q], 1)**2
                meas = '--'
                if L != 1000:
                    dd = e['phi'][q] - ref['phi'][q]
                    meas = r'%.2f' % (np.average(dd, weights=w) / 1e5)
                    pf = np.mean((e['dphi'] - ref['dphi'])[q]) / 1e5
                    meas += r' & %.2f & %.2f' % (pf, (e['dphi_rs'] - ref['dphi_rs']) / 1e5)
                else:
                    meas = '-- & -- & --'
                rows.append(r'%.1f & %d & %d & %.2f & %.2f & %.2f & %s & %.0f \\' % (
                    z, b, L, e['bias'], np.mean(e['dphi'][q]) / 1e5, e['dphi_rs'] / 1e5, meas,
                    vbulk(e['gbar'], z)))
        rows.append(r'\midrule')
    with open(out, 'w') as f:
        f.write('% written by make_boxsize_figures.py\n')
        f.write(r'\begin{tabular}{cccc cc ccc c}' + '\n' + r'\toprule' + '\n')
        f.write(r' & & & & \multicolumn{2}{c}{shift of $\langle\phi\rangle$} & \multicolumn{3}{c}{'
                r'$\langle\phi\rangle_L-\langle\phi\rangle_{1000}$, 1--20\,\Mpch} & missing \\' + '\n')
        f.write(r'$z$ & bin & $L$ & $b$ & Fourier & real sp. & measured & Fourier & real sp. & $v_{\rm bulk}$ \\'
                + '\n' + r'\midrule' + '\n')
        f.write('\n'.join(rows[:-1]) + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n')


def main():
    """Command line: make the box-size figures and table from the reduced data (``--d``, ``--figs``,
    ``--table``).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', default=os.path.join(HERE, 'data', 'box512_L200_L500_L1000'))
    ap.add_argument('--figs', default=os.path.join(HERE, 'figs'))
    ap.add_argument('--table', default=os.path.join(HERE, 'boxsize_table.tex'))
    A = ap.parse_args()
    D = load_all(A.d)
    fig_profiles(D, 'vprof', os.path.join(A.figs, 'boxsize_vprof.pdf'))
    fig_diff(D, 'vprof', os.path.join(A.figs, 'boxsize_vprof_dphi.pdf'))
    fig_profiles(D, 'xcorr', os.path.join(A.figs, 'boxsize_xcorr.pdf'))
    fig_diff(D, 'xcorr', os.path.join(A.figs, 'boxsize_xcorr_dsf.pdf'))
    table(D, A.table)
    for z in (0.0, 0.5):
        for L in BOXES:
            e = D[L, z, 1]
            r10 = np.argmin(abs(e['rx'] - 10))
            print('z=%.1f L%4d: xbar=%.5f  dphi(b=1) Fourier %.3f real %.3f e5;  dsf(10) Fourier %.3f real %.3f e10; '
                  'gbar %.0f (km/s)^2/(Mpc/h)' % (z, L, e['xbar'], np.mean(e['dphi']) / e['bias'] / 1e5,
                                                 e['dphi_rs'] / e['bias'] / 1e5, e['dsf'][r10] / 1e10,
                                                 e['dsf_rs'][r10] / 1e10, e['gbar']))


if __name__ == '__main__':
    main()
