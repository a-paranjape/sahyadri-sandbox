#!/usr/bin/env python3
"""P(k) checks of a run (verify_run.py) and the summary plot <halo dir>/pk/pk_summary.png.

For every output, from out_<snap>.pk (pk_measure.py) and <halo dir>/pk/class_* (class_pk.py), on the linear bins
k_mean <= KLIN (PK_KLIN, default 0.1 h/Mpc, at least the two lowest bins):
  absolute  R = <P_mm / P_ref>_modes (raw P_mm: see below) with P_ref = P_CLASS(z_start) [D(z)/D(z_start)]^2 x halofit/linear
            (CLASS at z): cosmic variance of the realisation, sigma = sqrt(2 / N_modes); FAIL if |R - 1| >
            4 sigma + PK_ABS_TOL (0.02): catches normalisation, unit or growth errors, not the realisation's scatter
  growth    G = <[P(z) / P(z_first)] / [P_ref(z) / P_ref(z_first)]>_modes over the bins that are still linear at z,
            Delta^2_lin(k, z) = k^3 P_lin / 2 pi^2 <= PK_DELTA2_LIN (0.01): the same modes at both times, so the
            realisation's scatter cancels; FAIL if |G - 1| > PK_GROWTH_TOL (0.03). Snapshots without such a bin (late
            times) are reported, not checked. Why 0.01: the lowest bins hold only 6-12 modes, and their growth departs
            from linear theory by ~(2-3) x Delta^2_lin per bin through mode coupling, with either sign (measured:
            128^3, 200 Mpc/h, z = 1.24, Delta^2_lin = 0.02-0.04: bins at +6 %, -6 %, +10 %; 64^3, 100 Mpc/h,
            k ~ 0.09 h/Mpc, z = 0: -12 %); a code error in the growth shows at high z, where many bins are linear
  CLASS     the ratio to CLASS linear and halofit at z itself (CLASS includes radiation): reported, not checked
  IC table  CLASS at z_start against the table the ICs were made from: FAIL if it differs by more than 2 %
  bias      large-scale b1 per halo mass bin: reported (warning if not positive where a bin has >= 100 halos)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'post-process'))


def load_class(pkdir):
    meta = json.load(open(os.path.join(pkdir, 'class_meta.json')))
    lin = np.loadtxt(os.path.join(pkdir, 'class_pk_lin.txt'))
    nl = np.loadtxt(os.path.join(pkdir, 'class_pk_nl.txt'))
    return meta, lin, nl


def interp_col(tab, col, k):
    return np.exp(np.interp(np.log(k), np.log(tab[:, 0]), np.log(tab[:, col])))


def check(E, P, outs, report):
    """Fill report.check('pk'); returns the per-snapshot rows (for the plot)."""
    import pk_measure
    name = 'pk'
    c = report.check(name)
    pkdir = os.path.join(P['halo'], 'pk')
    try:
        cm, lin, nl = load_class(pkdir)
    except (OSError, ValueError) as e:
        report.fail(name, 'no CLASS spectra in %s (%s): class_pk.py runs in the vahc stage' % (pkdir, e))
        return []
    klin = float(E.get('PK_KLIN', '0.1') or 0.1)
    abs_tol = float(E.get('PK_ABS_TOL', '0.02') or 0.02)
    g_tol = float(E.get('PK_GROWTH_TOL', '0.03') or 0.03)
    d2lin = float(E.get('PK_DELTA2_LIN', '0.01') or 0.01)
    ic = cm.get('ic_check')
    if ic is None:
        report.warn(name, 'no IC table to compare CLASS at z_start with')
    elif ic['max_abs_dev'] > 0.02:
        report.fail(name, 'CLASS at z_start differs from the IC table %s by up to %.1f%%: the reference is not the '
                          'spectrum the ICs were made from' % (ic['file'], 100 * ic['max_abs_dev']))
    D0 = cm['growth_D']['z_start']
    Ds = cm['growth_D']['outputs']
    rows = []
    first = None
    for snap, step, _a, z in outs:
        f = os.path.join(P['halo'], 'out_%d.pk' % snap)
        if not os.path.isfile(f):
            report.fail(name, 'no out_%d.pk' % snap)
            continue
        m, d = pk_measure.read(f)
        k, n = d['k_mean'], d['Nmodes']
        # raw P(k), no shot-noise subtraction: the particles start on a lattice, so on these (linear) scales their
        # discreteness noise is far below the Poisson 1/nbar (subtracting it removed ~7 % of the z = 12 power at
        # k = 0.063 h/Mpc in a 64^3, 100 Mpc/h test and faked 7 % excess growth); 1/nbar stays in the .pk files
        Pm = d['Pmm']
        col = 2 + snap                      # class tables: k, z_start, outputs...
        boost = interp_col(nl, col, k) / interp_col(lin, col, k)
        ref = interp_col(lin, 1, k) * (Ds[snap] / D0)**2 * boost
        sel = (k <= klin) & (n > 0)
        if sel.sum() < 2:
            sel = np.zeros_like(sel)
            sel[np.argsort(k)[:2]] = True
        R = float(np.sum(n[sel] * Pm[sel] / ref[sel]) / np.sum(n[sel]))
        sig = float(np.sqrt(2.0 / np.sum(n[sel])))
        row = dict(snap=snap, z=float(m.get('redshift', z)), R=R, sigma=sig, nmodes=int(np.sum(n[sel])),
                   class_lin=float(np.sum(n[sel] * Pm[sel] / interp_col(lin, col, k[sel])) / np.sum(n[sel])),
                   b1=[hb.get('b1') for hb in m.get('halo_bins', [])], nh=[hb.get('n') for hb in m.get('halo_bins', [])],
                   k=k.tolist(), Pm=Pm.tolist(), ref=ref.tolist(), shot=float(d['Pmm_shot'][0]),
                   kNy=m.get('kNy'), halo_bins=m.get('halo_bins', []),
                   bhm={i: d['bhm_%d' % i].tolist() for i in range(len(m.get('halo_bins', []))) if 'bhm_%d' % i in d})
        if abs(R - 1) > 4 * sig + abs_tol:
            report.fail(name, 'snapshot %d (z=%.2f): P(k <= %.2g) / reference = %.3f (sigma %.3f)' % (snap, row['z'], klin, R, sig))
        if first is None:
            first = (snap, Pm, ref, sel)
            row['G'] = 1.0
        else:
            s0, Pm0, ref0, sel0 = first
            plin = interp_col(lin, col, k)
            s2 = sel & sel0 & (k**3 * plin / (2 * np.pi**2) <= d2lin)
            row['n_growth_modes'] = int(np.sum(n[s2]))
            if not s2.any():
                row['G_note'] = 'no bin with Delta^2_lin <= %g' % d2lin
            if s2.any():
                G = float(np.sum(n[s2] * (Pm[s2] / Pm0[s2]) / (ref[s2] / ref0[s2])) / np.sum(n[s2]))
                row['G'] = G
                if abs(G - 1) > g_tol:
                    report.fail(name, 'snapshot %d (z=%.2f): growth since snapshot %d = %.3f of linear theory (tolerance %g)'
                                % (snap, row['z'], s0, G, g_tol))
        for hb in m.get('halo_bins', []):
            if hb.get('n', 0) >= 100 and not (hb.get('b1') or 0) > 0:
                report.warn(name, 'snapshot %d: b1 = %s for %d halos in [%g, %s) particles' % (
                    snap, hb.get('b1'), hb['n'], hb['np_lo'], hb['np_hi']))
        rows.append(row)
    c['info']['klin'] = klin
    c['info']['delta2_lin'] = d2lin
    c['info']['growth_checked'] = sum(1 for r in rows if r.get('n_growth_modes'))
    c['info']['ic_check'] = ic
    c['info']['snapshots'] = [{k: r[k] for k in ('snap', 'z', 'R', 'sigma', 'G', 'n_growth_modes', 'class_lin', 'b1', 'nh')
                               if k in r} for r in rows]
    if rows:
        c['info']['R_range'] = [min(r['R'] for r in rows), max(r['R'] for r in rows)]
        gs = [r['G'] for r in rows if r.get('n_growth_modes') or r is rows[0]]
        c['info']['G_range'] = [min(gs), max(gs)]
    return rows


# sequential (light = early -> dark = z 0) and categorical colours of the dataviz reference palette
SEQ = ['#86b6ef', '#5598e7', '#2a78d6', '#256abf', '#1c5cab', '#104281', '#0d366b']
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def plot(rows, cm_path, png, title):
    """Three panels: P(k) at up to 7 redshifts with CLASS; ratio to the reference; b(k) per mass bin at the last."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not rows:
        return None
    def zs(z):                       # no "-0.00" for the last output
        return '%.2f' % (0.0 if abs(z) < 0.005 else z)
    lin = np.loadtxt(os.path.join(cm_path, 'class_pk_lin.txt'))
    nl = np.loadtxt(os.path.join(cm_path, 'class_pk_nl.txt'))
    want = [rows[0]['z'], 5, 3, 2, 1, 0.5, 0]
    pick = []
    for zw in want:
        r = min(rows, key=lambda r: abs(r['z'] - zw))
        if r not in pick:
            pick.append(r)
    pick.sort(key=lambda r: -r['z'])
    cols = SEQ[-len(pick):] if len(pick) < len(SEQ) else SEQ
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    for r, colr in zip(pick, cols):
        k = np.array(r['k'])
        Pm = np.array(r['Pm'])
        ok = (k < r['kNy']) & (Pm > 0)
        ax[0].loglog(k[ok], Pm[ok], '-', color=colr, lw=2, label='z = ' + zs(r['z']))
        shot = r['shot']
        ax[0].loglog(lin[:, 0], lin[:, 2 + r['snap']], '--', color=colr, lw=1)
        ax[0].loglog(nl[:, 0], nl[:, 2 + r['snap']], ':', color=colr, lw=1)
        ref = np.array(r['ref'])
        ax[1].semilogx(k[ok], Pm[ok] / ref[ok], '-', color=colr, lw=2, label='z = ' + zs(r['z']))
    ax[0].axhline(shot, color=MUTED, lw=0.8, ls='-.', zorder=0)
    ax[0].text(ax[0].get_xlim()[1] / 1.5, shot * 1.25, '1/nbar', color=MUTED, fontsize=8, ha='right')
    k0 = np.array(pick[-1]['k'])
    ax[0].axvline(pick[-1]['kNy'] / 2, color=GRID, lw=1, zorder=0)
    ax[0].set_xlabel('k  [h/Mpc]')
    ax[0].set_ylabel('P(k)  [(Mpc/h)$^3$]')
    ax[0].set_title('matter P(k), no shot noise subtracted (solid); CLASS linear (dashed), halofit (dotted)', fontsize=9, color=MUTED)
    ax[0].legend(frameon=False, fontsize=8, loc='lower left')
    ax[0].set_ylim(max(1e-2, min(np.array(r['Pm'])[np.array(r['Pm']) > 0].min() for r in pick) / 3), None)
    ax[1].axhline(1, color=MUTED, lw=1)
    ax[1].axvline(pick[-1]['kNy'] / 2, color=GRID, lw=1, zorder=0)
    ax[1].set_ylim(0.5, 1.5)
    ax[1].set_xlim(k0.min() * 0.9, pick[-1]['kNy'])
    ax[1].set_xlabel('k  [h/Mpc]')
    ax[1].set_ylabel('measured / reference')
    ax[1].set_title('reference: CLASS(z$_{start}$) x D$^2$ x halofit/linear; grey: k$_{Ny}$/2', fontsize=9, color=MUTED)
    last = rows[-1]
    k = np.array(last['k'])
    for i, hb in enumerate(last['halo_bins']):
        if str(i) not in last['bhm'] and i not in last['bhm']:
            continue
        b = np.array(last['bhm'].get(i, last['bhm'].get(str(i))))
        ok = np.isfinite(b) & (k < last['kNy'] / 2)
        if hb.get('n', 0) < 2 or not ok.any():
            continue
        lab = 'M %.1e-%s: %d halos' % (hb['m_lo'], ('%.1e' % hb['m_hi']) if hb['m_hi'] else '', hb['n'])
        ax[2].semilogx(k[ok], b[ok], '-', color=CAT[i % len(CAT)], lw=2, label=lab)
    ax[2].set_xlabel('k  [h/Mpc]')
    ax[2].set_ylabel('b$_{hm}$(k) = P$_{hm}$ / P$_{mm}$')
    ax[2].set_title('halo bias at z = %s (host halos, M$_{vir}$ [M$_\\odot$/h])' % zs(last['z']), fontsize=9, color=MUTED)
    ax[2].legend(frameon=False, fontsize=8)
    for a in ax:
        a.grid(True, color=GRID, lw=0.6)
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(png), exist_ok=True)
    fig.savefig(png + '.part.png', dpi=110)
    plt.close(fig)
    os.replace(png + '.part.png', png)
    return png
