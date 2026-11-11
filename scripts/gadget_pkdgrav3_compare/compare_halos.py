#!/usr/bin/env python3
"""Compare the halo catalogues of a Gadget-4 and a pkdgrav3 run from the same ICs (consistent-trees hlists
out_<snap>.trees and value-added catalogues out_<snap>.vahc of the Sahyadri pipeline; Rockstar out_<snap>.list
where there is no hlist).

usage: compare_halos.py <gadget_halo_dir> <pkd_halo_dir> <out_dir> --outs 20,50,80,100 [--mbins ...]

Per output:
  mass functions dn/dlog10M (h^3 Mpc^-3) of host halos (pid = -1) in Mvir and M200b, Vmax function,
  subhalo fraction;
  host halos matched one to one: mutual nearest centres (periodic) within 0.5 Rvir_gadget and |log10 Mvir ratio|
  < 0.5; per Mvir_gadget bin: matched fraction and the median and 16/84 percentiles of pkd / gadget ratios
  (Mvir, M200b, Vmax, Rvir/rs, Spin_Bullock, b_to_a, c_to_a, T/|U|), of the centre offset / Rvir and |v_pkd - v_gadget|
  (bulk velocity, km/s), and of the VAHC quantities (b1, lam1..3 at R = 2 and 4 R200b) of matched halos.
Output in out_dir: halo_mf.csv, halo_match.csv, halo_match_out<i>.npz (matched pairs, for the report notebook).
"""
import argparse
import csv
import math
import os
import re
import sys

import numpy as np
from scipy.spatial import cKDTree

COLS = ['scale', 'id', 'pid', 'upid', 'mvir', 'rvir', 'rs', 'vmax', 'vrms', 'x', 'y', 'z', 'vx', 'vy', 'vz',
        'Spin_Bullock', 'M200b', 'M200c', 'Xoff', 'b_to_a', 'c_to_a', 'T/|U|']
RATIOS = ['mvir', 'M200b', 'vmax', 'conc', 'Spin_Bullock', 'b_to_a', 'c_to_a', 'T/|U|']
VAHC = ['b1', 'lam1_R2R200b', 'lam2_R2R200b', 'lam3_R2R200b', 'lam1_R4R200b', 'lam3_R4R200b']


def read_cat(d, i):
    """-> dict of arrays (lower-case Rockstar names as in the hlist) or None"""
    p = os.path.join(d, 'out_%d.trees' % i)
    kind = 'trees'
    if not os.path.exists(p):
        p, kind = os.path.join(d, 'out_%d.list' % i), 'list'
        if not os.path.exists(p):
            return None
    scale = np.nan
    with open(p) as f:
        head = f.readline().lstrip('#').split()
        for l in f:                                   # Rockstar .list / hlist header: '#a = 0.5'
            if not l.startswith('#'):
                break
            m = re.match(r'#\s*a\s*=\s*([0-9.eE+-]+)', l)
            if m:
                scale = float(m.group(1))
    names = [re.sub(r'\(\d+\)$', '', h) for h in head]
    lower = {n.lower(): k for k, n in enumerate(names)}
    want = {}
    for c in COLS:
        if c in names:
            want[c] = names.index(c)
        elif c.lower() in lower:
            want[c] = lower[c.lower()]
    # several columns share a name (b_to_a and b_to_a(500c)): the first one is the Mvir definition
    d_ = np.loadtxt(p, comments='#', usecols=sorted(set(want.values())), ndmin=2)
    col = {v: k for k, v in enumerate(sorted(set(want.values())))}
    h = {c: d_[:, col[j]] for c, j in want.items()}
    n = len(d_)
    if 'pid' not in h:
        h['pid'] = np.full(n, -1.0)
    h['conc'] = h['rvir'] / h['rs'] if 'rs' in h else np.full(n, np.nan)
    h['kind'] = kind
    h['a'] = scale if scale == scale else (float(h['scale'][0]) if 'scale' in h and n else np.nan)
    v = os.path.join(d, 'out_%d.vahc' % i)
    if os.path.exists(v):
        with open(v) as f:
            f.readline()
            vn = f.readline().lstrip('#').split()
        vd = np.loadtxt(v, comments='#', ndmin=2)
        row = {int(k): j for j, k in enumerate(vd[:, 0])} if len(vd) else {}
        idx = np.array([row.get(int(k), -1) for k in h['id']])
        for c in VAHC:
            if c in vn:
                a = np.full(n, np.nan)
                a[idx >= 0] = vd[idx[idx >= 0], vn.index(c)]
                h[c] = a
    return h


def mf(m, edges, vol):
    n = np.histogram(m, edges)[0]
    dl = np.diff(np.log10(edges))
    return n, n / vol / dl, np.sqrt(n) / vol / dl


def match(hg, hp, L):
    if len(hg['x']) == 0 or len(hp['x']) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    cg = np.mod(np.vstack([hg['x'], hg['y'], hg['z']]).T, L)
    cp = np.mod(np.vstack([hp['x'], hp['y'], hp['z']]).T, L)
    cg[cg >= L] = 0; cp[cp >= L] = 0
    tg, tp = cKDTree(cg, boxsize=L), cKDTree(cp, boxsize=L)
    d, ip = tp.query(cg)
    _, ig_back = tg.query(cp)
    ig = np.arange(len(cg))
    ok = (ig_back[ip] == ig) & (d < 0.5 * hg['rvir'] / 1e3) & (np.abs(np.log10(hp['mvir'][ip] / hg['mvir'])) < 0.5)
    return ig[ok], ip[ok], d[ok]


def pct(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return tuple(float(v) for v in np.percentile(x, [50, 16, 84]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gadget_halo_dir'); ap.add_argument('pkd_halo_dir'); ap.add_argument('out_dir')
    ap.add_argument('--outs', required=True)
    ap.add_argument('--box', type=float, default=200.0)
    ap.add_argument('--mbins', default='1e11,3e11,1e12,3e12,1e13,3e13,1e14,3e14,1e15,3e15')
    ap.add_argument('--vbins', default='30,50,80,120,200,300,500,800,1300,2000')
    A = ap.parse_args()
    os.makedirs(A.out_dir, exist_ok=True)
    L, vol = A.box, A.box**3
    me = np.array([float(v) for v in A.mbins.split(',')])
    ve = np.array([float(v) for v in A.vbins.split(',')])
    mfrows, mrows = [], []
    for i in [int(v) for v in A.outs.split(',') if v.strip()]:
        hg, hp = read_cat(A.gadget_halo_dir, i), read_cat(A.pkd_halo_dir, i)
        if hg is None or hp is None or len(hg['x']) == 0:
            print('out %d: no catalogue for both codes' % i); continue
        a = hg['a']
        z = 1 / a - 1 if a == a else np.nan
        hostg, hostp = hg['pid'] == -1, hp['pid'] == -1
        for q, e in (('mvir', me), ('M200b', me), ('vmax', ve)):
            if q not in hg or q not in hp:
                continue
            ng, fg, eg = mf(hg[q][hostg], e, vol)
            np_, fp, ep = mf(hp[q][hostp], e, vol)
            for k in range(len(e) - 1):
                mfrows.append(dict(out=i, z=z, quantity=q, lo=e[k], hi=e[k + 1], n_gadget=ng[k], n_pkd=np_[k],
                                   dn_gadget=fg[k], dn_pkd=fp[k], err_gadget=eg[k], err_pkd=ep[k],
                                   ratio=fp[k] / fg[k] if fg[k] else np.nan,
                                   ratio_err=(fp[k] / fg[k] * math.sqrt(1 / max(ng[k], 1) + 1 / max(np_[k], 1)))
                                   if fg[k] and np_[k] else np.nan))
        G = {k: v[hostg] for k, v in hg.items() if isinstance(v, np.ndarray)}
        P = {k: v[hostp] for k, v in hp.items() if isinstance(v, np.ndarray)}
        ig, ip, d = match(G, P, L)
        dv = np.sqrt((P['vx'][ip] - G['vx'][ig])**2 + (P['vy'][ip] - G['vy'][ig])**2 + (P['vz'][ip] - G['vz'][ig])**2) \
            if 'vx' in G and 'vx' in P else np.full(len(ig), np.nan)
        pairs = dict(z=z, out=i, d_over_rvir=d / (G['rvir'][ig] / 1e3), dv=dv)
        for q in set(RATIOS + VAHC + ['mvir', 'rvir', 'x', 'y', 'z', 'vx', 'vy', 'vz']):
            if q in G and q in P:
                pairs[q + '_gadget'] = G[q][ig]
                pairs[q + '_pkd'] = P[q][ip]
        np.savez_compressed(os.path.join(A.out_dir, 'halo_match_out%d.npz' % i),
                            mvir_all_gadget=G['mvir'], mvir_all_pkd=P['mvir'], **pairs)
        mg = G['mvir'][ig]
        for k in range(len(me) - 1):
            inb = (mg >= me[k]) & (mg < me[k + 1])
            nall = int(((G['mvir'] >= me[k]) & (G['mvir'] < me[k + 1])).sum())
            if nall == 0:
                continue
            r = dict(out=i, z=z, lo=me[k], hi=me[k + 1], n_gadget=nall, n_matched=int(inb.sum()),
                     matched_frac=inb.sum() / nall)
            for q in RATIOS:
                if q in G and q in P:
                    r[q + '_ratio_p50'], r[q + '_ratio_p16'], r[q + '_ratio_p84'] = pct(P[q][ip][inb] / G[q][ig][inb])
            for q in VAHC:
                if q in G and q in P:
                    r[q + '_diff_p50'], r[q + '_diff_p16'], r[q + '_diff_p84'] = pct(P[q][ip][inb] - G[q][ig][inb])
            r['offset_rvir_p50'], _, r['offset_rvir_p84'] = pct(pairs['d_over_rvir'][inb])
            r['dv_p50'], _, r['dv_p84'] = pct(dv[inb])
            mrows.append(r)
        print('out %d z %.3f (%s/%s): hosts %d gadget, %d pkd; matched %d (%.1f%% of gadget hosts); vahc %s'
              % (i, z, hg['kind'], hp['kind'], hostg.sum(), hostp.sum(), len(ig), 100 * len(ig) / max(1, hostg.sum()),
                 'b1' in G), flush=True)

    def write(rows, path):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, keys)
            w.writeheader(); w.writerows(rows)
    write(mfrows, os.path.join(A.out_dir, 'halo_mf.csv'))
    write(mrows, os.path.join(A.out_dir, 'halo_match.csv'))
    print('wrote %s/{halo_mf.csv,halo_match.csv,halo_match_out*.npz}' % A.out_dir)


if __name__ == '__main__':
    sys.exit(main())
