"""
compare_vpot.py: halo-by-halo comparison of the pipeline's mean particle potential around halos (.vpot columns
phi_<R> of out_<snap>_vahc.fits.gz, vahc_fast.py) between the Gadget-4 and the pkdgrav3 run of the same ICs.

Host halos (pid = -1) are matched as in compare_halos.py (mutual nearest centres within 0.5 Rvir_gadget,
|log10 Mvir ratio| < 0.5). Each code's phi_<R> uses its own halo centre and radius (rs, R200b, Rvir, ...) and
its own particles; the radii in Mpc/h (R2Mpch, R3Mpch, R5Mpch) are the same for both.
Per output and Mvir_gadget bin, for every radius: median and 16/84 percentiles of phi_pkd / phi_gadget and of
phi_pkd - phi_gadget in (km/s)^2; with --particles (compare_potential.py's particles.csv) also the difference after
removing the snapshot's zero-point offset (median of phi_pkd - phi_gadget over all particles; every 5th output).

Usage: compare_vpot.py <gadget compressed dir> <pkd compressed dir> <out.csv> [--box 200] [--particles csv]
Reads only the FITS files (fitsio); runs on the Mac on copies of out_*_{basic,vahc}.fits.gz.
"""
import argparse
import csv
import glob
import os
import re
import sys

import fitsio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_halos import match  # noqa: E402

RADII = ['rs', 'R0p5vir', 'R200b', 'Rvir', 'R1p5vir', 'R2R200b', 'R4R200b', 'R6R200b', 'R8R200b',
         'R2Mpch', 'R3Mpch', 'R5Mpch']


def read(d, i):
    b = os.path.join(d, 'out_%d_basic.fits.gz' % i)
    v = os.path.join(d, 'out_%d_vahc.fits.gz' % i)
    if not (os.path.isfile(b) and os.path.isfile(v)):
        return None
    with fitsio.FITS(b) as f:
        B = f[1].read(columns=['id', 'pid', 'Mvir', 'Rvir', 'x', 'y', 'z'])
        a = float(f[1].read_header().get('SCALE', np.nan))
    with fitsio.FITS(v) as f:
        if not f[1].read_header().get('HAS_VPOT', False):
            return None
        V = f[1].read(columns=['haloID'] + ['phi_' + r for r in RADII])
    host = B['pid'] == -1
    B = B[host]
    row = {int(k): j for j, k in enumerate(V['haloID'])}
    idx = np.array([row.get(int(k), -1) for k in B['id']], dtype=np.int64)
    h = dict(x=B['x'], y=B['y'], z=B['z'], rvir=B['Rvir'], mvir=B['Mvir'], a=a)
    for r in RADII:
        p = np.full(len(B), np.nan)
        p[idx >= 0] = V['phi_' + r][idx[idx >= 0]]
        h[r] = p
    return h


def pct(x):
    x = x[np.isfinite(x)]
    return (np.nan,) * 3 if len(x) < 5 else tuple(float(v) for v in np.percentile(x, [50, 16, 84]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gadget_dir'); ap.add_argument('pkd_dir'); ap.add_argument('out_csv')
    ap.add_argument('--box', type=float, default=200.0)
    ap.add_argument('--mbins', default='1e11,1e12,1e13,1e14,1e16')
    ap.add_argument('--particles', help='particles.csv of compare_potential.py (zero-point offsets)')
    A = ap.parse_args()
    me = [float(v) for v in A.mbins.split(',')]
    off = {}
    if A.particles:
        for r in csv.DictReader(open(A.particles)):
            off[int(r['out'])] = float(r['offset_median'])
    outs = sorted(int(re.search(r'out_(\d+)_vahc', p).group(1)) for p in glob.glob(os.path.join(A.gadget_dir, 'out_*_vahc.fits.gz')))
    rows = []
    for i in outs:
        g, p = read(A.gadget_dir, i), read(A.pkd_dir, i)
        if g is None or p is None or len(g['x']) == 0:
            continue
        ig, ip, _ = match(g, p, A.box)
        z = 1 / g['a'] - 1
        mg = g['mvir'][ig]
        for k in range(len(me) - 1):
            s = (mg >= me[k]) & (mg < me[k + 1])
            if s.sum() < 5:
                continue
            for r in RADII:
                pg, pp = g[r][ig[s]], p[r][ip[s]]
                ok = np.isfinite(pg) & np.isfinite(pp)
                pg, pp = pg[ok], pp[ok]
                q, d = pct(pp / pg), pct(pp - pg)
                row = dict(out=i, z=round(z, 4), lo=me[k], hi=me[k + 1], radius=r, n=int(ok.sum()),
                           phi_gadget_med=float(np.median(pg)) if len(pg) else np.nan,
                           ratio_med=q[0], ratio_p16=q[1], ratio_p84=q[2], diff_med=d[0], diff_p16=d[1], diff_p84=d[2],
                           offset=off.get(i, np.nan))
                if i in off:
                    q2 = pct((pp - off[i]) / pg)
                    row.update(ratio_aligned_med=q2[0], diff_aligned_med=d[0] - off[i])
                else:
                    row.update(ratio_aligned_med=np.nan, diff_aligned_med=np.nan)
                rows.append(row)
        print('out %d z %.3f: %d gadget hosts, %d pkd hosts, %d matched' % (i, z, len(g['x']), len(p['x']), len(ig)))
    with open(A.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('wrote %s (%d rows)' % (A.out_csv, len(rows)))


if __name__ == '__main__':
    main()
