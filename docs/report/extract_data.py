#!/usr/bin/env python3
"""Reduce the pawna validation and pipeline outputs to the small data files used by make_figures.ipynb.

Runs on pawna (needs the Rockstar catalogs, which are too large for the repository):
    $CODES/venv/bin/python extract_data.py --rsval /storage/shadab/pkc_rockstar/rsval2 \
        --val512 /storage/shadab/pkc_rockstar/val512 --logs /storage/shadab/pipeline_test/data/logs \
        --rsval-log /storage/shadab/pkc_rockstar/logs/rsval.<job>.log --out data/
Writes into --out:
    halo_routes_512.npz   per Rockstar input route and output: Mvir and Vmax histograms; for pairs of routes the
                          matched-halo differences (Mvir, Vmax, Rs, spin, centre, bulk velocity)
    ctrees_512.npz        consistent-trees z = 0 hlists (pkc and raw routes): host and subhalo mass histograms
    quantization_512.npz  per-particle .pkc - raw (HDF5) position and velocity differences (1% subsample, z = 0)
    routes_512.csv        Rockstar wall time, peak memory and halo counts per route (from the validation job log)
    mem_*.tsv             memory samples (memmon.sh) of every pipeline stage and validation step
    stages.csv            wall time and peak memory per pipeline stage (from the pipeline job logs)
Matching: nearest halo centre in the other catalog (periodic KD tree), within 10 kpc/h below Mvir = 1e12 Msun/h
and 50 kpc/h above (as in scripts/rockstar/pkc_validation/compare_quantization.py).
"""
import argparse, glob, os, re, shutil, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument('--rsval', required=True); ap.add_argument('--val512', required=True)
ap.add_argument('--logs', required=True); ap.add_argument('--rsval-log', required=True)
ap.add_argument('--pp', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../scripts/post-process'))
ap.add_argument('--out', default='data')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
L = 50.0
ROUTES = ['pkc', 'pkc2', 'gad', 'h5g', 'pkcsub']
PAIRS = [('pkc', 'h5g'), ('pkc', 'pkc2'), ('pkc', 'gad'), ('pkcsub', 'pkc')]
COLS = ['Mvir', 'Vmax', 'Rs', 'Spin', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'Np']
MBINS = np.logspace(9, 15, 61)
VBINS = np.logspace(1, 3.5, 51)


def load(route, i):
    p = f'{a.rsval}/{route}/out_{i}.list'
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        h = fh.readline()[1:].split()
    return pd.read_csv(p, comment='#', sep=r'\s+', header=None, names=h, usecols=COLS)[COLS].to_numpy()


def scale_factors(route):
    a_ = []
    for f in sorted(glob.glob(f'{a.rsval}/{route}/out_*.list'), key=lambda s: int(re.findall(r'out_(\d+)', s)[0])):
        with open(f) as fh:
            for line in fh:
                if line.startswith('#a = '):
                    a_.append(float(line.split('=')[1])); break
    return np.array(a_)


out = {'mbins': MBINS, 'vbins': VBINS}
aa = scale_factors('pkc'); out['z'] = 1 / aa - 1
nz = len(aa)
cat = {}
for r in ROUTES:
    for i in range(nz):
        c = load(r, i)
        if c is None:
            continue
        cat[r, i] = c
        out[f'hm_{r}_{i}'] = np.histogram(c[:, 0], MBINS)[0]
        out[f'hv_{r}_{i}'] = np.histogram(c[:, 1], VBINS)[0]
        out[f'n_{r}_{i}'] = len(c)
for A, B in PAIRS:
    for i in range(nz):
        if (A, i) not in cat or (B, i) not in cat:
            continue
        x, y = cat[A, i], cat[B, i]
        m = x[:, 0] >= 1e10
        x = x[m]
        t = cKDTree(y[:, 4:7] % L, boxsize=L)
        d, j = t.query(x[:, 4:7] % L)
        ok = d < np.where(x[:, 0] >= 1e12, 0.05, 0.01)
        y = y[j]
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = lambda k: np.where(ok, x[:, k] / y[:, k] - 1, np.nan)
        k = f'{A}_{B}_{i}'
        out['pair_' + k + '_logM'] = np.log10(x[:, 0]).astype(np.float32)
        out['pair_' + k + '_matched'] = ok
        for name, col in (('dM', 0), ('dV', 1), ('dRs', 2), ('dSpin', 3)):
            out['pair_' + k + '_' + name] = rel(col).astype(np.float16)   # relative differences: 3 digits suffice
        out['pair_' + k + '_dpos'] = np.where(ok, d * 1e3, np.nan).astype(np.float16)          # kpc/h
        out['pair_' + k + '_dvel'] = np.where(ok, np.linalg.norm(x[:, 7:10] - y[:, 7:10], axis=1), np.nan).astype(np.float16)
np.savez_compressed(f'{a.out}/halo_routes_512.npz', **out)
print('halo_routes_512.npz', os.path.getsize(f'{a.out}/halo_routes_512.npz') / 1e6, 'MB')

# consistent-trees z = 0
ct = {'mbins': MBINS}
for r in ('pkc', 'h5g'):
    p = f'{a.rsval}/{r}/hlists/hlist_1.00000.list'
    if not os.path.exists(p):
        continue
    with open(p) as fh:
        h = [s.split('(')[0] for s in fh.readline()[1:].split()]
    im, ip = h.index('Mvir'), h.index('pid')               # hlist column names repeat (A[x], b_to_a, ...)
    d = pd.read_csv(p, comment='#', sep=r'\s+', header=None, usecols=[im, ip])[[im, ip]]   # (usecols keeps file order)
    d.columns = ['Mvir', 'pid']
    ct[f'host_{r}'] = np.histogram(d.Mvir[d.pid == -1], MBINS)[0]
    ct[f'sub_{r}'] = np.histogram(d.Mvir[d.pid != -1], MBINS)[0]
np.savez_compressed(f'{a.out}/ctrees_512.npz', **ct)

# per-particle quantization: .pkc 1% subsample vs the raw HDF5 output of the same step. The run is in ordered
# mode, so HDF5 row = particle ID. HDF5 coordinates are in box units [-0.5, 0.5), velocities in code units
# (x vel_to_kms of the .pkc header = km/s). Set HDF5_USE_FILE_LOCKING=FALSE on NFS (/storage), or h5py hangs.
sys.path.insert(0, a.pp)
import pkcreader, h5py
snap = sorted(glob.glob(f'{a.val512}/snap.*.done'))[-1][:-5]
X, V, ids = [], [], []
for f in sorted(glob.glob(snap + '.s01.*.pkc')):
    pf = pkcreader.PKCFile(f)
    x, v, i = pf.read(units='box')
    X.append(x); V.append(v); ids.append(i)
X, V, ids = np.concatenate(X), np.concatenate(V), np.concatenate(ids).astype(np.int64)
hdr = pf.header
with h5py.File(snap, 'r') as f:
    xs = f['PartType1/Coordinates'][...][ids]
    vs = f['PartType1/Velocities'][...][ids] * hdr['vel_to_kms']
dx = X - xs
dx -= np.round(dx)                                         # periodic, box units
step = hdr['box_size'] / (hdr['ncell'] * hdr['sub']) * 1e3  # kpc/h
q = {'pos_step_kpch': step, 'vel_step_kms': hdr['vel_step'], 'n_particles': len(ids)}
dxa = dx * hdr['box_size'] * 1e3; dva = V - vs
q['hbins'] = np.linspace(-0.7, 0.7, 141)                     # in units of the step
q['hist_dx'] = np.histogram(dxa.ravel() / step, q['hbins'])[0]
q['hist_dv'] = np.histogram(dva.ravel() / hdr['vel_step'], q['hbins'])[0]
q['summary'] = np.array([np.abs(dxa).max(), dxa.std(), np.abs(dva).max(), dva.std()])
np.savez_compressed(f'{a.out}/quantization_512.npz', **q)
print('quantization (%d particles): pos step %.4f kpc/h, max |dx| %.4f, rms %.4f; vel step %.3f km/s, max |dv| %.3f, rms %.3f'
      % (len(ids), step, *q['summary'][:2], hdr['vel_step'], *q['summary'][2:]))

# Rockstar routes: wall time, memory, halo counts (validation job log)
rows = []
log = open(a.rsval_log).read()
for m in re.finditer(r'^(\w+): exit (\d+), (\d+) s, halos per output: ([\d ]+)$', log, re.M):
    mm = re.search(rf'^MEM {m.group(1)}: peak PSS ([\d.]+) GB \((\d+) processes\), peak RSS sum ([\d.]+) GB', log, re.M)
    rows.append(dict(route=m.group(1), exit=int(m.group(2)), wall_s=int(m.group(3)),
                     halos=' '.join(m.group(4).split()), pss_gb=float(mm.group(1)) if mm else np.nan,
                     nproc=int(mm.group(2)) if mm else -1, rss_gb=float(mm.group(3)) if mm else np.nan))
for m in re.finditer(r'^MEM (convert|ctrees_\w+): peak PSS ([\d.]+) GB \((\d+) processes\), peak RSS sum ([\d.]+) GB, (\d+) s', log, re.M):
    rows.append(dict(route=m.group(1), exit=0, wall_s=int(m.group(5)), halos='', pss_gb=float(m.group(2)),
                     nproc=int(m.group(3)), rss_gb=float(m.group(4))))
rows.append(dict(route='exact_pkc1_vs_gad1', exit=0, wall_s=0, halos=' '.join(re.findall(r'EXACT: (\S+) identical', log)),
                 pss_gb=np.nan, nproc=-1, rss_gb=np.nan))
pd.DataFrame(rows).to_csv(f'{a.out}/routes_512.csv', index=False)
for f in glob.glob(f'{a.rsval}/mem.*.tsv'):
    shutil.copy(f, f'{a.out}/mem_rsval_{os.path.basename(f)[4:]}')

# pipeline stages: wall time and peak memory from the job logs (runs pkdtest/mem128 and pkdtest/lcdm512)
rows = []
for f in sorted(glob.glob(f'{a.logs}/*.pkdtest_*_r1.*.log')):
    st, run, job = re.match(r'(\w+)\.pkdtest_(\w+)_r1\.(\d+)\.log', os.path.basename(f)).groups()
    if run not in ('mem128', 'lcdm512'):
        continue
    s = open(f).read()
    ts = re.findall(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]', s, re.M)
    pk = re.search(r'peak memory: PSS ([\d.]+) GB \((\d+) processes, at (\d+) s\), RSS sum ([\d.]+) GB', s)
    mr = re.findall(r'Maximum resident set size \(kbytes\): (\d+)', s)
    if not pk:
        continue
    rows.append(dict(run=run, stage=st, job=job, wall_s=(pd.Timestamp(ts[-1]) - pd.Timestamp(ts[0])).total_seconds(),
                     pss_gb=float(pk.group(1)), nproc=int(pk.group(2)), rss_gb=float(pk.group(4)),
                     maxrss_single_gb=int(mr[-1]) / 1048576 if mr else np.nan))
    tsv = f.replace('.log', '.mem.tsv')
    if os.path.exists(tsv):
        shutil.copy(tsv, f'{a.out}/mem_{run}_{st}.tsv')
pd.DataFrame(rows).to_csv(f'{a.out}/stages.csv', index=False)
print(pd.DataFrame(rows).to_string())
