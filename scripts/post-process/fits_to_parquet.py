#!/usr/bin/env python3
"""Convert a run's compressed FITS halo outputs (compressed/out_<snap>_<table>.fits.gz, compressed/trees/*.fits.gz)
into Parquet in a parallel directory with the same file names (.parquet instead of .fits.gz). FITS stays untouched.

Encodings, from parquet_layout_test.py (L500N512, snapshot 100 and the tree file): zstd level 3, row groups of 2^20 rows,
statistics on; dictionary encoding for basic, extended and the trees (smallest there), and for tens and vahc no
dictionary on float columns plus BYTE_STREAM_SPLIT (smallest there). Every file is read back and compared bit for bit
with the FITS columns; a mismatch fails the file. A file whose .parquet is newer than its .fits.gz is skipped.

usage: fits_to_parquet.py <compressed dir> <parquet dir> [--procs N] [--report report.json]
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import fitsio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SPLIT_TABLES = ('tens', 'vahc')


def native(a):
    return a.astype(a.dtype.newbyteorder('='))


def convert(src, dst):
    t0 = time.perf_counter()
    D = fitsio.read(src, ext=1)
    T = pa.table({c: native(D[c]) for c in D.dtype.names})
    hdr = fitsio.read_header(src, ext=1)
    meta = {k: str(hdr[k]) for k in hdr.keys() if k and not k.startswith(('TTYPE', 'TFORM', 'TDIM', 'NAXIS', 'PCOUNT',
                                                                            'GCOUNT', 'TFIELDS', 'XTENSION', 'BITPIX'))}
    T = T.replace_schema_metadata({'fits_header': json.dumps(meta), 'source': os.path.basename(src)})
    t1 = time.perf_counter()
    kw = dict(compression='zstd', compression_level=3, row_group_size=1 << 20, write_statistics=True)
    name = os.path.basename(src)
    if any('_%s.' % t in name for t in SPLIT_TABLES):
        fl = [c for c in T.column_names if pa.types.is_floating(T.schema.field(c).type)]
        kw.update(use_dictionary=[c for c in T.column_names if c not in fl], use_byte_stream_split=fl)
    tmp = dst + '.tmp'
    pq.write_table(T, tmp, **kw)
    t2 = time.perf_counter()
    R = pq.read_table(tmp)

    def same(c):
        if pa.types.is_string(T.schema.field(c).type) or pa.types.is_large_string(T.schema.field(c).type):
            return R[c].to_pylist() == T[c].to_pylist()                  # strings: value equality
        return np.array_equal(np.asarray(R[c]).view(np.uint8), np.asarray(T[c]).view(np.uint8))   # numbers: bit-exact
    ok = R.num_rows == T.num_rows and all(same(c) for c in T.column_names)
    t3 = time.perf_counter()
    if not ok:
        os.unlink(tmp)
        return dict(file=name, ok=False)
    os.replace(tmp, dst)
    return dict(file=name, ok=True, rows=T.num_rows, cols=T.num_columns, fits=os.path.getsize(src),
                parquet=os.path.getsize(dst), read_s=t1 - t0, write_s=t2 - t1, check_s=t3 - t2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('dst')
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--report')
    a = ap.parse_args()
    jobs = []
    for f in sorted(glob.glob(os.path.join(a.src, '*.fits.gz')) + glob.glob(os.path.join(a.src, 'trees', '*.fits.gz'))):
        rel = os.path.relpath(f, a.src)
        d = os.path.join(a.dst, rel[:-len('.fits.gz')] + '.parquet')
        if os.path.exists(d) and os.path.getmtime(d) >= os.path.getmtime(f):
            continue
        os.makedirs(os.path.dirname(d), exist_ok=True)
        jobs.append((f, d))
    jobs.sort(key=lambda j: -os.path.getsize(j[0]))          # largest (the tree file) first
    t0 = time.perf_counter()
    out = []
    with ProcessPoolExecutor(a.procs) as ex:
        for r in ex.map(convert, *zip(*jobs)) if jobs else []:
            out.append(r)
            if not r['ok']:
                print('MISMATCH', r['file'], flush=True)
    wall = time.perf_counter() - t0
    bad = [r for r in out if not r['ok']]
    good = [r for r in out if r['ok']]
    print('%d files converted, %d failed, wall %.1f s with %d processes' % (len(good), len(bad), wall, a.procs))
    if a.report:
        with open(a.report, 'w') as f:
            json.dump(dict(wall_s=wall, procs=a.procs, files=out), f, indent=0)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
