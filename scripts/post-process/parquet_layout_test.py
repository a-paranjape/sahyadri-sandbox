#!/usr/bin/env python3
"""Second Parquet test: float encodings that the first test (parquet_vs_fits_test.py) missed, one combined table per
snapshot against separate files, and DuckDB queries on the files.

usage: parquet_layout_test.py <out_dir> <dir with out_<snap>_{basic,extended,tens,vahc}.fits.gz and tree_*.fits.gz>
  1. encodings, per table of snapshot 100 and the tree file: zstd 3 with dictionary (pyarrow's default, as in the first
     test), zstd 3 with no dictionary for float columns, and that plus BYTE_STREAM_SPLIT for float columns (pyarrow
     uses BYTE_STREAM_SPLIT only for columns not dictionary-encoded); bit-exact round trip of every column
  2. layout, snapshots 100 and 50: the four tables as separate files against one file with all columns (rows aligned
     by id, as compress_halo_outputs.py writes them; checked), best encoding of step 1
  3. DuckDB (if importable): queries on the files, wall clock, best of 3 (warm page cache)
"""
import glob
import os
import sys
import time

import fitsio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TABLES = ['basic', 'extended', 'tens', 'vahc']


def best(f, n=3):
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); t.append(time.perf_counter() - t0)
    return min(t), r


def native(a):
    return a.astype(a.dtype.newbyteorder('='))


def table(path):
    D = fitsio.read(path, ext=1)
    return pa.table({c: native(D[c]) for c in D.dtype.names}), D


def write(T, f, mode, level=3):
    fl = [c for c in T.column_names if pa.types.is_floating(T.schema.field(c).type)]
    kw = dict(compression='zstd', compression_level=level, row_group_size=1 << 20, write_statistics=True)
    if mode == 'dict':
        pq.write_table(T, f, **kw)
    elif mode == 'nodict':
        pq.write_table(T, f, use_dictionary=[c for c in T.column_names if c not in fl], **kw)
    elif mode == 'nodict_split':
        pq.write_table(T, f, use_dictionary=[c for c in T.column_names if c not in fl], use_byte_stream_split=fl, **kw)
    return os.path.getsize(f)


def exact(T, f):
    R = pq.read_table(f)
    return all(np.array_equal(np.asarray(R[c]).view(np.uint8), np.asarray(T[c]).view(np.uint8)) for c in T.column_names)


def main():
    out, d = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    print('== 1. encodings (MB; write s)')
    files = [os.path.join(d, 'out_100_%s.fits.gz' % t) for t in TABLES] + sorted(glob.glob(os.path.join(d, 'tree_*.fits.gz')))
    for path in files:
        name = os.path.basename(path).replace('.fits.gz', '')
        T, _ = table(path)
        row = ['%-18s fits.gz %8.2f' % (name, os.path.getsize(path) / 1e6)]
        for mode in ('dict', 'nodict', 'nodict_split'):
            f = os.path.join(out, '%s.%s.parquet' % (name, mode))
            t0 = time.perf_counter(); size = write(T, f, mode); tw = time.perf_counter() - t0
            row.append('%s %8.2f (%.1f s%s)' % (mode, size / 1e6, tw, '' if exact(T, f) else ', NOT EXACT'))
        print(' | '.join(row), flush=True)
    print('== 2. layout: separate files vs one file per snapshot (zstd 3, no dictionary, split floats)')
    for snap in (100, 50):
        parts, ids = [], None
        sep = 0
        for t in TABLES:
            p = os.path.join(d, 'out_%d_%s.fits.gz' % (snap, t))
            if not os.path.isfile(p):
                continue
            T, D = table(p)
            idc = 'id' if 'id' in T.column_names else ('haloID' if 'haloID' in T.column_names else None)
            if ids is None:
                ids = np.asarray(T[idc]) if idc else None
            elif idc is not None and not np.array_equal(np.asarray(T[idc]), ids):
                print('  snapshot %d: %s rows not aligned with basic by %s' % (snap, t, idc))
            f = os.path.join(out, 'out_%d_%s.parquet' % (snap, t))
            sep += write(T, f, 'nodict_split')
            parts.append((t, T))
        if not parts:
            continue
        cols, seen = {}, set()
        for t, T in parts:
            for c in T.column_names:
                k = c if c not in seen else '%s_%s' % (t, c)
                if c in seen and np.array_equal(np.asarray(T[c]), np.asarray(cols.get(c, T[c]))):
                    continue               # the same column in two tables (e.g. the id): keep one
                cols[k] = T[c]; seen.add(c)
        C = pa.table(cols)
        fc = os.path.join(out, 'out_%d_all.parquet' % snap)
        comb = write(C, fc, 'nodict_split')
        print('  snapshot %d: %d tables, %d columns in all; separate %.2f MB, combined %.2f MB; exact %s' % (
            snap, len(parts), C.num_columns, sep / 1e6, comb / 1e6, exact(C, fc)), flush=True)
    try:
        import duckdb
    except ImportError:
        print('== 3. DuckDB not importable: skipped')
        return 0
    print('== 3. DuckDB %s (s, best of 3)' % duckdb.__version__)
    con = duckdb.connect()
    o = out.rstrip('/')
    mcol = 'Mvir'
    Q = [
        ('count Mvir > 1e14, snapshot 100 (one table)',
         "select count(*) from '%s/out_100_basic.parquet' where %s > 1e14" % (o, mcol)),
        ('the same over all snapshots (glob, snapshot from the file name)',
         "select regexp_extract(filename, 'out_(\\d+)_basic', 1)::int as snap, count(*) n, max(%s) mmax "
         "from read_parquet('%s/out_*_basic.parquet', filename=true) where %s > 1e14 group by snap order by snap"
         % (mcol, o, mcol)),
        ('join basic x vahc on id = haloID, Mvir > 1e14 (separate files)',
         "select count(*), avg(v.lam1_R2R200b) from '%s/out_100_basic.parquet' b join '%s/out_100_vahc.parquet' v "
         "on b.id = v.haloID where b.%s > 1e14" % (o, o, mcol)),
        ('the same from the combined file (no join)',
         "select count(*), avg(lam1_R2R200b) from '%s/out_100_all.parquet' where %s > 1e14" % (o, mcol)),
        ('mass function: 20 log bins (combined file)',
         "select floor(log10(%s)*4)/4 as lm, count(*) from '%s/out_100_all.parquet' group by lm order by lm" % (mcol, o)),
    ]
    tf = sorted(glob.glob(os.path.join(o, 'tree_*.nodict_split.parquet')))
    if tf:
        Q += [('trees: one tree (Tree_root_ID of the most massive z=0 halo), %d M rows' % (pq.ParquetFile(tf[0]).metadata.num_rows // 10**6),
               "with r as (select Tree_root_ID t from '%s' where scale = (select max(scale) from '%s') order by Mvir desc limit 1) "
               "select count(*) from '%s' where Tree_root_ID = (select t from r)" % (tf[0], tf[0], tf[0]))]
    for name, sql in Q:
        try:
            t, r = best(lambda: con.execute(sql).fetchall())
            print('  %-62s %7.3f s  -> %s' % (name, t, str(r[:4])[:90]))
        except Exception as e:  # noqa: BLE001
            print('  %-62s failed: %s' % (name, str(e).splitlines()[0][:120]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
