#!/usr/bin/env python3
"""Test: the pipeline's FITS halo tables (compress_halo_outputs.py, .fits.gz) rewritten as Parquet. For each file:
sizes, write time, read times (whole table, two columns, a mass cut), and a bit-exact round trip of every column.

usage: parquet_vs_fits_test.py <out_dir> <file.fits.gz> [...]
Parquet variants (pyarrow): zstd level 3; zstd 3 + BYTE_STREAM_SPLIT for float columns; zstd 9 + split; gzip 6
(the same codec as .fits.gz). Row groups of 1M rows. Header cards and the PROVENANCE HDU go into the schema metadata
(key 'fits_header_<hdu>' = JSON of the cards, 'provenance' = JSON of the PROVENANCE table), so nothing is lost.
Timings: wall clock, best of 3, files read from the file system just written to (page cache warm for both).
"""
import json
import os
import sys
import time

import fitsio
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

VARIANTS = [('zstd3', dict(compression='zstd', compression_level=3), False),
            ('zstd3_split', dict(compression='zstd', compression_level=3), True),
            ('zstd9_split', dict(compression='zstd', compression_level=9), True),
            ('gzip6', dict(compression='gzip', compression_level=6), False)]


def best(f, n=3):
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); t.append(time.perf_counter() - t0)
    return min(t), r


def header_json(h):
    return json.dumps([[r['name'], str(r['value']), r.get('comment', '')] for r in h.records() if r['name']])


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    print('%-34s %-12s %9s %8s %8s %8s %8s  %s' % ('file', 'format', 'MB', 'write s', 'read all', 'read 2c', 'cut', 'round trip'))
    for path in sys.argv[2:]:
        name = os.path.basename(path).replace('.fits.gz', '')
        with fitsio.FITS(path) as F:
            hdu = 1
            cols = F[hdu].get_colnames()
            meta = {'fits_header_0': header_json(F[0].read_header()), 'fits_header_1': header_json(F[hdu].read_header())}
            if 'PROVENANCE' in [h.get_extname() for h in F]:
                P = F['PROVENANCE'].read()
                meta['provenance'] = json.dumps({n: [str(v) for v in P[n]] for n in P.dtype.names})
        mcol = next((c for c in ('Mvir', 'mvir', 'M200b', 'Mvir_all') if c in cols), cols[1])
        two = [c for c in (mcol, 'x', 'X') if c in cols][:2] or cols[:2]
        tf, D = best(lambda: fitsio.read(path, ext=hdu), 1)
        tf2, _ = best(lambda: fitsio.read(path, ext=hdu, columns=two))
        thr = np.nanpercentile(D[mcol], 99)
        tfc, _ = best(lambda: (lambda d: d[d[mcol] > thr])(fitsio.read(path, ext=hdu)), 1)
        print('%-34s %-12s %9.2f %8s %8.2f %8.2f %8.2f  %d rows, %d columns' % (
            name, 'fits.gz', os.path.getsize(path) / 1e6, '-', tf, tf2, tfc, len(D), len(cols)))
        T = pa.table({c: D[c].astype(D[c].dtype.newbyteorder('=')) for c in cols})
        T = T.replace_schema_metadata({k: v.encode() for k, v in meta.items()})
        for vname, kw, split in VARIANTS:
            f = os.path.join(out, '%s.%s.parquet' % (name, vname))
            fl = [c for c in cols if D[c].dtype.kind == 'f']
            tw, _ = best(lambda: pq.write_table(T, f, row_group_size=1 << 20, use_byte_stream_split=fl if split else False,
                                                write_statistics=True, **kw), 1)
            ta, R = best(lambda: pq.read_table(f))
            t2, _ = best(lambda: pq.read_table(f, columns=two))
            tc, _ = best(lambda: pq.read_table(f, filters=[(mcol, '>', float(thr))]))
            ok = all(np.array_equal(np.asarray(R[c]).view(np.uint8), np.ascontiguousarray(D[c].astype(D[c].dtype.newbyteorder('='))).view(np.uint8))
                     for c in cols)
            md = R.schema.metadata or {}
            okm = all(md.get(k.encode()) == v.encode() for k, v in meta.items())
            print('%-34s %-12s %9.2f %8.2f %8.2f %8.2f %8.2f  %s' % (
                '', vname, os.path.getsize(f) / 1e6, tw, ta, t2, tc,
                ('bit-exact' if ok else 'DIFFERS') + (', metadata kept' if okm else ', METADATA LOST')))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
