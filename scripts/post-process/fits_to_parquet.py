#!/usr/bin/env python3
"""Stage parquet: a Parquet copy of a run's compressed FITS halo tables and trees, in a parallel directory with the
same names (FITS stays the reference; the Parquet directory can be dropped without touching it).

  <halo>/compressed/out_<snap>_<table>.fits.gz  ->  <halo>/parquet/out_<snap>_<table>.parquet
  <halo>/compressed/trees/<name>.fits.gz        ->  <halo>/parquet/trees/<name>.parquet

Memory stays low at any run size: each .fits.gz is decompressed as a stream (pyarrow's zlib) and the binary table
converted in row blocks of about --block-mb MB, each written as one Parquet row group; nothing holds a whole file.
Within a file, decompression, decoding/hashing and Parquet encoding overlap (threads; pipeline()).

Checks, per file (a failure leaves no .parquet behind and the exit code is 1):
  * the FITS data unit as read (decompressed bytes) against the table's DATASUM card (FITS checksum convention), so
    the stream was read completely and exactly;
  * the Parquet file read back row group by row group: per-column BLAKE2b hashes of the values equal those of the
    values converted from FITS (numbers bit for bit, strings by value), row and column counts equal NAXIS2/TFIELDS.
The FITS column decoding itself (TFORM/TZERO to types) is the one checked against fitsio in the tests (--selftest).

Metadata kept in the Parquet key-value store (footer): the primary and table headers (all cards, JSON), the
PROVENANCE table (header and rows, JSON), and the source file's name, size and mtime; a file whose .parquet records the current source size and
mtime is skipped (restart = rerun). .parquet files without a FITS source any more are removed.

Encodings (parquet_layout_test.py, L500N512 snapshot 100 and the tree file): zstd level 3, statistics on;
dictionary encoding for basic, extended and trees; for tens and vahc no dictionary on float columns and
BYTE_STREAM_SPLIT (smaller there).

A JSON line per file is appended to <parquet dir>/conversion.jsonl (rows, sizes, seconds, check results).

usage: fits_to_parquet.py <compressed dir> <parquet dir> [--procs N] [--block-mb MB] [--threads T]
       fits_to_parquet.py --selftest <fits.gz> [...]      (decoder against fitsio, whole files: small files only)
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BLOCK = 2880
SPLIT_TABLES = ('tens', 'vahc')
TYPES = dict(L='u1', B='u1', I='>i2', J='>i4', K='>i8', E='>f4', D='>f8')
UNSIGNED = {('I', 32768): ('>u2', 1 << 15), ('J', 2147483648): ('>u4', 1 << 31),
            ('K', 9223372036854775808): ('>u8', 1 << 63)}


# ------------------------------------------------------------------------------------------------ FITS stream
class Stream:
    """A .fits.gz (or .fits) read sequentially, with the FITS checksum of every byte read since mark()."""

    def __init__(self, path):
        raw = pa.OSFile(path, 'rb')
        self.f = pa.CompressedInputStream(raw, 'gzip') if path.endswith('.gz') else raw
        self.sum, self.tail = 0, b''

    def read(self, n, checksum=False):
        out = []
        left = n
        while left > 0:
            b = self.f.read(min(left, 64 << 20))
            if not b:
                raise EOFError('unexpected end of file (%d bytes short)' % left)
            out.append(b)
            left -= len(b)
        data = b''.join(bytes(x) if not isinstance(x, bytes) else x for x in out)
        if checksum:
            self._add(data)
        return data

    def mark(self):
        self.sum, self.tail = 0, b''

    def _add(self, data):
        d = self.tail + data
        k = len(d) - len(d) % 4
        self.tail = d[k:]
        if k:
            s = int(np.frombuffer(d[:k], dtype='>u4').sum(dtype=np.uint64))
            self.sum += s

    def datasum(self):
        s = self.sum
        if self.tail:
            s += int.from_bytes(self.tail + b'\0' * (4 - len(self.tail)), 'big')
        while s >> 32:
            s = (s & 0xffffffff) + (s >> 32)
        return s

    def close(self):
        self.f.close()


def card_value(card):
    if card[8:10] != '= ':
        return None
    v = card[10:].strip()
    if v.startswith("'"):
        m = re.match(r"'((?:[^']|'')*)'", v)
        return m.group(1).replace("''", "'").rstrip() if m else v
    v = v.split('/')[0].strip()
    if v in ('T', 'F'):
        return v == 'T'
    try:
        return int(v)
    except ValueError:
        try:
            return float(v.replace('D', 'E'))
        except ValueError:
            return v


def read_header(s):
    cards, H = [], {}
    while True:
        blk = s.read(BLOCK).decode('ascii', 'replace')
        for i in range(0, BLOCK, 80):
            c = blk[i:i + 80]
            k = c[:8].strip()
            if k == 'END':
                return cards, H
            if k or c.strip():
                cards.append(c.rstrip())
            if k and c[8:10] == '= ':
                H[k] = card_value(c)


def data_bytes(H):
    if H.get('NAXIS', 0) == 0:
        return 0
    n = abs(H['BITPIX']) // 8
    for i in range(1, H['NAXIS'] + 1):
        n *= H['NAXIS%d' % i]
    return n * H.get('GCOUNT', 1) + H.get('PCOUNT', 0)


def padded(n):
    return (n + BLOCK - 1) // BLOCK * BLOCK


def columns(H):
    """[(name, numpy dtype of the raw field, offset, decode spec)] of a BINTABLE header."""
    out, off = [], 0
    for i in range(1, H['TFIELDS'] + 1):
        name = H.get('TTYPE%d' % i) or 'col%d' % i
        m = re.match(r'^(\d*)([LXBIJKAEDCMPQ])', str(H['TFORM%d' % i]).strip())
        if not m:
            raise ValueError('TFORM%d %r not understood' % (i, H['TFORM%d' % i]))
        r, t = int(m.group(1) or 1), m.group(2)
        if t not in TYPES and t != 'A':
            raise ValueError('column %s: TFORM %s not supported (X/C/M/P/Q)' % (name, H['TFORM%d' % i]))
        zero, scale = H.get('TZERO%d' % i, 0), H.get('TSCAL%d' % i, 1)
        if t == 'A':
            dt, kind = np.dtype('S%d' % r), ('str', None)
        else:
            base = TYPES[t]
            kind = ('bool', None) if t == 'L' else ('num', None)
            if (t, zero) in UNSIGNED and scale == 1:
                base, flip = UNSIGNED[(t, zero)]
                kind = ('unsigned', flip)
            elif t == 'B' and zero == -128 and scale == 1:
                kind = ('int8', None)
            elif zero != 0 or scale != 1:
                raise ValueError('column %s: TZERO %s / TSCAL %s not supported' % (name, zero, scale))
            dt = np.dtype((base, (r,))) if r > 1 else np.dtype(base)
        out.append((name, dt, off, kind, r))
        off += dt.itemsize
    if off != H['NAXIS1']:
        raise ValueError('row size %d from TFORMs, NAXIS1 %d' % (off, H['NAXIS1']))
    return out


def decode(raw, cols, rowbytes):
    """Raw rows (bytes) -> {name: pyarrow array}."""
    rec = np.frombuffer(raw, dtype=np.dtype(dict(names=[c[0] for c in cols], formats=[c[1] for c in cols],
                                                 offsets=[c[2] for c in cols], itemsize=rowbytes)))
    out = {}
    for name, dt, _off, (kind, flip), r in cols:
        a = rec[name]
        if kind == 'str':
            v = np.char.rstrip(np.ascontiguousarray(a), b' ')
            out[name] = pa.array(np.char.decode(v, 'ascii'), type=pa.string()) if len(v) else pa.array([], pa.string())
            continue
        if kind == 'bool':
            a = (a == ord('T'))
        elif kind == 'unsigned':
            a = a.astype(a.dtype.newbyteorder('=')) ^ a.dtype.type(flip)
        elif kind == 'int8':
            a = (a.astype(np.int16) - 128).astype(np.int8)
        else:
            a = a.astype(a.dtype.newbyteorder('='))
        a = np.ascontiguousarray(a)
        if r > 1:
            out[name] = pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1)), r)
        else:
            out[name] = pa.array(a)
    return out


def table_rows(path):
    """(rows, columns) of HDU 1 from the headers only: decompresses the first blocks, not the file (verify)."""
    s = Stream(path)
    try:
        _c, H0 = read_header(s)
        s.read(padded(data_bytes(H0)))
        _c, H = read_header(s)
        return H['NAXIS2'], H['TFIELDS']
    finally:
        s.close()


# ------------------------------------------------------------------------------------------------ hashing
def col_hash_update(h, arr):
    """Canonical bytes of a column chunk: numbers as little-endian values, strings as length + UTF-8, so the hash does
    not depend on how the rows are split into chunks."""
    if isinstance(arr, pa.ChunkedArray):
        for c in arr.chunks:
            col_hash_update(h, c)
        return
    t = arr.type
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        for v in arr.to_pylist():
            b = v.encode('utf-8')
            h.update(len(b).to_bytes(4, 'little')); h.update(b)
    elif pa.types.is_fixed_size_list(t):
        col_hash_update(h, arr.flatten())
    elif pa.types.is_boolean(t):
        h.update(np.asarray(arr.to_numpy(zero_copy_only=False), dtype=np.uint8).tobytes())
    else:
        h.update(np.ascontiguousarray(arr.to_numpy(zero_copy_only=False)).tobytes())


class _Stopped(Exception):
    pass


def pipeline(s, w, cols, rowbytes, nrows, block, schema, hashes, threads):
    """The row blocks of one file through three overlapping steps (the C code of each releases the GIL):
    reader thread (gzip decompression + FITS checksum) -> this thread (decode, byte order, per-column hashes on
    `threads` threads) -> writer thread (Parquet encoding + zstd). Queues of one block each (memory: ~5 blocks).
    Blocks keep their order in every step, so the file, the hashes and the checksum are those of the serial loop.
    An exception in any step stops the others and is raised here."""
    q_raw, q_tab = queue.Queue(1), queue.Queue(1)
    stop, err = threading.Event(), []

    def put(q, x):
        while True:
            try:
                q.put(x, timeout=0.2)
                return True
            except queue.Full:
                if stop.is_set():
                    return False

    def get(q):
        while True:
            try:
                return q.get(timeout=0.2)
            except queue.Empty:
                if stop.is_set():
                    raise _Stopped()

    def fail(e):
        err.append(e)
        stop.set()

    def reader():
        try:
            done = 0
            while done < nrows:
                n = min(block, nrows - done)
                if not put(q_raw, (n, s.read(n * rowbytes, checksum=True))):
                    return
                done += n
            put(q_raw, None)
        except BaseException as e:  # noqa: BLE001
            fail(e)

    def writer():
        try:
            while True:
                x = get(q_tab)
                if x is None:
                    return
                w.write_table(x[1], row_group_size=x[0])
        except _Stopped:
            pass
        except BaseException as e:  # noqa: BLE001
            fail(e)

    th = [threading.Thread(target=reader, daemon=True), threading.Thread(target=writer, daemon=True)]
    for t in th:
        t.start()
    try:
        with ThreadPoolExecutor(max(1, threads)) as pool:
            while True:
                x = get(q_raw)
                if x is None:
                    break
                n, raw = x
                cols_ = decode(raw, cols, rowbytes)
                del raw, x
                list(pool.map(lambda c: col_hash_update(hashes[c], cols_[c]), cols_))
                if not put(q_tab, (n, pa.table(cols_, schema=schema))):
                    break
                del cols_
        put(q_tab, None)
    except _Stopped:
        pass
    except BaseException as e:  # noqa: BLE001
        fail(e)
    finally:
        if err:
            stop.set()
        for t in th:
            t.join()
    if err:
        raise err[0]


# ------------------------------------------------------------------------------------------------ one file
def convert(src, dst, block_mb, threads=4):
    t0 = time.time()
    name = os.path.basename(src)
    s = Stream(src)
    try:
        cards0, H0 = read_header(s)
        s.read(padded(data_bytes(H0)))
        cards1, H = read_header(s)
        if H.get('XTENSION') != 'BINTABLE':
            raise ValueError('HDU 1 is %s, not BINTABLE' % H.get('XTENSION'))
        if H.get('PCOUNT', 0):
            raise ValueError('heap (variable-length columns) not supported')
        cols = columns(H)
        nrows, rowbytes = H['NAXIS2'], H['NAXIS1']
        block = max(1, min(1 << 20, (block_mb << 20) // max(1, rowbytes)))
        split = any('_%s.' % t in name for t in SPLIT_TABLES)
        fields, hashes = [], {}
        for cname, dt, _o, (kind, _f), r in cols:
            if kind == 'str':
                ty = pa.string()
            else:
                base = dt.base if r > 1 else dt
                ty = pa.bool_() if kind == 'bool' else (pa.int8() if kind == 'int8' else pa.from_numpy_dtype(base.newbyteorder('=')))
                if r > 1:
                    ty = pa.list_(ty, r)
            fields.append(pa.field(cname, ty))
            hashes[cname] = hashlib.blake2b(digest_size=16)
        floats = [f.name for f in fields if pa.types.is_floating(f.type)]
        kw = dict(compression='zstd', compression_level=3, write_statistics=True)
        if split:
            kw.update(use_dictionary=[f.name for f in fields if f.name not in floats], use_byte_stream_split=floats)
        st = os.stat(src)
        meta = {'fits_primary_header': json.dumps(cards0), 'fits_table_header': json.dumps(cards1),
                'source': name, 'source_size': str(st.st_size), 'source_mtime': repr(st.st_mtime),
                'converter': 'fits_to_parquet.py'}
        schema = pa.schema(fields)
        part = dst + '.part'
        w = pq.ParquetWriter(part, schema.with_metadata(meta), **kw)
        s.mark()
        try:
            pipeline(s, w, cols, rowbytes, nrows, block, schema, hashes, threads)
            rest = padded(nrows * rowbytes) - nrows * rowbytes
            if rest:
                s.read(rest, checksum=True)
            fits_sum = s.datasum()
            # the PROVENANCE table (HDU 2, small): read whole, into the metadata
            prov, cards2 = [], []
            try:
                cards2, H2 = read_header(s)
                if H2.get('XTENSION') == 'BINTABLE' and H2.get('NAXIS2', 0):
                    prov = pa.table(decode(s.read(H2['NAXIS2'] * H2['NAXIS1']), columns(H2), H2['NAXIS1'])).to_pylist()
            except EOFError:
                pass
            w.add_key_value_metadata({'fits_provenance_header': json.dumps(cards2),
                                      'fits_provenance': json.dumps(prov, default=str)})
        except BaseException:
            w.close()
            os.unlink(part)
            raise
        w.close()
    finally:
        s.close()
    t1 = time.time()
    want = str(H.get('DATASUM', '')).strip()
    sum_ok = (want == '' or int(want) == fits_sum)
    # read back, row group by row group
    pf = pq.ParquetFile(part)
    back = {c: hashlib.blake2b(digest_size=16) for c in hashes}
    with ThreadPoolExecutor(max(1, threads)) as pool:
        for g in range(pf.num_row_groups):
            T = pf.read_row_group(g)
            list(pool.map(lambda c: col_hash_update(back[c], T[c]), back))
    same = [c for c in hashes if hashes[c].digest() == back[c].digest()]
    ok = sum_ok and len(same) == len(hashes) and pf.metadata.num_rows == nrows and pf.metadata.num_columns == len(cols)
    t2 = time.time()
    res = dict(file=os.path.relpath(src), parquet=dst, ok=ok, rows=nrows, cols=len(cols), fits_bytes=st.st_size,
               datasum=('ok' if want and sum_ok else ('absent' if not want else 'MISMATCH %d != %s' % (fits_sum, want))),
               hash_mismatch=[c for c in hashes if c not in same], read_write_s=round(t1 - t0, 2),
               check_s=round(t2 - t1, 2), row_groups=pf.num_row_groups)
    if not ok:
        os.unlink(part)
        return res
    os.replace(part, dst)
    res['parquet_bytes'] = os.path.getsize(dst)
    return res


def up_to_date(src, dst):
    if not os.path.isfile(dst):
        return False
    try:
        m = pq.read_metadata(dst).metadata or {}
        st = os.stat(src)
        return (m.get(b'source_size') == str(st.st_size).encode() and m.get(b'source_mtime') == repr(st.st_mtime).encode())
    except Exception:  # noqa: BLE001
        return False


def plan(src_dir, dst_dir):
    jobs, want = [], set()
    for f in sorted(glob.glob(os.path.join(src_dir, '*.fits.gz')) + glob.glob(os.path.join(src_dir, 'trees', '*.fits.gz'))):
        rel = os.path.relpath(f, src_dir)[:-len('.fits.gz')] + '.parquet'
        d = os.path.join(dst_dir, rel)
        want.add(os.path.abspath(d))
        if not up_to_date(f, d):
            jobs.append((f, d))
    orphans = [p for p in glob.glob(os.path.join(dst_dir, '*.parquet')) + glob.glob(os.path.join(dst_dir, 'trees', '*.parquet'))
               if os.path.abspath(p) not in want]
    return jobs, orphans, len(want)


def main_convert(a):
    jobs, orphans, n_all = plan(a.src, a.dst)
    for p in orphans:
        print('removing %s (no FITS source)' % p, flush=True)
        os.unlink(p)
    for p in glob.glob(os.path.join(a.dst, '*.part')) + glob.glob(os.path.join(a.dst, 'trees', '*.part')):
        os.unlink(p)                             # left by a killed job
    jobs.sort(key=lambda j: -os.path.getsize(j[0]))          # largest (the tree file) first
    print('%d FITS files, %d to convert (%d up to date), %d processes x (%d hash + reader + writer threads), blocks of %d MB'
          % (n_all, len(jobs), n_all - len(jobs), a.procs, a.threads, a.block_mb), flush=True)
    t0 = time.time()
    bad = 0
    log = os.path.join(a.dst, 'conversion.jsonl')
    os.makedirs(os.path.join(a.dst, 'trees'), exist_ok=True)
    with ProcessPoolExecutor(a.procs) as ex:
        fut = {ex.submit(convert, f, d, a.block_mb, a.threads): f for f, d in jobs}
        for x in as_completed(fut):
            try:
                r = x.result()
            except Exception as e:  # noqa: BLE001
                r = dict(file=fut[x], ok=False, error='%s: %s' % (type(e).__name__, e))
            r['t'] = time.time()
            with open(log, 'a') as f:
                f.write(json.dumps(r) + '\n')
            if not r['ok']:
                bad += 1
                print('FAIL %s' % json.dumps(r), flush=True)
    tot = sum(os.path.getsize(p) for p in glob.glob(os.path.join(a.dst, '*.parquet')) + glob.glob(os.path.join(a.dst, 'trees', '*.parquet')))
    print('converted %d, failed %d, wall %.1f s; parquet dir %.3f GB' % (len(jobs) - bad, bad, time.time() - t0, tot / 1e9))
    return 1 if bad else 0


def selftest(paths):
    import fitsio
    bad = 0
    for p in paths:
        D = fitsio.read(p, ext=1)
        s = Stream(p)
        read_header(s);
        _c, H0 = None, None
        s.close()
        s = Stream(p)
        _c0, H0 = read_header(s); s.read(padded(data_bytes(H0)))
        _c1, H = read_header(s)
        cols = columns(H)
        got = decode(s.read(H['NAXIS2'] * H['NAXIS1']), cols, H['NAXIS1'])
        s.close()
        for c in D.dtype.names:
            a, b = D[c], got[c]
            if a.dtype.kind in 'SU':
                same = [str(x).rstrip() for x in a.astype(str)] == b.to_pylist()
            elif a.ndim > 1:
                same = np.array_equal(a.reshape(len(a), -1), np.asarray(b.flatten()).reshape(len(a), -1))
            else:
                bb = b.to_numpy(zero_copy_only=False)
                same = a.dtype.newbyteorder('=') == bb.dtype and np.array_equal(
                    a.astype(a.dtype.newbyteorder('=')).view(np.uint8), np.ascontiguousarray(bb).view(np.uint8))
            if not same:
                bad += 1
                print('DIFF %s %s: fitsio %s, decoder %s' % (os.path.basename(p), c, a.dtype, b.type))
        print('%s: %d rows, %d columns checked against fitsio' % (os.path.basename(p), len(D), len(D.dtype.names)))
    return 1 if bad else 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--selftest':
        return selftest(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('dst')
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--block-mb', type=int, default=256)
    ap.add_argument('--threads', type=int, default=4, help='per file: column-hash threads (plus a reader and a writer '
                    'thread); 1 = one hash thread, still overlapped')
    return main_convert(ap.parse_args())


if __name__ == '__main__':
    sys.exit(main())
