"""
halo_table_io.py: the value-added halo tables out_<snap>.vahc, .tens and .vpot (vahc_fast.py), as text or binary.

Text (as before): '#' header lines (line 2 = column names), then one row per halo, values space separated.
Binary (same file name): the same '#' header lines, then one line
    #SAHYADRI-BINARY v1 rows=<n> dtype=<JSON list of [name, numpy type]>
and the rows as packed little-endian records (numpy structured array, tobytes()). Values are those the text
would hold: the text writes str() of the same float32/int64 numbers (the shortest string that reads back to the
same float32), so the FITS tables made from either are identical.

  write(path, arr, header, binary=True)   via <path>.part and rename (a killed job leaves no complete-looking file)
  header_lines(path)       the '#' lines before the data (without the #SAHYADRI-BINARY line)
  names(path)              column names (header line 2)
  is_binary(path)
  read(path)               float64 array (rows, columns), as np.loadtxt(path, ndmin=2)
  read_records(path)       structured array (binary: memory map; text: parsed with the header names)
  iter_chunks(path, chunk) float64 arrays of <= chunk rows, in order
  column(path, name)       one column (binary: without reading the others)
"""
import json
import os
import re

import numpy as np

MAGIC = b'#SAHYADRI-BINARY'


def _scan(path):
    """(header lines as str, binary info dict or None, data offset in bytes)."""
    lines, info = [], None
    with open(path, 'rb') as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith(b'#'):
                return lines, info, pos
            if line.startswith(MAGIC):
                m = re.match(r'#SAHYADRI-BINARY v1 rows=(\d+) dtype=(.*)$', line.decode().rstrip('\n'))
                if not m:
                    raise ValueError('%s: unknown binary header %r' % (path, line[:80]))
                info = dict(rows=int(m.group(1)), dtype=np.dtype([(n, t) for n, t in json.loads(m.group(2))]))
                return lines, info, f.tell()
            lines.append(line.decode('utf-8', 'replace').rstrip('\n'))


def header_lines(path):
    """The ``#`` header lines of a halo table (text or binary)."""
    return _scan(path)[0]


def names(path):
    """Column names from the second header line."""
    h = header_lines(path)
    return h[1].lstrip('#').split()


def is_binary(path):
    """True if the table is in the binary format (header line with row count and dtype)."""
    return _scan(path)[1] is not None


def _memmap(path):
    """Memory map of the rows of a binary table, or None for a text table; raises ValueError if the file is
    shorter than its header says.
    """
    lines, info, off = _scan(path)
    if info is None:
        return None
    need = off + info['rows'] * info['dtype'].itemsize
    size = os.path.getsize(path)
    if size != need:
        raise ValueError('%s: %d bytes, the header says %d (truncated or damaged)' % (path, size, need))
    if info['rows'] == 0:
        return np.zeros(0, info['dtype'])
    return np.memmap(path, dtype=info['dtype'], mode='r', offset=off, shape=(info['rows'],))


def _to_float(rec):
    """Structured array -> 2-D float64 array (one column per field)."""
    out = np.empty((rec.size, len(rec.dtype.names)))
    for j, n in enumerate(rec.dtype.names):
        out[:, j] = rec[n]
    return out


def read(path):
    """The whole table as a 2-D float64 array (text: ``np.loadtxt``; binary: memory map)."""
    m = _memmap(path)
    if m is None:
        return np.loadtxt(path, comments='#', ndmin=2)
    return _to_float(m)


def read_records(path):
    """The whole table as a structured array with the column names (binary: the memory map itself)."""
    m = _memmap(path)
    if m is not None:
        return m
    nm = names(path)
    a = np.loadtxt(path, comments='#', ndmin=2)
    out = np.zeros(len(a), [(n, 'i8' if n == 'haloID' or n.startswith('Np_') else 'f4') for n in nm])
    for j, n in enumerate(nm):
        out[n] = a[:, j]
    return out


def iter_chunks(path, chunk):
    """Yield the table as 2-D float64 arrays of at most ``chunk`` rows."""
    m = _memmap(path)
    if m is None:
        import pandas as pd
        for c in pd.read_csv(path, sep=r'\s+', comment='#', header=None, dtype=np.float64, engine='c', chunksize=chunk):
            yield c.values
        return
    for i in range(0, m.size, chunk):
        yield _to_float(m[i:i + chunk])


def column(path, name):
    """One column by name as a 1-D array (binary: from the memory map; text: read with pandas)."""
    m = _memmap(path)
    if m is not None:
        return np.asarray(m[name])
    j = names(path).index(name)
    import pandas as pd
    return pd.read_csv(path, sep=r'\s+', comment='#', header=None, usecols=[j], engine='c')[j].to_numpy()


def write(path, arr, header, binary=True, step=200000):
    """header: the '#' lines (one string, each line ending in '\\n'; line 2 = column names, as many as arr has
    fields). Text rows: str() of every value, as AddValue wrote them, formatted step rows at a time (bounded
    memory). Binary: the records under the header names, little endian."""
    nm = header.splitlines()[1].lstrip('#').split()
    if len(nm) != len(arr.dtype.names):
        raise ValueError('%s: %d header names, %d columns' % (path, len(nm), len(arr.dtype.names)))
    part = path + '.part'
    with open(part, 'wb') as f:
        f.write(header.encode())
        if binary:
            dt = np.dtype([(n, arr.dtype[i].newbyteorder('<') if arr.dtype[i].byteorder != '|' else arr.dtype[i])
                           for i, n in enumerate(nm)])
            f.write(b'%s v1 rows=%d dtype=%s\n' % (MAGIC, arr.size,
                                                   json.dumps([[n, dt[n].str] for n in dt.names]).encode()))
            for i in range(0, arr.size, step):
                c = arr[i:i + step]
                out = np.empty(c.size, dt)
                for a, b in zip(arr.dtype.names, nm):
                    out[b] = c[a]
                f.write(out.tobytes())
        else:
            for i in range(0, arr.size, step):
                cols = [arr[n][i:i + step].astype(str) for n in arr.dtype.names]
                f.write(''.join(' '.join(r) + '\n' for r in zip(*cols)).encode())
    os.replace(part, path)
    return path
