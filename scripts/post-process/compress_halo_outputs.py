#!/usr/bin/env python3
"""
compress_halo_outputs.py: Rockstar/consistent-trees ASCII outputs -> FITS (.fits.gz) with full metadata.

Replaces ConvertToFITS.convert_fits (halo catalogs) and consistent_trees_compress.py (trees) for the
Sahyadri pipeline (2026-11-11; plan: docs/rockstar_pkc_reader_plan.md, section 3d). Output names, HDU
layout and column names are those of the existing files, so readers.py (prep_halos_fits) works unchanged:

  halos  <indir>/out_<snap>.trees (+ .vahc, .tens)  ->  <outdir>/out_<snap>_{basic,extended,vahc,tens}.fits.gz
         basic    = BASIC_COLS from .trees (+ b1 from .vahc, rows aligned by id == haloID, checked)
         extended = all other .trees columns;  vahc / tens = their columns except BASIC_COLS
  trees  <treedir>/tree_X_Y_Z.dat  ->  <outdir>/tree_X_Y_Z.fits.gz      (rows in file order, HDU "HALOS")
         <treedir>/locations.dat ->  <outdir>/locations.fits.gz       (all 6 columns + FitsRow, FitsRows)
         <treedir>/forests.list  ->  <outdir>/forests.fits.gz         (TreeRootID, ForestID)

Every file: HDU 0 and HDU 1 headers carry the metadata (cosmology, box, scale factor / redshift, snapshot,
particle mass, halo-finder and merger-tree settings, simulation header from a .pkc snapshot, provenance);
HDU 2 "PROVENANCE" holds the full texts (ASCII header block, rockstar.cfg, merger_tree.cfg, .pkc info JSON).

Lossless: integer-valued columns (IDs, counts, flags) are int64 and positions / scale factors float64, so
they read back exactly as printed; all other columns are float32, as computed by Rockstar/consistent-trees
(single precision), i.e. rounding only at the 6e-8 relative level. The fitsio/zlib path is verified after
writing: row count and per-column sums of the stored values must match exactly (--no-verify to skip).

Efficiency: pandas C parser in chunks (bounded memory), files converted in parallel (--nproc), FITS written
uncompressed to --tmpdir (use node-local disk) and then gzip-compressed with pigz if present, else python-isal
if installed, else zlib.

usage:
  compress_halo_outputs.py halos --indir R1 [--outdir R1/compressed] [--snaps 0-100] [--types basic,extended,vahc]
  compress_halo_outputs.py trees --indir R1/trees [--outdir R1/compressed/trees]
  compress_halo_outputs.py all   --indir R1      (halos from R1, trees from R1/trees)
  common: --nproc N --tmpdir DIR --level 6 --rockstar-cfg F --ctrees-cfg F --pkc PREFIX --meta KEY=VALUE ...
          --overwrite --delete-ascii (only after a verified write) --chunk ROWS
"""
import argparse
import datetime
import glob
import gzip
import json
import multiprocessing as mp
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
import zlib

import fitsio
import numpy as np
import pandas as pd

TOOL = "compress_halo_outputs.py 1.0 (Sahyadri, 2026-11-11)"

# same split as ConvertToFITS.out_column_list (pegasus version of 22 Jun 2026)
BASIC_COLS = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'Mvir', 'Mvir_all', 'M200b', 'M200c', 'M500c', 'M2500c', 'Mpeak',
              'Rvir', 'rs', 'vrms', 'id', 'pid', 'T/|U|', 'Vpeak', 'b1']

# dtype policy (names compared in lower case, trailing '?' removed)
INT64_COLS = {
    'id', 'desc_id', 'descid', 'num_prog', 'pid', 'upid', 'desc_pid', 'phantom', 'mmp', 'np', 'num_p',
    'breadth_first_id', 'depth_first_id', 'tree_root_id', 'orig_halo_id', 'snap_idx', 'snap_num',
    'next_coprogenitor_depthfirst_id', 'last_progenitor_depthfirst_id', 'last_mainleaf_depthfirst_id',
    'tidal_id', 'future_merger_mmp_id', 'haloid', 'treerootid', 'forestid', 'fileid', 'offset',
    'num_halos', 'num_bytes'}
FLOAT64_COLS = {'scale', 'desc_scale', 'scale_of_last_mm', 'x', 'y', 'z'}


def col_dtype(name):
    n = name.lower().strip().rstrip('?')
    if n in INT64_COLS:
        return '>i8'
    if n in FLOAT64_COLS:
        return '>f8'
    return '>f4'


def safe_tree_colname(name):
    """Column names of the tree files, as consistent_trees_compress.py made them (no brackets, slashes, '?')."""
    n = re.sub(r'\[', '_', name.strip())
    n = re.sub(r'\]', '', n)
    n = re.sub(r'[/\\ ]', '_', n)
    return n.rstrip('?')


def log(msg):
    print('[%s] %s' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg), flush=True)


# ------------------------------------------------------------------------------------------------------
# metadata


def read_cfg(path):
    """Rockstar / consistent-trees style 'KEY = value' file -> (dict, text)."""
    d = {}
    text = open(path).read()
    for line in text.splitlines():
        line = line.split('#', 1)[0]
        if '=' not in line:
            continue
        k, v = line.split('=', 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k:
            d[k] = v
    return d, text


def read_pkc_header(prefix):
    """Header of one file of a pkdgrav3 compressed snapshot (format version 3): dict + info JSON text."""
    files = sorted(glob.glob(prefix + '.s*.pkc'))
    if not files:
        raise FileNotFoundError('no files %s.s*.pkc' % prefix)
    b = open(files[0], 'rb').read(4096)
    if b[:8] != b'PKCOMP\x00\x01':
        raise ValueError(files[0] + ': not a snapcomp file')
    version = struct.unpack_from('<I', b, 8)[0]
    u64 = lambda o: struct.unpack_from('<Q', b, o)[0]
    f64 = lambda o: struct.unpack_from('<d', b, o)[0]
    h = {'version': version, 'n_total': u64(40), 'lattice_n': u64(32), 'box_size': f64(72), 'redshift': f64(80),
         'omega_m': f64(96), 'omega_l': f64(104), 'h': f64(112), 'vel_step': f64(56), 'step': u64(280)}
    q = struct.unpack_from('<I', b, 48)[0] * struct.unpack_from('<I', b, 52)[0]
    h['pos_step'] = h['box_size'] / q
    info_end = 4000 if version >= 3 else (4080 if version == 2 else 4096)
    if version >= 3:
        for i, k in enumerate(['omega_b', 'omega_rad', 'omega_de', 'w0', 'wa', 'sigma8', 'n_s', 'running',
                               'normalization', 'pivot']):
            h[k] = f64(4000 + 8 * i)
    info = b[288:info_end].split(b'\x00', 1)[0].decode('utf-8', 'replace')
    return h, info


ROCKSTAR_KEYS = ['FILE_FORMAT', 'PARTICLE_MASS', 'BOX_SIZE', 'FORCE_RES', 'FORCE_RES_PHYS_MAX', 'FOF_LINKING_LENGTH',
                 'FOF_FRACTION', 'MIN_HALO_PARTICLES', 'MIN_HALO_OUTPUT_SIZE', 'UNBOUND_THRESHOLD', 'OVERLAP_LENGTH',
                 'MASS_DEFINITION', 'MASS_DEFINITION2', 'MASS_DEFINITION3', 'MASS_DEFINITION4', 'MASS_DEFINITION5',
                 'STRICT_SO_MASSES', 'BOUND_PROPS', 'BOUND_OUT_TO_HALO_EDGE', 'TEMPORAL_HALO_FINDING', 'PERIODIC',
                 'SHAPE_ITERATIONS', 'WEIGHTED_SHAPES', 'INITIAL_METRIC_SCALING', 'INCLUDE_HOST_POTENTIAL_RATIO',
                 'Om', 'Ol', 'h0', 'W0', 'WA', 'PKC_SUBSAMPLES', 'NUM_BLOCKS', 'NUM_WRITERS', 'TOTAL_PARTICLES']


def card(name, value, comment=''):
    if isinstance(value, (bool, np.bool_)):
        value = bool(value)
    elif isinstance(value, (np.integer,)):
        value = int(value)
    elif isinstance(value, (np.floating,)):
        value = float(value)
    elif isinstance(value, str):
        try:  # numbers in config files become numbers in the header
            value = int(value) if re.fullmatch(r'[-+]?\d+', value) else float(value)
        except ValueError:
            pass
    return {'name': name, 'value': value, 'comment': comment}


def build_metadata(args):
    """Run-level header cards and provenance texts (same for every file of the run)."""
    cards, prov = [], []
    cards.append(card('CONVTOOL', TOOL, 'converter'))
    cards.append(card('CONVDATE', datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'UTC'))
    cards.append(card('CONVHOST', socket.gethostname()))
    if args.rockstar_cfg:
        d, text = read_cfg(args.rockstar_cfg)
        prov.append(('rockstar.cfg', text))
        for k in ROCKSTAR_KEYS:
            if k in d:
                cards.append(card('RS_' + k, d[k], 'Rockstar config'))
    if args.ctrees_cfg:
        d, text = read_cfg(args.ctrees_cfg)
        prov.append(('merger_tree.cfg', text))
        for k, v in d.items():
            if not re.search(r'(BASE|FILE|LABELS|DESCRIPTIONS|INTERPOLATIONS)$', k):
                cards.append(card('CT_' + k, v, 'consistent-trees config'))
    if args.pkc:
        h, info = read_pkc_header(args.pkc)
        prov.append(('pkc header info (' + os.path.basename(args.pkc) + ')', info))
        for k, v in h.items():
            if k not in ('redshift', 'step'):  # these belong to one snapshot, not the run
                cards.append(card('SIM_' + k.upper(), v, 'pkdgrav3 .pkc header'))
        try:
            j = json.loads(info)
            for grp, keys in (('sim', ['name', 'set', 'comment', 'code', 'code_git']),
                              ('box', ['n_grid', 'particle_mass_msunh', 'softening_kpch']),
                              ('ic', ['z_initial', 'lpt_order', 'seed', 'fixed_amplitude']),
                              ('cosmology', ['Omega_K', 'H0', 'class', 'tf_file'])):
                for k in keys:
                    if k in j.get(grp, {}):
                        cards.append(card('SIM_%s_%s' % (grp.upper(), k.upper()), j[grp][k], 'pkdgrav3 info JSON'))
        except ValueError:
            pass
    for kv in args.meta or []:
        k, _, v = kv.partition('=')
        cards.append(card(k.strip().upper(), v.strip(), 'user'))
    return cards, prov


def ascii_header(path, max_lines=400):
    """Leading '#' lines of an ASCII file and cosmology cards parsed from them."""
    lines = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            lines.append(line.rstrip('\n'))
            if len(lines) >= max_lines:
                break
    cards = []
    text = '\n'.join(lines)
    for pat, name, com in ((r'Omega_M\s*=\s*([-\d.eE+]+)', 'OMEGA_M', 'from catalog header'),
                           (r'Omega_L\s*=\s*([-\d.eE+]+)', 'OMEGA_L', 'from catalog header'),
                           (r'h0\s*=\s*([-\d.eE+]+)', 'H0', 'H0/100, from catalog header'),
                           (r'w0\s*=\s*([-\d.eE+]+)', 'W0', 'dark energy w0, from catalog header'),
                           (r'wa\s*=\s*([-\d.eE+]+)', 'WA', 'dark energy wa, from catalog header'),
                           (r'[Bb]ox size\s*[=:]\s*([-\d.eE+]+)', 'BOXSIZE', 'Mpc/h, from catalog header'),
                           (r'Particle mass:\s*([-\d.eE+]+)', 'PARTMASS', 'Msun/h, from catalog header'),
                           (r'Consistent Trees Version\s*(\S+)', 'CTVERS', 'consistent-trees version')):
        m = re.search(pat, text)
        if m:
            cards.append(card(name, m.group(1), com))
    return lines, cards


def provenance_table(prov):
    """(source, line) rows for HDU 2."""
    rows = [(src, line) for src, text in prov for line in (text.splitlines() or [''])]
    if not rows:
        rows = [('none', '')]
    ws = max(len(s) for s, _ in rows)
    wl = max(1, max(len(l) for _, l in rows))
    arr = np.zeros(len(rows), dtype=[('source', 'S%d' % ws), ('line', 'S%d' % wl)])
    arr['source'] = [s.encode('utf-8', 'replace') for s, _ in rows]
    arr['line'] = [l.encode('utf-8', 'replace') for _, l in rows]
    return arr


# ------------------------------------------------------------------------------------------------------
# writing


def gzip_file(src, dst, level, threads):
    if shutil.which('pigz'):
        with open(dst, 'wb') as out:
            subprocess.run(['pigz', '-%d' % level, '-p', str(max(1, threads)), '-c', src], stdout=out, check=True)
        return 'pigz'
    try:
        from isal import igzip  # python-isal: several times faster than zlib
        opener, how = (lambda p: igzip.open(p, 'wb', compresslevel=min(level, 3))), 'isal'
    except ImportError:
        opener, how = (lambda p: gzip.open(p, 'wb', compresslevel=level)), 'zlib'
    with open(src, 'rb') as fi, opener(dst) as fo:
        shutil.copyfileobj(fi, fo, length=16 << 20)
    return how


class TableWriter:
    """Writes HDU 0 (header), HDU 1 (table, appended chunk by chunk), HDU 2 (provenance) to an uncompressed
    temporary file, keeping per-column sums of what was written for the verification."""

    def __init__(self, tmp, dtype, extname, cards):
        self.tmp, self.dtype, self.extname, self.cards = tmp, dtype, extname, cards
        self.f = fitsio.FITS(tmp, 'rw', clobber=True)
        self.f.write(None, header=cards)
        self.nrows = 0
        self.sums = {n: 0 for n in dtype.names}
        self.started = False

    def append(self, arr):
        if len(arr) == 0:
            return
        for n in arr.dtype.names:
            if arr.dtype[n].kind == 'S':  # strings: running CRC of the bytes
                self.sums[n] = zlib.crc32(np.ascontiguousarray(arr[n]).tobytes(), self.sums[n])
            elif arr.dtype[n].kind in 'iu':
                self.sums[n] += int(arr[n].astype(np.int64).sum())
            else:
                self.sums[n] += float(arr[n].astype(np.float64).sum())
        if not self.started:
            self.f.write(arr, extname=self.extname, header=self.cards)
            self.started = True
        else:
            self.f[1].append(arr)
        self.nrows += len(arr)

    def close(self, prov_arr, extra_cards=()):
        if not self.started:  # empty table: still write HDU 1 with the right columns
            self.f.write(np.zeros(0, dtype=self.dtype), extname=self.extname, header=self.cards)
        for c in extra_cards:
            for h in (0, 1):
                self.f[h].write_key(c['name'], c['value'], comment=c.get('comment', ''))
        self.f[1].write_key('NROWS', self.nrows, comment='rows in this table')
        self.f.write(prov_arr, extname='PROVENANCE')
        for h in self.f:
            h.write_checksum()
        self.f.close()


def finish(writer, prov_arr, out, args, extra_cards=()):
    """Close, compress, verify, move into place. Returns a summary dict."""
    writer.close(prov_arr, extra_cards)
    part = out + '.part'
    t = time.time()
    how = gzip_file(writer.tmp, part, args.level, args.threads)
    tz = time.time() - t
    os.remove(writer.tmp)
    if not args.no_verify:
        with fitsio.FITS(part) as f:
            n = f[1].get_nrows()
            if n != writer.nrows:
                raise RuntimeError('%s: %d rows read back, %d written' % (out, n, writer.nrows))
            data = f[1].read() if n else np.zeros(0, dtype=writer.dtype)
            for name, s in writer.sums.items():
                col = data[name]
                if writer.dtype[name].kind == 'S':  # fitsio returns strings as unicode
                    back = zlib.crc32(np.ascontiguousarray(col.astype(writer.dtype[name])).tobytes(), 0)
                elif col.dtype.kind in 'iu':
                    back = int(col.astype(np.int64).sum())
                else:
                    back = float(col.astype(np.float64).sum())
                if back != s:
                    raise RuntimeError('%s: column %s does not read back identically' % (out, name))
            for h in range(len(f)):
                f[h].verify_checksum()  # raises ValueError if the FITS checksums do not match
    os.replace(part, out)
    return {'out': out, 'rows': writer.nrows, 'gzip': how, 'gzip_s': tz, 'bytes': os.path.getsize(out)}


# ------------------------------------------------------------------------------------------------------
# halo catalogs


def halo_header_names(path, kind):
    """Column names as ConvertToFITS.get_header_dic makes them: .trees line 1 without '(N)'; .vahc/.tens line 2."""
    with open(path) as fh:
        lines = [fh.readline() for _ in range(2)]
    line = lines[0] if kind == 'trees' else lines[1]
    assert line.startswith('#'), path + ': no column header line'
    toks = line[1:].split()
    if kind != 'trees':
        return toks
    # ConvertToFITS kept the text before the first '(' and, through a dict, silently dropped the first of two
    # equal names (b_to_a, c_to_a, A[x..z] appear for all particles and for (500c)). Here the first keeps the
    # plain name and later ones get the bracket text: b_to_a_500c, A[x]_500c (as readers.py names them).
    names, seen = [], set()
    for t in toks:
        base = t.split('(')[0]
        name = base
        if name in seen:
            inner = re.sub(r'\(\d+\)$', '', t)[len(base):]
            name = base + '_' + re.sub(r'[^0-9A-Za-z]+', '', inner)
            k = 2
            while name in seen:
                name, k = '%s_%d' % (base, k), k + 1
        seen.add(name)
        names.append(name)
    return names


def convert_halo_snapshot(task):
    snap, args, run_cards, run_prov = task
    base = os.path.join(args.indir, 'out_%s' % snap)
    src = {k: base + '.' + k for k in ('trees', 'vahc', 'tens')}
    have = {k: os.path.isfile(p) for k, p in src.items()}
    if not have['trees']:
        return [{'snap': snap, 'error': 'missing ' + src['trees']}]
    want = [t for t in args.types if t in ('basic', 'extended') or have.get(t)]
    todo = [t for t in want if args.overwrite or not os.path.isfile(os.path.join(args.outdir, 'out_%s_%s.fits.gz' % (snap, t)))]
    if not todo:
        return [{'snap': snap, 'skipped': 'exists'}]

    names = {k: halo_header_names(p, k) for k, p in src.items() if have[k]}
    hdr_lines, hdr_cards = ascii_header(src['trees'])
    prov = run_prov + [('out_%s.trees header' % snap, '\n'.join(hdr_lines))]
    if have['vahc']:
        prov.append(('out_%s.vahc header' % snap, '\n'.join(ascii_header(src['vahc'])[0])))
    prov_arr = provenance_table(prov)

    # columns of each output: list of (source kind, column name)
    plan = {}
    for t in todo:
        if t == 'basic':
            cols = [('trees', c) for c in names['trees'] if c in BASIC_COLS]
            if have['vahc']:
                cols += [('vahc', c) for c in names['vahc'] if c in BASIC_COLS and c not in names['trees']]
        elif t == 'extended':
            cols = [('trees', c) for c in names['trees'] if c not in BASIC_COLS]
        else:
            cols = [(t, c) for c in names[t] if c not in BASIC_COLS]
        plan[t] = cols
    sources = sorted({k for cols in plan.values() for k, _ in cols} | ({'vahc'} if 'basic' in todo and have['vahc'] else set()))

    tmpdir = args.tmpdir or args.outdir
    writers = {}
    common = run_cards + hdr_cards + [card('SNAPNUM', int(snap) if snap.isdigit() else snap, 'snapshot index')]
    for t, cols in plan.items():
        dt = np.dtype([(c, col_dtype(c)) for _, c in cols])
        cards = common + [card('CATTYPE', t, 'basic / extended / vahc / tens'),
                          card('SRCFILE', ','.join(sorted({os.path.basename(src[k]) for k, _ in cols})))]
        if t == 'basic':
            cards.append(card('HAS_B1', bool(have['vahc']), 'b1 from .vahc included'))
        tmp = os.path.join(tmpdir, '.tmp_out_%s_%s_%d.fits' % (snap, t, os.getpid()))
        writers[t] = (TableWriter(tmp, dt, 'HALOS', cards), cols)

    readers = {k: pd.read_csv(src[k], sep=r'\s+', comment='#', header=None, dtype=np.float64, engine='c',
                              chunksize=args.chunk) for k in sources}
    t0 = time.time()
    scale = None
    for chunks in zip(*[readers[k] for k in sources]):
        data = {k: c.values for k, c in zip(sources, chunks)}
        n = len(next(iter(data.values())))
        for k in sources:
            if data[k].shape != (n, len(names[k])):
                raise RuntimeError('%s: chunk shape %s, expected (%d, %d)' % (src[k], data[k].shape, n, len(names[k])))
        if 'vahc' in data:  # rows must be the same halos
            if not np.array_equal(data['trees'][:, names['trees'].index('id')], data['vahc'][:, names['vahc'].index('haloID')]):
                raise RuntimeError('out_%s: .trees id and .vahc haloID differ row by row' % snap)
        if scale is None and n and 'scale' in names['trees']:
            scale = float(data['trees'][0, names['trees'].index('scale')])
        for t, (w, cols) in writers.items():
            arr = np.empty(n, dtype=w.dtype)
            for k, c in cols:
                v = data[k][:, names[k].index(c)]
                if arr.dtype[c].kind == 'i':
                    if not np.all(v == np.round(v)):
                        raise RuntimeError('out_%s: column %s is not integer' % (snap, c))
                    arr[c] = v.astype(np.int64)
                else:
                    arr[c] = v
            w.append(arr)
    # a shorter source would end the zip early: check every reader is exhausted
    for k in sources:
        if next(iter(readers[k]), None) is not None:
            raise RuntimeError('%s has more rows than the other sources' % src[k])
    tread = time.time() - t0
    extra = []
    if scale is not None:
        extra = [card('SCALE', scale, 'scale factor of this snapshot'), card('REDSHIFT', 1.0 / scale - 1.0)]
    out = []
    for t, (w, _) in writers.items():
        r = finish(w, prov_arr, os.path.join(args.outdir, 'out_%s_%s.fits.gz' % (snap, t)), args, extra)
        r.update(snap=snap, type=t, read_s=tread)
        out.append(r)
    if args.delete_ascii and not args.no_verify:  # only inputs whose every output now exists (verified)
        done = lambda t: os.path.isfile(os.path.join(args.outdir, 'out_%s_%s.fits.gz' % (snap, t)))
        if done('basic') and done('extended') and (not have['vahc'] or done('vahc')):
            os.remove(src['trees'])
            if have['vahc']:
                os.remove(src['vahc'])
        if have['tens'] and done('tens'):
            os.remove(src['tens'])
    return out


# ------------------------------------------------------------------------------------------------------
# trees


def parse_tree_header(path):
    """Column names (consistent_trees_compress.py naming), header lines, and lines before the data."""
    col_re = re.compile(r'^\w[\w\?\[\]/|]*\(\d+\)')
    cols, lines, nskip, ntrees = None, [], 0, None
    with open(path) as fh:
        for raw in fh:
            s = raw.strip()
            if not s.startswith('#'):
                if s.isdigit():
                    ntrees = int(s)
                    nskip += 1
                break
            nskip += 1
            lines.append(raw.rstrip('\n'))
            content = s[1:].strip()
            if cols is None and col_re.match(content):
                cols = [safe_tree_colname(re.sub(r'\(\d+\)$', '', c)) for c in content.split()]
    return cols, lines, nskip, ntrees


def convert_tree_file(task):
    path, args, run_cards, run_prov = task
    name = os.path.basename(path)
    out = os.path.join(args.outdir, name.replace('.dat', '.fits.gz'))
    if os.path.isfile(out) and not args.overwrite:
        return [{'out': out, 'skipped': 'exists'}]
    cols, lines, nskip, ntrees = parse_tree_header(path)
    if not cols:
        return [{'out': out, 'error': 'no column header in ' + name}]
    _, hdr_cards = ascii_header(path)
    m = re.search(r'tree_(\d+)_(\d+)_(\d+)\.dat', name)
    tile = [int(g) for g in m.groups()] if m else [-1, -1, -1]
    dt = np.dtype([(c, col_dtype(c)) for c in cols])
    cards = run_cards + hdr_cards + [card('SRCFILE', name), card('CHUNK_X', tile[0]), card('CHUNK_Y', tile[1]),
                                     card('CHUNK_Z', tile[2]), card('NTREES', ntrees if ntrees is not None else -1,
                                                                    'trees in the ASCII file')]
    prov_arr = provenance_table(run_prov + [(name + ' header', '\n'.join(lines))])
    tmp = os.path.join(args.tmpdir or args.outdir, '.tmp_%s_%d.fits' % (name, os.getpid()))
    w = TableWriter(tmp, dt, 'HALOS', cards)
    itree = cols.index('Tree_root_ID') if 'Tree_root_ID' in cols else None
    roots, starts = [], []
    last = None
    t0 = time.time()
    # '#tree <id>' separator lines are comments; the tree a row belongs to is in Tree_root_ID
    for chunk in pd.read_csv(path, sep=r'\s+', comment='#', header=None, skiprows=nskip, dtype=np.float64,
                             engine='c', chunksize=args.chunk):
        v = chunk.values
        if v.shape[1] != len(cols):
            raise RuntimeError('%s: %d columns, header has %d' % (name, v.shape[1], len(cols)))
        arr = np.empty(len(v), dtype=dt)
        for j, c in enumerate(cols):
            if dt[c].kind == 'i':
                if not np.all(v[:, j] == np.round(v[:, j])):
                    raise RuntimeError('%s: column %s is not integer' % (name, c))
                arr[c] = v[:, j].astype(np.int64)
            else:
                arr[c] = v[:, j]
        if itree is not None and len(arr):  # first row of every tree (trees are contiguous)
            r = arr['Tree_root_ID']
            new = np.flatnonzero(np.r_[last is None or r[0] != last, r[1:] != r[:-1]])
            roots.extend(r[new].tolist())
            starts.extend((new + w.nrows).tolist())
            last = r[-1]
        w.append(arr)
    tread = time.time() - t0
    if ntrees is not None and itree is not None and len(roots) != ntrees:
        raise RuntimeError('%s: %d trees found in the rows, header says %d' % (name, len(roots), ntrees))
    res = finish(w, prov_arr, out, args)
    res.update(read_s=tread, tile=tile, roots=np.array(roots, dtype=np.int64), starts=np.array(starts, dtype=np.int64))
    return [res]


def convert_tree_aux(args, run_cards, run_prov, tile_rows):
    """locations.dat and forests.list (small) in the main process."""
    res = []
    loc = os.path.join(args.indir, 'locations.dat')
    out = os.path.join(args.outdir, 'locations.fits.gz')
    if os.path.isfile(loc) and (args.overwrite or not os.path.isfile(out)):
        hdr = open(loc).readline()
        names = hdr[1:].split() if hdr.startswith('#') else ['TreeRootID', 'FileID', 'Offset', 'Filename', 'Num_Halos', 'Num_Bytes']
        df = pd.read_csv(loc, sep=r'\s+', comment='#', header=None, names=names, engine='c')
        fn_w = max(1, int(df['Filename'].str.len().max())) if len(df) else 1
        dt = [(c, 'S%d' % fn_w) if c == 'Filename' else (c, col_dtype(c)) for c in names]
        dt += [('FitsRow', '>i8'), ('FitsRows', '>i8')]
        arr = np.zeros(len(df), dtype=dt)
        for c in names:
            arr[c] = df[c].values.astype(np.bytes_) if c == 'Filename' else df[c].values
        # row of each tree in its tree_X_Y_Z.fits.gz (-1 if that tile was not converted in this run)
        arr['FitsRow'] = -1
        arr['FitsRows'] = arr['Num_Halos'] if 'Num_Halos' in names else -1
        for fname, (roots, starts) in tile_rows.items():
            sel = np.flatnonzero(arr['Filename'] == fname.encode())
            pos = {int(r): int(s) for r, s in zip(roots, starts)}
            arr['FitsRow'][sel] = [pos.get(int(t), -1) for t in arr['TreeRootID'][sel]]
        cards = run_cards + [card('SRCFILE', 'locations.dat'),
                             card('FITSROW', 'first row of the tree in tree_X_Y_Z.fits.gz (-1: unknown)')]
        tmp = os.path.join(args.tmpdir or args.outdir, '.tmp_locations_%d.fits' % os.getpid())
        w = TableWriter(tmp, arr.dtype, 'LOCATIONS', cards)
        w.append(arr)
        res.append(finish(w, provenance_table(run_prov), out, args))
    fl = os.path.join(args.indir, 'forests.list')
    out = os.path.join(args.outdir, 'forests.fits.gz')
    if os.path.isfile(fl) and (args.overwrite or not os.path.isfile(out)):
        df = pd.read_csv(fl, sep=r'\s+', comment='#', header=None, names=['TreeRootID', 'ForestID'], dtype=np.int64,
                         engine='c')
        arr = np.zeros(len(df), dtype=[('TreeRootID', '>i8'), ('ForestID', '>i8')])
        arr['TreeRootID'], arr['ForestID'] = df['TreeRootID'].values, df['ForestID'].values
        cards = run_cards + [card('SRCFILE', 'forests.list'), card('NFORESTS', int(np.unique(arr['ForestID']).size))]
        tmp = os.path.join(args.tmpdir or args.outdir, '.tmp_forests_%d.fits' % os.getpid())
        w = TableWriter(tmp, arr.dtype, 'FORESTS', cards)
        w.append(arr)
        res.append(finish(w, provenance_table(run_prov), out, args))
    return res


# ------------------------------------------------------------------------------------------------------
# driver


def select_snaps(indir, spec):
    snaps = sorted({os.path.basename(p)[4:-6] for p in glob.glob(os.path.join(indir, 'out_*.trees'))},
                   key=lambda s: (len(s), s))
    if not spec or spec == 'all':
        return snaps
    keep = set()
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            keep |= {str(i) for i in range(int(a), int(b) + 1)}
        else:
            keep.add(part)
    return [s for s in snaps if s in keep]


class Safe:
    """Worker wrapper: an exception in one file is reported, the other files go on."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, task):
        try:
            return self.fn(task)
        except Exception as e:  # noqa: BLE001
            return [{'error': '%s: %s' % (type(e).__name__, e), 'task': str(task[0])}]


def run_pool(fn, tasks, nproc):
    fn = Safe(fn)
    results = []
    if nproc <= 1:
        for t in tasks:
            results.extend(fn(t))
            log_result(results[-1])
        return results
    with mp.get_context('fork').Pool(nproc) as pool:
        for r in pool.imap_unordered(fn, tasks):
            results.extend(r)
            for x in r:
                log_result(x)
    return results


def log_result(r):
    if 'error' in r:
        log('ERROR %s' % r)
    elif 'skipped' in r:
        log('skip %s (%s)' % (r.get('out', r.get('snap')), r['skipped']))
    else:
        log('wrote %s: %d rows, %.1f MB, read %.0f s, %s %.0f s' % (r['out'], r['rows'], r['bytes'] / 1e6,
                                                                    r.get('read_s', 0), r['gzip'], r['gzip_s']))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['halos', 'trees', 'all'])
    ap.add_argument('--indir', required=True)
    ap.add_argument('--outdir')
    ap.add_argument('--snaps', default='all', help="e.g. 'all', '0-100', '5,7,99'")
    ap.add_argument('--types', default='basic,extended,vahc,tens', help='halo outputs (vahc/tens only if present)')
    ap.add_argument('--nproc', type=int, default=max(1, min(16, os.cpu_count() or 1)))
    ap.add_argument('--threads', type=int, default=4, help='pigz threads per file')
    ap.add_argument('--tmpdir', help='uncompressed temporary FITS (node-local disk recommended)')
    ap.add_argument('--level', type=int, default=6, help='gzip level')
    ap.add_argument('--chunk', type=int, default=2000000, help='rows per parsing chunk')
    ap.add_argument('--rockstar-cfg', help='Rockstar rockstar.cfg (as written in OUTBASE)')
    ap.add_argument('--ctrees-cfg', help='consistent-trees merger_tree.cfg')
    ap.add_argument('--pkc', help='prefix of one .pkc snapshot of the run (simulation header)')
    ap.add_argument('--meta', action='append', help='extra header card KEY=VALUE (repeatable)')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--no-verify', action='store_true')
    ap.add_argument('--delete-ascii', action='store_true', help='remove ASCII inputs after a verified write')
    args = ap.parse_args()
    args.types = [t.strip() for t in args.types.split(',') if t.strip()]
    if args.tmpdir:
        os.makedirs(args.tmpdir, exist_ok=True)
    run_cards, run_prov = build_metadata(args)
    log('%s  %s  indir %s  nproc %d  verify %s' % (TOOL, args.what, args.indir, args.nproc, not args.no_verify))
    t0 = time.time()
    results = []
    if args.what in ('halos', 'all'):
        a = argparse.Namespace(**vars(args))
        a.outdir = args.outdir or os.path.join(args.indir, 'compressed')
        os.makedirs(a.outdir, exist_ok=True)
        snaps = select_snaps(a.indir, a.snaps)
        log('halos: %d snapshots -> %s' % (len(snaps), a.outdir))
        results += run_pool(convert_halo_snapshot, [(s, a, run_cards, run_prov) for s in snaps], a.nproc)
    if args.what in ('trees', 'all'):
        a = argparse.Namespace(**vars(args))
        if args.what == 'all':
            a.indir = os.path.join(args.indir, 'trees')
            a.outdir = os.path.join(args.outdir or os.path.join(args.indir, 'compressed'), 'trees')
        else:
            a.outdir = args.outdir or os.path.join(args.indir, 'compressed')
        os.makedirs(a.outdir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(a.indir, 'tree_*.dat')))
        log('trees: %d tree files -> %s' % (len(files), a.outdir))
        res = run_pool(convert_tree_file, [(p, a, run_cards, run_prov) for p in files], a.nproc)
        tile_rows = {os.path.basename(r['out']).replace('.fits.gz', '.dat'): (r['roots'], r['starts'])
                     for r in res if 'roots' in r}
        try:
            aux = convert_tree_aux(a, run_cards, run_prov, tile_rows)
        except Exception as e:  # noqa: BLE001
            aux = [{'error': '%s: %s' % (type(e).__name__, e), 'task': 'locations/forests'}]
        for r in aux:
            log_result(r)
            res.append(r)
        if args.delete_ascii and not args.no_verify and not any('error' in r for r in res):
            for p in files + [os.path.join(a.indir, f) for f in ('locations.dat', 'forests.list')]:
                if os.path.isfile(p):
                    os.remove(p)
        results += res
    nerr = sum('error' in r for r in results)
    log('done: %d files written, %d skipped, %d errors, %.0f s' % (
        sum('rows' in r for r in results), sum('skipped' in r for r in results), nerr, time.time() - t0))
    return 1 if nerr else 0


if __name__ == '__main__':
    sys.exit(main())
