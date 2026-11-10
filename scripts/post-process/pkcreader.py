# Copied from pkdgrav3 tools/pkcreader/pkcreader.py (commit 5697d3c3, format versions 1-3); keep in sync there.
"""Reader for compressed pkdgrav3 snapshots (.pkc, docs/snapcomp_format.md). Needs numpy and zstandard.

Written from the format spec only, independent of the C++ code. One file, so it can be copied next to
Sahyadri's readers.py.

    from pkcreader import PkcSnapshot
    s = PkcSnapshot("snapshots/snap.00100")            # all <prefix>.s*.pkc files of one snapshot
    s.redshift, s.scale, s.Lbox, s.npart, s.mpart, s.Om, s.Olam, s.hubble, s.percents
    d = s.read(subsamples=[1, 3])                       # dict: pos (n,3) Mpc/h in [0, L), vel (n,3) km/s, ids
    d = s.read(fields=("pos", "pot"))                   # pot (n,): peculiar potential in (km/s)^2, if stored
    s.has_potential, s.pot_step
    pos = s.read_block("pos", subsamples=[1])           # SnapshotReader style: (3, n); "vel" (3, n); "ids" (n,)
    for pos, vel, ids in s.iter_blocks():               # streaming, one block (<= 65536 particles) at a time
        ...

Velocities are peculiar velocities in km/s (not Gadget's v/sqrt(a)). Subsamples are chosen by percent
(1, 3, 5, 9, 10, 12, 15, 20, 25 by default); a particle is in the same subsample in every snapshot.
Particles come out in file order, then block order; the order is the same on every call, so
read_block("pos") and read_block("ids") line up.

Lower level: PKCFile(path).header / .read(), decode_block(header, bytes), read_snapshot(prefix).
"""
import glob
import json
import mmap
import os
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zstandard

HEADER_SIZE = 4096
_HEADER_MAGIC = b"PKCOMP\x00\x01"
_TRAILER_MAGIC = b"PKCIDX\x00\x01"
_BLOCK_MAGIC = b"PKB1"
ID_RAW, ID_DELTA_IN_CELL, ID_LAGRANGIAN = 0, 1, 2
K_POT = 8
POT_OFFSET = 4080  # versions 2, 3: pot_step, pot_to_kms2 (doubles)
COSMO_OFFSET = 4000  # version 3: 10 doubles of cosmology; the info text ends before it
COSMO_KEYS = ("Omega_b", "Omega_rad", "Omega_DE", "w0", "wa", "sigma8", "n_s", "running", "normalization", "pivot")
RHO_CRIT = 2.77536627e11  # Msun/h per (Mpc/h)^3

_FIELDS = [  # (offset, struct code, name)
    (16, "Q", "n_particles"), (24, "Q", "n_blocks"), (32, "Q", "lattice_n"), (40, "Q", "n_total"),
    (48, "I", "ncell"), (52, "I", "sub"), (56, "d", "vel_step"), (64, "d", "vel_to_kms"),
    (72, "d", "box_size"), (80, "d", "redshift"), (88, "d", "time"), (96, "d", "omega_m"),
    (104, "d", "omega_l"), (112, "d", "h"), (120, "I", "subsample"), (124, "I", "n_subsamples"),
    (256, "Q", "subsample_seed"), (264, "I", "writer"), (268, "I", "n_writers"), (272, "I", "block_size"),
    (276, "i", "zstd_level"), (280, "Q", "step"),
]


def parse_header(b):
    if bytes(b[:8]) != _HEADER_MAGIC:
        raise ValueError("not a snapcomp file")
    version, hsize = struct.unpack_from("<II", b, 8)
    if version not in (1, 2, 3) or hsize != HEADER_SIZE:
        raise ValueError(f"unsupported version {version} / header size {hsize}")
    h = {name: struct.unpack_from("<" + code, b, off)[0] for off, code, name in _FIELDS}
    h["version"] = version
    h["percents"] = list(struct.unpack_from("<16d", b, 128))[: h["n_subsamples"]]
    end = COSMO_OFFSET if version >= 3 else POT_OFFSET if version >= 2 else HEADER_SIZE
    h["info"] = bytes(b[288:end]).split(b"\x00", 1)[0].decode(errors="replace")
    h["pot_step"], h["pot_to_kms2"] = struct.unpack_from("<2d", b, POT_OFFSET) if version >= 2 else (0.0, 1.0)
    if version == 2 and not h["pot_step"] > 0:
        raise ValueError("version 2 header without a potential step")
    if not h["pot_step"] > 0:
        h["pot_step"], h["pot_to_kms2"] = 0.0, 1.0
    # version 3: binary cosmology (Omega_DE with w(a) = w0 + wa (1 - a); pivot in 1/Mpc); {} before version 3
    h["cosmology"] = dict(zip(COSMO_KEYS, struct.unpack_from("<10d", b, COSMO_OFFSET))) if version >= 3 else {}
    h["q"] = h["ncell"] * h["sub"]
    return h


def _unzigzag(u):
    u = u.astype(np.uint64)
    return (u >> np.uint64(1)).astype(np.int64) ^ -(u & np.uint64(1)).astype(np.int64)


def decode_block(h, b):
    """Decode one block: integer positions X (n,3), velocities V (n,3) in vel_step units, IDs (n)."""
    return decode_block_full(h, b)[:3]


def decode_block_full(h, b):
    """As decode_block, plus the potential P (n,) in pot_step units (None if the file has none)."""
    if bytes(b[:4]) != _BLOCK_MAGIC:
        raise ValueError("bad block magic")
    n, n_cells, ns, coder = struct.unpack_from("<IIBB", b, 4)
    dctx = zstandard.ZstdDecompressor()  # one per call: thread-safe
    streams, off, data = [], 16, 16 + 12 * ns
    for _ in range(ns):
        w, kind, _, count, cs = struct.unpack_from("<BBHII", b, off)
        raw = dctx.decompress(bytes(b[data:data + cs]), max_output_size=count * w)
        if len(raw) != count * w:
            raise ValueError("stream size mismatch")
        planes = np.frombuffer(raw, np.uint8).reshape(w, count)
        v = np.zeros(count, np.uint64)
        for k in range(w):
            v |= planes[k].astype(np.uint64) << np.uint64(8 * k)
        streams.append(v)
        off += 12
        data += cs
    nc, sub, Q = h["ncell"], h["sub"], h["q"]
    keys = np.cumsum(streams[0].astype(np.int64))
    counts = streams[1].astype(np.int64)
    key = np.repeat(keys, counts)
    cell = np.stack([key % nc, (key // nc) % nc, key // (nc * nc)], 1)
    X = cell * sub + np.stack([streams[2 + d].astype(np.int64) for d in range(3)], 1)
    V = np.stack([_unzigzag(streams[5 + d]) for d in range(3)], 1)
    if coder == ID_RAW:
        ids = streams[8]
    elif coder == ID_DELTA_IN_CELL:
        d = _unzigzag(streams[8]).astype(np.uint64)  # wrapping differences
        start = np.concatenate(([0], np.cumsum(counts)[:-1]))
        first = np.cumsum(d[start], dtype=np.uint64)
        within = d.copy()
        within[start] = 0
        cw = np.cumsum(within, dtype=np.uint64)
        ids = np.repeat(first, counts) + (cw - np.repeat(cw[start], counts))
    elif coder == ID_LAGRANGIAN:
        N = h["lattice_n"]
        l = (X * N) // Q  # X < 2^32 and N < 2^21 fit in int64
        q = (np.stack([_unzigzag(streams[8 + d]) for d in range(3)], 1) + l) % N
        ids = ((q[:, 2] * N + q[:, 1]) * N + q[:, 0]).astype(np.uint64)
    else:
        raise ValueError(f"unknown ID coder {coder}")
    if len(ids) != n:
        raise ValueError("block particle count mismatch")
    P = None
    if h["pot_step"] > 0:
        k = 8 + (3 if coder == ID_LAGRANGIAN else 1)
        if ns != k + 1 or struct.unpack_from("<B", b, 16 + 12 * k + 1)[0] != K_POT:
            raise ValueError("expected a potential stream")
        P = np.cumsum(_unzigzag(streams[k]))
    elif ns != 8 + (3 if coder == ID_LAGRANGIAN else 1):
        raise ValueError("unexpected stream count")
    return X, V, ids.astype(np.uint64), P


class PKCFile:
    """One .pkc file, memory-mapped; blocks are decoded on demand."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mm = self._mm
        if len(mm) < HEADER_SIZE + 32 or mm[-32:-24] != _TRAILER_MAGIC:
            raise ValueError(f"{path}: no trailer (incomplete file)")
        idx, nb, npart = struct.unpack_from("<QQQ", mm, len(mm) - 24)
        self.header = parse_header(mm[:HEADER_SIZE])
        if self.header["n_blocks"] != nb or self.header["n_particles"] != npart:
            raise ValueError(f"{path}: header and trailer disagree")
        ent = np.frombuffer(mm, np.uint8, nb * 32, idx).reshape(nb, 32)
        self.offsets = ent[:, 0:8].copy().view("<u8").ravel()
        self.sizes = ent[:, 8:16].copy().view("<u8").ravel()
        self.counts = ent[:, 16:20].copy().view("<u4").ravel().astype(np.int64)
        self.index = list(zip(self.offsets.tolist(), self.sizes.tolist(), self.counts.tolist()))  # (offset, size, n)
        self.percent = self.header["percents"][self.header["subsample"]]

    def block(self, i, pot=False):
        """(X, V, ids) of block i; with pot=True also P (potential in pot_step units, None if absent)."""
        o, s = int(self.offsets[i]), int(self.sizes[i])
        r = decode_block_full(self.header, memoryview(self._mm)[o:o + s])
        return r if pot else r[:3]

    def blocks(self):
        for i in range(len(self.offsets)):
            yield self.block(i)

    def read(self, units="mpc"):
        h = self.header
        parts = list(self.blocks())
        if not parts:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.uint64)
        X = np.concatenate([p[0] for p in parts])
        V = np.concatenate([p[1] for p in parts])
        ids = np.concatenate([p[2] for p in parts])
        x = (X + 0.5) / h["q"]  # [0, 1)
        x = x - 0.5 if units == "box" else x * h["box_size"]
        return x, V * h["vel_step"], ids


def read_snapshot(prefix, percents=None, units="mpc"):
    """Read all files of one snapshot (or only the listed subsample percents) and concatenate."""
    d = PkcSnapshot(prefix).read(subsamples=percents, units=units)
    return d["pos"], d["vel"], d["ids"]


class PkcSnapshot:
    """All files of one compressed snapshot (every subsample, every writer)."""

    def __init__(self, prefix, threads=None):
        self.prefix = prefix
        paths = sorted(glob.glob(glob.escape(prefix) + ".s*.pkc"))
        if not paths:
            raise FileNotFoundError(f"no files {prefix}.s*.pkc")
        self.files = [PKCFile(p) for p in paths]
        self.files.sort(key=lambda f: (f.header["subsample"], f.header["writer"]))
        h = self.header = self.files[0].header
        self.threads = threads or min(32, os.cpu_count() or 1)
        self.redshift = h["redshift"]
        self.scale = 1.0 / (1.0 + h["redshift"])
        self.Lbox = h["box_size"]
        self.npart = h["n_total"]
        self.Om, self.Olam, self.hubble = h["omega_m"], h["omega_l"], h["h"]
        self.mpart = RHO_CRIT * self.Om * self.Lbox**3 / self.npart  # Msun/h
        self.percents = h["percents"]
        self.lattice_n = h["lattice_n"]
        self.has_potential = h["pot_step"] > 0
        # Free-text info (JSON from pkdgrav3): full cosmology (w0, wa, Omega_DE, Omega_rad, Omega_b, sigma8, n_s,
        # E_z = H(z)/H0, CLASS/transfer files) and run settings; empty dicts for files written before it existed.
        try:
            self.info = json.loads(h["info"]) if h["info"] else {}
        except ValueError:
            self.info = {}
        # info JSON cosmology (with values at the snapshot), completed with the version-3 binary block
        self.cosmology = dict(h["cosmology"])
        self.cosmology.update(self.info.get("cosmology", {}))
        self.pot_step = h["pot_step"]  # (km/s)^2, max error is half of it
        got = sum(f.header["n_particles"] for f in self.files)
        if got != self.npart:
            raise ValueError(f"{prefix}: files hold {got} particles, header says {self.npart} (missing files?)")

    def _select(self, subsamples):
        if subsamples is None or len(subsamples) == 0:
            return self.files
        out = [f for f in self.files if any(abs(f.percent - p) < 1e-9 for p in subsamples)]
        if not out:
            raise ValueError(f"no subsample in {subsamples}; available {self.percents}")
        return out

    def count(self, subsamples=None):
        return int(sum(f.header["n_particles"] for f in self._select(subsamples)))

    def read(self, fields=("pos", "vel", "ids"), subsamples=None, units="mpc", dtype=np.float32):
        """Return a dict of the requested fields ("pos", "vel", "ids", "pot") for the chosen subsamples (percents)."""
        if "pot" in fields and not self.has_potential:
            raise ValueError(f"{self.prefix}: no potential stored (written with dCompPotStep = 0)")
        files = self._select(subsamples)
        jobs, start = [], 0
        for f in files:
            for i, n in enumerate(f.counts.tolist()):
                jobs.append((f, i, start, n))
                start += n
        out = {}
        if "pos" in fields:
            out["pos"] = np.empty((start, 3), dtype)
        if "vel" in fields:
            out["vel"] = np.empty((start, 3), dtype)
        if "ids" in fields:
            out["ids"] = np.empty(start, np.uint64)
        if "pot" in fields:
            out["pot"] = np.empty(start, dtype)

        def work(job):
            f, i, s, n = job
            h = f.header
            X, V, ids, P = f.block(i, pot=True)
            if "pot" in out:
                out["pot"][s:s + n] = P * h["pot_step"]
            if "pos" in out:
                x = (X + 0.5) / h["q"]
                out["pos"][s:s + n] = x - 0.5 if units == "box" else x * h["box_size"]
            if "vel" in out:
                out["vel"][s:s + n] = V * h["vel_step"]
            if "ids" in out:
                out["ids"][s:s + n] = ids

        with ThreadPoolExecutor(self.threads) as ex:
            for _ in ex.map(work, jobs):
                pass
        return out

    def read_block(self, block="pos", subsamples=None):
        """SnapshotReader-style: pos and vel as (3, n), ids as (n,)."""
        name = {"pos": "pos", "vel": "vel", "ids": "ids", "pot": "pot"}[block]
        a = self.read(fields=(name,), subsamples=subsamples)[name]
        return a if name in ("ids", "pot") else a.T

    def iter_blocks(self, subsamples=None, units="mpc"):
        """Yield (pos (n,3), vel (n,3), ids) per block, in the same order as read()."""
        for f in self._select(subsamples):
            h = f.header
            for i in range(len(f.counts)):
                X, V, ids = f.block(i)
                x = (X + 0.5) / h["q"]
                yield (x - 0.5 if units == "box" else x * h["box_size"]), V * h["vel_step"], ids
