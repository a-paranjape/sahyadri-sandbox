#!/usr/bin/env python3
"""Test helper only: turn a version-1 .pkc set (e.g. from pkc_convert, which writes no cosmology block)
into version 3 in place by writing the cosmology block into each 4096-byte header. Blocks, index and
trailer are unchanged (the header has a fixed size). Not for production snapshots: pkdgrav3 writes
version 3 itself.

usage: pkc_v1_to_v3.py <prefix> omega_de w0 wa [omega_b omega_rad sigma8 n_s]
"""
import glob
import struct
import sys

prefix = sys.argv[1]
ode, w0, wa = (float(x) for x in sys.argv[2:5])
rest = [float(x) for x in sys.argv[5:9]] + [0.0] * (4 - len(sys.argv[5:9]))
ob, orad, s8, ns = rest
files = sorted(glob.glob(prefix + ".s*.pkc"))
if not files:
    sys.exit("no files " + prefix + ".s*.pkc")
for f in files:
    with open(f, "r+b") as fh:
        h = bytearray(fh.read(4096))
        if h[:8] != b"PKCOMP\x00\x01":
            sys.exit(f + ": not a snapcomp file")
        version = struct.unpack_from("<I", h, 8)[0]
        if version != 1:
            sys.exit(f"{f}: version {version}, expected 1")
        h[3999:4096] = bytes(97)  # info ends before 4000; cosmology 4000-4079; no potential (4080-4095 = 0)
        struct.pack_into("<I", h, 8, 3)
        struct.pack_into("<10d", h, 4000, ob, orad, ode, w0, wa, s8, ns, 0.0, 0.0, 0.0)
        fh.seek(0)
        fh.write(h)
print(f"{len(files)} files of {prefix} rewritten as version 3 (omega_de {ode}, w0 {w0}, wa {wa})")
