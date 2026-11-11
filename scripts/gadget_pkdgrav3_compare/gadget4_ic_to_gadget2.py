#!/usr/bin/env python3
"""Convert a Gadget-4 HDF5 initial-conditions file (dark matter only, PartType1) to a Gadget-2 "format 1"
binary file that pkdgrav3 reads directly (achInFile; io/fio.c gadgetOpen).

usage: gadget4_ic_to_gadget2.py <snapshot_ics_000.hdf5 | snapshot_ics_000 (multi-file base)> <out.dat>

- Particles are written sorted by ParticleIDs, with ID - min(ID) as the ID. pkdgrav3's Gadget reader sets iOrder to
  the file's ID (pkd.cxx pkdReadFIO) and its Reorder() needs iOrder = 0..N-1, so the Gadget IDs must be contiguous.
  With NGenIC's IDs 1..N: pkdgrav3 iOrder (= the ID in its .pkc files, = the row of its ordered HDF5 output)
  = Gadget ID - 1. The offset is printed.
- Values are copied unchanged (float32 positions in Mpc/h, Gadget velocities sqrt(a) dx/dt in km/s, uint64 IDs,
  mass from the MassTable, rescaled to pkdgrav3's G, see below). pkdgrav3 converts units itself (positions / BoxSize - 0.5,
  velocities * sqrt(8 pi/3) / (100 BoxSize sqrt(a)) and then * a^2 in pkdReadFIO, mass to code units).
- The header has Omega0, OmegaLambda, h, BoxSize and Time of the HDF5 header; pkdgrav3 takes the starting
  time from it.
"""
import glob
import struct
import sys

import h5py
import numpy as np


def files_of(path):
    if path.endswith('.hdf5'):
        return [path]
    fl = sorted(glob.glob(path + '.*.hdf5'), key=lambda f: int(f.split('.')[-2]))
    if not fl:
        sys.exit('no file %s.hdf5 or %s.<n>.hdf5' % (path, path))
    return fl


def main():
    src, out = sys.argv[1], sys.argv[2]
    fl = files_of(src)
    pos, vel, ids = [], [], []
    for f in fl:
        with h5py.File(f, 'r') as h:
            hdr = dict(h['Header'].attrs)
            par = dict(h['Parameters'].attrs) if 'Parameters' in h else {}
            g = h['PartType1']
            pos.append(g['Coordinates'][:].astype(np.float32))
            vel.append(g['Velocities'][:].astype(np.float32))
            ids.append(g['ParticleIDs'][:].astype(np.uint64))
    pos, vel, ids = np.concatenate(pos), np.concatenate(vel), np.concatenate(ids)
    ntot = int(hdr['NumPart_Total'][1])
    assert len(ids) == ntot, (len(ids), ntot)
    for k in range(6):
        if k != 1:
            assert int(hdr['NumPart_Total'][k]) == 0, 'only dark matter (PartType1) is supported'
    order = np.argsort(ids, kind='stable')
    pos, vel, ids = pos[order], vel[order], ids[order]
    assert np.all(np.diff(ids.astype(np.int64)) > 0), 'duplicate particle IDs'
    off = int(ids[0])
    contiguous = bool(ids[-1] - ids[0] == ntot - 1)
    print('%d particles from %d file(s); IDs %d..%d, contiguous: %s -> pkdgrav3 iOrder = Gadget ID - %d'
          % (ntot, len(fl), ids[0], ids[-1], contiguous, off))
    assert contiguous, 'Gadget IDs must be contiguous (pkdgrav3 iOrder must be 0..N-1)'
    ids = ids - np.uint64(off)
    L = float(hdr['BoxSize'])
    pos = np.mod(pos, np.float32(L))   # Gadget can leave x == L after rounding; pkdgrav3 wants [0, L)
    pos[pos >= L] = 0.0
    a = float(hdr['Time'])
    om = float(par.get('Omega0', hdr.get('Omega0', 0)))
    ol = float(par.get('OmegaLambda', hdr.get('OmegaLambda', 0)))
    hh = float(par.get('HubbleParam', hdr.get('HubbleParam', 0)))
    mass = np.array(hdr['MassTable'], dtype=np.float64)
    assert mass[1] > 0, 'MassTable[1] must be set (equal-mass dark matter)'

    # pkdgrav3's reader converts the mass with G = 4.30172e-9 Mpc (km/s)^2/Msun (io/fio.c); Gadget-4 used its own G.
    # Write the mass that gives total mass = Omega0 in pkdgrav3 units (rho_crit = 3/(8 pi), L = 1), so that the
    # particles and pkdgrav3's background agree; G*m is then the same in both codes.
    mass_fac = 8 * np.pi * 4.30172 / (L**3 * 3.0e3)
    m_g4 = mass[1]
    mass[1] = om / (ntot * mass_fac)
    print('mass: Gadget-4 %.9g, written %.9g 1e10 Msun/h (ratio %.6f): pkdgrav3 total = Omega0 = %.7f'
          % (m_g4, mass[1], mass[1] / m_g4, mass[1] * ntot * mass_fac))
    npart = [0, ntot, 0, 0, 0, 0]
    assert ntot < 2**32
    h = struct.pack('<6I6dddii6Iiiddddii6Ii', *npart, *mass, a, 1.0 / a - 1.0, 0, 0, *npart, 0, 1, L, om, ol, hh,
                    0, 0, *([0] * 6), 0)
    h = h + b'\0' * (256 - len(h))
    assert len(h) == 256

    def block(f, arr):
        b = np.ascontiguousarray(arr).tobytes()
        f.write(struct.pack('<I', len(b))); f.write(b); f.write(struct.pack('<I', len(b)))

    with open(out, 'wb') as f:
        block(f, np.frombuffer(h, dtype=np.uint8))
        block(f, pos.astype('<f4'))
        block(f, vel.astype('<f4'))
        block(f, ids.astype('<u8'))
    print('wrote %s: a = %.6g (z = %.6g), L = %g Mpc/h, m = %.6g 1e10 Msun/h, Om = %g, OL = %g, h = %g'
          % (out, a, 1 / a - 1, L, mass[1], om, ol, hh))


if __name__ == '__main__':
    main()
