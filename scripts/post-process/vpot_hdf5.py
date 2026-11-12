"""
vpot_hdf5.py: the .vpot fields (vahc_fast.py) for Gadget-4 HDF5 snapshots that store the potential
(EVALPOTENTIAL + OUTPUT_POTENTIAL), so that run_addvalue.py --fast --pot also writes out_<snap>.vpot for Gadget-4 runs.

hdf5_fields(sr, ps, threads, halos, hpos, Om) returns (density, n, vpot) like vahc_fast.streamed_fields:
  density  ps.density_field(pos) of all particles: exactly the density of the existing HDF5 path of run_addvalue.py,
           so .vahc/.tens are unchanged by --pot
  vpot     CIC grid of phi and the sphere sums within vahc_fast.SPHERES, from the same particles and the same C
           routines (vpot_sums.c) as the .pkc path
phi = Potential / a in (km/s)^2: Gadget-4 writes the comoving potential with the a^-1 scaling attribute (snap_io.cc)
and removes the self-interaction itself, so the self term is 0 (pkdgrav3's -(105/64) m/eps is removed in the .pkc
path; with it removed the two codes' zero points agree to ~1e3 (km/s)^2, reports/gadget_pkdgrav3_potential).
Memory: positions and potential of all particles at once (24 + 8 bytes per particle, float64 copies per chunk).
"""
import time

import numpy as np

from vahc_fast import SPHERES, vpot_lib


def has_potential(sr):
    """True when the Gadget-4 HDF5 snapshot of this SnapshotReader has PartType<ptype>/Potential."""
    import h5py
    if getattr(sr, 'pkc', None) is not None or not str(getattr(sr, 'snapshot_ext', '')).endswith('.hdf5'):
        return False
    with h5py.File(sr.snapshot_file + sr.snapshot_ext, 'r') as f:
        return ('PartType%d/Potential' % sr.ptype) in f


def hdf5_fields(sr, ps, threads, halos, hpos, Om, chunk=1 << 24):
    pos = sr.read_block('pos')                      # (3, N), Mpc/h
    n = pos.shape[1]
    density = ps.density_field(pos)
    phi = np.asarray(sr.read_block('potential'), np.float64) / float(sr.scale)
    if phi.size != n:
        raise ValueError('%d potentials for %d particles' % (phi.size, n))
    lib = vpot_lib()
    G = ps.grid
    L = float(ps.Lbox)
    potgrid = np.zeros(G**3)
    nh = halos.size
    hx, hy, hz = (np.ascontiguousarray(hpos[:, i] % L, np.float64) for i in range(3))
    nr = len(SPHERES)
    radii = np.ascontiguousarray(np.stack([f(halos, Om) for _, f in SPHERES], axis=1), np.float64)
    ssum = np.zeros((nh, nr))
    scnt = np.zeros((nh, nr), np.int64)
    nc = int(max(1, min(256, L / 0.5)))
    tsph = 0.0
    for i in range(0, n, chunk):
        # float32 positions as the density path (ps.density_field) and the .pkc path deposit them
        x, y, z = (np.ascontiguousarray(pos[k, i:i + chunk].astype(np.float32), np.float64) for k in range(3))
        p = np.ascontiguousarray(phi[i:i + chunk])
        lib.deposit_weighted(x.size, G, potgrid, x, y, z, p, ps.cell_size_x, ps.cell_size_y, ps.cell_size_z)
        t = time.time()
        rc = lib.sphere_sums(x.size, x, y, z, p, nh, hx, hy, hz, nr, radii, L, nc, max(1, threads), ssum, scnt)
        if rc:
            raise MemoryError('sphere_sums: out of memory')
        tsph += time.time() - t
    del pos, phi
    with np.errstate(invalid='ignore', divide='ignore'):
        vpot = dict(potgrid=np.reshape(potgrid, (G, G, G), order='F'), nbar=n / float(G)**3, self_term=0.0,
                    n_sph=scnt, phi_sph=np.where(scnt > 0, ssum / scnt, np.nan), sphere_s=tsph, chunk=chunk,
                    code='Gadget-4')
    return density, n, vpot
