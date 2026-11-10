import sys, numpy as np
from scipy.spatial import cKDTree
def load(d):
    a = np.loadtxt(d + "/out_0.list", comments="#")
    # columns: ID DescID Mvir Vmax Vrms Rvir Rs Np X Y Z VX VY VZ ...
    return dict(m=a[:,2], vmax=a[:,3], r=a[:,5], np_=a[:,7], x=a[:,8:11], v=a[:,11:14], pid=None)
A, B = load(sys.argv[1]), load(sys.argv[2])  # A exact (HDF5), B compressed (PKC)
L = 200.0
print("halos: exact %d, pkc %d" % (len(A["m"]), len(B["m"])))
for n in (20, 50, 100, 1000):
    print("  Np >= %4d: exact %5d pkc %5d" % (n, (A["np_"]>=n).sum(), (B["np_"]>=n).sum()))
t = cKDTree(B["x"], boxsize=L)
sel = A["np_"] >= 100
d, j = t.query(A["x"][sel], k=1)
ok = d < 0.1 * A["r"][sel] / 1000.0
mA, mB = A["m"][sel][ok], B["m"][j[ok]]
print("Np >= 100: %d halos, %d matched within 0.1 Rvir (%.2f%%)" % (sel.sum(), ok.sum(), 100.0*ok.mean()))
dl = np.log10(mB / mA)
print("  dlog10 Mvir: median %.2e, rms %.2e, 99%% |.| < %.2e" % (np.median(dl), np.sqrt(np.mean(dl**2)), np.percentile(np.abs(dl), 99)))
dv = B["vmax"][j[ok]] / A["vmax"][sel][ok] - 1
print("  dVmax/Vmax: median %.2e, rms %.2e" % (np.median(dv), np.sqrt(np.mean(dv**2))))
dx = d[ok] * 1000.0 / A["r"][sel][ok]
print("  position offset / Rvir: median %.2e, 99%% %.2e" % (np.median(dx), np.percentile(dx, 99)))
dvel = np.linalg.norm(B["v"][j[ok]] - A["v"][sel][ok], axis=1)
print("  bulk velocity difference: median %.2f km/s, 99%% %.2f km/s" % (np.median(dvel), np.percentile(dvel, 99)))
