# Rockstar catalogs of different input routes (pkc, pkc2, gad, h5g) at z = 0, matched to gad by nearest centre.
import numpy as np, sys
from scipy.spatial import cKDTree
D='/storage/shadab/pkc_rockstar/rsval/'
def load(r, f):
    a=np.loadtxt(D+r+'/'+f, usecols=(2,8,9,10,5), comments='#')  # Mvir Vmax? -> pick by header
    return a
def hdr(r,f):
    with open(D+r+'/'+f) as fh: return fh.readline()[1:].split()
cols=hdr('pkc','out_5.list'); ix={c:i for i,c in enumerate(cols)}
use=[ix['Mvir'],ix['Vmax'],ix['X'],ix['Y'],ix['Z']]
L=50.
data={}
for r in ['pkc','pkc2','gad','h5g']:
    data[r]=np.loadtxt(D+r+'/out_5.list',usecols=use,comments='#')
ref=data['gad']
bins=np.array([1e10,1e11,1e12,1e13,1e15])
print('z=0 counts (host+sub) by Mvir bin [Msun/h]:', bins)
for r,a in data.items():
    print(f'{r:5s} N={len(a):7d}', np.histogram(a[:,0],bins)[0])
t=cKDTree(ref[:,2:5]%L, boxsize=L)
for r in ['pkc','pkc2','h5g']:
    a=data[r]; m=a[:,0]>=1e11   # >= ~1230 particles
    d,j=t.query(a[m,2:5]%L)
    ok=d<0.01
    dm=a[m,0][ok]/ref[j[ok],0]-1; dv=a[m,1][ok]/ref[j[ok],1]-1
    print(f'{r:5s} vs gad, Mvir>=1e11: {m.sum()} halos, matched within 10 kpc/h: {ok.mean()*100:.2f}%, '
          f'|dM/M|>1e-3: {(abs(dm)>1e-3).mean()*100:.2f}%, median |dM/M|={np.median(abs(dm)):.1e}, median |dVmax/Vmax|={np.median(abs(dv)):.1e}')
