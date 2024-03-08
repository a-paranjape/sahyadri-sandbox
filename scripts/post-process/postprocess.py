import os,sys
import numpy as np
import gc
from readers import SnapshotReader,HaloReader
from correlations import PowerSpectrum
from time import time

if(len(sys.argv)==7):
    sim_stem = sys.argv[1]
    snap = int(sys.argv[2])
    real = int(sys.argv[3])
    grid = int(sys.argv[4])
    downsample = int(sys.argv[5])
    Npmin = int(sys.argv[6])
else: 
    sim_stem = input("Specify path (e.g., `su128/delta0.0' or 'scm1024'): ")
    snap = int(input("Specify snapshot (e.g., 0-200): "))
    real = int(input("Specify realisation (e.g., 1-10): "))
    out_Pk = input("Specify output filename for P(k) (e.g., `Pk_000'): ")
    grid = int(input("Specify 1-d gridsize (e.g. 256) for density field calculation: "))
    downsample = int(input("Specify 1-d n_particles (e.g. 128) for downsampling; 0 to use full sample: "))
    Npmin = int(input("Specify minimum particle mass for halo catalog: "))

Seed = 42
start_time = time()
sr = SnapshotReader(sim_stem=sim_stem,real=real,snap=snap)
pos = sr.read_block('pos',down_to=downsample,seed=Seed)
# vel = sr.read_block('vel',down_to=downsample,seed=Seed)
# ids = sr.read_block('ids',down_to=downsample,seed=Seed)

hr = HaloReader(sim_stem='testTF128',real=real,snap=snap)
hpos,halos = hr.prep_halos(massdef='m200b',QE=0.5,Npmin=Npmin)

ps = PowerSpectrum(grid=256,Lbox=sr.Lbox)
delta_dm = ps.density_field(pos)
FT_delta_dm = ps.fourier_transform_density(delta_dm)
Pk_mm = ps.Pk_grid(FT_delta_dm,input_is_FTdensity=True)

delta_h = ps.density_field(hpos)
FT_delta_h = ps.fourier_transform_density(delta_h)
Pk_hh = ps.Pk_grid(FT_delta_h,input_is_FTdensity=True)
Pk_hm = ps.Pk_grid(FT_delta_h,input_array2=FT_delta_dm,input_is_FTdensity=True)

Pk_hh -= ps.Lbox**3/(halos.size + ps.TINY)
if downsample > 0:
    Pk_mm -= ps.Lbox**3/(downsample**3)
else:
    Pk_mm -= ps.Lbox**3/(sr.npart)

del pos#,vel,ids
del delta_dm, FT_delta_dm,delta_h,FT_delta_h
gc.collect()

sr.print_this('... done',ps.logfile)

outfile_Pk = sr.sim_path + sim_stem + '/r'+str(real)+'/Pk_{0:03d}.txt'
sr.print_this('Writing to file: '+outfile_Pk,ps.logfile)
f = open(outfile_Pk,'w')
f.write("# P(k) (DM,halos,cross) from snapshot_{0:03d}".format(snap))
f.write("# grid = {0:d}; downsampled to ({1:d})^3 particles\n".format(grid,downsample))
f.write("# Halos satisfy {0:.2f} < 2T/|U| < {1:.2f}\n".format(1.-QE,1.+QE))
f.write("# "+massdef+" > {0:.4e} Msun/h\n".format(mmin))
f.write("# k (h/Mpc) | P(k) (Mpc/h)^3 | Phalo | Pcross\n")
f.close()
for k in range(powspec.nbin):
    write_to_file(outfile_Pk,[powspec.ktab[k],Pk[k],Pkh[k],Pkx[k]])
sr.print_this('... done',ps.logfile)

sr.time_this(start_time)
