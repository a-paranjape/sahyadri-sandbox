#!/usr/bin/python

import paths 
import constants as con

import sys
sys.path.append(paths.python_path)
import numpy as ny
from utilities import write_to_file,writelog,quit_,time_this_log
from time import time
from halo_reader import HaloReader
from gadget2_reader import Gadget2Reader
from correlations import PowerSpectrum
import gc

########################################################
# Read GADGET-2 snapshot and ROCKSTAR halo catalog 
# and measure power spectrum P(k). 
########################################################


if(len(sys.argv)==12):
    sim_stem = sys.argv[1]
    snapshot = sys.argv[2]
    snapnum = int(sys.argv[3])
    real = int(sys.argv[4])
    out_Pk = sys.argv[5]
    rsd = int(sys.argv[6])
    GRID = int(sys.argv[7])
    DOWNSAMPLE = int(sys.argv[8])
    su = int(sys.argv[9])
    NFILE = int(sys.argv[10])
    TREES = int(sys.argv[11])
else: 
    sim_stem = raw_input("Specify path (e.g., `su128/delta0.0' or 'scm1024'): ")
    snapshot = raw_input("Specify common snapshot filename (e.g., `snapshot_000'): ")
    snapnum = int(raw_input("Specify snapshot (e.g., 0-200): "))
    real = int(raw_input("Specify realisation (e.g., 1-10): "))
    out_Pk = raw_input("Specify output filename for P(k) (e.g., `Pk_000'): ")
    rsd = int(raw_input("Specify whether (1) or not (0) to account for redshift distortions: "))
    GRID = int(raw_input("Specify 1-d gridsize (e.g. 256) for density field calculation: "))
    DOWNSAMPLE = int(raw_input("Specify 1-d n_particles (e.g. 128) for downsampling; 0 to use full sample: "))
    su = int(raw_input("Specify whether to use Mpc/h (0) or Mpc (1) : "))
    NFILE = int(raw_input("Specify number of files per gadget snapshot : "))
    TREES = int(raw_input("Specify whether (1) or not (0) merger trees available: "))


start_time = time()

EXCLUDE_SUBHALOS = True # default True

outfile_Pk = paths.sim_path + sim_stem + '/r'+str(real)+'/' + out_Pk
if rsd:
    outfile_Pk += '_RSD'
outfile_Pk += '.txt'

logfile = outfile_Pk[:-3]+'log'
f = open(logfile,'w')
f.close()

if GRID > 512:
    writelog(logfile,"Grid is too fine. Switching to 512^3\n")
    GRID = 512

READ_VEL = True if rsd else False
g2reader = Gadget2Reader(read_pos=True,read_vel=READ_VEL,read_ID=False,logfile=logfile)


snapshot_read = sim_stem + '/r'+str(real)+'/' + snapshot
if NFILE > 1:
    snapshot_read = snapshot_read + '.0'
writelog(logfile,"Reading file "+snapshot_read+'\n')
infile_snap = paths.sim_path + snapshot_read
if DOWNSAMPLE:
    pos,vel,IDs = g2reader.downsample(infile_snap,down_to=DOWNSAMPLE,seed=42)
else:
    pos,vel,IDs = g2reader.read_this(infile_snap)

if NFILE > 1:
    for f in range(1,NFILE):
        snapshot_read = sim_stem + '/r'+str(real)+'/' + snapshot + '.'+str(f)
        writelog(logfile,"Reading file "+snapshot_read+'\n')
        infile_snap = paths.sim_path + snapshot_read
        if DOWNSAMPLE:
            f_pos,f_vel,f_IDs = g2reader.downsample(infile_snap,down_to=DOWNSAMPLE,seed=42)
        else:
            f_pos,f_vel,f_IDs = g2reader.read_this(infile_snap)
        pos = ny.append(pos.T,f_pos.T,axis=0).T
        if READ_VEL:
            vel = ny.append(vel.T,f_vel.T,axis=0).T

# must appear after reading data
Lbox = g2reader.Lbox 
hubble = g2reader.hubble
Om = g2reader.Om 
Npart = g2reader.npart_tot[g2reader.ptype]
Npmin = 100 # 40 also in AssemblyBias-SU.sh

mmin = Om*con.rhoc*Lbox**3*Npmin/Npart

hr = HaloReader(sim_stem=sim_stem,TREES=TREES,logfile=logfile)
halos = hr.read_this(real,snapnum)

QE = 0.5 # 0 < QE < 1
writelog(logfile,"Halos will satisfy {0:.2f} < 2T/|U| < {1:.2f}\n".format(1.-QE,1.+QE))
cond_QE = (2*halos['TbyU'] < (1+QE)) & ((1-QE) < 2*halos['TbyU'])
halos = halos[cond_QE]

Nhalo_all = halos.size
if EXCLUDE_SUBHALOS:
    halos = halos[halos['pid']==-1]
    Nhalo_parents = halos.size
    writelog(logfile,"... satellite fraction = {0:.5f}\n".format(1 - 1.0*Nhalo_parents/Nhalo_all))
else:
    writelog(logfile,'... keeping subhalos too\n')
massdef = 'mCustom' if su else 'm200b'

halos = halos[halos[massdef] > mmin]
Nhalos = halos.size
writelog(logfile,"... keeping {0:d} objects\n".format(Nhalos))
hpos = ny.array(zip(halos['x'],halos['y'],halos['z'])).T

if rsd:
    writelog(logfile,'... including RSD on z-axis\n')
    pos[2] = ny.mod(pos[2]+0.01*vel[2],Lbox)
    hpos[2] = ny.mod(hpos[2]+0.01*halos['vz'],Lbox)
    

powspec = PowerSpectrum(grid=GRID,Lbox=Lbox,logfile=logfile,nbin=90,lgbin=0) # 90 = 3x default. 

density = powspec.density_field(pos)
writelog(logfile,'... done with dark matter density\n')
if Nhalos > 0:
    halo_density = powspec.density_field(hpos) 
    writelog(logfile,'... done with halo density\n')
else:
    writelog(logfile,'... not computing halo density\n')
del pos,vel,IDs,hpos,halos
gc.collect()

writelog(logfile,'... Fourier transforming dark matter density\n')
FTdensity = powspec.fourier_transform_density(density)
del density
if Nhalos > 0:
    writelog(logfile,'... Fourier transforming halo density\n')
    FThalo_density = powspec.fourier_transform_density(halo_density) 
    del halo_density
gc.collect()

Pk = powspec.Pk_grid(FTdensity,input_is_FTdensity=1)
Pkh = powspec.Pk_grid(FThalo_density,input_is_FTdensity=1) if Nhalos > 0 else ny.zeros(Pk.size,dtype=float)
Pkx = powspec.Pk_grid(FTdensity,FThalo_density,input_is_FTdensity=1) if Nhalos > 0 else ny.zeros(Pk.size,dtype=float)

if Nhalos > 0:
    Pkh -= Lbox**3/(1.0*Nhalos)
if DOWNSAMPLE:
    Pk -= Lbox**3/(1.0*DOWNSAMPLE**3)
else:
    Pk -= Lbox**3/(1.0*Npart)

del FTdensity
if Nhalos > 0:
    del FThalo_density
gc.collect()

writelog(logfile,'... done\n')

writelog(logfile,'Writing to file: '+outfile_Pk+'\n')
f = open(outfile_Pk,'w')
f.write("# P(k) (DM,halos,cross) from "+snapshot+"\n")
f.write("# grid = {0:d}; downsampled to ({1:d})^3 particles\n".format(GRID,DOWNSAMPLE))
f.write("# Halos satisfy {0:.2f} < 2T/|U| < {1:.2f}\n".format(1.-QE,1.+QE))
if su:
    f.write("# "+massdef+" > {0:.4e} Msun\n".format(mmin/hubble))
    f.write("# k (1/Mpc) | P(k) (Mpc)^3 | Phalo | Pcross\n")
    f.close()
    for k in range(powspec.nbin):
        write_to_file(outfile_Pk,[powspec.ktab[k]*hubble,Pk[k]/hubble**3,Pkh[k]/hubble**3,Pkx[k]/hubble**3])
else:
    f.write("# "+massdef+" > {0:.4e} Msun/h\n".format(mmin))
    f.write("# k (h/Mpc) | P(k) (Mpc/h)^3 | Phalo | Pcross\n")
    f.close()
    for k in range(powspec.nbin):
        write_to_file(outfile_Pk,[powspec.ktab[k],Pk[k],Pkh[k],Pkx[k]])

time_this_log(start_time,logfile)
