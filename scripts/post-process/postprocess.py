import os,sys
import numpy as np
import gc
from readers import SnapshotReader,HaloReader
from correlations import PowerSpectrum
from addvalue import AddValue
from time import time

if(len(sys.argv)==8):
    sim_stem = sys.argv[1]
    snap_start = int(sys.argv[2])
    snap_end = int(sys.argv[3])
    real = int(sys.argv[4])
    grid = int(sys.argv[5])
    downsample = int(sys.argv[6])
    Lbox = float(sys.argv[7])
else: 
    sim_stem = input("Specify path (e.g., `su128/delta0.0' or 'scm1024'): ")
    snap_start = int(input("Specify starting snapshot (e.g., 0-200): "))
    snap_end = int(input("Specify final snapshot (e.g., 0-200): "))
    real = int(input("Specify realisation (e.g., 1-10): "))
    grid = int(input("Specify 1-d gridsize (e.g. 256) for density field calculation: "))
    downsample = int(input("Specify 1-d n_particles (e.g. 128) for downsampling; 0 to use full sample: "))
    Lbox = float(input("Specify box size (Mpc/h): "))

if snap_end < snap_start:
    raise ValueError("Need snap_start <= snap_end.")

logfile = None #'log_'+sim_stem+'_r{0:d}'.format(real)+'_grid{0:d}'.format(grid)+'.log'
ps = PowerSpectrum(grid=grid,Lbox=Lbox,logfile=logfile)

Seed = 42
kmax = 0.1
massdef = 'm200b'
QE = 0.5

for snap in range(snap_start,snap_end+1):
    start_time = time()
    sr = SnapshotReader(sim_stem=sim_stem,real=real,snap=snap,logfile=logfile)
    pos = sr.read_block('pos',down_to=downsample,seed=Seed)
    # vel = sr.read_block('vel',down_to=downsample,seed=Seed)
    # ids = sr.read_block('ids',down_to=downsample,seed=Seed)

    hr = HaloReader(sim_stem=sim_stem,real=real,snap=snap,logfile=logfile)
    Npmin = hr.calc_Npmin_default(grid)

    mmin = hr.mpart*Npmin
    hpos,halos = hr.prep_halos(massdef=massdef,QE=QE,Npmin=Npmin)

    delta_dm = ps.density_field(pos)
    FT_delta_dm = ps.fourier_transform_density(delta_dm)
    Pk_mm = ps.Pk_grid(FT_delta_dm,input_is_FTdensity=True)

    delta_h = ps.density_field(hpos.T)
    FT_delta_h = ps.fourier_transform_density(delta_h)
    Pk_hh = ps.Pk_grid(FT_delta_h,input_is_FTdensity=True)
    Pk_hm = ps.Pk_grid(FT_delta_h,input_array2=FT_delta_dm,input_is_FTdensity=True)

    Pk_hh -= ps.Lbox**3/(halos.size + ps.TINY)
    if downsample > 0:
        Pk_mm -= ps.Lbox**3/(downsample**3)
    else:
        Pk_mm -= ps.Lbox**3/(sr.npart)

    sr.print_this('... done',ps.logfile)

    outfile_Pk = sr.sim_path + sim_stem + '/r'+str(real)+'/Pk_{0:03d}.txt'.format(snap)
    sr.print_this('Writing to file: '+outfile_Pk,sr.logfile)
    f = open(outfile_Pk,'w')
    f.write("# P(k) (DM,halos,cross) from snapshot_{0:03d}\n".format(snap))
    down = downsample if downsample > 0 else np.rint(sr.npart**(1/3.)).astype(int)
    f.write("# grid = {0:d}; downsampled to ({1:d})^3 particles\n".format(grid,down))
    f.write("# Halos satisfy {0:.2f} < 2T/|U| < {1:.2f}\n".format(1.-QE,1.+QE))
    f.write("# "+massdef+" > {0:.4e} Msun/h\n".format(mmin))
    f.write("# k (h/Mpc) | P(k) (Mpc/h)^3 | Phalo | Pcross\n")
    f.close()
    for k in range(ps.ktab.size):
        sr.write_to_file(outfile_Pk,[ps.ktab[k],Pk_mm[k],Pk_hh[k],Pk_hm[k]])

    tags = ['mvir','m200b','m200c'] 
    ntags = len(tags)
    mass_string = ''
    for t in tags[:-1]: mass_string += t+','
    mass_string += tags[-1]

    lgmmin = 11.0 #np.log10(mmin) # hard-code to enable easier comparison across cosmologies
    lgmmax = np.log10(hr.MHALO_MAX)
    dlgm = 0.1
    nlgm = int((lgmmax-lgmmin)/dlgm)
    mbins = np.logspace(lgmmin,lgmmax,nlgm+1)
    mcenter = np.sqrt(mbins[1:]*mbins[:-1])
    dlnm = np.log(mbins[1]/mbins[0])

    dndlnm = np.zeros((ntags,nlgm),dtype=float)
    Vbox = sr.Lbox**3
    for t in range(ntags):
        tag = tags[t]
        dndlnm[t],temp = np.histogram(halos[tag],bins=mbins,density=False)
        dndlnm[t] = dndlnm[t]/dlnm/Vbox
        if tag==massdef:
            sr.print_this("Nhalos: direct = {0:d}; integrated = {1:.1f}"
                          .format(halos.size,Vbox*dlnm*np.sum(dndlnm[t])),sr.logfile)

    outfile_mf = sr.halo_path + sim_stem + '/r'+str(real)+'/mf_{0:d}.txt'.format(snap)
    sr.print_this('Writing to file: '+outfile_mf,sr.logfile)
    fh = open(outfile_mf,'w')
    fh.write("#\n# Mass functions for " + sim_stem + '/'+'r'+str(real)+'/out_' + str(snap)+"\n")
    fh.write("# This file contains dn/dlnm (h/Mpc)^3 for various mass definitions.\n")
    fh.write("#\n# mass (Msun/h) | dndlnm["+mass_string+"]\n")
    fh.close()
    for m in range(nlgm):
        mlist = [mcenter[m]]
        for t in range(ntags):
            mlist.append(dndlnm[t,m])
        sr.write_to_file(outfile_mf,mlist)


    # more inclusive catalog for .vahc file
    hpos,halos = hr.prep_halos(QE=None,Npmin=0.0,keep_subhalos=True)
    av = AddValue(sim_stem=sim_stem,real=real,snap=snap,grid=grid,kmax=kmax,massdef=massdef,
                  ps=ps,density=FT_delta_dm,hpos=hpos,halos=halos,input_is_FTdensity=True)
    va = av.add_value(write_vahc=True)

    del pos#,vel,ids
    del delta_dm, FT_delta_dm,delta_h,FT_delta_h
    del hpos,halos
    del va
    gc.collect()
    sr.print_this('... snap {0:d} done'.format(snap),sr.logfile)

ps.print_this('... all done',ps.logfile)
ps.time_this(start_time)
