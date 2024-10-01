import os,sys
import numpy as np
import gc
from readers import SnapshotReader,HaloReader
from correlations import PowerSpectrum
from addvalue import AddValue
from voronoi_web import Voronoi
from time import time
import multiprocessing as mp

if(len(sys.argv)==9):
    sim_stem = sys.argv[1]
    snap_start = int(sys.argv[2])
    snap_end = int(sys.argv[3])
    real = int(sys.argv[4])
    grid = int(sys.argv[5])
    downsample = int(sys.argv[6])
    Lbox = float(sys.argv[7])
    N_Proc = int(sys.argv[8])
else: 
    sim_stem = input("Specify path (e.g., `su128/delta0.0' or 'scm1024'): ")
    snap_start = int(input("Specify starting snapshot (e.g., 0-200): "))
    snap_end = int(input("Specify final snapshot (e.g., 0-200): "))
    real = int(input("Specify realisation (e.g., 1-10): "))
    grid = int(input("Specify 1-d gridsize (e.g. 256) for density field calculation: "))
    downsample = int(input("Specify 1-d n_particles (e.g. 128) for downsampling; 0 to use full sample: "))
    Lbox = float(input("Specify box size (Mpc/h): "))
    N_Proc = int(input("Specify number of cores (e.g., 16; use 1 for serial job): "))

if snap_end < snap_start:
    raise ValueError("Need snap_start <= snap_end.")

logfile = None #'log_'+sim_stem+'_r{0:d}'.format(real)+'_grid{0:d}'.format(grid)+'.log'
ps = PowerSpectrum(grid=grid,Lbox=Lbox,logfile=logfile)


###########################################
# move these hard-coded values to a file
calc_Pk = False
calc_mf = False
add_value = False
calc_vvf = False
calc_knn = True

if calc_Pk | calc_vvf | calc_knn:
    Seed = 42
if calc_Pk | calc_mf | calc_vvf | calc_knn:
    massdef = 'm200b'
    QE = 0.5
if calc_mf:
    tags = ['mvir','m200b','m200c']
    lgmmin = 11.0 # hard-coded to enable easier comparison across cosmologies
    dlgm = 0.1
if add_value:
    kmax = 0.1
if calc_vvf | calc_knn:
    lgm_cuts = [11.5,12.5,13.5] # adjust as needed. expect ~250 halos >~ 1e14 Msun/h in 200 Mpc/h box
    Ntrc_Min = 100 # minimum number of halos in any population for calculating VVF/kNN stats
if calc_vvf:
    ran_fac = 5000 # default 5000 for < 3% convergence of 2.5pc (and sub-percent for > 16pc) at z ~ 0
    force_ran_fac = True # set to True to by-pass adjustment and force above value to be used as-is,
                         # else will be adjusted according to grid size and number of halos
    vvf_percs = [2.5,16.0,50.0,84.0,97.5] # VVF percentile values to report
if calc_knn:
    target_number_density = 1e-4 # (h/Mpc)^3
    rmin = 1.0 # Mpc/h
    rmax = 40.0 # Mpc/h
    nbin = 80 # int
    n_query_points = 4000000 # int
    k_list = [1,2,3,4] # list of ints
###########################################


if calc_mf:
    ntags = len(tags)
    mass_string = ''
    for t in tags[:-1]: mass_string += t+','
    mass_string += tags[-1]


def do_this_snap(snap):
    start_time_snap = time()
    sr = SnapshotReader(sim_stem=sim_stem,real=real,snap=snap,logfile=logfile)
    if calc_Pk:
        pos = sr.read_block('pos',down_to=downsample,seed=Seed)
        # vel = sr.read_block('vel',down_to=downsample,seed=Seed)
        # ids = sr.read_block('ids',down_to=downsample,seed=Seed)

    if calc_Pk | calc_mf | calc_vvf | calc_knn:
        hr = HaloReader(sim_stem=sim_stem,real=real,snap=snap,logfile=logfile)
        Npmin = hr.calc_Npmin_default(grid)
        # this is 5 particles for sinhagad with grid=256, i.e. a very inclusive cut
        # for sahyadri with grid=512, value is 320 particles

        mmin = hr.mpart*Npmin
        hpos,halos = hr.prep_halos(massdef=massdef,QE=QE,Npmin=Npmin)

    if calc_Pk:
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

        outfile_Pk = sr.sim_path + sim_stem + '/r'+str(real)+'/Pk/Pk_{0:03d}.txt'.format(snap)
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

    if calc_mf:
        lgmmax = np.log10(hr.MHALO_MAX)
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

        outfile_mf = sr.halo_path + sim_stem + '/r'+str(real)+'/mf/mf_{0:d}.txt'.format(snap)
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

    if add_value:
        # more inclusive catalog for .vahc file
        hpos,halos = hr.prep_halos(QE=None,Npmin=0.0,keep_subhalos=True)
        av = AddValue(sim_stem=sim_stem,real=real,snap=snap,grid=grid,kmax=kmax,massdef=massdef,
                      ps=ps,density=FT_delta_dm,hpos=hpos,halos=halos,input_is_FTdensity=True)
        va = av.add_value(write_vahc=True)
        
    if calc_vvf:
        vor = Voronoi(sim_stem=sim_stem,real=real,snap=snap,Ran_Fac=ran_fac,logfile=logfile,seed=Seed)
        stats = np.zeros((len(lgm_cuts),len(vvf_percs)),dtype=float)
        for m in range(stats.shape[0]):
            mmin = 10**lgm_cuts[m]
            cond = (halos[massdef] >= mmin)
            halos_cut = halos[cond]
            hpos_cut = hpos[cond]
            Ntrc = halos_cut.size

            if Ntrc >= Ntrc_Min:
                if not force_ran_fac:
                    vor.set_ran_fac(Ntrc,grid)

                delta_trc = vor.voronoi_periodic_box(hpos_cut,ret_ran=False)
                V_trc = 1/(1.0 + delta_trc + vor.TINY)

                for s in range(stats.shape[1]):
                    stats[m,s] = np.percentile(V_trc,vvf_percs[s])

                del delta_trc,V_trc
            
            del cond,halos_cut,hpos_cut
            gc.collect()

        outfile_vvf = sr.halo_path + sim_stem + '/r'+str(real)+'/vvf/vvf_{0:d}.txt'.format(snap)
        sr.print_this('Writing to file: '+outfile_vvf,sr.logfile)
        fv = open(outfile_vvf,'w')
        fv.write("#\n# Voronoi volume functions for " + sim_stem + '/'+'r'+str(real)+'/out_' + str(snap)+"\n")
        fv.write("# This file contains VVF percentiles for various mass cuts.\n")
        fv.write("#\n# percentile | lg(m_min/h-1Msun) = ["+','.join([str(lgm) for lgm in lgm_cuts])+"]\n")
        fv.close()
        for s in range(stats.shape[1]):
            vlist = [vvf_percs[s]]
            for m in range(stats.shape[0]):
                vlist.append(stats[m,s])
            sr.write_to_file(outfile_vvf,vlist)

    if calc_knn:
        vor = Voronoi(sim_stem=sim_stem,real=real,snap=snap,logfile=logfile,seed=Seed)
        stats = np.zeros((len(lgm_cuts),len(k_list),nbin),dtype=float)
        for m in range(stats.shape[0]):
            mmin = 10**lgm_cuts[m]
            cond = (halos[massdef] >= mmin)
            halos_cut = halos[cond]
            hpos_cut = hpos[cond]
            Ntrc = halos_cut.size

            if Ntrc >= Ntrc_Min:
                bins,knn_data_vector = vor.get_knn_data_vector(hpos_cut,target_number_density=target_number_density,
                                                               rmin=rmin,rmax=rmax,nbin=nbin,n_query_points=n_query_points,k_list=k_list)
                for k in range(stats.shape[1]):
                    stats[m,k] = knn_data_vector[k]
            
            del cond,halos_cut,hpos_cut
            gc.collect()

        sr.print_this('Writing to files... ',sr.logfile)
        for k in range(len(k_list)):
            outfile_knn = sr.halo_path + sim_stem + '/r'+str(real)+'/knn/knn_k{0:d}_{1:d}.txt'.format(k_list[k],snap)
            sr.print_this('... '+outfile_knn,sr.logfile)
            fv = open(outfile_knn,'w')
            fv.write("#\n# kNN CDFs for " + sim_stem + '/'+'r'+str(real)+'/out_' + str(snap)+"\n")
            fv.write("# This file contains kNN CDFs for k = {0:d} and various mass cuts.\n".format(k_list[k]))
            fv.write("#\n# bin (Mpc/h) | lg(m_min/h-1Msun) = ["+','.join([str(lgm) for lgm in lgm_cuts])+"]\n")
            fv.close()
            for s in range(stats.shape[2]):
                vlist = [bins[s]]
                for m in range(stats.shape[0]):
                    vlist.append(stats[m,k,s])
                sr.write_to_file(outfile_knn,vlist)
            
       
    if calc_Pk:
        del pos#,vel,ids
        del delta_dm, FT_delta_dm,delta_h,FT_delta_h    
    if calc_Pk | calc_mf | calc_vvf:
        del hpos,halos
    if add_value:
        del va
    gc.collect()
    sr.print_this('... snap {0:d} done'.format(snap),sr.logfile)
    sr.time_this(start_time_snap)
    
    return


start_time = time()
if N_Proc==1:
    print('serial calculation')
    for snap in range(snap_start,snap_end+1):
        do_this_snap(snap)
else:
    pool = mp.Pool(processes=N_Proc)
    print('pool pooled')
    results = [pool.apply_async(do_this_snap,args=(snap,)) for snap in range(snap_start,snap_end+1)]
    print('results setup')
    for s in range(len(results)):
        results[s].get()
    print('results got')
    pool.close()
    pool.join()
    print('pool closed')

ps.print_this('... all done',ps.logfile)
ps.time_this(start_time)

# OLDER, SERIAL CODE
# for snap in range(snap_start,snap_end+1):
#     start_time_snap = time()
#     sr = SnapshotReader(sim_stem=sim_stem,real=real,snap=snap,logfile=logfile)
#     if calc_Pk:
#         pos = sr.read_block('pos',down_to=downsample,seed=Seed)
#         # vel = sr.read_block('vel',down_to=downsample,seed=Seed)
#         # ids = sr.read_block('ids',down_to=downsample,seed=Seed)

#     if calc_Pk | calc_mf:
#         hr = HaloReader(sim_stem=sim_stem,real=real,snap=snap,logfile=logfile)
#         Npmin = hr.calc_Npmin_default(grid)

#         mmin = hr.mpart*Npmin
#         hpos,halos = hr.prep_halos(massdef=massdef,QE=QE,Npmin=Npmin)

#     if calc_Pk:
#         delta_dm = ps.density_field(pos)
#         FT_delta_dm = ps.fourier_transform_density(delta_dm)
#         Pk_mm = ps.Pk_grid(FT_delta_dm,input_is_FTdensity=True)

#         delta_h = ps.density_field(hpos.T)
#         FT_delta_h = ps.fourier_transform_density(delta_h)
#         Pk_hh = ps.Pk_grid(FT_delta_h,input_is_FTdensity=True)
#         Pk_hm = ps.Pk_grid(FT_delta_h,input_array2=FT_delta_dm,input_is_FTdensity=True)

#         Pk_hh -= ps.Lbox**3/(halos.size + ps.TINY)
#         if downsample > 0:
#             Pk_mm -= ps.Lbox**3/(downsample**3)
#         else:
#             Pk_mm -= ps.Lbox**3/(sr.npart)

#         sr.print_this('... done',ps.logfile)

#         outfile_Pk = sr.sim_path + sim_stem + '/r'+str(real)+'/Pk_{0:03d}.txt'.format(snap)
#         sr.print_this('Writing to file: '+outfile_Pk,sr.logfile)
#         f = open(outfile_Pk,'w')
#         f.write("# P(k) (DM,halos,cross) from snapshot_{0:03d}\n".format(snap))
#         down = downsample if downsample > 0 else np.rint(sr.npart**(1/3.)).astype(int)
#         f.write("# grid = {0:d}; downsampled to ({1:d})^3 particles\n".format(grid,down))
#         f.write("# Halos satisfy {0:.2f} < 2T/|U| < {1:.2f}\n".format(1.-QE,1.+QE))
#         f.write("# "+massdef+" > {0:.4e} Msun/h\n".format(mmin))
#         f.write("# k (h/Mpc) | P(k) (Mpc/h)^3 | Phalo | Pcross\n")
#         f.close()
#         for k in range(ps.ktab.size):
#             sr.write_to_file(outfile_Pk,[ps.ktab[k],Pk_mm[k],Pk_hh[k],Pk_hm[k]])

#     if calc_mf:
#         lgmmax = np.log10(hr.MHALO_MAX)
#         nlgm = int((lgmmax-lgmmin)/dlgm)
#         mbins = np.logspace(lgmmin,lgmmax,nlgm+1)
#         mcenter = np.sqrt(mbins[1:]*mbins[:-1])
#         dlnm = np.log(mbins[1]/mbins[0])

#         dndlnm = np.zeros((ntags,nlgm),dtype=float)
#         Vbox = sr.Lbox**3
#         for t in range(ntags):
#             tag = tags[t]
#             dndlnm[t],temp = np.histogram(halos[tag],bins=mbins,density=False)
#             dndlnm[t] = dndlnm[t]/dlnm/Vbox
#             if tag==massdef:
#                 sr.print_this("Nhalos: direct = {0:d}; integrated = {1:.1f}"
#                               .format(halos.size,Vbox*dlnm*np.sum(dndlnm[t])),sr.logfile)

#         outfile_mf = sr.halo_path + sim_stem + '/r'+str(real)+'/mf_{0:d}.txt'.format(snap)
#         sr.print_this('Writing to file: '+outfile_mf,sr.logfile)
#         fh = open(outfile_mf,'w')
#         fh.write("#\n# Mass functions for " + sim_stem + '/'+'r'+str(real)+'/out_' + str(snap)+"\n")
#         fh.write("# This file contains dn/dlnm (h/Mpc)^3 for various mass definitions.\n")
#         fh.write("#\n# mass (Msun/h) | dndlnm["+mass_string+"]\n")
#         fh.close()
#         for m in range(nlgm):
#             mlist = [mcenter[m]]
#             for t in range(ntags):
#                 mlist.append(dndlnm[t,m])
#             sr.write_to_file(outfile_mf,mlist)

#     if add_value:
#         # more inclusive catalog for .vahc file
#         hpos,halos = hr.prep_halos(QE=None,Npmin=0.0,keep_subhalos=True)
#         av = AddValue(sim_stem=sim_stem,real=real,snap=snap,grid=grid,kmax=kmax,massdef=massdef,
#                       ps=ps,density=FT_delta_dm,hpos=hpos,halos=halos,input_is_FTdensity=True)
#         va = av.add_value(write_vahc=True)

#     if calc_Pk:
#         del pos#,vel,ids
#         del delta_dm, FT_delta_dm,delta_h,FT_delta_h    
#     if calc_Pk | calc_mf:
#         del hpos,halos
#     if add_value:
#         del va
#     gc.collect()
#     sr.print_this('... snap {0:d} done'.format(snap),sr.logfile)
#     sr.time_this(start_time_snap)

# ps.print_this('... all done',ps.logfile)
# ps.time_this(start_time)
