#Example run: python postprocess.py sinhagad/default256 200 200 1 100 0 200 1
import os,sys
import numpy as np
import gc
from readers import SnapshotReader,HaloReader
from correlations import PowerSpectrum
from addvalue import AddValue
from voronoi_web import Voronoi
from time import time
import multiprocessing as mp
import re

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
calc_knn = False
calc_2pcf = True 

if calc_Pk | calc_vvf | calc_knn | add_value:
    Seed = 42
if calc_Pk | calc_mf | calc_vvf | calc_knn | calc_2pcf | add_value:
    massdef = 'm200b'
    QE = 0.5
if calc_mf:
    tags = ['mvir','m200b','m200c']
    lgmmin = 11.0 # hard-coded value in log10(Msun) to enable easier comparison across cosmologies
    dlgm = 0.1
if add_value:
    kmax = 0.1
if calc_vvf | calc_knn | calc_2pcf:
    Ntrc_Min = 100 # minimum number of halos in any population for calculating VVF/kNN/2pcf stats
if calc_vvf | calc_2pcf:
    Np_min = 40 # minimum particle count for halos to define VVF
    n_cuts = [7e-4,2e-4,7e-5,2e-5] # no. density cuts in Mpc^-3: 7e-4 ~ 2.5e-3 (h/Mpc)^3 ~ 10^12.2 Msun/h at z=0 (40 particles)
if calc_vvf:
    ran_fac = 5000 # default 5000 for < 3% convergence of 2.5pc (and sub-percent for > 16pc) at z ~ 0
    force_ran_fac = True # set to True to by-pass adjustment and force above value to be used as-is,
                         # else will be adjusted according to grid size and number of halos
    vvf_percs = [2.5,16.0,50.0,84.0,97.5] # VVF percentile values to report

#!!!!!!######
if calc_2pcf or calc_vvf:
    njn=100 #Number of jackknife regions to calculate
    los=1
    obs_space=['real','rsd']
    # SHADAB CAN ADD NEW VARIABLES HERE SPECIFIC TO 2PCF
#!!!!!!######
if calc_knn:
    lgm_cuts = [12.5,13.0,13.5] # adjust as needed. expect ~250 halos >~ 1e14 Msun/h in 200 Mpc/h box
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

def out_file_root(sr,snap,quant='Pk',tag=''):
   outdir = '%s%s%s/%s/'%(sr.sim_path, sim_stem, '/stats/',quant)
   outroot='%s_%03d'%(tag,snap)
   #check if directory exists, otherwise create the directory
   res=create_nested_directory(outdir, max_depth=20)

   return outdir+outroot

def do_this_snap(snap):
    start_time_snap = time()
    sr = SnapshotReader(sim_stem=sim_stem,real=real,snap=snap,logfile=logfile,read_header=True)
    if calc_Pk | add_value:
        pos = sr.read_block('pos',down_to=downsample,seed=Seed)
        # vel = sr.read_block('vel',down_to=downsample,seed=Seed)
        # ids = sr.read_block('ids',down_to=downsample,seed=Seed)

    if calc_Pk | calc_mf | calc_vvf | calc_knn | calc_2pcf | add_value:
        hr = HaloReader(sim_stem=sim_stem,real=real,snap=snap,logfile=logfile)
        Npmin = hr.calc_Npmin_default(grid)
        # this is 5 particles for sinhagad with grid=256, i.e. a very inclusive cut
        # for sahyadri with grid=512, value is 320 particles

        mmin = hr.mpart*Npmin
        hpos,halos = hr.prep_halos(massdef=massdef,QE=QE,Npmin=Npmin)

    if calc_Pk | add_value:
        delta_dm = ps.density_field(pos)
        FT_delta_dm = ps.fourier_transform_density(delta_dm)
        
    if calc_Pk:
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
        lgmmax = np.log10(hr.MHALO_MAX/hr.hubble) # value in log10(Msun)
        nlgm = int((lgmmax-lgmmin)/dlgm)
        mbins = np.logspace(lgmmin,lgmmax,nlgm+1)
        mcenter = np.sqrt(mbins[1:]*mbins[:-1])
        dlnm = np.log(mbins[1]/mbins[0])

        dndlnm = np.zeros((ntags,nlgm),dtype=float)
        Vbox = sr.Lbox**3
        for t in range(ntags):
            tag = tags[t]
            dndlnm[t],temp = np.histogram(halos[tag]/hr.hubble,bins=mbins,density=False) # bin values in Msun
            dndlnm[t] = dndlnm[t]/dlnm/(Vbox/hr.hubble**3) # value in Mpc^-3
            if tag==massdef:
                sr.print_this("Nhalos: direct = {0:d}; integrated = {1:.1f}"
                              .format(halos.size,Vbox/hr.hubble**3*dlnm*np.sum(dndlnm[t])),sr.logfile)

        outfile_mf = sr.halo_path + sim_stem + '/r'+str(real)+'/mf/mf_{0:d}.txt'.format(snap)
        sr.print_this('Writing to file: '+outfile_mf,sr.logfile)
        fh = open(outfile_mf,'w')
        fh.write("#\n# Mass functions for " + sim_stem + '/'+'r'+str(real)+'/out_' + str(snap)+"\n")
        fh.write("# This file contains dn/dlnm (1/Mpc)^3 for various mass definitions.\n")
        fh.write("#\n# mass (Msun) | dndlnm["+mass_string+"]\n")
        fh.close()
        for m in range(nlgm):
            mlist = [mcenter[m]]
            for t in range(ntags):
                mlist.append(dndlnm[t,m])
            sr.write_to_file(outfile_mf,mlist)
        
    if calc_vvf | calc_2pcf:
        # hpos,halos will be available at this point
        if calc_vvf:
            vor = Voronoi(sim_stem=sim_stem,real=real,snap=snap,Ran_Fac=ran_fac,logfile=logfile,seed=Seed)
            #stats = np.zeros((len(n_cuts),len(vvf_percs)),dtype=float)
            vvf_arr = np.zeros((len(vvf_percs),njn+4),dtype=float)
        #!!!!!!!#########
        if calc_2pcf:
            sys.path.insert(0,sr.home_path+'/CorrelationFunction/')
            pass
            # SHADAB CAN ADD CLASS/ARRAY INITIALIZATION SPECIFIC TO 2PCF. DON'T RE-USE THE ARRAY stats
            # storage array should have first dimension equal to number of density thresholds
            # stats_2pcf = np.zeros((len(n_cuts),...),dtype=float)
        #!!!!!!!#########

        for m in range(len(n_cuts)): # ensure stats_2pcf.shape[0] == stats.shape[0] == len(n_cuts)
            # note: halos already sorted in increasing order of massdef by HaloReader
            N_req = int(n_cuts[m]*(sr.Lbox/sr.hubble)**3) # required no. of halos. recall n_cuts in 1/Mpc^3
            if (N_req <= halos.size) & (N_req >= Ntrc_Min):
                mmin = halos[massdef][-N_req]
                if (mmin > Np_min*sr.mpart): # ensure only resolved halos retained
                    halos_cut = halos[-N_req:]
                    hpos_cut = hpos[-N_req:]
                    Ntrc = halos_cut.size # should equal N_req
                    if Ntrc != N_req:
                        raise Exception('detected Ntrc = {0:d} which is not N_req = {1:d}'.format(Ntrc,N_req))

                    ncut_str=small_number_to_filename(n_cuts[m])
                    for oo, ospace in enumerate(obs_space):
                        hpos_this=hpos_cut.copy()
                        #import pylab as pl
                        #pl.hist(halos_cut['vz'],bins=100); print(100*hr.scale, Lbox)
                        #pl.show()
                        #exit()
                        if(ospace=='rsd'):
                            hpos_this[:,2] =np.mod( hpos_this[:,2]+ (halos_cut['vz']/(100*hr.scale)),Lbox)
                        if calc_vvf:
                            if not force_ran_fac:
                                vor.set_ran_fac(Ntrc,grid)
        

                            delta_trc = vor.voronoi_periodic_box(hpos_this,ret_ran=False)
                            V_trc = 1/(1.0 + delta_trc + vor.TINY)

                            vvf_arr[:,0]=vvf_percs
                            if(njn>0):
                               hpos_jn,_=add_pbc_jncol(hpos_this,rand=None,Lbox=Lbox,njn=njn,los=los)
                               #for s in range(stats.shape[1]):
                               #    stats[m,s] = np.percentile(V_trc,vvf_percs[s]i)
                               vvf_arr[:,3]=np.percentile(V_trc,vvf_percs)
                               for ii in range(0,njn):
                                   ind_mask=hpos_jn[:,-1]!=ii
#                                   print(ii,ind_mask.sum())
                                   vvf_arr[:,ii+4]=np.percentile(V_trc[ind_mask],vvf_percs)
                               vvf_arr[:,1]=np.mean(vvf_arr[:,4:],axis=1)
                               vvf_arr[:,2]=np.sqrt(njn-1)*np.std(vvf_arr[:,4:],axis=1)
                               head='perc, vvf_mean, vvf_err, vvf_all, vvf_realizations'
                            else:
                               head='perc, vvf'
                               vvf_arr[:,1]=np.percentile(V_trc,vvf_percs)

                            #out put file name
                            outroot=out_file_root(sr,snap,quant='vvf',tag='vvf_%s_%s'%(ospace,ncut_str))
                            outroot=outroot+'.txt'
                            np.savetxt(outroot,vvf_arr,header=head)
                            print('written: ',outroot)
                            #stats[m,:] = np.percentile(V_trc,vvf_percs)

                            del delta_trc,V_trc

                        #!!!!!!!#########
                        if calc_2pcf:
                            outroot=out_file_root(sr,snap,quant='2pcf',tag='2pcf_%s_%s'%(ospace,ncut_str))
                            xi_dic=estimate_2pcf(hpos_this,njn=njn,Lbox=Lbox,los=los,space=ospace,
                                    randfactor=10,interactive=0,xi2droot=None,xiprojroot=outroot)
                    # SHADAB CAN ADD 2PCF EVALUATION CODE HERE. STORE RESULTS IN stats_2pcf
                    #!!!!!!!#########

                    del halos_cut,hpos_cut
                    gc.collect()

        if calc_vvf and False:
            outfile_vvf = sr.halo_path + sim_stem + '/r'+str(real)+'/vvf/vvf_{0:d}.txt'.format(snap)
            sr.print_this('Writing to file: '+outfile_vvf,sr.logfile)
            fv = open(outfile_vvf,'w')
            fv.write("#\n# Voronoi volume functions for " + sim_stem + '/'+'r'+str(real)+'/out_' + str(snap)+"\n")
            fv.write("# This file contains VVF percentiles for various number density thresholds.\n")
            fv.write("#\n# percentile | n (Mpc-3) = 1e4 * ["+','.join(['{0:.2f}'.format(1e4*n) for n in n_cuts])+"]\n")
            fv.close()
            for s in range(stats.shape[1]):
                vlist = [vvf_percs[s]]
                for m in range(stats.shape[0]):
                    vlist.append(stats[m,s])
                sr.write_to_file(outfile_vvf,vlist)
                

        #!!!!!!!#########
        if calc_2pcf:
            pass # SHADAB CAN ADD 2PCF WRITE-OUT CODE HERE
        #!!!!!!!#########

    if calc_knn:
        vor = Voronoi(sim_stem=sim_stem,real=real,snap=snap,logfile=logfile,seed=Seed)
        stats = np.zeros((len(lgm_cuts),len(k_list)+1,nbin),dtype=float)
        for m in range(stats.shape[0]):
            mmin = 10**lgm_cuts[m]
            cond = (halos[massdef] >= mmin)
            halos_cut = halos[cond]
            hpos_cut = hpos[cond]
            Ntrc = halos_cut.size

            if Ntrc >= Ntrc_Min:
                bins,knn_data_vector = vor.get_knn_data_vector(hpos_cut,target_number_density=target_number_density,
                                                               rmin=rmin,rmax=rmax,nbin=nbin,n_query_points=n_query_points,k_list=k_list)
                stats[m,0] = bins
                for k in range(len(k_list)):
                    stats[m,k+1] = knn_data_vector[k]
            
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
                vlist = [stats[m,0,s]]
                for m in range(stats.shape[0]):
                    vlist.append(stats[m,k+1,s])
                sr.write_to_file(outfile_knn,vlist)

    if add_value:
        # more inclusive catalog for .vahc file
        hpos,halos = hr.prep_halos(QE=None,Npmin=0.0,keep_subhalos=True)
        av = AddValue(sim_stem=sim_stem,real=real,snap=snap,grid=grid,kmax=kmax,massdef=massdef,
                      ps=ps,density=FT_delta_dm,hpos=hpos,halos=halos,input_is_FTdensity=True)
        va = av.add_value(write_vahc=True)
            
       
    if calc_Pk:
        del pos#,vel,ids
        del delta_dm, FT_delta_dm,delta_h,FT_delta_h    
    if calc_Pk | calc_mf | calc_vvf | calc_knn | add_value:
        del hpos,halos
    if add_value:
        del va
    gc.collect()
    sr.print_this('... snap {0:d} done'.format(snap),sr.logfile)
    sr.time_this(start_time_snap)
    
    return

def estimate_2pcf(hpos,njn=0,Lbox=250,los=1,space='real',randfactor=2,interactive=1,xi2droot=None,xiprojroot=None):
    # Direct calling the function
    import General_FITS_selection as GFS
    import ccorr_utility_nopc as corrnopc

    #set the parameters
    pbc=1

    if(space=='real'):
        sampmode=3
        samplim=np.array([0,1.6,-40,40])
        nbins=np.array([15,60])
        xylab=['rper','rpar']
    elif(space=='rsd'):
        sampmode=0
        samplim=np.array([0,50,0,1])
        nbins=np.array([25,100])
        xylab=['r','mu']

    #datafile='../data/HOD_cat_model-bolshoi-ICTS-1.gcat'
    #Load the data
    #data=np.loadtxt(datafile)
    #remove additional axis
    #if(rsd):
    #    data[:,2]=np.mod(data[:,2]+(data[:,5]/100.0),Lbox)
    #data=data[:,:4]
    #add weight as the 4th columns
    if(hpos.shape[1]==3):
        hpos=np.column_stack([hpos,np.ones(hpos.shape[0])])


    #if(sampmode==1):
    #    samplim=np.array([0,30,-30,30])
    #    nbins=np.array([30,60])
    #elif(sampmode==0):
    #    samplim=np.array([0,30,0,1])
    #    nbins=np.array([30,100])
    #elif(sampmode==3):
    #    samplim=np.array([-1,1,-30,30])
    #    nbins=np.array([15,60])


    #Number of randoms
    nrand=randfactor*hpos.shape[0]
    # generate randoms uniformaly
    rand=GFS.randoms_fixedseed(hpos,nrand,Lbox)

    if(njn>0):
        #Add jackknife regions
        hpos,rand=add_pbc_jncol(hpos,rand,Lbox=Lbox,njn=njn,los=los)

    sumwt_dic={}
    data_c, rand_c, blen, POS_min, POS_max, sumwt_dic['SDwt'], sumwt_dic['SRwt']=GFS.prep_data_random_jn_wt(
        hpos ,rand,njn=njn,los=los,pbc=pbc,interactive=interactive,outarr='c',
        datafile='array',randfile='array')

    rlim_c =np.ascontiguousarray(np.array(samplim,dtype='double'))
    blen_c =np.ascontiguousarray(blen)
    pos_min_c=np.ascontiguousarray(POS_min)

    xi_dic=corrnopc.compute_auto(data_c, rand_c,rlim_c, nbins, blen_c, pos_min_c=pos_min_c,
            sampmode=sampmode, njn=njn, nproc=4, pbc=pbc, los=los,interactive=interactive,
                xi2droot=xi2droot,xiprojroot=xiprojroot,sumwt_dic=sumwt_dic)

    #convert the xi2d in two matrix
    if(njn==0):
        xi_dic['xi2d']=np.transpose(xi_dic['xi2d'].reshape((xi_dic[xylab[0]].size,xi_dic[xylab[1]].size)))
    else:
         xi_dic['xi2d']=np.transpose(xi_dic['xi2d'][:,0].reshape((xi_dic[xylab[0]].size,xi_dic[xylab[1]].size)))

    return xi_dic

def create_nested_directory(path, max_depth=20):
    # Convert the path to an absolute path
    abs_path = os.path.abspath(path)

    # Split the path into individual components
    components = abs_path.split(os.sep)

    # Check if the number of nested directories exceeds the maximum allowed depth
    if len(components) > max_depth:
        raise ValueError(f"Maximum directory depth of {max_depth} exceeded")

    # Initialize the current path
    current_path = os.sep if abs_path.startswith(os.sep) else ""

    # Iterate through the path components
    for component in components:
        current_path = os.path.join(current_path, component)

        # Check if the current path exists
        if not os.path.exists(current_path):
            try:
                # Create the directory
                os.mkdir(current_path)
                print(f"Created directory: {current_path}")
            except OSError as e:
                print(f"Error creating directory {current_path}: {e}")
                return False

    #print(f"Successfully created or verified the path: {abs_path}")
    return True

def small_number_to_filename(number):
    # Convert the number to a string in scientific notation
    sci_notation = f"{number:.2e}"

    # Use regex to separate the coefficient and exponent
    match = re.match(r"(\d+\.\d+)e([+-]\d+)", sci_notation)
    if not match:
        raise ValueError("Invalid number format")

    coefficient, exponent = match.groups()
    # Remove leading zeros from the exponent
    exponent_strip = exponent.lstrip('+0').lstrip('-0')
    if exponent_strip == "":
        exponent_strip = "0"

    # Construct the filename-friendly representation
    if exponent.startswith('-'):
        return f"{coefficient.replace('.', 'p')}em{exponent_strip}"
    else:
        return f"{coefficient.replace('.', 'p')}e{exponent_strip}"

def is_perfect_power(number,power):
    """
    Indicates (with True/False) if the provided number is a perfect power.
    """
    number = abs(number)  # Prevents errors due to negative numbers
    return round(number ** (1 / power)) ** power == number

def add_pbc_jncol(data,rand=None,Lbox=200,njn=0,los=1):
   '''If the input is a periodic box and los is along z axis then jacknife region is simply equal area region in the x-y space which can be done in using this function and not needed to be supplied with data file.
   If njn is perfect cube then 3d jacknife is implemented
   If njn is perfect square then jn split is in x-y plane
   '''


   if(is_perfect_power(njn,3)):
       jntype='3d'
       NJNx=int(np.round(njn ** (1. / 3)))
       NJNy=NJNx;NJNz=NJNx
       print('Using 3d Jacknife:', NJNx, NJNy, NJNz)
   elif(is_perfect_power(njn,2)):
       jntype='2d'
       NJNx=int(np.sqrt(njn))
       NJNy=int(njn/NJNx)
       print('Using 2d Jacknife:', NJNx, NJNy)
   else:
       print('''error: njn must be either a perfect square or perfect cube''')
       sys.exit()

   #adding jacknife regions
   if(njn>0 and los==1):
      POS_min=[0,0,0]; POS_max=[Lbox,Lbox,Lbox];blen=POS_max
      #POS_min,POS_max, blen=getminmax(data,rand=rand)
      #NJNx=int(np.sqrt(args.njn))
      #NJNy=int(args.njn/NJNx)
      for ii in range(0,2):
         if(ii==0): 
             mat=data
         elif(rand is None):
             continue
         else:
             mat=rand

         #get the x and y indx as integers
         indx=np.zeros(mat[:,0].size,dtype=int)
         indy=np.zeros(mat[:,0].size,dtype=int)

         for kk in range(0,indx.size):
             indx[kk]=int(NJNx*(mat[kk,0]-POS_min[0])/blen[0])
             indy[kk]=int(NJNy*(mat[kk,1]-POS_min[1])/blen[1])
         #apply modulo operation on x an y index
         indx=np.mod(indx,NJNx)
         indy=np.mod(indy,NJNy)

         if(jntype=='2d'):
            jnreg=NJNy*indx+indy
         elif(jntype=='3d'):
            indz=int_(NJNz*(mat[:,2]-POS_min[2])/blen[2])
            indz=np.mod(indz,NJNz)
            jnreg=NJNz*(NJNy*indx+indy)+indz

         #convert index to integers
         #indx.astype(np.int64); indy.astype(np.int64);
         mat=np.column_stack([mat,jnreg])

         if(ii==0): data=mat
         else: rand=mat
      return data,rand
   else:
      print('not appropriate input to add jacknife internally')
      #sys.exit()
      return 0

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
