#!/mnt/csoft/tools/anaconda2/bin/python2

import paths 
import constants as con

import sys
sys.path.append(paths.python_path)
import numpy as ny
import scipy.linalg as syal
from voronoi_web import Voronoi
from time import time
from correlations import PowerSpectrum
import gc

########################################################
# Read GADGET-2 snapshot and ROCKSTAR halo catalog 
# and calculate following info:
# -- tidal environment of each halo
# -- halo-by-halo bias
# These additional properties written to ***.vahc
########################################################


if(len(sys.argv)==9):
    sim_stem = sys.argv[1]
    GRID = int(sys.argv[2])
    NFILE = int(sys.argv[3])
    TREES = int(sys.argv[4])
    sreal = int(sys.argv[5])
    nreal = int(sys.argv[6])
    ssnap = int(sys.argv[7])
    esnap = int(sys.argv[8])
else: 
    sim_stem = raw_input("Specify path (e.g., `su128/delta0.0' or 'scm1024'): ")
    GRID = int(raw_input("Specify 1-d gridsize (e.g. 256) for density field calculation: "))
    NFILE = int(raw_input("Specify number of files per gadget snapshot : "))
    TREES = int(raw_input("Specify whether (1) or not (0) merger trees available : "))
    sreal = int(raw_input("Specify starting realisation (e.g., 1-10): "))
    nreal = int(raw_input("Specify number of realisations (e.g., 1-10): "))
    ssnap = int(raw_input("Specify starting snapshot number (e.g., 0-9): "))
    esnap = int(raw_input("Specify ending snapshot number (e.g., 1-10): "))

# TREES and NFILE not used, can be removed from input args.
#TREES = bool(TREES)
ereal = sreal + nreal - 1
REALS = ny.arange(sreal,ereal+1,dtype=int)
SNAPS = ny.arange(ssnap,esnap+1,dtype=int)


# number of grid cells enclosed inside 2*R_200b
NENCL_2R200B = 0.2 # 0.2

# size of grid on which to interpolate smoothing scales
N_RG = 60 # 60 is fine for ~1-2% accuracy

OM_DEF = 0.276         # default Om for RG calculations
M_NORM = 6.4173e13     # normalising mass scale for RG calculations
RG_2R200b_NORM = 0.894 # RGeff(2R200b) at normalising mass scale and default Om.

MASSDEF = 'm200b'

# whether or not to deconvolve CIC filter
CIC = False
# setting False based on tests done during tidal_maps analysis. 
# see email to oliver+ravi dated 26 Jun 2018 sub: grid artifacts

for real in REALS:
    out_dir = paths.halo_path + sim_stem + '/r'+str(real)+'/'
    for snapnum in SNAPS[::-1]:
        start_time = time()
        va_cat = 'out_' + str(snapnum) + '.vahc'
        LOGFILE = out_dir + 'vahc_'+str(snapnum)+'.log'

        vor = Voronoi(sim_stem=sim_stem,RSD=False,
                      real=real,snap=snapnum,logfile=LOGFILE,firstcall=True) 
        # consider setting nproc=NPROC

        Npmin = 200*NENCL_2R200B*(vor.Npart/1024**3.)*(512./GRID)**3
        mmin = vor.mpart*Npmin

        if GRID > 688:
            vor.print_this("Grid is too fine. Switching to 688^3",vor.logfile)
            GRID = 688

        vor.print_this("Value added halo catalogs. All quantities computed on {0:d}^3 grid".format(GRID),vor.logfile)

        pos,vel,IDs = vor.prep_sim_data()

        # MHALO_MAX = 7.7e14 if vor.Lbox > 150.0 else 1e14
        MHALO_MAX = 3e15
        # 7.7e14 gives N(>mmax) = 10 for SUwmap7 with Lbox=300Mpc/h


        # hpos,halos = vor.prep_halo_data(va=False,QE=ny.inf,massdef=MASSDEF,Npmin=0,keep_subhalos=True)
        halos = vor.read_this(vor.real,vor.snap,VA=False)
        hpos = ny.array(zip(halos['x'],halos['y'],halos['z']))
        Nhalo = halos.size
        vor.print_this("... ... {0:d} objects in catalog".format(Nhalo),vor.logfile)

        hpos_grid = hpos*GRID/vor.Lbox
        hgrid_round = ny.fmod(ny.round(hpos_grid),GRID).astype('int')
        # shape(Nhalo,3)
        # hgrid_round contains integer vectors identifying the cell (round) corresponding to each halo: equiv to NGP

        vor.print_this("... ignoring halos smaller than mmin = {0:.3e}Msun/h for tidal tensor".format(mmin),vor.logfile)
        allowed_h = ny.where(halos[MASSDEF] > mmin)[0]
        N_allowed = allowed_h.size
        vor.print_this("... ... {0:d} halos will be used".format(N_allowed),vor.logfile)

        va_props = ny.zeros(Nhalo,dtype=con.halo_data_vac)
        va_props['ID'] = halos['ID']

        vor.print_this("... setting up k-space quantities",vor.logfile)
        powspec = PowerSpectrum(grid=GRID,Lbox=vor.Lbox,logfile=vor.logfile)

        # real space matter density field
        delta_matter = powspec.density_field(pos.T)

        ###################
        # Halo-by-halo bias
        ###################
        NTRC_SPLIT = 80000 # for sample size larger than this, split into chunks 

        RANGE_MAX = ny.max([1,
                            ny.rint(Nhalo/NTRC_SPLIT*(vor.kmax - 2*ny.pi/vor.Lbox)/(0.1 - 2*ny.pi/300)).astype(int)])

        if RANGE_MAX == 1:
            b1_tmp,b1_k,ktab_trunc = vor.haloByhalo_b1(delta_matter,hpos,input_is_density=True,CIC=True) 
            va_props['b1wtd'][:] = b1_tmp 
            del b1_tmp
            gc.collect()
        else:
            vor.print_this("... ... splitting tracers into {0:d} ranges".format(RANGE_MAX),vor.logfile)
            for r in range(RANGE_MAX):
                sl = ny.s_[r*Nhalo/RANGE_MAX:(r+1)*Nhalo/RANGE_MAX]
                b1_tmp,b1_k,ktab_trunc = vor.haloByhalo_b1(delta_matter,hpos[sl],input_is_density=True,CIC=True)
                va_props['b1wtd'][sl] = b1_tmp 
                del b1_tmp
                gc.collect()
                vor.print_this("... ... ... range {0:d} of {1:d} done".format(r+1,RANGE_MAX),vor.logfile)

        del hpos,hpos_grid
        gc.collect()

        vor.print_this("... Fourier transforming density",vor.logfile)
        FTdensity = powspec.fourier_transform_density(delta_matter,CIC=CIC)

        ###################
        # Tidal environment
        ###################
        RGmin = 0.325*(vor.Lbox/300.0)*(NENCL_2R200B/8.0)**(1/3.)*(512.0/GRID) 
        # min defined by enclosed cells
        RGmax = 5*RG_2R200b_NORM*((MHALO_MAX/M_NORM)*(OM_DEF/vor.Om))**(1/3.) 
        # max defined by 10*R200b of largest allowed halo (encloses 8x)
        RGtab = ny.logspace(ny.log10(RGmin),ny.log10(RGmax),N_RG)

        smoothing_scales = ny.concatenate((RGtab,ny.array([2.0,3.0,5.0])))/vor.Lbox
        N_scales = smoothing_scales.size

        psi_eigvals = ny.zeros((Nhalo,N_scales,3),dtype=float)

        vor.print_this("... calculating tidal environment for {0:d} of {1:d} halos".format(N_allowed,Nhalo),vor.logfile)
        for s in range(N_scales):
            Rs = smoothing_scales[s]
            vor.print_this("... ... Rs = {0:.2f}Mpc/h".format(Rs*vor.Lbox),vor.logfile)
            psi11,psi22,psi33,psi12,psi13,psi23 = powspec.tidal_tensor_field(FTdensity,Rs,input_is_FTdensity=1)

            vor.print_this('... ... looping over halos',vor.logfile)
            cnt = 0
            for h in allowed_h:
                hcell = hgrid_round[h]

                # consider using interpolation over nbrs below
                psi11_h = psi11[hcell[0],hcell[1],hcell[2]].real
                psi22_h = psi22[hcell[0],hcell[1],hcell[2]].real
                psi33_h = psi33[hcell[0],hcell[1],hcell[2]].real
                psi12_h = psi12[hcell[0],hcell[1],hcell[2]].real
                psi13_h = psi13[hcell[0],hcell[1],hcell[2]].real
                psi23_h = psi23[hcell[0],hcell[1],hcell[2]].real

                psi_mat = ny.matrix(ny.zeros((3,3),dtype=float))
                psi_mat[ny.diag_indices(3)]    = ny.array([psi11_h,psi22_h,psi33_h])
                psi_mat[ny.triu_indices(3,1)]  = ny.array([psi12_h,psi13_h,psi23_h])
                psi_mat[ny.tril_indices(3,-1)] = ny.array([psi12_h,psi13_h,psi23_h])

                psi_eigvals[h,s] = syal.eigvalsh(psi_mat)
                if ((cnt+1) % int(1.0*N_allowed/5) == 0):
                    vor.print_this("... ... ... {0:.1f}% done".format(100.0*(cnt+1)/N_allowed),vor.logfile)
                cnt += 1

            del psi11,psi22,psi33,psi12,psi13,psi23
            gc.collect()


        psi_eigvals = ny.transpose(psi_eigvals,(2,0,1))
        # shape (3,Nhalo,N_scales)

        scale_strings = con.scale_strings
        N_scalestrings = len(scale_strings)
        psi_eigvals_final = ny.zeros((N_scalestrings,3,Nhalo),dtype=float)

        for s in range(-3,0):
            psi_eigvals_final[s] = psi_eigvals[:,:,s]

        vor.print_this('... interpolating to Nx R200b for all halos',vor.logfile)
        for s in range(N_scalestrings-3):
            vor.print_this("... ... {0:d}x R200b".format(2*(s+1)),vor.logfile)
            RGeff = (s+1)*RG_2R200b_NORM*((halos[MASSDEF]/M_NORM)*(OM_DEF/vor.Om))**(1/3.) 
            # works because s=0,1,2,3 so s+1=1x,2x,3x,4x base value
            vor.print_this('... ... ... looping over halos',vor.logfile)
            cnt = 0
            for h in allowed_h:
                for eig in range(3):
                    psi_eigvals_final[s,eig,h] = ny.interp(ny.log10(RGeff[h]),ny.log10(RGtab),psi_eigvals[eig,h,:N_RG])
                if ((cnt+1) % int(1.0*N_allowed/5) == 0):
                    vor.print_this("... ... ... ... {0:.1f}% done".format(100.0*(cnt+1)/N_allowed),vor.logfile)
                cnt += 1

        for s in range(N_scalestrings):
            va_props['lam1_'+scale_strings[s]] = psi_eigvals_final[s,0]
            va_props['lam2_'+scale_strings[s]] = psi_eigvals_final[s,1]
            va_props['lam3_'+scale_strings[s]] = psi_eigvals_final[s,2]

        del psi_eigvals,psi_eigvals_final
        gc.collect()


        H_eigvals = ny.zeros((Nhalo,2,3),dtype=float)

        vor.print_this("... calculating (dimensionless) density Hessian for {0:d} of {1:d} halos"
                       .format(N_allowed,Nhalo),vor.logfile)
        for s in [-2,-1]:
            Rs = smoothing_scales[s]
            vor.print_this("... ... Rs = {0:.2f}Mpc/h".format(Rs*vor.Lbox),vor.logfile)
            H11,H22,H33,H12,H13,H23 = powspec.density_hessian_field(FTdensity,Rs,input_is_FTdensity=1)

            vor.print_this('... ... looping over halos',vor.logfile)
            cnt = 0
            for h in allowed_h:
                hcell = hgrid_round[h]

                # consider using interpolation over nbrs below
                H11_h = H11[hcell[0],hcell[1],hcell[2]].real
                H22_h = H22[hcell[0],hcell[1],hcell[2]].real
                H33_h = H33[hcell[0],hcell[1],hcell[2]].real
                H12_h = H12[hcell[0],hcell[1],hcell[2]].real
                H13_h = H13[hcell[0],hcell[1],hcell[2]].real
                H23_h = H23[hcell[0],hcell[1],hcell[2]].real

                H_mat = ny.matrix(ny.zeros((3,3),dtype=float))
                H_mat[ny.diag_indices(3)]    = ny.array([H11_h,H22_h,H33_h])
                H_mat[ny.triu_indices(3,1)]  = ny.array([H12_h,H13_h,H23_h])
                H_mat[ny.tril_indices(3,-1)] = ny.array([H12_h,H13_h,H23_h])

                H_eigvals[h,s] = syal.eigvalsh(H_mat)
                if ((cnt+1) % int(1.0*N_allowed/5) == 0):
                    vor.print_this("... ... ... {0:.1f}% done".format(100.0*(cnt+1)/N_allowed),vor.logfile)
                cnt += 1

            del H11,H22,H33,H12,H13,H23
            gc.collect()

        H_eigvals = ny.transpose(H_eigvals,(1,2,0))
        # shape (N_scales,3,Nhalo)

        for s in [-2,-1]:
            va_props['lamH1_'+scale_strings[s]] = H_eigvals[s,0]
            va_props['lamH2_'+scale_strings[s]] = H_eigvals[s,1]
            va_props['lamH3_'+scale_strings[s]] = H_eigvals[s,2]

        del H_eigvals
        gc.collect()

        vor.print_this("... comparing to T10 fit",vor.logfile)
        dlgm = 0.5
        nm = ny.rint((15.0-11.0)/dlgm).astype(int) # hard-coded for range of T10 at z=0
        mbin = ny.logspace(11.0,15.0,nm+1)
        mmid = ny.zeros(nm,dtype=float)
        b1msd = ny.zeros(nm,dtype=float)
        vor.print_this("... ... binning bias values by m200b for parents",vor.logfile)
        for m in range(nm):
            cond_bin = (halos['m200b'] >= mbin[m]) & (halos['m200b'] < mbin[m+1]) & (halos['pid'] == -1)
            nhbin = ny.where(cond_bin)[0].size
            if nhbin > 1:
                mmid[m] = ny.median(halos['m200b'][cond_bin])  
                b1msd[m] = ny.mean(va_props['b1wtd'][cond_bin])  
            else: 
                mmid[m] = ny.sqrt(mbin[m]*mbin[m+1])
            del cond_bin
            gc.collect()
        vor.print_this("... ... calculating binned T10 fit",vor.logfile)
        b1bin = ny.zeros(nm+1,dtype=float)
        nbin = ny.zeros(nm+1,dtype=float)
        for m in range(nm+1):
            nT10,bias,biasmass = vor.massfuncbias_thresh(mbin[m],z=vor.redshift) # appropriate for m200b
            b1bin[m] = bias
            nbin[m] = nT10
        b1T10 = ny.diff(nbin*b1bin)/ny.diff(nbin)
        b1res = b1msd/b1T10 - 1.0
        vor.print_this("... ... residuals    : ["+','.join(["{0:.2f}".format(b1res[m]) for m in range(nm)])+"]",vor.logfile)
        vor.print_this("... ... for lg(m200b): ["+','.join(["{0:.2f}".format(ny.log10(mmid[m])) for m in range(nm)])+"]",
                       vor.logfile)
        # vor.time_this_log(start_time,vor.logfile)
        # raise ValueError("")

        del halos
        gc.collect()

        vor.print_this('Writing to file: '+ sim_stem + '/r'+str(real)+'/' + va_cat,vor.logfile)
        va_cat = out_dir + va_cat
        header_string =  "# Value added halo catalog. All quantities computed on {0:d}^3 grid\n".format(GRID)
        header_string += '# haloID'
        for s in scale_strings:
            header_string += ' lam1_'+s+' lam2_'+s+' lam3_'+s
        for s in scale_strings[-2:]:
            header_string += ' lamH1_'+s+' lamH2_'+s+' lamH3_'+s
        header_string += ' b1 b1wtd'
        header_string += '\n'
        f = open(va_cat,'w')
        f.write(header_string)
        f.close()
        vor.write_structured(va_cat,va_props)

        vor.time_this_log(start_time,vor.logfile)

