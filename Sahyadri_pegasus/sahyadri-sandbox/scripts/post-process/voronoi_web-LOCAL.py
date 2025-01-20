#!/home/aseem/anaconda3/bin/python

import numpy as np
import scipy.spatial as spatial
import scipy.ndimage as syndim
import scipy.linalg as syal
import multiprocessing as mp
import sys
sys.path.append('/home/aseem/iucaa/PICASA/picasa/code')
from picasa import PICASA
sys.path.append('/home/aseem/python_modules/cosmology/')
from universe import *
sys.path.append('/home/aseem/simulations_and_mocks/code/N-body/Gadget-Analysis/gadget2-lite/')
import paths
import constants as con
from environment import TidalAnisotropy
from halo_reader import HaloReader
from gadget2_reader import Gadget2Reader
from correlations import PowerSpectrum
import gc

#######################################################################

class Voronoi(HaloReader,Cosmology,Constants,Utilities,TidalAnisotropy,PICASA):
    """ Voronoi methods for density estimation in halo (and, later, galaxy) catalogs."""
    ############################################################
    ############################################################
    def __init__(self,sim_stem='scm1024',EXTERNAL_STORAGE=False,DB=1,RSD=False,
                 logfile=None,real=1,snap=200,have_hdf=True,
                 RAN_FAC=30000,fit_bias=False,
                 verbose=True,seed=None,firstcall=False):
        """ Initialise cosmology class, constants, halo reader, utilities, and various constants and flags. 
        
             Methods:
             -- get_snapshot_file() [convenience]
             -- set_ran_fac() [convenience]
             -- prep_halo_data()
             -- prep_sim_data()
             -- voronoi_periodic_box()
             -- voronoi_to_grid()
             -- haloByhalo_b1()
             -- voronoi_dm()
        """

        self.NFILE = con.nfile_dict[sim_stem]
        Constants.__init__(self)
        Utilities.__init__(self)
        PICASA.__init__(self)
        TidalAnisotropy.__init__(self)
        HaloReader.__init__(self,sim_stem,EXTERNAL_STORAGE,DB,logfile)
        self.RSD = RSD
        self.READ_VEL = self.RSD # can be made more complex later
        self.READ_ID = False # can be altered later
        self.logfile = logfile
        self.verbose = verbose
        self.have_hdf = have_hdf

        self.real = real
        self.snap = snap

        self.seed = seed

        self.FIT_BIAS = fit_bias 
        # If True, h-by-h bias will return b10 where b1(k) = b10 + b11*k**2
        # else will return weighted mean of b1(k)

        self.NHALO_HI = 10000
        self.NRAN_PER_CELL = 4.0 # 4.0: minimum average number of randoms in smallest cell
        self.GRID_MAX = 384 # max grid size we can handle
        # use these to adjust self.RAN_FAC for individual halo populations

        self.NTRC_SPLIT = 80000 # for use in speeding up hbyh bias calc

        self.RADIUS_FAC = 4.0
        # factor to scale minimum halo radius by, for Gaussian smoothing

        # maximum number of randoms allowed
        # used 2e8 for VVF paper [most VVF stats convgd at 6e7]
        # voronoi_dm seems ~10% converged between 4e8-8e8
        self.MAX_RANDOMS = 5e8 
        
        # factor to multiply with Nhalo to set number of randoms for uniform sampling of volume.
        self.RAN_FAC_DEFAULT = 1*RAN_FAC 
        self.RAN_FAC = 1*RAN_FAC 
        # Default 30k based on output of convergence_ranfac.py (<~1% conv of 1+dhalo) 
        # and also based on having enough randoms in 256^3 grid for 1000-halo sample.
        # 10k is also fine (<~2% conv of 1+dhalo) and may be useful for large samples.
        # RAN_FAC_DEFAULT will be kept fixed.
        # RAN_FAC may change from population to population of tracers.

        if self.verbose:
            print_string = "--------------------------------\n"
            print_string += "Voronoi number density estimator\n"
            print_string += "--------------------------------\n"
            print_string += "... using simulation " + self.sim_stem
            overwrite = True if firstcall else False
            self.print_this(print_string,self.logfile,overwrite=overwrite)

        self.infile_snap = self.get_snapshot_file(self.real,self.snap)
        self.g2reader = Gadget2Reader(read_pos=True,read_vel=self.READ_VEL,read_ID=self.READ_ID,
                                      verbose=self.verbose,seed=self.seed)
        self.g2header = Gadget2Reader(read_pos=False,read_vel=False,read_ID=False,
                                      verbose=self.verbose,seed=self.seed)
        pos,vel,IDs = self.g2header.read_this(self.infile_snap)
        del pos,vel,IDs

        self.Lbox = self.g2header.Lbox 
        self.hubble = self.g2header.hubble
        self.Om = self.g2header.Om
        self.OLam = self.g2header.OLam
        self.Ok = 1.0 - self.Om - self.OLam
        self.Npart = self.g2header.npart_tot[self.g2header.ptype]
        self.redshift = self.g2header.redshift

        self.Mtot = self.Lbox**3*self.Om*self.rhoc
        self.mpart = self.Mtot/self.Npart

        # kmax in h/Mpc for halo-by-halo bias calculation. Should be redshift-dependent.
        self.kmax = 0.1 # 0.2?? if self.FIT_BIAS else 0.1

        Cosmology.__init__(self,Om=self.Om,Ok=self.Ok,hubble=self.hubble,Tcmb=0.0,ns=self.ns,sig8=self.sig8,
                           Pklin=self.sim_stem,verbose=self.verbose)
        # Note Tcmb = 0 since simulations don't have radiation.
        # Ob is irrelevant so leave at default value.

        self.Om_z = self.Om*(1+self.redshift)**3/self.EHub(self.redshift)**2

        if self.verbose:
            print_string = "--------------------------------\n"
            print_string += "... simulation box size Lbox = {0:.1f} Mpc/h\n".format(self.Lbox)
            print_string += ("... cosmology (Om,OLam,hubble,sig8,ns) = ({0:.4f},{1:.4f},{2:.3f},{3:.3f},{4:.4f})\n"
                             .format(self.Om,self.OLam,self.hubble,self.sig8,self.ns))
            rsd_str = 'ON' if self.RSD else 'OFF'
            print_string += "... redshift space distortions are " + rsd_str + '\n'
            print_string += "... working at redshift z = {0:.3f}".format(self.redshift)
            self.print_this(print_string,self.logfile)

        # if self.verbose:
        #     self.print_this("... calculating linear 2pcf",self.logfile)
        # #########################
        # self.dlnr = 0.003 # 0.001 is safe, 0.003 also fine
        # self.rmin = 0.01*self.Lbox/self.MAX_RANDOMS**(1/3.) # use 0.01 out front
        # self.rmax = 0.87*self.Lbox # 0.87 = sqrt(3)/2
        # self.nr = np.min([int(np.log(self.rmax/self.rmin)/self.dlnr),15000]) # max 15k due to memory on 32GB machine.
        # #########################
        # self.rtab = np.logspace(np.log10(self.rmin),np.log10(self.rmax),self.nr)
        # self.dlnr = np.log(self.rtab[1]/self.rtab[0])
        # self.xi_lin = self.calc_xi_lin(self.rtab)

        # internal variables for binning in stochastic dust model
        self.stoch_dlgm = 8.0
        self.stoch_nmbin = 65

        if self.verbose:
            self.print_this("... initialisation complete\n--------------------------------",self.logfile)

    ############################################################
    ############################################################

    ############################################################
    def get_snapshot_file(self,real,snap,nf=0):
        """ Convenience function. 
             Returns snapshot filename with full path for specific realisation and snapshot (and file number nf).
        """
        snapshot = "snapshot_{0:03d}".format(snap)
        if nf > self.NFILE-1:
            raise ValueError("File number can't exceed {0:d}.".format(self.NFILE-1))
        if self.NFILE > 1:
            snapshot += ".{0:d}".format(nf)
        if self.have_hdf:
            snapshot += '.hdf5'
        snapshot_read = self.sim_stem + '/r'+str(real)+'/' + snapshot
        infile_snap = paths.sim_path + snapshot_read
        return infile_snap
    ############################################################

    ############################################################
    def prep_halo_data(self,va=True,QE=0.5,massdef='mvir',Npmin=100,keep_subhalos=True):
        """ Reads halo (+ vahc) catalogs for given realisation and snapshot. 
             Cleans catalog by selecting relaxed objects in range max(0,1-QE) <= 2T/|U| <= 1+QE 
             where QE > 0 (default QE=0.5; Bett+07).
             Selects objects with at least Npmin particles for given massdef.
             Optionally removes subhalos (set keep_subhalos=False).
             Returns array of shape (Ndata,3) for positions (Mpc/h); structured array(s) for full halo properties (+ vahc).
             Halos will be sorted by (increasing) massdef.
             If RSD flag is True, position arrays will be in redshift space (accounting for periodic boundaries 
             of box of comoving size Lbox). 
        """ 

        if self.verbose:
            self.print_this("... preparing halo data",self.logfile)
        halos = self.read_this(self.real,self.snap,VA=False,verbose=self.verbose)
        Nhalos_all = halos.size
        mmin = self.mpart*Npmin
        cond_clean = (halos[massdef] >= mmin)
        TbyU_max = 0.5*(1+QE)
        TbyU_min = np.max([0.0,0.5*(1-QE)])
        cond_clean = cond_clean & ((halos['TbyU'] < TbyU_max) & (TbyU_min < halos['TbyU']))
        if not keep_subhalos:
            cond_clean = cond_clean & (halos['pid'] == -1)
        if self.verbose:
            self.print_this("... ... using mass definition " + massdef + " > {0:.3e} Msun/h".format(mmin),self.logfile)
            self.print_this("... ... only relaxed objects retained with {0:.2f} < 2T/|U| < {1:.2f}"
                            .format(2*TbyU_min,2*TbyU_max),self.logfile)
            if not keep_subhalos:
                self.print_this("... ... discarding subhalos",self.logfile)

        halos = halos[cond_clean]
        if self.verbose:
            self.print_this("... ... kept {0:d} of {1:d} objects in catalog".format(halos.size,Nhalos_all),self.logfile)

        # pos = np.zeros((3,halos.size),dtype=float)
        pos = np.array([halos['x'],halos['y'],halos['z']])
        if (self.RSD) & (halos.size > 0):
            if self.verbose:
                self.print_this("... ... applying redshift space displacement",self.logfile)
            pos[2] = pos[2] + 0.01*halos['vz']*(1+self.redshift)/self.EHub(self.redshift)
            # pos[2] = pos[2] % self.Lbox
        if va:
            vahc = self.read_this(self.real,self.snap,VA=True)
            vahc = vahc[cond_clean]

        pos = pos % self.Lbox
        pos = pos.T  # shape (Ntrc,3)

        del cond_clean
        gc.collect()

        if self.verbose:
            self.print_this("... ... sorting by "+massdef,self.logfile)
        sorter = halos[massdef].argsort()
        halos = halos[sorter]
        pos = pos[sorter]
        if va:
            vahc = vahc[sorter]

        del sorter
        gc.collect()

        return (pos,halos,vahc) if va else (pos,halos)
    ############################################################

    ############################################################
    def prep_sim_data(self,down=None):
        """ Reads particle position data for given realisation and snapshot. 
             Optionally downsample data to down*self.NFILE particles.
             Returns array of shape (self.Npart,3) for positions (Mpc/h).
             If RSD flag is True, position arrays will be in redshift space (accounting for periodic boundaries 
             of box of comoving size Lbox). 
        """ 
        if self.verbose:
            self.print_this("... preparing dark matter particle data",self.logfile)

        if down is not None:
            if self.verbose:
                self.print_this("... ... downsampling snapshot to {0:d} x {1:d}^3 particles".format(self.NFILE,down),self.logfile)
            ppos,pvel,pIDs = self.g2reader.downsample(self.infile_snap,down_to=down)
            ppos = ppos.T
            if self.READ_VEL:
                pvel = pvel.T
            # shape (Npart,3)
            if self.NFILE > 1:
                for f in range(1,self.NFILE):
                    infile_snap = self.get_snapshot_file(self.real,self.snap,nf=f)
                    f_pos,f_vel,f_IDs = self.g2reader.downsample(infile_snap,down_to=down)
                    ppos = np.append(ppos,f_pos.T,axis=0)
                    if self.READ_VEL:
                        pvel = np.append(pvel,f_vel.T,axis=0)
                    if self.READ_ID:
                        pIDs = np.append(pIDs,f_IDs)
                    del f_pos,f_vel,f_IDs
                    gc.collect()
        else:
            if self.verbose:
                self.print_this("... ... keeping all {0:d} particles".format(self.Npart),self.logfile)
            ppos,pvel,pIDs = self.g2reader.read_this(self.infile_snap)
            ppos = ppos.T
            if self.READ_VEL:
                pvel = pvel.T
            if self.NFILE > 1:
                for f in range(1,self.NFILE):
                    infile_snap = self.get_snapshot_file(self.real,self.snap,nf=f)
                    f_pos,f_vel,f_IDs = self.g2reader.read_this(infile_snap)
                    ppos = np.append(ppos,f_pos.T,axis=0)
                    if self.READ_VEL:
                        pvel = np.append(pvel,f_vel.T,axis=0)
                    if self.READ_ID:
                        pIDs = np.append(pIDs,f_IDs)
                    del f_pos,f_vel,f_IDs
                    gc.collect()

        if self.RSD:
            # if self.redshift > 0.0:
            #     self.print_this("... WARNING: check Gadget velocity conventions for RSD calculation!",self.logfile)
            ppos = ppos.T
            pvel = pvel.T
            if self.verbose:
                self.print_this("... ... applying redshift space displacement",self.logfile)
            ppos[2] += 0.01*pvel[2]*np.sqrt(1+self.redshift)/self.EHub(self.redshift) 
            # pec vel = gadget vel * sqrt(a) = gadget vel / sqrt(1+z)
            # vel correction = 0.01 * pec vel * (1+z) / E(z) = 0.01 * gadget vel * sqrt(1+z) / E(z)
            ppos[2] = ppos[2] % self.Lbox
            ppos = ppos.T
            pvel = pvel.T

        return ppos,pvel,pIDs

    ############################################################

    ############################################################
    def prep_wts(self,wts,Ntrc):
        """ Convenience function to set up normalised tracer weights."""
        wts_out = None
        if wts is not None:
            if ((len(wts.shape) == 1) & (wts.size == Ntrc)):
                if np.any(wts) < 0.0:
                    raise ValueError("Negative weights detected")
                else:
                    wts_out = wts/wts.mean() # NOTE mean not sum!!
            else:
                raise TypeError("Incompatible shape for array wts.")

        return wts_out
    ############################################################


    ############################################################
    def voronoi_periodic_box(self,pos,ret_ran=True,seed=None,pos_wts=None,dm_pos=None,n_wt_bins=15):
        """ Estimate Voronoi-based number density field for halos in a periodic box of side self.Lbox. 
             Expect array-like pos containing positions in range 0..self.Lbox with shape (Ntrc,3).
             If seed is None, use self.seed to initialise RandomState() [useful for consistency across function calls], 
             else use seed. Default None.
             If dm_pos is not None, expect dark matter particles of shape (Nran,3).
             If pos_wts is not None, expect non-negative array of shape (Ntrc,). In this case, 
             distance(ran,trc) = euclidean_distance(ran,trc)*pos_wts(trc) is used for calculating pseudo-Voronoi volumes. 
             n_wt_bins: number of log[pos_wt] bins to use. Default 15.
             Output behaviour:
             - ret_ran == True:
                 1a. (Ntrc,): contains values of delta_trc(t) where t indexes the tracer catalog and
                      1+delta_trc(t) = (Nrantot/Ntrc)/Nran(t) = self.RAN_FAC/Nran(t),
                 OR
                 1b. if dm_pos is not None: (Ntrc,): contains values of w_dm(t) where
                      w_dm(t) = (1+delta_dm(t))/(1 +  delta_trc(t)) = Nran,mass(t)/Nran,mass,tot * Ntrc
                 2. (Nran,3) containing random positions sorted by nearest tracer nbr (array in 3.),
                 3. (Nran,) containing indices of tracer nbr closest to each random position.
                 4. (Ntrc,) containing number of random positions assigned to each tracer.
                 5. (Nran,) containing Euclidean distances between each random position and its assigned tracer.
             - ret_ran == False:
                 1. from above
        """
        if len(pos.shape) != 2:
            raise TypeError("Incompatible shape for position data. Need (Ntrc,3), detected (" 
                            + ','.join(['%d' % (i,) for i in pos.shape]) +').')
        if pos.shape[1] != 3:
            dim = 'dimension'
            if pos.shape[1] > 1:
                dim += 's'
            raise TypeError("Only 3-d data sets supported. Detected {0:d} ".format(pos.shape[1]) + dim + '.')

        Ntrc = pos.shape[0]
        SEED = self.seed if seed is None else seed
        rng = np.random.RandomState(seed=SEED)

        if dm_pos is None:
            Nran = self.RAN_FAC*Ntrc
            if self.verbose:
                self.print_this("... ... generating {0:d} x Ntrc random positions".format(self.RAN_FAC),self.logfile)
            ran_pos = self.Lbox*rng.rand(Nran,3)
        else:
            # below was stupid
            # if pos_wts is not None:
            #     raise ValueError('Both pos_wts and dm_pos cannot be different from None in voronoi_periodic_box().')
            if len(dm_pos.shape) != 2:
                raise TypeError("Incompatible shape for random position data. Need (Nran,3), detected (" 
                                + ','.join(['%d' % (i,) for i in dm_pos.shape]) +').')
            if dm_pos.shape[1] != 3:
                dim = 'dimension'
                if dm_pos.shape[1] > 1:
                    dim += 's'
                raise TypeError("Only 3-d data sets supported. Detected {0:d} ".format(dm_pos.shape[1]) + dim + '.')
            if self.verbose:
                self.print_this("... ... using {0:d} (~ {1:d} x Ntrc) supplied random positions"
                                .format(dm_pos.shape[0],dm_pos.shape[0]//Ntrc),self.logfile)
            ran_pos = dm_pos.copy()
            Nran = ran_pos.shape[0]

        if pos_wts is not None:
            if pos_wts.shape != (Ntrc,):
                raise TypeError("Incompatible shape for position weights data. Need (Ntrc,), detected (" 
                                + ','.join(['%d' % (i,) for i in pos_wts.shape]) +').')
            if np.any(pos_wts < 0.0):
                raise ValueError("Negative position weights detected.")
            if self.verbose:
                self.print_this('... ... finding nearest nbrs of each random in binned tracer data',self.logfile)
            ind_nbr = -1*np.ones(Nran,dtype=int)
            dist_nbr = -1.0*np.ones(Nran,dtype=float)
            use_bin = np.ones(n_wt_bins,dtype=bool)
            pos_wts_med = np.zeros(n_wt_bins,dtype=float)
            wt_bins = np.logspace(np.log10(pos_wts.min()),np.log10(pos_wts.max()),n_wt_bins+1)
            dwt = np.log10(wt_bins[1]/wt_bins[0])
            wt_bins = np.logspace(np.log10(pos_wts.min())-0.1*dwt,np.log10(pos_wts.max())+0.1*dwt,n_wt_bins+1)
            dwt = np.log10(wt_bins[1]/wt_bins[0])
            if self.verbose:
                self.print_this('... ... weighted distances will be accurate within {0:.3f} dex'.format(0.5*dwt),self.logfile)

            n_Nr = np.max([int(Nran/2e7),1]) # 10 is reasonable for Nran=2e8
            Nr_mins = (Nran//n_Nr)*np.arange(n_Nr)
            Nr_maxs = Nr_mins + (Nran//n_Nr)
            Nr_maxs[-1] += Nran % n_Nr

            tree_data = []
            ind_sel = []
            if self.verbose:
                self.print_this('... ... setting up trees in {0:d} bins of pos_wts'.format(n_wt_bins),self.logfile)
            for w in range(n_wt_bins):
                ind_sel_w = np.where((pos_wts >= wt_bins[w]) & (pos_wts < wt_bins[w+1]))[0]
                if ind_sel_w.size:
                    pos_wts_med[w] = np.median(pos_wts[ind_sel_w])
                    tree_data.append(spatial.KDTree(pos[ind_sel_w],boxsize=self.Lbox))
                    ind_sel.append(ind_sel_w)
                else:
                    use_bin[w] = False
                    tree_data.append([np.nan])
                    ind_sel.append([np.nan])
                del ind_sel_w
                gc.collect()
                if self.verbose:
                    self.status_bar(w,n_wt_bins)

            some_bins_empty = (use_bin.sum() < n_wt_bins)
            if self.verbose:
                self.print_this('... ... {0:d} of {1:d} bins are useable'.format(use_bin.sum(),n_wt_bins),self.logfile)
                self.print_this('... ... looping over {0:d} bins of randoms and {1:d} bins of pos_wts'
                                .format(n_Nr,n_wt_bins),self.logfile)

            for r in range(n_Nr):
                Nr_step = Nr_maxs[r] - Nr_mins[r]
                dist_nbr_bins = self.Lbox*np.ones((Nr_step,n_wt_bins),dtype=float)
                ind_nbr_bins = -1*np.ones((Nr_step,n_wt_bins),dtype=int)
                for w in range(n_wt_bins):
                    if use_bin[w]:
                        tmp_dist,tmp_ind = tree_data[w].query(ran_pos[Nr_mins[r]:Nr_maxs[r]],k=1,workers=self.NPROC)
                        # tmp_dist,tmp_ind have shape (Nr_step,)
                        dist_nbr_bins[:,w] = tmp_dist 
                        ind_nbr_bins[:,w] = ind_sel[w][np.asarray(tmp_ind)]
                        del tmp_dist,tmp_ind
                        gc.collect()
                    if self.verbose:
                        self.status_bar(w + r*n_wt_bins,n_wt_bins*n_Nr)

                if some_bins_empty:
                    ind_nbr_bins = ind_nbr_bins[:,use_bin].squeeze()
                    dist_nbr_bins = dist_nbr_bins[:,use_bin].squeeze()
                    gc.collect()

                argmins = np.argmin(dist_nbr_bins*pos_wts_med[use_bin],axis=1) # KEY STEP minimising weighted dist

                ind_nbr[Nr_mins[r]:Nr_maxs[r]] = ind_nbr_bins[np.arange(Nr_step),argmins]
                dist_nbr[Nr_mins[r]:Nr_maxs[r]] = dist_nbr_bins[np.arange(Nr_step),argmins] # reporting Euclidean dist
                del dist_nbr_bins,ind_nbr_bins,argmins
                gc.collect()
            if self.verbose:
                self.print_this('... ... ... all bins done',self.logfile)
            del tree_data
            gc.collect()
        else:
            if self.verbose:
                self.print_this('... ... generating tracer tree',self.logfile)
            tree_data = spatial.KDTree(pos,boxsize=self.Lbox)
            if self.verbose:
                self.print_this('... ... finding nearest nbr of each random in tracer data',self.logfile)
            dist_nbr,ind_nbr = tree_data.query(ran_pos,k=1,workers=self.NPROC)
            # dist_nbr,ind_nbr have shape (Nran,)
            # ind_nbr gives indices in halo data of nearest neighbour of each random
            # dist_nbr gives distance of nearest neighbour of each random
            del tree_data
            gc.collect()

        if self.verbose:
            self.print_this('... ... counting number of random nbrs for each tracer',self.logfile)
        nbr_count = np.bincount(ind_nbr,minlength=Ntrc)
        if dm_pos is None:
            out = self.RAN_FAC/(nbr_count + self.TINY) - 1.0 
        else:
            out = 1.0*nbr_count*Ntrc/ran_pos.shape[0]
        
        if ret_ran: 
            if self.verbose:
                self.print_this("... ... sorting randoms by nearest tracers",self.logfile)
            sorter = ind_nbr.argsort()
            ind_nbr = ind_nbr[sorter]
            dist_nbr = dist_nbr[sorter]
            ran_pos = ran_pos[sorter]
            del sorter
            gc.collect()

        if not ret_ran:
            del ran_pos,ind_nbr,nbr_count,dist_nbr
        if self.verbose:
            self.print_this("... ... done with voronoi density",self.logfile)

        if ret_ran:
            return (out,ran_pos,ind_nbr,nbr_count,dist_nbr)
        else:
            return out
    ############################################################


    ############################################################
    def in_hull(self,p, hull):
        """
        Test if points in `p` are in `hull`

        courtesy:https://stackoverflow.com/questions/16750618/\
        whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed.
        """
        if not isinstance(hull,spatial.Delaunay):
            hull = spatial.Delaunay(hull)

        return hull.find_simplex(p) >= 0 # True if p is inside hull, False if outside
    ############################################################


    ############################################################
    def find_substructure(self,ran_pos_sph,nbr_count_sph,ran_pos=None,nbr_count=None,spherical=True):
        """ Identify substructure by asking whether any random assigned to some tracer t'
             is inside Rtrc-spheres centered on any other more massive tracer t. 
             In this case, label t' as substructure.
             spherical: If True, use spherical randoms instead of Voronoi randoms for t'. 
                              Effectively compares spherical overlap. Default True.
                              If False, then ran_pos (Nran,r) and nbr_count (Ntrc,) should be specified.
             Inputs: ran_pos_sph and nbr_count_sph as returned by spherical_randoms() [already sorted by mass proxy].
                optionally: ran_pos and nbr_count as returned by voronoi_periodic_box() [already sorted by mass proxy].
             Returns: boolean array of shape (Ntrc,) with substructure labelled True.
        """
        # are inside the convex hull defined by randoms assigned to any other larger (by mass proxy) tracer t. 
        if self.verbose:            
            print_str = "... identifying substructure "
            self.print_this(print_str,self.logfile)

        if len(ran_pos_sph.shape) != 2:
            raise TypeError("Incompatible shape for spherical random position data. Need (Nran,3), detected (" 
                            + ','.join(['%d' % (i,) for i in ran_pos_sph.shape]) +').')
        if ran_pos_sph.shape[1] != 3:
            dim = 'dimension'
            if ran_pos_sph.shape[1] > 1:
                dim += 's'
            raise TypeError("Only 3-d data sets supported. Detected {0:d} ".format(ran_pos_sph.shape[1]) + dim + '.')
        Nran_sph = ran_pos_sph.shape[0]

        if len(nbr_count_sph.shape) != 1:
            raise TypeError("Incompatible shape for ran_sph neighbour count data. Need (Ntrc,), detected (" 
                            + ','.join(['%d' % (i,) for i in nbr_count_sph.shape]) +').')
        if nbr_count_sph.sum() != Nran_sph:
            raise TypeError("Incompatible Nran_sph")

        Ntrc = nbr_count_sph.size

        if not spherical:
            if (ran_pos is None) | (nbr_count is None):
                raise TypeError('Need random positions and neighbour counts for non-spherical substructure comparision.')

            if len(nbr_count.shape) != 1:
                raise TypeError("Incompatible shape for neighbour count data. Need (Ntrc,), detected (" 
                                + ','.join(['%d' % (i,) for i in nbr_count.shape]) +').')

            if len(ran_pos.shape) != 2:
                raise TypeError("Incompatible shape for random position data. Need (Nran,3), detected (" 
                                + ','.join(['%d' % (i,) for i in ran_pos.shape]) +').')
            if ran_pos.shape[1] != 3:
                dim = 'dimension'
                if ran_pos.shape[1] > 1:
                    dim += 's'
                raise TypeError("Only 3-d data sets supported. Detected {0:d} ".format(ran_pos.shape[1]) + dim + '.')
            Nran = ran_pos.shape[0]

        is_sub = np.zeros(Ntrc,dtype=bool)
        if not spherical:
            is_sub[nbr_count == 0] = True 
            # `zero-volume' cells are for tracers with `no' tidal influence, hence substructure
            if self.verbose:            
                self.print_this("... ... {0:d} tracers labelled as substructure due to zero volume"
                                .format(np.where(is_sub)[0].size),self.logfile)

        if self.verbose:
            print_str = "... ... looping over tracers to identify " 
            if spherical:
                print_str += "(spherical) "
            print_str += "overlaps"
            self.print_this(print_str,self.logfile)

        nu_compare = nbr_count_sph if spherical else nbr_count
        ran_compare = ran_pos_sph if spherical else ran_pos 
        ################
        # THIS LOOP IS VERY SLOW, especially for large Nran with spherical=False
        ################
        count = Nran_sph
        for t in range(Ntrc-1,-1,-1): # loop from largest to smallest tracer
            if not is_sub[t]: # only create convex hull if not already a substructure
                hull = spatial.Delaunay(ran_pos_sph[count-nbr_count_sph[t]:count]) 
                # USE ACCUMULATE TO REPLACE count_lo UPDATE? THEN PARALLELISE?
                count_lo = 0
                for t_lo in range(t): # check with all smaller tracers...
                    if not is_sub[t_lo]: # ... which are not already substructure
                        is_sub[t_lo] = np.any(self.in_hull(ran_compare[count_lo:count_lo+nu_compare[t_lo]],hull))
                    count_lo += nu_compare[t_lo]
                del hull
                gc.collect()
            count -= nbr_count_sph[t]
            if self.verbose:            
                self.status_bar(Ntrc-t-1,Ntrc)
        ################

        if self.verbose:            
            self.print_this("... ... {0:d} of {1:d} tracers labelled as substructure in all".format(np.where(is_sub)[0].size,Ntrc),
                            self.logfile)

        return is_sub

    ############################################################


    ############################################################
    def voronoi_to_grid(self,delta_trc,wts_trc,wts_dust,Delta_m,eta_trc,ran_pos,ind_nbr,nbr_count,dist_nbr,
                        grid=128,split_cell=True,pos=None): 
        """ Use output of voronoi_periodic_box along with tracer and dust weights to generate density estimate on grid, 
             optionally splitting each cell into halo and dust. 
             if split_cell is True, pos must contain tracer positions of shape (Ntrc,3).
             Returns delta_grid of shape (grid,grid,grid).
        """
        if self.verbose:            
            print_str = "... gridding density contrast"
            if split_cell:
                print_str += " with split cells"
            self.print_this(print_str,self.logfile)
        
        if split_cell & (pos is None):
            raise ValueError('Splitting cells needs tracer positions to be supplied')
            # self.print_this('Tracer positions not specified: hence not splitting cells',self.logfile)
            # split_cells = False
        Nran = ind_nbr.size

        onepdtrc = 1+delta_trc

        n_Nr = np.max([int(Nran/2e7),1]) # 10 is reasonable for Nran=2e8
        Nr_mins = (Nran//n_Nr)*np.arange(n_Nr)
        Nr_maxs = Nr_mins + (Nran//n_Nr)
        Nr_maxs[-1] += Nran % n_Nr
        grid_edges = np.linspace(0.0,self.Lbox,grid+1)
        delta_grid = np.zeros((grid,grid,grid),dtype=float)
        Nran_grid = np.zeros((grid,grid,grid),dtype=float)

        if split_cell:
            ###############################
            # 1 + delta_g = (1/N_ran,g) [ Sum_{r in g} Delta_ran[r] 
            #                                       + (eta_trc/Delta_m)(Nran/Nran_sph) Sum_{r_sph in g} (Delta_m - Delta_ran[r_sph]) ]
            ###############################
            f_less = (eta_trc/Delta_m)*wts_trc*onepdtrc
            cond_set = (f_less < 1.0)
            if self.verbose:
                self.print_this("... ... {0:d} of {1:d} tracers have no dust"
                                .format(np.where(~cond_set)[0].size,wts_trc.size),self.logfile)
            if self.verbose:
                self.print_this("... ... setting up densities",self.logfile)
            Delta = np.zeros(wts_trc.size,dtype=float)
            Delta[cond_set] = (1.0-eta_trc)*wts_dust[cond_set]*onepdtrc[cond_set]/(1-f_less[cond_set])
            Delta[~cond_set] = f_less[~cond_set]*Delta_m

            del f_less,onepdtrc
            gc.collect()

            if self.verbose:
                self.print_this("... ... setting up tracer sphere sizes",self.logfile)
            Rtrc = self.Lbox*(3*wts_trc*eta_trc/(4*np.pi*Delta_m*wts_trc.size))**(1/3.)

            Delta_ran = np.zeros(Nran,dtype=float)
            if self.verbose:
                self.print_this("... ... looping over halos to set Delta_ran",self.logfile)
            counter = 0
            for t in range(Delta.size):
                sl = slice(counter,counter+nbr_count[t])
                Delta_ran[sl] = Delta[t] 
                counter += nbr_count[t]

            if self.verbose:
                self.print_this("... ... histogramming on {0:d}^3 grid".format(grid),self.logfile)
                self.print_this('... ... ... looping over {0:d} bins of randoms'.format(n_Nr),self.logfile)
            for r in range(n_Nr):
                Nr_step = Nr_maxs[r] - Nr_mins[r]
                sl = slice(Nr_mins[r],Nr_maxs[r])
                tmp_hist,tmp_bins = np.histogramdd(ran_pos[sl],weights=Delta_ran[sl],
                                                   bins=(grid_edges,grid_edges,grid_edges),density=False)
                delta_grid += tmp_hist
                del tmp_hist,tmp_bins
                gc.collect()

                tmp_hist,tmp_bins = np.histogramdd(ran_pos[sl],weights=None,
                                                   bins=(grid_edges,grid_edges,grid_edges),density=False)
                Nran_grid += tmp_hist

                del tmp_hist,tmp_bins
                del sl
                gc.collect()
                if self.verbose:
                    self.status_bar(r,n_Nr)

            # delta_grid now contains 1st term in sum
            # Nran_grid contains number of randoms in each cell (denominator of final expression)

            del Delta_ran
            gc.collect()

            #####################
            # spherical randoms calculation
            #####################
            if self.verbose:
                self.print_this("... ... generating spherical randoms",self.logfile)
            ran_pos_sph,nu_ran = self.spherical_randoms(pos,Rtrc,cond_set=cond_set)
            Nran_sph = ran_pos_sph.shape[0]

            del Rtrc
            gc.collect()

            Delta_ran_sph = np.zeros(Nran_sph,dtype=float)
            if self.verbose:
                self.print_this("... ... looping over halos to set Delta_ran_sph",self.logfile)
            counter = 0
            Delta = Delta[cond_set]
            for t in range(Delta.size):
                sl = slice(counter,counter+nu_ran[t])
                Delta_ran_sph[sl] = Delta_m - Delta[t] 
                counter += nu_ran[t]

            del cond_set,Delta
            gc.collect()

            Delta_ran_sph_grid = np.zeros_like(Nran_grid)

            n_Nr = np.max([int(Nran_sph/2e7),1]) # 10 is reasonable for Nran=2e8
            Nr_mins = (Nran_sph//n_Nr)*np.arange(n_Nr)
            Nr_maxs = Nr_mins + (Nran_sph//n_Nr)
            Nr_maxs[-1] += Nran_sph % n_Nr
            if self.verbose:
                self.print_this("... ... histogramming spherical randoms on {0:d}^3 grid".format(grid),self.logfile)
                self.print_this('... ... ... looping over {0:d} bins of randoms'.format(n_Nr),self.logfile)
            for r in range(n_Nr):
                Nr_step = Nr_maxs[r] - Nr_mins[r]
                sl = slice(Nr_mins[r],Nr_maxs[r])

                tmp_hist,tmp_bins = np.histogramdd(ran_pos_sph[sl],weights=Delta_ran_sph[sl],
                                                   bins=(grid_edges,grid_edges,grid_edges),density=False)
                Delta_ran_sph_grid += tmp_hist

                del tmp_hist,tmp_bins
                del sl
                gc.collect()
                if self.verbose:
                    self.status_bar(r,n_Nr)

            del ran_pos_sph,grid_edges,Delta_ran_sph
            gc.collect()

            delta_grid += eta_trc/Delta_m*Nran/Nran_sph*Delta_ran_sph_grid

            del Delta_ran_sph_grid
            gc.collect()

            # delta_grid contains numerator of final expression
            # Nran_grid contains number of randoms in each cell (denominator of final expression)
        else:
            Delta = (eta_trc*wts_trc + (1-eta_trc)*wts_dust)*onepdtrc# - 1.0
            Delta_ran = -100.0*np.ones(Nran,dtype=float)
            if self.verbose:
                self.print_this("... ... looping over halos to set Delta_ran",self.logfile)
            # below is 'infinitely' faster than h-by-h boolean indexing
            counter = 0
            for t in range(Delta.size):
                sl = slice(counter,counter+nbr_count[t])
                Delta_ran[sl] = Delta[t] 
                counter += nbr_count[t]
            del Delta,onepdtrc
            gc.collect()

            if self.verbose:
                self.print_this("... ... histogramming on {0:d}^3 grid".format(grid),self.logfile)
                # self.print_this('... ... ... looping over {0:d} bins of randoms'.format(n_Nr),self.logfile)
            # powspec = PowerSpectrum(grid=grid,Lbox=self.Lbox,logfile=self.logfile,verbose=self.verbose)
            # delta_grid = powspec.wtd_ngp_field(ran_pos.T,weights=Delta_ran)
            # Nran_grid = powspec.wtd_ngp_field(ran_pos.T,weights=None)
            for r in range(n_Nr):
                Nr_step = Nr_maxs[r] - Nr_mins[r]
                sl = slice(Nr_mins[r],Nr_maxs[r])
                tmp_hist,tmp_bins = np.histogramdd(ran_pos[sl],weights=Delta_ran[sl],
                                                   bins=(grid_edges,grid_edges,grid_edges),density=False)
                delta_grid += tmp_hist
                del tmp_hist,tmp_bins
                gc.collect()

                tmp_hist,tmp_bins = np.histogramdd(ran_pos[sl],weights=None,
                                                   bins=(grid_edges,grid_edges,grid_edges),density=False)
                Nran_grid += tmp_hist

                del tmp_hist,tmp_bins
                del sl
                gc.collect()
                if self.verbose:
                    self.status_bar(r,n_Nr)
            del grid_edges,Delta_ran
            gc.collect()
            # delta_grid contains sum of Delta_ran in each cell
            # Nran_grid contains number of randoms in each cell

        ##################
        # # below is older, more memory inefficient version (because of not splitting randoms)
        # if powspec is None:
        #     powspec = PowerSpectrum(grid=grid,Lbox=self.Lbox,logfile=self.logfile,verbose=self.verbose)
        # delta_grid = powspec.wtd_ngp_field(ran_pos.T,weights=delta_ran)
        # # delta_grid contains sum of delta_ran in each cell
        # Nran_grid = powspec.wtd_ngp_field(ran_pos.T,weights=None)
        # # Nran_grid contains number of randoms in each cell

        delta_grid /= (Nran_grid + self.TINY) 
        grid_wts = Nran_grid/Nran_grid.sum()

        del Nran_grid
        gc.collect()

        delta_grid -= 1.0
        if self.verbose:
            self.print_this("... ... {0:.1e}% of cells have negative density: setting to zero density"
                            .format(1e2*delta_grid[delta_grid < -1.0].size/delta_grid.size),self.logfile)
        delta_grid[delta_grid < -1.0] = -1.0

        d_mean = np.sum(grid_wts*delta_grid)
        d_std = np.sum(grid_wts*(delta_grid - d_mean)**2)/(1.0 - np.sum(grid_wts**2) + self.TINY)

        if self.verbose:
            self.print_this("... ... stats: vol wtd mean = {0:.3e}; vol wtd std dev = {1:.3f}"
                            .format(d_mean,d_std),self.logfile)
            self.print_this("... ... adjusting mean",self.logfile)
        delta_grid = (delta_grid - d_mean)/(1 + d_mean)        

        return delta_grid
    ############################################################


    ############################################################
    def spherical_randoms(self,pos,Rtrc,cond_set=None,nustar=300,seed=None):
        """Generate uniform random spheres around tracer locations with given tracer sizes.
            Assumes tracer masses ~ Rtrc^3.
            nustar: number of randoms sampling mean tracer mass.
            Returns array of shape (Nran_sph,3) with positions and array of shape (Ntrc,) with number counts per tracer.
        """
        if len(pos.shape) != 2:
            raise TypeError("Incompatible shape for position data in spherical_randoms(). Need (Ntrc,3), detected (" 
                            + ','.join(['%d' % (i,) for i in pos.shape]) +').')
        if pos.shape[1] != 3:
            dim = 'dimension'
            if pos.shape[1] > 1:
                dim += 's'
            raise TypeError("Only 3-d data sets supported. Detected {0:d} ".format(pos.shape[1]) + dim + '.')
        
        Ntrc = pos.shape[0]
        if Rtrc.size != Ntrc:
            raise ValueError("Incompatible shapes detected for pos and Rtrc in spherical_randoms()")

        SEED = self.seed if seed is None else seed
        rng = np.random.RandomState(seed=SEED)

        cond_set_trc = np.ones(Ntrc,dtype=bool) if cond_set is None else cond_set.copy()
        pos_set = pos[cond_set_trc]
        Rtrc_set = Rtrc[cond_set_trc]
        Ntrc = pos_set.shape[0]

        del cond_set_trc
        gc.collect()

        reject_frac = np.pi/6 # 4pi/3 / 2^3
        nbr_count_sph = np.rint(self.prep_wts(Rtrc_set**3,Ntrc)*nustar/reject_frac).astype(int)
        nbr_count_sph[nbr_count_sph < 10/reject_frac] = np.rint(10/reject_frac).astype(int)
        # eventually sample each tracer with at least 10 randoms.
        Nran_sph = nbr_count_sph.sum()
        if self.verbose:
            self.print_this("... ... setting up {0:d} uniforms (~ {1:d} x Ntrc) in {2:d} spheres"
                            .format(Nran_sph,Nran_sph//Ntrc,Ntrc),
                            self.logfile)
        ran_pos_sph = 2*rng.rand(Nran_sph,3) - 1.0 # ~ uniform(-1,1)
        ran_pos_sph = ran_pos_sph[(np.sum(ran_pos_sph**2,axis=1) <= 1.0)] # unit sphere centered at origin
        Nran_sph = ran_pos_sph.shape[0] # now smaller than nbr_count_sph.sum() by factor ~ reject_frac
        nbr_count_sph = np.rint(nbr_count_sph*reject_frac).astype(int) #rescale all nbr_count_sph values
        nbr_count_sph_sum = nbr_count_sph.sum()
        if self.verbose:
            self.print_this("... ... {0:d} uniforms left after rejection; partition sums to {1:d}"
                            .format(Nran_sph,nbr_count_sph_sum),self.logfile)
        if nbr_count_sph_sum != Nran_sph:
            if self.verbose:
                self.print_this("... ... ... adjusting partition",self.logfile)
            if nbr_count_sph_sum < Nran_sph:
                # in this case just delete a few existing randoms
                ran_pos_sph = ran_pos_sph[:nbr_count_sph_sum]
                if self.verbose:
                    self.print_this("... ... ... deleted {0:d} randoms to leave {1:d} (should match partition sum)"
                                    .format(Nran_sph-ran_pos_sph.shape[0],ran_pos_sph.shape[0]),self.logfile)
            else:
                # in this case adjust the value of the largest nbr_count_sph (which will be best-sampled anyway)
                i_max = nbr_count_sph.argmax()
                nbr_count_sph_max = nbr_count_sph[i_max]
                nbr_count_sph[i_max] = nbr_count_sph_max - (nbr_count_sph_sum - Nran_sph)
                if self.verbose:
                    self.print_this("... ... ... reset max count from {0:d} to {1:d}; partition sums to {2:d} (should match array size)"
                                    .format(nbr_count_sph_max,nbr_count_sph[i_max],nbr_count_sph.sum()),
                                    self.logfile)
        else:
            if self.verbose:
                self.print_this("... ... ... partition unchanged",self.logfile)
        # check:
        if nbr_count_sph.sum() != ran_pos_sph.shape[0]:
            raise ValueError('Something wrong with spherical random rejection output.')
        Nran_sph = nbr_count_sph.sum() # reset total Nran_sph

        # now ran_pos_sph contains unit-spheres partitioned, with nbr_count_sph[t] randoms for each t        
        if self.verbose:
            self.print_this("... ... looping over halos to scale randoms by radius and shift position",self.logfile)
        counter = 0
        for t in range(Ntrc):
            sl = slice(counter,counter+nbr_count_sph[t])
            ran_pos_sph[sl] *= Rtrc_set[t]
            ran_pos_sph[sl] += pos_set[t]
            counter += nbr_count_sph[t]
            if self.verbose:
                self.status_bar(t,Ntrc)

        del Rtrc_set,pos_set
        gc.collect()

        if self.verbose:
            self.print_this("... ... applying periodic boundaries",self.logfile)
        ran_pos_sph %= self.Lbox
        
        if self.verbose:
            self.print_this("... ... done with random spheres",self.logfile)

        return ran_pos_sph,nbr_count_sph
    ############################################################


    ############################################################
    def set_ran_fac(self,Ntrc,grid=None):
        """Convenience function. Changes value of self.RAN_FAC."""

        GRID = grid if grid is not None else self.GRID_MAX
        self.RAN_FAC = int(self.NRAN_PER_CELL*GRID**3/Ntrc)
        if (Ntrc < self.NHALO_HI):
            if (self.RAN_FAC < 30000):
                self.RAN_FAC = 30000
        else:
            if (self.RAN_FAC > 15000):
                self.RAN_FAC = 15000
            elif (self.RAN_FAC < 10000):
                self.RAN_FAC = 10000
        self.RAN_FAC = np.min([self.RAN_FAC,int(self.MAX_RANDOMS/Ntrc)])

        return
    ############################################################


    ############################################################
    def voronoi_dm(self,arg0,predictor,interpolator,massdef='m200b',grid=128,ran_fac=None,pos_wt_exp=None,
                   split_cell=False,model_dust=True,Delta_m=200.0,keep_subhalos=False,seed=None,galaxies=False):
        """ Wrapper for Voronoi-based matter density estimator. Estimates dark matter density on a grid 
             of size grid^3. 
            Arguments:
            - arg0: scalar integer or float specifying minimum particle threshold if galaxies is False, else tuple containing
             galaxy positions of shape (Ntrc,3) and galaxy property of shape (Ntrc,).
            - predictor: GPRTrainer.predict() instance having call signature predictor(params,interpolator). Should be 
                                compatible with model_dust, split_cell and pos_wt_exp.
            - interpolator: output of GPRTrainer.train_gpr(), compatible with model_dust, split_cell and pos_wt_exp.
             kwargs:
             - grid: grid resolution
             - ran_fac: user specified RAN_FAC. Default None (auto generated).
             - massdef: mass definition
             - model_dust: If True, use interpolator to model dust weights rather than total weights.
             - keep_subhalos: whether or not to include subhalos. Default False.
             - seed: passed to self.voronoi_periodic_box()
             - pos_wt_exp: If not None, use d(ran,trc) = d_Euc(ran,trc)/mass(trc)^(pos_wt_exp) to define Voronoi volume.
             - split_cell: If True, use density estimator for each cell assigning tracer mass inside Rvir and dust mass outside.
             Returns 1. delta_dm_grid: 1 array of shape (grid,grid,grid)
                           2. (if galaxies is False) hpos: 1 array of shape (Ntrc,3) with tracer positions
        """
        if self.verbose:
            self.print_this("Estimating dark matter density...",self.logfile)

        if galaxies & (not model_dust):
            self.print_this("... galaxies only supported with explicit dust model; switching on model_dust",self.logfile)
            model_dust = True

        if split_cell & galaxies:
            self.print_this("... split cell model not implemented for galaxies, modelling constant w_dm instead",self.logfile)
            split_cell = False

        if grid > self.GRID_MAX:
            self.print_this("... grid is too fine. Switching to {0:d}^3".format(self.GRID_MAX),self.logfile)
            grid = self.GRID_MAX

        if not galaxies:
            Npmin = arg0
            if self.verbose:
                self.print_this("... using tracer mass definition: "+massdef,self.logfile)
            pos,halos = self.prep_halo_data(va=False,massdef=massdef,Npmin=Npmin,keep_subhalos=keep_subhalos) 
            Ntrc = halos.size
            eta_trc = halos[massdef].sum()/(self.Npart*self.mpart)
        else:
            pos,gprop = arg0
            if len(pos.shape) != 2:
                raise TypeError("Incompatible shape for galaxy position data. Need (Ntrc,3), detected (" 
                                + ','.join(['%d' % (i,) for i in pos.shape]) +').')
            if pos.shape[1] != 3:
                raise TypeError("Only 3-d data sets supported")
            if gprop.shape != (pos.shape[0],):
                raise TypeError("Incompatible shape for galaxy property data. Need ({0:d},), detected (".format(pos.shape[0])
                                + ','.join(['%d' % (i,) for i in gprop.shape]) +').')
            Ntrc = gprop.size
            if self.verbose:
                self.print_this("... found {0:d} galaxies".format(Ntrc),self.logfile)

        if ran_fac is not None:
            self.RAN_FAC = ran_fac
        else:
            self.set_ran_fac(Ntrc,grid=grid)

        # calculate number- and mass-weighted field
        if self.verbose:
            self.print_this("... number- and mass-weighted fields",self.logfile)

        if self.verbose:
            self.print_this("... ... calculating density field",self.logfile)
        pos_wts=None
        if (pos_wt_exp is not None):
            if galaxies:
                if self.verbose:
                    self.print_this("... ... weighted positions not yet implemented for galaxies: using Euclidean distances",
                                    self.logfile)
            else:
                if self.verbose:
                    self.print_this("... ... using weighted distances d(r,t) = d_Euc(r,t)/m(t)^{0:.3f}".format(pos_wt_exp),
                                    self.logfile)
                pos_wts = 1.0/halos[massdef]**pos_wt_exp
        else:
            if self.verbose:
                self.print_this("... ... using Euclidean distances",self.logfile)
        delta_trc,ran_pos,ind_nbr,nbr_count,dist_nbr = self.voronoi_periodic_box(pos,ret_ran=True,
                                                                                 seed=seed,pos_wts=pos_wts)

        del pos_wts
        gc.collect()

        if self.verbose:
            self.print_this('... ... estimating weighted density',self.logfile)

        wts_trc = self.prep_wts(halos[massdef],Ntrc) if not galaxies else self.prep_wts(10**gprop,Ntrc)
        params = np.array([np.log10(1+delta_trc),np.log10(wts_trc)]).T
        if model_dust:
            if self.verbose:
                self.print_this("... ... ... explicitly modelling w_dust using eta_trc = {0:.3f}".format(eta_trc),self.logfile)        
            wts_dust = 10**predictor(params,interpolator)
            # wts_dust = self.prep_wts(wts_trc**gammaM*(1+delta_trc)**gammaV,Ntrc)
        else:
            if self.verbose:
                self.print_this("... ... ... directly modelling w_dm",self.logfile)        
            # wts = self.prep_wts(wts_trc**gammaM*(1+delta_trc)**gammaV,Ntrc)
            wts = 10**predictor(params,interpolator)
            wts_dust = (wts - eta_trc*wts_trc)/(1-eta_trc)
            del wts
            gc.collect()
        del params
        gc.collect()

        ################

        delta_dm_grid = self.voronoi_to_grid(delta_trc,wts_trc,wts_dust,Delta_m,eta_trc,
                                             ran_pos,ind_nbr,nbr_count,dist_nbr,grid=grid,split_cell=split_cell,pos=pos)

        if not galaxies:
            del halos
        del delta_trc,ran_pos,ind_nbr,nbr_count,dist_nbr,wts_trc,wts_dust
        gc.collect()

        if self.verbose:
            self.print_this("... done",self.logfile)

        self.RAN_FAC = self.RAN_FAC_DEFAULT # do this last, so that calls to other functions don't get affected.

        if galaxies:
            return delta_dm_grid
        else:
            return delta_dm_grid,pos
    ############################################################



    ############################################################
    def voronoi_dm_DEPRECATED(self,arg0,massdef='m200b',grid=128,ran_fac=None,pos_wt_exp=None,
                              split_cell=False,model_dust=True,gammaM=1.0,gammaV=0.0,Delta_m=200.0,
                              keep_subhalos=True,seed=None,galaxies=False):
        """ Wrapper for Voronoi-based matter density estimator. Given value of particle threshold 
             (scalar integer or float) to define the tracer population, estimates dark matter density on a grid 
             of size grid^3 and grid_coarse^3. If galaxies is True, first argument should be tuple containing
             galaxy positions of shape (Ntrc,3) and galaxy property of shape (Ntrc,).
             kwargs:
             - grid: grid resolution
             - grid_coarse: coarse grid resolution
             - ran_fac: user specified RAN_FAC. Default None (auto generated).
             - massdef: mass definition
             - gammaM,gammaV: normalisation and exponent for mass-weighted field
                                                 Delta_dm(t) = [f(t) / <f>] * Delta_trc(t)
                                                 where f(t) = m(t)**gammaM * Delta_trc(t)**gammaV
                                                 Default (1.0,0.0) [mass wtd]
                                                 Other examples: (0.0,0.0) [no.-wtd]
             - keep_subhalos: whether or not to include subhalos. Default True.
             - seed: passed to self.voronoi_periodic_box()
             - pos_wt_exp: If not None, use d(ran,trc) = d_Euc(ran,trc)/mass(trc)^(pos_wt_exp) to define Voronoi volume.
             - model_dust: If true, use gammaM,gammaV to model dust weights rather than total weights.
             - split_cell: If True, use density estimator for each cell assigning tracer mass inside Rvir and dust mass outside.
             Returns 1. delta_dm_grid: 1 array of shape (grid,grid,grid)
                           2. delta_dm_grid_coarse: 1 array of shape (grid_coarse,grid_coarse,grid_coarse)
                           3. (if galaxies is False) hpos: 1 array of shape (Ntrc,3) with tracer positions
        """
        if self.verbose:
            self.print_this("Estimating dark matter density...",self.logfile)

        if galaxies & (not model_dust):
            self.print_this("... galaxies only supported with explicit dust model; switching on model_dust",self.logfile)
            model_dust = True

        if split_cell & galaxies:
            self.print_this("... split cell model not implemented for galaxies, modelling constant w_dm instead",self.logfile)
            split_cell = False

        if grid > self.GRID_MAX:
            self.print_this("... grid is too fine. Switching to {0:d}^3".format(self.GRID_MAX),self.logfile)
            grid = self.GRID_MAX

        if not galaxies:
            Npmin = arg0
            if self.verbose:
                self.print_this("... using tracer mass definition: "+massdef,self.logfile)
            pos,halos = self.prep_halo_data(va=False,massdef=massdef,Npmin=Npmin,keep_subhalos=keep_subhalos) 
            Ntrc = halos.size
            eta_trc = halos[massdef].sum()/(self.Npart*self.mpart)
        else:
            pos,gprop = arg0
            if len(pos.shape) != 2:
                raise TypeError("Incompatible shape for galaxy position data. Need (Ntrc,3), detected (" 
                                + ','.join(['%d' % (i,) for i in pos.shape]) +').')
            if pos.shape[1] != 3:
                raise TypeError("Only 3-d data sets supported")
            if gprop.shape != (pos.shape[0],):
                raise TypeError("Incompatible shape for galaxy property data. Need ({0:d},), detected (".format(pos.shape[0])
                                + ','.join(['%d' % (i,) for i in gprop.shape]) +').')
            Ntrc = gprop.size
            if self.verbose:
                self.print_this("... found {0:d} galaxies".format(Ntrc),self.logfile)

        if ran_fac is not None:
            self.RAN_FAC = ran_fac
        else:
            self.set_ran_fac(Ntrc,grid=grid)

        # calculate number- and mass-weighted field
        if self.verbose:
            self.print_this("... number- and mass-weighted fields",self.logfile)

        if self.verbose:
            self.print_this("... ... calculating density field",self.logfile)
        pos_wts=None
        if (pos_wt_exp is not None):
            if galaxies:
                if self.verbose:
                    self.print_this("... ... weighted positions not yet implemented for galaxies: using Euclidean distances",
                                    self.logfile)
            else:
                if self.verbose:
                    self.print_this("... ... using weighted distances d(r,t) = d_Euc(r,t)/m(t)^{0:.3f}".format(pos_wt_exp),
                                    self.logfile)
                pos_wts = 1.0/halos[massdef]**pos_wt_exp
        else:
            if self.verbose:
                self.print_this("... ... using Euclidean distances",self.logfile)
        delta_trc,ran_pos,ind_nbr,nbr_count,dist_nbr = self.voronoi_periodic_box(pos,ret_ran=True,
                                                                                 seed=seed,pos_wts=pos_wts)

        del pos_wts
        gc.collect()

        if self.verbose:
            self.print_this('... ... estimating weighted density',self.logfile)

        wts_trc = self.prep_wts(halos[massdef],Ntrc) if not galaxies else self.prep_wts(10**gprop,Ntrc)
        if model_dust:
            if self.verbose:
                self.print_this("... ... ... explicitly modelling w_dust using eta_trc = {0:.3f}".format(eta_trc),self.logfile)        
            wts_dust = self.prep_wts(wts_trc**gammaM*(1+delta_trc)**gammaV,Ntrc)
        else:
            if self.verbose:
                self.print_this("... ... ... directly modelling w_dm",self.logfile)        
            wts = self.prep_wts(wts_trc**gammaM*(1+delta_trc)**gammaV,Ntrc)
            wts_dust = (wts - eta_trc*wts_trc)/(1-eta_trc)
            del wts
            gc.collect()

        ################

        delta_dm_grid = self.voronoi_to_grid(delta_trc,wts_trc,wts_dust,Delta_m,eta_trc,
                                             ran_pos,ind_nbr,nbr_count,dist_nbr,grid=grid,split_cell=split_cell,pos=pos)
        # delta_dm_grid_coarse = self.voronoi_to_grid(delta_trc,wts_trc,wts_dust,Delta_m,eta_trc,
        #                                             ran_pos,ind_nbr,nbr_count,dist_nbr,grid=grid_coarse,split_cell=split_cell,pos=pos)

        if not galaxies:
            del halos
        del delta_trc,ran_pos,ind_nbr,nbr_count,dist_nbr,wts_trc,wts_dust
        gc.collect()

        if self.verbose:
            self.print_this("... done",self.logfile)

        self.RAN_FAC = self.RAN_FAC_DEFAULT # do this last, so that calls to other functions don't get affected.

        if galaxies:
            return delta_dm_grid#,delta_dm_grid_coarse
        else:
            return delta_dm_grid,pos#,delta_dm_grid_coarse
    ############################################################



    ############################################################
    def voronoi_dm_chi2_DEPRECATED(self,arg0,gamma_ranges,Pk_dm_CIC,powspec,freq_CIC,lgdbins,
                                   model_dust=True,Delta_m=200.0,split_cell=False,Nsamp=10,dsig=5.0,
                                   kmax=0.5,eps_density=0.1,pos_wt_exp=None,
                                   massdef='mvir',keep_subhalos=False,ran_fac=None,seed=None,galaxies=False):
        """ Wrapper to minimise chi2 for power spectrum (k < kmax) and delta on grid for voronoi_dm, using PICASA.
             gamma_ranges should be array of shape (4,) containing (gM_min,gM_max,gV_min,gV_max).
             Nsamp,dsig are int,float for use in simulated annealing (ASA) as part of PICASA.optimize().
             If galaxies is False, arg0 should be Npmin, else tuple containing (gpos,gprop).
             Pk_dm_CIC should be power spectrum array calculated using instance powspec of PowerSpectrum(). 
             freq_CIC,lgdbins should be CIC density histogram and bins of log(1+delta). 
             eps_density multiplies contribution of coarse grid residuals to chi2
             Returns output of PICASA.optimize()
        """
        if self.verbose:
            self.print_this("Chi2 calculation (2-d)...",self.logfile)

        if galaxies & (not model_dust):
            self.print_this("... galaxies only supported with explicit dust model; switching on model_dust",self.logfile)
            model_dust = True

        if split_cell & galaxies:
            self.print_this("... split cell model not implemented for galaxies, modelling constant w_dm instead",self.logfile)
            split_cell = False

        if not galaxies:
            Npmin = arg0
        else:
            gpos,gprop = arg0
            if len(gpos.shape) != 2:
                raise TypeError("Incompatible shape for galaxy position data. Need (Ntrc,3), detected (" 
                                + ','.join(['%d' % (i,) for i in gpos.shape]) +').')
            if gpos.shape[1] != 3:
                raise TypeError("Only 3-d data sets supported")
            if gprop.shape != (gpos.shape[0],):
                raise TypeError("Incompatible shape for galaxy property data. Need ({0:d},), detected ("
                                .format(gpos.shape[0])+ ','.join(['%d' % (i,) for i in glum.shape]) +').')

            Ntrc = gprop.size
            if self.verbose:
                self.print_this("... found {0:d} galaxies".format(Ntrc),self.logfile)

        if gamma_ranges.shape != (4,):
            raise TypeError("Incompatible array detected: gamma_ranges")

        if len(freq_CIC.shape) != 1:
            raise TypeError("Incompatible array detected: freq_CIC")
        if lgdbins.size != freq_CIC.size+1:
            raise TypeError("Incompatible array detected: lgdbins")

        if self.verbose:
            self.print_this("... density pdf downweighted by {0:.3f} in chi2".format(eps_density),self.logfile)

        # if len(dmgrid_coarse.shape) != 3:
        #     raise TypeError("Incompatible array detected: dmgrid_coarse")
        # grid_coarse = dmgrid_coarse.shape[0]
        # if self.verbose:
        #     self.print_this("... coarse grid is {0:d}^3; downweighted by {1:.3f} in chi2".format(grid_coarse,eps_density),
        #                     self.logfile)
        # if (dmgrid_coarse.shape[1] != grid_coarse) | (dmgrid_coarse.shape[2] != grid_coarse):
        #     raise TypeError("Incompatible array detected: dmgrid_coarse")

        grid = powspec.grid
        if self.verbose:
            self.print_this("... pow-spec using {0:d}^3 grid with kmax = {1:.3f} h/Mpc".format(grid,kmax),self.logfile)
        if Pk_dm_CIC.shape != powspec.ktab.shape:
            raise TypeError("Incompatible array detected: Pk_dm_CIC")

        ind_k = np.where((powspec.ktab < kmax) & (Pk_dm_CIC > 0.0))[0]
        Nk = powspec.I_SIZE[ind_k]

        if not galaxies:
            mmin = self.mpart*Npmin
            lgmmin = np.log10(mmin)

            if self.verbose:
                self.print_this("... using tracer mass definition: "+massdef,self.logfile)
            hpos,halos = self.prep_halo_data(va=False,massdef=massdef,Npmin=Npmin,keep_subhalos=keep_subhalos) 

            eta_trc = halos[massdef].sum()/(self.Npart*self.mpart)
            Ntrc = halos.size
        else:
            eta_trc = 0.0 # THIS NEEDS TO CHANGE

        if ran_fac is not None:
            self.RAN_FAC = ran_fac
        else:
            self.set_ran_fac(Ntrc,grid=grid)

        if self.verbose:
            self.print_this("... calculating number- and mass-weighted fields",self.logfile)
        if galaxies:
            if (pos_wt_exp is not None):
                if self.verbose:
                    self.print_this("... ... weighted positions not yet implemented for galaxies: using Euclidean distances",
                                    self.logfile)
            delta_trc,ran_pos,ind_nbr,nbr_count,dist_nbr = self.voronoi_periodic_box(gpos,ret_ran=True,seed=seed)
        else:
            pos_wts=None
            if (pos_wt_exp is not None):
                if self.verbose:
                    self.print_this("... ... using weighted distances d(r,t) = d_Euc(r,t)/m(t)^{0:.3f}".format(pos_wt_exp),
                                    self.logfile)
                pos_wts = 1.0/halos[massdef]**pos_wt_exp
            else:
                if self.verbose:
                    self.print_this("... ... using Euclidean distances",self.logfile)
            delta_trc,ran_pos,ind_nbr,nbr_count,dist_nbr = self.voronoi_periodic_box(hpos,ret_ran=True,seed=seed,
                                                                                     pos_wts=pos_wts)


        if self.verbose:
            self.print_this("... setting up data",self.logfile)        
        # data = np.concatenate((Pk_dm_CIC[ind_k],1+dmgrid_coarse.flatten()))
        data = np.concatenate((Pk_dm_CIC[ind_k],np.log10(freq_CIC + self.TINY)))
        invcov_data = np.zeros_like(data)
        wt_prop = 10**gprop if galaxies else halos[massdef]
        verbose_default = self.verbose
        powspec.verbose = False
        model_args = [wt_prop,delta_trc,ind_k,powspec,ran_pos,ind_nbr,nbr_count,lgdbins,eta_trc,
                      split_cell,dist_nbr,Delta_m,model_dust]
        if split_cell:
            if not galaxies:
                model_args.append(hpos)
            else:
                model_args.append(gpos) # redundant for now
        else:
            model_args.append(None) # redundant for now
        if model_dust:
            if self.verbose:
                self.print_this("... ... explicitly modelling w_dust using eta_trc = {0:.3f}".format(eta_trc),self.logfile)        
        else:
            if self.verbose:
                self.print_this("... ... directly modelling w_dm and using eta_trc = {0:.3f} for dust".format(eta_trc),self.logfile)
        del wt_prop
        gc.collect()
        cost_args = (ind_k.size,eps_density)

        if self.verbose:
            self.print_this("... iterating",self.logfile)

        param_mins = np.array([gamma_ranges[0],gamma_ranges[2]])
        param_maxs = np.array([gamma_ranges[1],gamma_ranges[3]])

        self.verbose = False

        data_pack = {}
        data_pack['chi2_file'] = 'data/voronoi_dm/chi2.txt'
        data_pack['Nsamp'] = Nsamp
        data_pack['param_mins'] = param_mins
        data_pack['param_maxs'] = param_maxs
        data_pack['dsig'] = dsig
        data_pack['data'] = data
        data_pack['invcov_data'] = invcov_data
        data_pack['model_func'] = self.model_func_chi2dm
        data_pack['model_args'] = model_args
        data_pack['cost_func'] = self.cost_func_chi2dm
        data_pack['cost_args'] = cost_args
        data_pack['verbose'] = verbose_default

        chi2,gMVvals,cov_asa,params_best,chi2_min,eigvals,rotate,flag_ok = self.optimize(data_pack)
        powspec.verbose = verbose_default
        self.verbose = verbose_default
        del data,invcov_data,model_args,cost_args,data_pack
        gc.collect()
        
        if not galaxies:
            del hpos,halos
        del delta_trc,ran_pos,ind_nbr,nbr_count
        gc.collect()

        if self.verbose:
            self.print_this("... all done",self.logfile)

        self.RAN_FAC = self.RAN_FAC_DEFAULT # do this last, so that calls to other functions don't get affected.

        return chi2,gMVvals,cov_asa,params_best,chi2_min,eigvals,rotate,flag_ok
    ############################################################


    ############################################################
    def model_func_chi2dm(self,model_args,params):
        """ Convenience function for use in voronoi_dm_chi2(). """ 
        #(halo_masses/gal_prop,delta_trc,ind_k,powspec,ran_pos,ind_nbr,nbr_count,
        #  lgdbins,eta_trc,split_cell,dist_nbr,Delta_m,model_dust,pos) = model_args
        gM,gV = params
        eta_trc = model_args[8]
        split_cell = model_args[9]
        Delta_m = model_args[11]
        model_dust = model_args[12]
        
        wts_trc = self.prep_wts(model_args[0],model_args[1].size)
        if model_dust:
            wts_dust = self.prep_wts(model_args[0]**gM*(1+model_args[1])**gV,model_args[1].size)
        else:
            wts = self.prep_wts(model_args[0]**gM*(1+model_args[1])**gV,model_args[1].size)
            wts_dust = (wts - eta_trc*wts_trc)/(1-eta_trc)
            del wts
            gc.collect()

        dmgrid = self.voronoi_to_grid(model_args[1],wts_trc,wts_dust,Delta_m,eta_trc,
                                      model_args[4],model_args[5],model_args[6],model_args[10],
                                      grid=model_args[3].grid,split_cell=split_cell,pos=model_args[13])
        # dmgrid_coarse = self.voronoi_to_grid(model_args[1],wts_trc,wts_dust,200.0,eta_trc,
        #                                      model_args[4],model_args[5],model_args[6],model_args[10],
        #                                      grid=model_args[7],split_cell=split_cell,pos=model_args[13])
        del wts_dust,wts_trc
        gc.collect()
        
        Pk_dm = model_args[3].Pk_grid(dmgrid,input_is_density=True,CIC=False)
        freq,bins = np.histogram(np.log10(1 + dmgrid + self.TINY),bins=model_args[7],density=False)
        model = np.concatenate((Pk_dm[model_args[2]],np.log10(freq + self.TINY)))
        del dmgrid,Pk_dm,freq#,dmgrid_coarse
        gc.collect()

        return model
    ############################################################



    ############################################################
    def cost_func_chi2dm(self,data,invcov_data,model,cost_args):
        """ Convenience function for use in voronoi_dm_chi2(). """ 
        # ind_k.size,eps_density = cost_args
        chi2 = np.sum((model[:cost_args[0]]/data[:cost_args[0]] - 1.0)**2)
        chi2 += cost_args[1]*np.sum((model[cost_args[0]:]/data[cost_args[0]:] - 1.0)**2)
        return chi2
    ############################################################



    ############################################################
    def haloByhalo_alpha(self,input_array,pos,Rh,input_is_density=False,grid=128,powspec=None,CIC=True):
        """ Halo-by-halo alpha. Given input_array containing either
             matter density estimate of shape (grid,grid,grid) [input_is_density=True] 
             or dark matter positions of shape (Npart,3) [input_is_density=False] 
             halo positions pos of shape (Ntrc,3) 
             and halo R200b values Rh (Mpc/h units) of shape (Ntrc,), 
             estimates tidal anisotropy alpha for each halo.
             If input_is_density == False, kwarg grid must be a valid grid size, 
             else powspec must be a valid PowerSpectrum instance with same grid and box as used for density.
             Returns alpha_trc: [shape (Ntrc,)]
        """
        if self.verbose:
            self.print_this("... Halo-by-halo alpha",self.logfile)

        if len(pos.shape) != 2:
            raise TypeError("Incompatible shape for position data. Need (Ntrc,3), detected (" 
                            + ','.join(['%d' % (i,) for i in pos.shape]) +').')

        if pos.shape[1] != 3:
            dim = 'dimension'
            if pos.shape[1] > 1:
                dim += 's'
            raise TypeError("Only 3-d data sets supported. Detected {0:d} ".format(pos.shape[1]) + dim + '.')

        Ntrc = pos.shape[0]

        if len(Rh.shape) != 1:
            raise TypeError("Incompatible shape for halo radius data. Need (Ntrc,), detected (" 
                            + ','.join(['%d' % (i,) for i in Rh.shape]) +').')

        if Rh.size != Ntrc:
            raise TypeError("Incompatible size for halo radius data. Need {0:d} detected {1:d}".format(Ntrc,Rh.size))

        if input_is_density:
            if len(input_array.shape) != 3:
                raise TypeError("Incompatible shape for input_array when input_is_density=True."
                                +" Need (grid,grid,grid), detected (" 
                                + ','.join(['%d' % (i,) for i in input_array.shape]) +').')
            delta_grid = input_array 
            GRID = delta_grid.shape[0]
            if self.verbose:
                self.print_this("... ... using pre-set k-space quantities on {0:d}^3 grid".format(GRID),self.logfile)
            powspec = powspec
        else:
            if (len(input_array.shape) != 2) or (input_array.shape[1]!=3):
                raise TypeError("Incompatible shape for input_array when input_is_density=True."
                                +" Need (Npart,3), detected (" 
                                + ','.join(['%d' % (i,) for i in input_array.shape]) +').')
            if grid is None:
                raise ValueError("kwarg grid cannot be None when input_is_density=False.")
            GRID = grid
            if self.verbose:
                self.print_this("... ... setting up k-space quantities on {0:d}^3 grid".format(GRID),self.logfile)
            powspec = PowerSpectrum(grid=GRID,Lbox=self.Lbox,logfile=self.logfile,verbose=self.verbose)
            delta_grid = powspec.density_field(input_array)

        hgrid_round = np.fmod(np.round(pos*GRID/self.Lbox),GRID).astype('int')
        # shape(Ntrc,3)
        # hgrid_round contains integer vectors identifying the cell (round) corresponding to each halo: equiv to NGP

        RGeff = 4*Rh/np.sqrt(5.)

        N_RG = 15
        RGtab = np.logspace(np.log10(RGeff.min()),np.log10(RGeff.max()),N_RG)
        smoothing_scales = RGtab/self.Lbox
        N_scales = smoothing_scales.size


        psi_eigvals = np.zeros((Ntrc,N_scales,3),dtype=float)

        FTdensity = powspec.fourier_transform_density(delta_grid,CIC=False)

        for s in range(N_scales):
            Rs = smoothing_scales[s]
            if self.verbose:
                self.print_this("... ... ... Rs = {0:.2f}Mpc/h ({1:d} of {2:d} scales)".format(Rs*self.Lbox,s+1,N_scales),self.logfile)
            psi11,psi22,psi33,psi12,psi13,psi23 = powspec.tidal_tensor_field(FTdensity,Rs,input_is_FTdensity=1)

            if self.verbose:
                self.print_this('... ... ... looping over tracers',self.logfile)

            for h in range(Ntrc):
                hcell = hgrid_round[h]

                # consider using interpolation over nbrs below
                psi11_h = psi11[hcell[0],hcell[1],hcell[2]].real
                psi22_h = psi22[hcell[0],hcell[1],hcell[2]].real
                psi33_h = psi33[hcell[0],hcell[1],hcell[2]].real
                psi12_h = psi12[hcell[0],hcell[1],hcell[2]].real
                psi13_h = psi13[hcell[0],hcell[1],hcell[2]].real
                psi23_h = psi23[hcell[0],hcell[1],hcell[2]].real

                psi_mat = np.matrix(np.zeros((3,3),dtype=float))
                psi_mat[np.diag_indices(3)]    = np.array([psi11_h,psi22_h,psi33_h])
                psi_mat[np.triu_indices(3,1)]  = np.array([psi12_h,psi13_h,psi23_h])
                psi_mat[np.tril_indices(3,-1)] = np.array([psi12_h,psi13_h,psi23_h])

                psi_eigvals[h,s] = syal.eigvalsh(psi_mat)
                if self.verbose:
                    self.status_bar(h,Ntrc)

            del psi11,psi22,psi33,psi12,psi13,psi23
            gc.collect()


        psi_eigvals = np.transpose(psi_eigvals,(2,0,1))
        # shape (3,Ntrc,N_scales)

        psi_eigvals_final = np.zeros((3,Ntrc),dtype=float)

        if self.verbose:
            self.print_this('... ... ... interpolating to 4 x R200b',self.logfile)
            self.print_this('... ... ... looping over halos',self.logfile)
        for h in range(Ntrc):
            for eig in range(3):
                psi_eigvals_final[eig,h] = np.interp(np.log10(RGeff[h]),np.log10(RGtab),psi_eigvals[eig,h])
            if self.verbose:
                self.status_bar(h,Ntrc)

        alpha = self.func_alpha(psi_eigvals_final[0],psi_eigvals_final[1],psi_eigvals_final[2])

        del psi_eigvals,psi_eigvals_final,hgrid_round
        gc.collect()

        return alpha
    ############################################################


    ############################################################
    def haloByhalo_aext_brute(self,ppos,hpos,Rh,rmax=10.0):
        """ Halo-by-halo external acceleration field. Given ppos containing 
             dark matter positions (Mpc/h units) of shape (Npart,3),
             halo positions hpos (Mpc/h units) of shape (Ntrc,3) 
             halo Rvir values Rh (Mpc/h units) of shape (Ntrc,)
             and maximum radius rmax (Mpc/h),
             estimates acceleration field aext for each halo in configuration space.
             Returns aext: [shape (Ntrc,3), units m/s^2]
        """
        if self.verbose:
            self.print_this("... Halo-by-halo aext",self.logfile)

        if len(hpos.shape) != 2:
            raise TypeError("Incompatible shape for tracer position data. Need (Ntrc,3), detected (" 
                            + ','.join(['%d' % (i,) for i in hpos.shape]) +').')

        if hpos.shape[1] != 3:
            dim = 'dimension'
            if hpos.shape[1] > 1:
                dim += 's'
            raise TypeError("Only 3-d data sets supported. Detected {0:d} ".format(hpos.shape[1]) + dim + '.')

        if len(ppos.shape) != 2:
            raise TypeError("Incompatible shape for dark matter particle position data. Need (Npart,3), detected (" 
                            + ','.join(['%d' % (i,) for i in ppos.shape]) +').')

        if ppos.shape[1] != 3:
            dim = 'dimension'
            if ppos.shape[1] > 1:
                dim += 's'
            raise TypeError("Only 3-d data sets supported. Detected {0:d} ".format(ppos.shape[1]) + dim + '.')

        Ntrc = hpos.shape[0]
        Npart = ppos.shape[0]

        if len(Rh.shape) != 1:
            raise TypeError("Incompatible shape for halo radius data. Need (Ntrc,), detected (" 
                            + ','.join(['%d' % (i,) for i in Rh.shape]) +').')

        if Rh.size != Ntrc:
            raise TypeError("Incompatible size for halo radius data. Need {0:d} detected {1:d}".format(Ntrc,Rh.size))

        Rh2_box = (Rh/self.Lbox)**2
        rmax2_box = (rmax/self.Lbox)**2

        aext_final = np.zeros((Ntrc,3),dtype=float)

        H02byh = 3.240779e-13 # h/Mpc m/s^2
        prefac = 1.5*self.Om*self.hubble*self.EHub(self.redshift)**2*self.Lbox*H02byh # (3/2) Om H^2 L in m/s^2
        prefac /= (4*np.pi*Npart)

        if self.verbose:
            self.print_this('... ... ... looping over tracers',self.logfile)
        for h in range(Ntrc):
            pos_eval = hpos[h]
            ppos_cen = ppos - (pos_eval - 0.5*self.Lbox) # shape (Npart,3)
            ppos_cen = ppos_cen % self.Lbox
            ppos_cen /= self.Lbox
            ppos_cen -= 0.5
            r2 = np.sum(ppos_cen**2,axis=1) # shape (Npart,)
            cond = (r2 >= Rh2_box[h]) & (r2 <= rmax2_box)
            ppos_cen = ppos_cen[cond]
            r2 = r2[cond]
            acc = (ppos_cen.T/r2**1.5).T # shape (Npart(reduced),3)
            acc *= prefac
            aext_final[h] = np.sum(acc,axis=0) # shape (3,)
            del cond,ppos_cen,r2,acc
            gc.collect()
            if self.verbose:
                self.status_bar(h,Ntrc)

        del Rh2_box
        gc.collect()

        return aext_final
    ############################################################


    ############################################################
    def haloByhalo_aext(self,ppos,hpos,Rh,rmax=10.0,weights=None):
        """ Halo-by-halo external acceleration field. Given ppos containing 
             dark matter positions (Mpc/h units) of shape (Npart,3),
             halo positions hpos (Mpc/h units) of shape (Ntrc,3) 
             halo Rvir values Rh (Mpc/h units) of shape (Ntrc,)
             and maximum radius rmax (Mpc/h),
             estimates acceleration field aext for each halo in configuration space.
             If weights is not None, must be array containing ratios m_halo/m_part. 
             In this case, ppos is assumed to be locations of halos of mass m_halo.
             Returns aext: [shape (Ntrc,3), units m/s^2]
        """
        if self.verbose:
            self.print_this("... Halo-by-halo aext",self.logfile)

        if weights is not None:
            if self.verbose:
                self.print_this("... ... summing over halos not particles",self.logfile)

        if len(hpos.shape) != 2:
            raise TypeError("Incompatible shape for tracer position data. Need (Ntrc,3), detected (" 
                            + ','.join(['%d' % (i,) for i in hpos.shape]) +').')

        if hpos.shape[1] != 3:
            dim = 'dimension'
            if hpos.shape[1] > 1:
                dim += 's'
            raise TypeError("Only 3-d data sets supported. Detected {0:d} ".format(hpos.shape[1]) + dim + '.')

        if len(ppos.shape) != 2:
            raise TypeError("Incompatible shape for dark matter particle position data. Need (Npart,3), detected (" 
                            + ','.join(['%d' % (i,) for i in ppos.shape]) +').')

        if ppos.shape[1] != 3:
            dim = 'dimension'
            if ppos.shape[1] > 1:
                dim += 's'
            raise TypeError("Only 3-d data sets supported. Detected {0:d} ".format(ppos.shape[1]) + dim + '.')

        Ntrc = hpos.shape[0]
        Npart = ppos.shape[0] if weights is None else self.Npart

        if weights is not None:
            if len(weights.shape) != 1:
                raise TypeError("Incompatible shape for halo position data. Need (Nhalo,), detected (" 
                                + ','.join(['%d' % (i,) for i in weights.shape]) +').')
            
            if weights.size != ppos.shape[0]:
                err_str = "Incompatible size for halo position data." 
                err_str += " Need {0:d} detected {1:d}".format(ppos.shape[0],weights.size)
                raise TypeError(err_str)


        if len(Rh.shape) != 1:
            raise TypeError("Incompatible shape for halo radius data. Need (Ntrc,), detected (" 
                            + ','.join(['%d' % (i,) for i in Rh.shape]) +').')

        if Rh.size != Ntrc:
            raise TypeError("Incompatible size for halo radius data. Need {0:d} detected {1:d}".format(Ntrc,Rh.size))

        if self.verbose:
            self.print_this('... ... ... using rmax = {0:.2f} Mpc/h'.format(rmax),self.logfile)

        Rh_box = (Rh/self.Lbox)
        rmax_box = (rmax/self.Lbox)

        aext_final = np.zeros((Ntrc,3),dtype=float)

        H02byh = 3.240779e-13 # h/Mpc m/s^2
        prefac = 1.5*self.Om*self.hubble*self.EHub(self.redshift)**2*self.Lbox*H02byh # (3/2) Om H^2 L in m/s^2
        prefac /= (4*np.pi*Npart)

        if self.verbose:
            self.print_this('... ... ... setting up trees',self.logfile)
        if self.verbose:
            self.print_this('... ... ... ... tracers',self.logfile)
        tree_h = spatial.KDTree(hpos/self.Lbox,boxsize=1.0)
        if self.verbose:
            self.print_this('... ... ... ... particles',self.logfile)
        tree_p = spatial.KDTree(ppos/self.Lbox,boxsize=1.0)

        if self.verbose:
            self.print_this('... ... ... querying for rmax',self.logfile)
        ind_p_out = tree_h.query_ball_tree(tree_p,rmax_box)
        # list of lists. element h is list of particle indices within rmax from halo h

        if self.verbose:
            self.print_this('... ... ... looping over tracers',self.logfile)
        for h in range(Ntrc):
            pos_eval = hpos[h]/self.Lbox
            ind_p_in = tree_p.query_ball_point(pos_eval,Rh_box[h])
            # list of particle indices within Rvir of halo h

            ind_p = list(set(list(ind_p_out[h])).difference(ind_p_in)) 
            # list of particle indices within Rvir to rmax of halo h

            ppos_h = np.remainder(ppos[ind_p]/self.Lbox - pos_eval + 0.5,1.0) - 0.5 # periodic difference vectors
            r2 = np.sum(ppos_h**2,axis=1)

            acc = (ppos_h.T/r2**1.5) # shape (3,Npart(reduced))
            if weights is not None:
                acc *= weights[ind_p]
            acc *= prefac 
            aext_final[h] = np.sum(acc,axis=1) # shape (3,)
            del ind_p_in,ind_p,ppos_h,r2,acc
            gc.collect()
            if self.verbose:
                self.status_bar(h,Ntrc)

        del Rh_box,tree_h,tree_p,ind_p_out
        gc.collect()

        return aext_final
    ############################################################


    ############################################################
    def haloByhalo_aext_kspace(self,input_array,pos,Rh,rmax=10.0,input_is_density=False,
                               grid=128,powspec=None,CIC=False,N_RG=15,sinc=False,
                               kspace_kernel=True,gauss_smooth=False):
        """ Halo-by-halo external acceleration field. Given input_array containing either
             matter density estimate of shape (grid,grid,grid) [input_is_density=True] 
             or dark matter positions of shape (Npart,3) [Mpc/h units; input_is_density=False] 
             halo positions pos (Mpc/h units) of shape (Ntrc,3) 
             halo Rvir values Rh (Mpc/h units) of shape (Ntrc,)
             and maximum radius rmax (Mpc/h),
             estimates acceleration field aext for each halo.
             If input_is_density == False, kwarg grid must be a valid grid size, 
             else powspec must be a valid PowerSpectrum instance with same grid and box as used for density.
             Returns aext: [shape (Ntrc,3), units m/s^2]
        """
        if self.verbose:
            self.print_this("... Halo-by-halo aext",self.logfile)

        if len(pos.shape) != 2:
            raise TypeError("Incompatible shape for position data. Need (Ntrc,3), detected (" 
                            + ','.join(['%d' % (i,) for i in pos.shape]) +').')

        if pos.shape[1] != 3:
            dim = 'dimension'
            if pos.shape[1] > 1:
                dim += 's'
            raise TypeError("Only 3-d data sets supported. Detected {0:d} ".format(pos.shape[1]) + dim + '.')

        Ntrc = pos.shape[0]

        if len(Rh.shape) != 1:
            raise TypeError("Incompatible shape for halo radius data. Need (Ntrc,), detected (" 
                            + ','.join(['%d' % (i,) for i in Rh.shape]) +').')

        if Rh.size != Ntrc:
            raise TypeError("Incompatible size for halo radius data. Need {0:d} detected {1:d}".format(Ntrc,Rh.size))

        if input_is_density:
            if len(input_array.shape) != 3:
                raise TypeError("Incompatible shape for input_array when input_is_density=True."
                                +" Need (grid,grid,grid), detected (" 
                                + ','.join(['%d' % (i,) for i in input_array.shape]) +').')
            delta_grid = input_array 
            GRID = delta_grid.shape[0]
            if self.verbose:
                self.print_this("... ... using pre-set k-space quantities on {0:d}^3 grid".format(GRID),self.logfile)
            powspec = powspec
        else:
            if (len(input_array.shape) != 2) or (input_array.shape[1] != 3):
                raise TypeError("Incompatible shape for input_array when input_is_density=True."
                                +" Need (Npart,3), detected (" 
                                + ','.join(['%d' % (i,) for i in input_array.shape]) +').')
            if grid is None:
                raise ValueError("kwarg grid cannot be None when input_is_density=False.")
            GRID = grid
            if self.verbose:
                self.print_this("... ... setting up k-space quantities on {0:d}^3 grid".format(GRID),self.logfile)
            powspec = PowerSpectrum(grid=GRID,Lbox=self.Lbox,logfile=self.logfile,verbose=self.verbose)
            delta_grid = powspec.density_field(input_array)

        hgrid_round = np.fmod(np.round(pos*GRID/self.Lbox),GRID).astype('int')
        # shape(Ntrc,3)
        # hgrid_round contains integer vectors identifying the cell (round) corresponding to each halo: equiv to NGP

        Rhtab = np.logspace(np.log10(Rh.min()),np.log10(Rh.max()),N_RG)
        smoothing_scales = Rhtab/self.Lbox
        N_scales = smoothing_scales.size
        Rmax = rmax/self.Lbox

        aext_vals = np.zeros((Ntrc,N_scales,3),dtype=float)

        FTdensity = powspec.fourier_transform_density(delta_grid,CIC=CIC) 
        H02byh = 3.240779e-13 # h/Mpc m/s^2
        prefac = 1.5*self.Om*self.hubble*self.EHub(self.redshift)**2*H02byh

        for s in range(N_scales):
            Rs = smoothing_scales[s]
            if self.verbose:
                self.print_this("... ... ... Rs = {0:.3f}Mpc/h ({1:d} of {2:d} scales)".format(Rs*self.Lbox,s+1,N_scales),self.logfile)
            a1,a2,a3 = powspec.aext_field(FTdensity,Rs,Rmax,prefac=prefac,input_is_FTdensity=1,
                                          CIC=CIC,sinc=sinc,kspace_kernel=kspace_kernel,gauss_smooth=gauss_smooth)

            if self.verbose:
                self.print_this('... ... ... looping over tracers',self.logfile)

            for h in range(Ntrc):
                hcell = hgrid_round[h]

                # consider using interpolation over nbrs below
                aext_vals[h,s,0] = a1[hcell[0],hcell[1],hcell[2]].real
                aext_vals[h,s,1] = a2[hcell[0],hcell[1],hcell[2]].real
                aext_vals[h,s,2] = a3[hcell[0],hcell[1],hcell[2]].real

                if self.verbose:
                    self.status_bar(h,Ntrc)

            del a1,a2,a3
            gc.collect()

        aext_vals = np.transpose(aext_vals,(2,0,1))
        # shape (3,Ntrc,N_scales)

        aext_final = np.zeros((Ntrc,3),dtype=float)
        lgRh = np.log10(Rh)
        lgRhtab = np.log10(Rhtab)

        if self.verbose:
            self.print_this('... ... ... interpolating to Rvir',self.logfile)
            self.print_this('... ... ... looping over halos',self.logfile)
        for h in range(Ntrc):
            for i in range(3):
                aext_final[h,i] = np.interp(lgRh[h],lgRhtab,aext_vals[i,h])
            if self.verbose:
                self.status_bar(h,Ntrc)

        del aext_vals,hgrid_round,lgRh,lgRhtab,Rhtab
        gc.collect()

        return aext_final
    ############################################################

    ############################################################
    def haloByhalo_b1(self,input_array,pos,delta_for_Pk=None,ret_Pk=False,
                      input_is_density=True,grid=None,CIC=True,NBIN=30,LGBIN=1):
        """ Halo-by-halo linear bias. Given input_array containing either
             matter density estimate of shape (grid,grid,grid) [input_is_density=True] 
             or dark matter positions of shape (Npart,3) [input_is_density=False] 
             and halo positions pos of shape (Ntrc,3), estimates linear bias b1 for each halo.
             If delta_for_Pk is not None, this should contain density estimate of shape (grid,grid,grid) for
             calculating normalising power spectrum in bias calculation.
             If input_is_density == False, kwarg grid must be a valid grid size.
             Returns 1. b1_trc: 1 array of shape (Ntrc,)
                           2. b1_k,ktab[krange]: 2 arrays of shape (krange.size,)
                           3. (optionally) Pk_delta,ktab: 2 arrays of shape (ktab.size,)
        """

        if self.verbose:
            self.print_this("... Halo-by-halo linear bias",self.logfile)

        if len(pos.shape) != 2:
            raise TypeError("Incompatible shape for position data. Need (Ntrc,3), detected (" 
                            + ','.join(['%d' % (i,) for i in pos.shape]) +').')

        if pos.shape[1] != 3:
            dim = 'dimension'
            if pos.shape[1] > 1:
                dim += 's'
            raise TypeError("Only 3-d data sets supported. Detected {0:d} ".format(pos.shape[1]) + dim + '.')

        if input_is_density:
            if len(input_array.shape) != 3:
                raise TypeError("Incompatible shape for input_array when input_is_density=True."
                                +" Need (grid,grid,grid), detected (" 
                                + ','.join(['%d' % (i,) for i in input_array.shape]) +').')
            delta_grid = input_array 
            GRID = delta_grid.shape[0]
            nbin = int(NBIN*np.log(GRID/2)/np.log(256)) 
            # nbin = int(NBIN*(1.0*GRID/512)**0.2)
        else:
            if (len(input_array.shape) != 2) or (input_array.shape[1]!=3):
                raise TypeError("Incompatible shape for input_array when input_is_density=True."
                                +" Need (Npart,3), detected (" 
                                + ','.join(['%d' % (i,) for i in input_array.shape]) +').')
            if grid is None:
                raise ValueError("kwarg grid cannot be None when input_is_density=False.")
            GRID = grid
            nbin = int(NBIN*np.log(GRID/2)/np.log(256)) 
            # nbin = int(NBIN*(1.0*grid/512)**0.2)            

        if self.verbose:
            self.print_this("... ... setting up k-space quantities",self.logfile)
        powspec = PowerSpectrum(grid=GRID,Lbox=self.Lbox,logfile=self.logfile,lgbin=LGBIN,nbin=nbin,
                                verbose=self.verbose)

        KMAX = np.min([np.where(powspec.ktab <= self.kmax)[0][-1],powspec.nbin/2])

        if input_is_density:
            if self.verbose:
                self.print_this("... ... using supplied density field",self.logfile)
        else:
            if self.verbose:
                self.print_this("... ... calculating CIC density field",self.logfile)
            delta_grid = powspec.density_field(input_array.T)

        if delta_for_Pk is not None:
            if delta_for_Pk.shape != delta_grid.shape:
                raise TypeError("Incompatible shape detected for array delta_for_Pk in haloByhalo_b1().")

        Ntrc = pos.shape[0]

        hpos_grid = pos*GRID/self.Lbox
        hgrid = np.fmod(hpos_grid,GRID).astype('int')
        # shape(Ntrc,3)
        # hgrid contains integer vectors identifying the cell (floor) corresponding to each halo: used by CIC

        if self.verbose:
            self.print_this("... ... Fourier transforming matter density",self.logfile)
        delta_matter = powspec.fourier_transform_density(delta_grid,CIC=CIC)

        Pk_delta = powspec.Pk_grid(delta_matter,input_is_FTdensity=1,CIC=CIC)
        # Pk_delta = powspec.Pk_grid(delta_grid,input_is_density=1,CIC=CIC)
        Pk_matter = (1.0*Pk_delta
                     if delta_for_Pk is None else
                     powspec.Pk_grid(delta_for_Pk,input_is_density=1,CIC=CIC))
            
        cond_posPk = (Pk_matter > 0.0)
        ind_posPk = np.where(cond_posPk)[0]
        if ind_posPk.size:
            KMIN = ind_posPk[0]
            # if KMIN==0:
            #     KMIN = 1
            if KMIN >= KMAX:
                raise ValueError("Incompatible Pk detected in haloByhalo_b1. Try again with finer grid.")
        else:
            raise ValueError("Pk = 0 detected at all k in haloByhalo_b1. Try again with finer grid.")

        if self.verbose:
            self.print_this("... ... setting up index arrays",self.logfile)
        krange = np.arange(KMIN,KMAX,dtype=int)

        ind = [[]]
        k1_array = [[]]
        k2_array = [[]]
        k3_array = [[]]
        for k in range(krange.size-1):
            ind.append([])
            k1_array.append([])
            k2_array.append([])
            k3_array.append([])

        i_size = np.zeros(krange.size,dtype=int)
        for k in range(krange.size):
            cond = ((powspec.KK >= powspec.k2_compare[krange[k]]) 
                    & (powspec.KK < powspec.k2_compare[krange[k]+1]))
            ind[k] = np.where(cond) # indices of cells contributing to bin
            i_size[k] = ind[k][0].size # number of cells contributing to bin
            k1_array[k] = powspec.K1[ind[k][0],:,:].flatten()*2*np.pi/GRID # 2pi/GRID is for later convenience
            k2_array[k] = powspec.K2[:,ind[k][1],:].flatten()*2*np.pi/GRID
            k3_array[k] = powspec.K3[:,:,ind[k][2]].flatten()*2*np.pi/GRID
            del cond

        i_size_max = np.max(i_size)
        K_ARRAY = np.zeros((krange.size,i_size_max,3),dtype=float) # create enough storage space as numpy array
        for k in range(krange.size):
            K_ARRAY[k,:i_size[k],0] = k1_array[k]
            K_ARRAY[k,:i_size[k],1] = k2_array[k]
            K_ARRAY[k,:i_size[k],2] = k3_array[k]

        del k1_array,k2_array,k3_array
        gc.collect()

        nmodes_k = 1.0*i_size
        nmodes_k[~cond_posPk[krange]] = 0.0 # don't use bins where Pk=0

        if self.verbose:
            self.print_this("... ... trimming matter modes",self.logfile)
        delta_k = np.zeros((krange.size,i_size_max),dtype=complex)
        for k in range(krange.size):
            delta_k[k,:i_size[k]] = delta_matter[ind[k]]

        del delta_matter,ind
        gc.collect()

        # For use in CIC weights
        cp1 = (hgrid + 1) % GRID

        cic_d = np.absolute(hpos_grid - np.floor(hpos_grid))# - 0.5) # incorrect -0.5, but makes no practical difference
        # shape (Ntrc,3)
        cic_d = cic_d.T
        cic_t = 1.0 - cic_d
        # shape (3,Ntrc)

        del hpos_grid
        gc.collect()

        cic_ttt = cic_t[0]*cic_t[1]*cic_t[2]
        cic_dtt = cic_d[0]*cic_t[1]*cic_t[2]
        cic_tdt = cic_t[0]*cic_d[1]*cic_t[2]
        cic_ttd = cic_t[0]*cic_t[1]*cic_d[2]
        cic_ddt = cic_d[0]*cic_d[1]*cic_t[2]
        cic_dtd = cic_d[0]*cic_t[1]*cic_d[2]
        cic_tdd = cic_t[0]*cic_d[1]*cic_d[2]
        cic_ddd = cic_d[0]*cic_d[1]*cic_d[2]
        # shape (Ntrc,)

        del cic_d,cic_t
        gc.collect()

        if self.verbose:
            self.print_this("... ... calculating bias",self.logfile)

        hgrid = hgrid.T
        cp1 = cp1.T
        # shape (3,Ntrc)

        if self.verbose:
            self.print_this("... ... ... cic ttt",self.logfile)
        tmp_phase = np.tensordot(K_ARRAY,hgrid,axes=1).astype('complex')
        halo_phase = cic_ttt*np.exp(-1.j*tmp_phase)
        if self.verbose:
            self.print_this("... ... ... cic dtt",self.logfile)
        tmp_phase = np.tensordot(K_ARRAY,np.array([cp1[0],hgrid[1],hgrid[2]]),axes=1).astype('complex')
        halo_phase += cic_dtt*np.exp(-1.j*tmp_phase)
        if self.verbose:
            self.print_this("... ... ... cic tdt",self.logfile)
        tmp_phase = np.tensordot(K_ARRAY,np.array([hgrid[0],cp1[1],hgrid[2]]),axes=1).astype('complex')
        halo_phase += cic_tdt*np.exp(-1.j*tmp_phase)
        if self.verbose:
            self.print_this("... ... ... cic ttd",self.logfile)
        tmp_phase = np.tensordot(K_ARRAY,np.array([hgrid[0],hgrid[1],cp1[2]]),axes=1).astype('complex')
        halo_phase += cic_ttd*np.exp(-1.j*tmp_phase)
        if self.verbose:
            self.print_this("... ... ... cic ddt",self.logfile)
        tmp_phase = np.tensordot(K_ARRAY,np.array([cp1[0],cp1[1],hgrid[2]]),axes=1).astype('complex')
        halo_phase += cic_ddt*np.exp(-1.j*tmp_phase)
        if self.verbose:
            self.print_this("... ... ... cic dtd",self.logfile)
        tmp_phase = np.tensordot(K_ARRAY,np.array([cp1[0],hgrid[1],cp1[2]]),axes=1).astype('complex')
        halo_phase += cic_dtd*np.exp(-1.j*tmp_phase)
        if self.verbose:
            self.print_this("... ... ... cic tdd",self.logfile)
        tmp_phase = np.tensordot(K_ARRAY,np.array([hgrid[0],cp1[1],cp1[2]]),axes=1).astype('complex')
        halo_phase += cic_tdd*np.exp(-1.j*tmp_phase)
        if self.verbose:
            self.print_this("... ... ... cic ddd",self.logfile)
        tmp_phase = np.tensordot(K_ARRAY,cp1,axes=1).astype('complex')
        halo_phase += cic_ddd*np.exp(-1.j*tmp_phase)
        ###############
        # shape (krange,i_size_max,Ntrc)
        halo_phase = np.transpose(halo_phase,(2,0,1))
        # shape (Ntrc,krange,i_size_max)
        bias_k = np.sum(halo_phase*delta_k,axis=2)/(i_size + self.TINY) # zeros in delta_k ensure this is actually mean
        # shape (Ntrc,krange)

        del tmp_phase,halo_phase
        gc.collect()

        del cp1,hgrid,delta_k,K_ARRAY
        del cic_ttt,cic_dtt,cic_tdt,cic_ttd,cic_ddt,cic_dtd,cic_tdd,cic_ddd
        gc.collect()

        bias_k *= self.Lbox**3
        bias_k = bias_k.real/(Pk_matter[krange]+self.TINY)

        if self.FIT_BIAS:
            # fit b1 = b10 + b11*k^2 and return b10. then self.kmax can be larger than 0.15.
            # recall bias_k has shape (Ntrc,krange.size)
            if self.verbose:
                self.print_this("... ... fitting b1(k) = b10 + b11*k**2",self.logfile)
            wts_k = 1.0*nmodes_k*(Pk_matter[krange]+self.TINY) # see notes for multiplication by P_mm
            F00 = np.sum(wts_k)
            F02 = np.sum(powspec.ktab[krange]**2*wts_k)
            F22 = np.sum(powspec.ktab[krange]**4*wts_k)
            detF = F22*F00 - F02**2 + self.TINY
            # shape (scalar)
            B0 = np.sum(bias_k*wts_k,axis=1) 
            B2 = np.sum(bias_k*powspec.ktab[krange]**2*wts_k,axis=1)
            # shape (Ntrc,)
            b1 = (F22*B0 - F02*B2)/detF
            err_b1 = np.sqrt(F22/detF) # not used, since overall normalisation of errors not yet valid
        else:
            if self.verbose:
                self.print_this("... ... reporting mode-weighted sum of b1(k)",self.logfile)
            wts_k = 1.0*nmodes_k*(Pk_matter[krange]+self.TINY) # see notes for multiplication by P_mm
            wts_k = wts_k/np.sum(wts_k)
            b1 = np.sum(wts_k*bias_k,axis=1)
            
        bias_k = np.mean(bias_k,axis=0)

        return ((b1,bias_k,powspec.ktab[krange]) 
                if not ret_Pk else 
                (b1,bias_k,powspec.ktab[krange],Pk_delta,powspec.ktab))

    ############################################################



#######################################################################

if __name__ == "__main__":
    import scipy.special as sysp
    start_time = time()

    vor = Voronoi(sim_stem='scm1024',EXTERNAL_STORAGE=False,RSD=False,real=1,snap=200)
    
    nbin = 50 
    ybin = np.logspace(-3,2,nbin+1)
    ycen = np.sqrt(ybin[1:]*ybin[:-1])
    dlny = np.log(ycen[1]/ycen[0])

    # a,b,c=3.24174,3.24269,1.26861
    a,b,c=4.8065,4.06342,1.16391

    LGMASS_VALS = np.array([11.0,12.5])
    MASS_VALS = 10**LGMASS_VALS
    NPMIN_VALS = np.rint(MASS_VALS/vor.mpart).astype(int)
    ypy = np.zeros((LGMASS_VALS.size+3,nbin),dtype=float)
    ypy[0] = ycen
    ypy[-1] = c*b**(a/c)/sysp.gamma(a/c)*ycen**a*np.exp(-b*ycen**c)

    for np in range(NPMIN_VALS.size):
        Npmin = NPMIN_VALS[np]
        mmin = vor.mpart*Npmin
        pos,halos = vor.prep_halo_data(va=False,massdef='m200b',Npmin=Npmin,keep_subhalos=False)
        Ntrc = halos.size

        vor.MAX_RANDOMS = 2e8
        vor.set_ran_fac(Ntrc)

        delta_trc = vor.voronoi_periodic_box(pos,ret_ran=False)
        y_trc = 1/(1.0 + delta_trc)

        temp,bins = np.histogram(y_trc,bins=ybin,density=False)
        ypy[np+1] = 1.0*temp/np.trapz(temp,dx=dlny)

        if np==0:
            vor.print_this('... generating random tracer locations',vor.logfile)
            pos_trc = vor.Lbox*vor.rng.rand(Ntrc,3) 
            vor.print_this("... tessellating {0:d} random tracers".format(pos_trc.shape[0]),vor.logfile)
            delta_trc = vor.voronoi_periodic_box(pos_trc,ret_ran=False)
            y_trc = 1/(1.0 + delta_trc)

            temp,bins = np.histogram(y_trc,bins=ybin,density=False)
            ypy[-2] = 1.0*temp/np.trapz(temp,dx=dlny)

        del delta_trc,y_trc,pos,halos
        gc.collect()

    outfile = 'data/vvf_pdf_scm1024.txt'
    vor.print_this('... writing to file: '+outfile,vor.logfile)
    hdr_str = "lg(mlim) = ("+','.join(map(str,LGMASS_VALS))+")\n"
    hdr_str += "y | p(y)_{mlim} | p(y)_random | Poisson fit"
    np.savetxt(outfile,ypy.T,fmt='%.6e',header=hdr_str)
    vor.time_this(start_time)

    # sim_stem = 'scm1024'
    # real = 1 
    # snap = 200 # 92
    # RSD = False 

    # # np.random.seed(42)

    # FIT_BIAS = True
    # DEBUG = True
    # MU_SEP = 5.0 # 300 gives no massive objects in scm1024
    # DELTA_TRC_SEP = 1.0 # 1e3 gives no lmhc objects

    # vor = Voronoi(sim_stem=sim_stem,TREES=True,EXTERNAL_STORAGE=False,
    #               DB=1,RSD=RSD,real=real,snap=snap,fit_bias=FIT_BIAS,
    #               mu_sep=MU_SEP,Delta_trc_sep=DELTA_TRC_SEP,
    #               debug=DEBUG)    

    # Npmin = 60000
    # MASSDEF = 'm200b'
    # GRID = 256
    # MAX_ITER = 10
    # SIG_EPS = 0.01
    # DUST_IS_CLUSTERED = True
    # SEGREGATE_TRACERS = True
    # SMOOTH = False

    # delta_dm_grid_seg,pos,iteration = vor.iterate_voronoi(Npmin,massdef=MASSDEF,grid=GRID,smooth=SMOOTH,
    #                                                       max_iter=MAX_ITER,sig_eps=SIG_EPS,
    #                                                       dust_is_clustered=True,segregate_tracers=True)
    
    # vor.time_this(start_time)

    # delta_halo = np.sort(delta_halo)
    # import matplotlib.pyplot as plt
    # plt.plot(1.0 + np.arange(delta_halo.size),1+delta_halo,lw=1.0)
    # # plt.xscale('log')
    # plt.yscale('log')
    # plt.ylim([5e-3,5e2])
    # plt.savefig('test.png')

