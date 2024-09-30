import numpy as np
import scipy.spatial as spatial
from readers import HaloReader
from utilities import Constants
import gc

class Voronoi(HaloReader,Constants):
    """ Voronoi methods for density estimation in halo (and, later, galaxy) catalogs."""
    ############################################################
    ############################################################
    def __init__(self,sim_stem='scm1024',real=1,snap=200,Ran_Fac=30000,
                 verbose=True,logfile=None,seed=None,firstcall=False,N_Proc=1):
        HaloReader.__init__(self,sim_stem=sim_stem,real=real,snap=snap,logfile=logfile,verbose=verbose)
        Constants.__init__(self)

        if self.verbose:
            print_string = "--------------------------------\n"
            print_string += "Voronoi number density estimator\n"
            print_string += "--------------------------------\n"
            print_string += "... using simulation " + self.sim_stem
            overwrite = True if firstcall else False
            self.print_this(print_string,self.logfile,overwrite=overwrite)

        self.seed = seed
        self.N_Proc = N_Proc
            
        self.Nhalo_Hi = 10000
        self.Nran_Per_Cell = 4.0 # 4.0: minimum average number of randoms in smallest cell
        self.Grid_Max = 512 # max grid size we can handle
        # use these to adjust self.Ran_Fac for individual halo populations

        # maximum number of randoms allowed
        # used 2e8 for VVF paper [most VVF stats convgd at 6e7]
        # voronoi_dm seems ~10% converged between 4e8-8e8
        self.Max_Randoms = 5e8 
        
        # factor to multiply with Nhalo to set number of randoms for uniform sampling of volume.
        self.RAN_FAC_DEFAULT = 1*Ran_Fac 
        self.Ran_Fac = 1*Ran_Fac 
        # Default 30k based on output of convergence_ranfac.py (<~1% conv of 1+dhalo) 
        # and also based on having enough randoms in 256^3 grid for 1000-halo sample.
        # 10k is also fine (<~2% conv of 1+dhalo) and may be useful for large samples.
        # RAN_FAC_DEFAULT will be kept fixed.
        # Ran_Fac may change from population to population of tracers.

        if self.verbose:
            print_string = "--------------------------------\n"
            print_string += "... simulation box size Lbox = {0:.1f} Mpc/h\n".format(self.Lbox)
            print_string += ("... cosmology (Om,OLam,hubble) = ({0:.4f},{1:.4f},{2:.3f})\n"
                             .format(self.Om,self.OLam,self.hubble))
            print_string += "... working at redshift z = {0:.3f} in realisation {1:d}".format(self.redshift,self.real)
            self.print_this(print_string,self.logfile)
            self.print_this("... initialisation complete\n--------------------------------",self.logfile)


    ############################################################
    def set_ran_fac(self,Ntrc,grid=None):
        """Convenience function. Changes value of self.Ran_Fac."""

        Grid = grid if grid is not None else self.Grid_Max
        self.Ran_Fac = int(self.Nran_Per_Cell*Grid**3/Ntrc)
        if (Ntrc < self.Nhalo_Hi):
            if (self.Ran_Fac < 30000):
                self.Ran_Fac = 30000
        else:
            if (self.Ran_Fac > 15000):
                self.Ran_Fac = 15000
            elif (self.Ran_Fac < 10000):
                self.Ran_Fac = 10000
        self.Ran_Fac = np.min([self.Ran_Fac,int(self.Max_Randoms/Ntrc)])

        return
    ############################################################
    

    ############################################################
    def voronoi_periodic_box(self,pos,ret_ran=True,seed=None):
        """ Estimate Voronoi-based number density field for halos in a periodic box of side self.Lbox. 
             Expect array-like pos containing positions in range 0..self.Lbox with shape (Ntrc,3).
             If seed is None, use self.seed to initialise RandomState() [useful for consistency across function calls], 
             else use seed. Default None.
             ** THIS IS BASIC, UN-WEIGHTED VERSION. WILL INCLUDE VERSION FOR PSEUDO-VORONOI VOLUMES IF NEEDED FROM voronoi_web-LOCAL.py. **
             Output behaviour:
             - ret_ran == True:
                 1. (Ntrc,): contains values of delta_trc(t) where t indexes the tracer catalog and
                     1+delta_trc(t) = (Nrantot/Ntrc)/Nran(t) = self.Ran_Fac/Nran(t),
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

        Nran = self.Ran_Fac*Ntrc
        if self.verbose:
            self.print_this("... ... generating {0:d} x Ntrc random positions".format(self.Ran_Fac),self.logfile)
        ran_pos = self.Lbox*rng.rand(Nran,3)

        if self.verbose:
            self.print_this('... ... generating tracer tree',self.logfile)
        tree_data = spatial.KDTree(pos,boxsize=self.Lbox)
        if self.verbose:
            self.print_this('... ... finding nearest nbr of each random in tracer data',self.logfile)
        dist_nbr,ind_nbr = tree_data.query(ran_pos,k=1,workers=self.N_Proc)
        # dist_nbr,ind_nbr have shape (Nran,)
        # ind_nbr gives indices in halo data of nearest neighbour of each random
        # dist_nbr gives distance of nearest neighbour of each random
        del tree_data
        gc.collect()

        if self.verbose:
            self.print_this('... ... counting number of random nbrs for each tracer',self.logfile)
        nbr_count = np.bincount(ind_nbr,minlength=Ntrc)
        out = self.Ran_Fac/(nbr_count + self.TINY) - 1.0 
        
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
            
