import numpy as np
import scipy.fftpack as fft
import scipy.spatial as syspat
import scipy.special as sysp

from utilities import Utilities,Paths,Constants
import gc

from numpy.ctypeslib import ndpointer
from ctypes import *

#################################################################
# New (very slow) 2pcf class added by Aseem 21 Feb 2026
#################################################################
class TwoPointCorrelationFunctionPeriodic(Utilities,Paths,Constants):
    """ 2-point (cross-)correlation functions in bins of separation. 
        Assumes complete, cubic periodic box throughout.
    """
    #############################################################
    def __init__(self,lgbin=False,smin=1e-2,smax=3e1,n_s=15,Lbox=300.0,
                 aniso=True,n_mu=161,los=2,L_Max=3,verbose=True,logfile=None):
        """ Initialise the following:
            -- lgbin    : use logarithmic (True) or linear (False) binning
            -- smin,smax: min,max separations in s.
            -- n_s      : number of bins in s.
            -- Lbox     : Box size in same units as smin,smax.
            -- aniso    : bool (default True). If True, calculate multipoles ell = 2*L for L in range(L_Max), else only monopole ell=0.
            -- n_mu     : number of bins in mu.               --> Only used if aniso=True.
            -- los      : int 0/1/2, line of sight direction. --> Only used if aniso=True.
            -- L_Max    : int >= 1, number of multipoles.     --> Only used if aniso=True.
            -- verbose  : write out messages
            -- logfile  : logfile (default None)
             
            Methods: 
            pair_counts,DD_split,DD_only,RR_theory,auto_CF
            """
        Paths.__init__(self)
        Constants.__init__(self)
        Utilities.__init__(self)
        self.lgbin = lgbin
        self.smin = smin
        self.smax = smax
        self.n_s = n_s
        self.verbose = verbose
        self.logfile = logfile
        self.Lbox = Lbox
        self.aniso = aniso
        # below only used if self.aniso is True
        self.n_mu = n_mu # tested 41,81,201. ell=0 converged at <1% at 81. set default to 161. 
        self.L_Max = L_Max
        self.los = los
        self.non_los = np.where(np.arange(3,dtype=int) != self.los)[0]

        self.N_SPLIT = 100000 # 1000000; max number of objects that can be handled on 1 cpu

        if self.verbose:
            print_string = "Initialising 2pcf calculator for periodic boxes\n"
            print_string += "--------------------------------"
            self.print_this(print_string,self.logfile)

        if self.smax >= 0.5*self.Lbox:
            self.print_this("Warning! : smax > Lbox/2 detected. Theory RR may be incorrect.",self.logfile)
            
            
        if self.lgbin:
            self.sbin = np.logspace(np.log10(self.smin),np.log10(self.smax),self.n_s+1)
            self.smid = np.sqrt(self.sbin[1:]*self.sbin[:-1])
        else:
            self.sbin = np.linspace(self.smin,self.smax,self.n_s+1)
            self.smid = 0.5*(self.sbin[1:]+self.sbin[:-1])

        if self.aniso:
            if self.verbose:
                self.print_this("... 2pcf multipoles calculated",self.logfile)
            self.mubin = np.linspace(-1,1,self.n_mu+1)
            self.mumid = 0.5*(self.mubin[1:]+self.mubin[:-1])
            self.dmu = self.mubin[1] - self.mubin[0] 
            self.Pell = np.ones((self.L_Max,self.n_mu),dtype=float)
            for L in range(self.L_Max):
                self.Pell[L] = sysp.legendre_p(2*L,self.mumid)[0]
        else:
            if self.verbose:
                self.print_this("... real-space/monopole 2pcf calculated",self.logfile)

        if self.verbose:
            print_string = "... initialisation complete\n"
            print_string += "--------------------------------"
            self.print_this(print_string,self.logfile)

    #############################################################


    #############################################################
    def pair_counts(self,pos_1,pos_2,mark_1=None,mark_2=None):
        """ Unnormalised binned pair counts between two data sets (can be the same data set).
            -- pos_j : tracer positions shape (ndata_j,3) 
            -- mark_j: tracer marks, None (default) or float shape (ndata_j,) 
            Returns array of shape (self.n_s,self.n_mu) if self.aniso==True else (self.n_s,). 
        """
        MARKED = False
        if (mark_1 is not None) & (mark_2 is not None):
            wts_1 = mark_1/np.mean(mark_1)
            wts_2 = mark_2/np.mean(mark_2)
            MARKED = True

        if MARKED & self.aniso:
            raise NotImplementedError("anisotropic marked pair counts still under construction!")

        # 3-d trees used for both aniso and monopole calcs
        tree_1 = syspat.cKDTree(pos_1,boxsize=self.Lbox)
        tree_2 = syspat.cKDTree(pos_2,boxsize=self.Lbox)
        
        if self.aniso:
            bin_counts = np.zeros((self.n_s,self.n_mu),dtype=int)                
            ########################
            ndata_1 = tree_1.n
            ########################
            ind_s = tree_1.query_ball_tree(tree_2,self.sbin[0])
            # ind_s is list of size len(data1)
            # each element j is list of s-ball-nbrs of data1[j] in data2
            s_len = np.array(list(map(len,ind_s)))
            # s_len is array of lengths of ind_s lists
            for s in range(1,self.n_s+1):
                ########################
                ind_sDs = tree_1.query_ball_tree(tree_2,self.sbin[s])
                # ind_sDs[j] is list of indices of data2 that are (s+Ds)-ball-nbrs of data1[j]
                sDs_len = np.array(list(map(len,ind_sDs)))

                cond_nonzero = (sDs_len > s_len)
                nonzero_ind = np.where(cond_nonzero)[0] # pick elements of data1 with at least one s-bin-nbr in data2

                ind_Ds = [np.concatenate([np.setdiff1d(ind_sDs[j],ind_s[j],assume_unique=True)]).astype(int)
                          for j in nonzero_ind]
                ########################
                # ind_Ds[j] is array of indices of data2 that are s-bin-nbrs of data1[j]
                # Use these for mu measurement
                ########################
                
                # CAN WE ELIMINATE THIS LOOP??
                for jj in range(len(nonzero_ind)):
                    j = nonzero_ind[jj]
                    delta_mu = pos_2[ind_Ds[jj],self.los] - pos_1[j,self.los]
                    cond_out = (np.fabs(delta_mu) > 0.5*self.Lbox)
                    delta_mu[cond_out] -= np.sign(delta_mu[cond_out])*self.Lbox
                    s_sq = delta_mu**2
                    for nl in self.non_los:
                        dsp = np.fabs(pos_2[ind_Ds[jj],nl] - pos_1[j,nl])
                        dsp[dsp > 0.5*self.Lbox] -= self.Lbox
                        s_sq += dsp**2
                    delta_mu /= np.sqrt(s_sq) 
                    counts,dummy = np.histogram(delta_mu,bins=self.mubin,density=False)
                    # which of these are also mu-bin-nbrs.
                    bin_counts[s-1] += counts
                    # we only care about number of (s,mu)-nbrs, not identities
                    ########################                                            
                del nonzero_ind,cond_nonzero,ind_Ds

                ind_s = ind_sDs.copy()
                s_len = sDs_len.copy()
                ########################
        ###################        
        else:
            if MARKED:
                bin_counts = tree_1.count_neighbors(tree_2,self.sbin,cumulative=False,weights=(wts_1,wts_2))
            else:
                bin_counts = tree_1.count_neighbors(tree_2,self.sbin,cumulative=False)
            bin_counts = bin_counts[1:] # first value contains all pairs with seps smaller than lower edge of first bin
            ########################
            
        del tree_1,tree_2
        gc.collect()

        return bin_counts 
    #############################################################


    #############################################################
    def RR_theory(self):
        if self.verbose:
            self.print_this("... ... calculating RR from theory",self.logfile)
        # valid for smax < Lbox/2, else see Deserno 04
        RR = 2*np.pi/3*(self.sbin[1:]**3-self.sbin[:-1]**3)/self.Lbox**3
        RR *= self.dmu if self.aniso else 2.0
        return RR
    #############################################################


    #############################################################
    def auto_CF(self,pos_data1,pos_data2=None,mark_data1=None,mark_data2=None):
        """ Auto/cross-correlation of data points.
            -- pos_dataj : tracer positions shape (ndata_j,3); pos_data2 = None gives auto, else cross CF 
            -- mark_dataj: tracer marks, None (default) or float shape (ndata_j,) 

            *** ASSUMES -Lbox/2 <= pos_dataj[] < Lbox/2 ***

            Applies recursive binary split for large data sets.
            Returns Peebles-Hauser estimator DD/RR - 1 
            of shape (self.L_Max,self.n_s) if self.aniso==True else (self.n_s,).
        """
        if pos_data1.shape[1] != 3:
            raise TypeError("Incompatible data shape. Expected (ndata,3).")

        DD = self.DD_split(pos_data1,pos_data2=pos_data2,mark_data1=mark_data1,mark_data2=mark_data2)
        # shape (s,mu) if self.aniso==True else (s,)

        RR = self.RR_theory()
        # shape (s,)
        
        cf_s = (DD.T/(RR + self.TINY)).T - 1.0    
        # shape (s,mu) if self.aniso==True else (s,)

        if self.aniso:
            if self.verbose:
                self.print_this("... ... computing multipoles",self.logfile)
            cf = np.zeros((self.L_Max,self.n_s),dtype=float)
            for L in range(self.L_Max):
                ell = 2*L
                cf[L] = 0.5*(2*ell+1)*np.trapezoid(self.Pell[L]*cf_s,dx=self.dmu)
        else:
            cf = 1.0*cf_s

        if self.verbose:
            self.print_this("... ... done",self.logfile)
        
        return cf
    #############################################################


    #############################################################
    def DD_only(self,pos_data1,pos_data2=None,mark_data1=None,mark_data2=None):
        """ DD (or D1D2) calculation.
            Assumes pos_data1.shape() = (ndata,3).
            If pos_data2 is not None, assumes pos_data2.shape() = (ndata2,3).
            Returns DD or D1D2.
        """
        ndata1 = pos_data1.shape[0]
        if pos_data2 is None:
            cf = 1.0*self.pair_counts(pos_data1,pos_data1,mark_1=mark_data1,mark_2=mark_data1)/ndata1/(ndata1-1)
        else:
            ndata2 = pos_data2.shape[0]
            cf = 1.0*self.pair_counts(pos_data1,pos_data2,mark_1=mark_data1,mark_2=mark_data2)/ndata1/ndata2

        return cf
    #############################################################


    #############################################################
    def DD_split(self,pos_data1,pos_data2=None,mark_data1=None,mark_data2=None):
        """ DD calculation wrapper.
            Assumes pos_data1.shape() = (ndata,3).
            If pos_data2 is not None, assumes pos_data2.shape() = (ndata2,3).
            Returns DD (or D1D2), applying binary recursive split for large data sets.
        """
        MARKED = False if mark_data1 is None else True
        ndata = pos_data1.shape[0]
        ndata_by_2 = ndata//2
        if ndata < self.N_SPLIT:
            if self.verbose:
                print_string = '... ... calculating '
                if MARKED:
                    print_string += 'marked '
                print_string += "DD" if pos_data2 is None else "D1D2"
                self.print_this(print_string,self.logfile)
            cf = self.DD_only(pos_data1,pos_data2=pos_data2,mark_data1=mark_data1,mark_data2=mark_data2)
        else:
            if self.verbose:
                self.print_this("... ... binary split",self.logfile)
            if pos_data2 is None:
                pd1s = pos_data1[:ndata_by_2].copy()
                ndata1s = pd1s.shape[0]
                pd1spr = pos_data1[ndata_by_2:].copy()
                ndata1spr = pd1spr.shape[0]
                if MARKED:
                    md1s = mark_data1[:ndata_by_2].copy()
                    md1spr = mark_data1[ndata_by_2:].copy()                    
                    D1D1 = self.DD_split(pd1s,mark_data1=md1s)
                    D1D1pr = self.DD_split(pd1s,pos_data2=pd1spr,mark_data1=md1s,mark_data2=md1spr)
                    D1prD1pr = self.DD_split(pd1spr,mark_data1=md1spr)
                else:
                    D1D1 = self.DD_split(pd1s)
                    D1D1pr = self.DD_split(pd1s,pos_data2=pd1spr)
                    D1prD1pr = self.DD_split(pd1spr)
                cf = ndata1s*(ndata1s-1)*D1D1 + 2*ndata1s*ndata1spr*D1D1pr + ndata1spr*(ndata1spr-1)*D1prD1pr
                cf = 1.0*cf/ndata/(ndata-1)
                del pd1s,pd1spr
                if MARKED:
                    del md1s,md1spr
                gc.collect()
            else:
                ndata2 = pos_data2.shape[0]
                ndata2_by_2 = ndata//2
                pd1s = pos_data1[:ndata_by_2].copy()
                ndata1s = pd1s.shape[0]
                pd1spr = pos_data1[ndata_by_2:].copy()
                ndata1spr = pd1spr.shape[0]
                pd2s = pos_data2[:ndata2_by_2].copy()
                ndata2s = pd2s.shape[0]
                pd2spr = pos_data2[ndata2_by_2:].copy()
                ndata2spr = pd2spr.shape[0]
                if MARKED:
                    if mark_data2 is None:
                        raise TypeError('Need marks for second data set')
                    md1s = mark_data1[:ndata_by_2].copy()
                    md1spr = mark_data1[ndata_by_2:].copy()
                    md2s = mark_data2[:ndata2_by_2].copy()
                    md2spr = mark_data2[ndata2_by_2:].copy()
                    D1D2 = self.DD_split(pd1s,pos_data2=pd2s,mark_data1=md1s,mark_data2=md2s)
                    D1D2pr = self.DD_split(pd1s,pos_data2=pd2spr,mark_data1=md1s,mark_data2=md2spr)
                    D1prD2 = self.DD_split(pd1spr,pos_data2=pd2s,mark_data1=md1spr,mark_data2=md2s)
                    D1prD2pr = self.DD_split(pd1spr,pos_data2=pd2spr,mark_data1=md1spr,mark_data2=md2spr)
                else:
                    D1D2 = self.DD_split(pd1s,pos_data2=pd2s)
                    D1D2pr = self.DD_split(pd1s,pos_data2=pd2spr)
                    D1prD2 = self.DD_split(pd1spr,pos_data2=pd2s)
                    D1prD2pr = self.DD_split(pd1spr,pos_data2=pd2spr)
                    
                cf = ndata1s*ndata2s*D1D2 + ndata1s*ndata2spr*D1D2pr 
                cf = cf + ndata1spr*ndata2s*D1prD2 + ndata1spr*ndata2spr*D1prD2pr
                cf = 1.0*cf/ndata/ndata2
                del pd1s,pd1spr,pd2s,pd2spr
                if MARKED:
                    del md1s,md1spr,md2s,md2spr
                gc.collect()

        return cf
    #############################################################
#################################################################


################################################################
# Power spectrum and various (Fourier) fields from particle data
################################################################
class PowerSpectrum(Utilities,Paths,Constants):
    """ Power spectrum and various (Fourier) fields from particle data in cubic periodic box. """    
    ###############################################
    def __init__(self,grid=256,Lbox=200.0,lgbin=True,nbin=30,KSEP=3,NFAC=10,logfile=None,verbose=True):
        """ Initialise the following:
            -- grid: resolution at which to compute density in case of particles
            -- Lbox: box size along one dimension
            -- lgbin: use logarithmic (True, default) or linear+log (False) binning
            -- nbin: number of bins in k
            -- KSEP: dynamic range separator for linear+log binning.
            -- NFAC: factor to separate number of bins for linear+log binning.
            Additionally defines the variables/arrays:
            cell_size,Delta_k,kmin,kmax,kbin,ktab,(dlnk or dk)
            Methods: 
            density_field,fourier_transform_density,
            tidal_tensor_field,density_hessian_field,Pk_grid            
            """
        Paths.__init__(self)
        Utilities.__init__(self)
        Constants.__init__(self)
        self.lib_cic = cdll.LoadLibrary(self.python_path+'cic_update.so')

        self.lgbin = lgbin
        self.nbin = nbin
        self.Lbox = Lbox
        self.grid = grid
        self.logfile = logfile
        self.verbose = verbose
        self.dynran = np.log2(self.grid/2) # dynamic range from fundamental mode to Nyquist frequency
        self.NFAC = NFAC
        self.KSEP = KSEP        

        if self.verbose:
            self.print_this('Setting up k-space...',self.logfile)
            
        self.cell_size = self.Lbox/self.grid
        self.Delta_k = 2*np.pi/self.Lbox
        self.grid_inv = 2*np.pi/self.grid
        
        self.kmin = 1.0*self.Delta_k
        self.kNy = np.pi/self.cell_size # don't change this!
        if self.lgbin:
            if self.verbose:
                self.print_this('... using log binning',self.logfile)
            self.dlnk = np.log(self.kNy/self.kmin)/(self.nbin)
            self.kbin = self.kmin*np.exp(self.dlnk*np.arange(self.nbin+1,dtype=float))
            self.ktab = np.sqrt(self.kbin[1:]*self.kbin[:-1])
        else:
            if self.dynran > self.KSEP: 
                self.KSEP = int(self.dynran/2) 
            if self.KSEP > 0:
                if self.verbose:
                    self.print_this('... using linear+log binning',self.logfile)
                self.dk = (self.kmin*(2**self.KSEP - 1))/(self.nbin/self.NFAC)
                self.kbin1 = self.kmin + self.dk*np.arange(self.nbin/self.NFAC+1,dtype=float)
                self.ktab1 = 0.5*(self.kbin1[1:]+self.kbin1[:-1])
                self.dlnk = np.log(self.kNy/(self.kmin*2**self.KSEP))/(self.nbin-self.ktab1.size)
                self.kbin2 = self.kbin1[-1]*np.exp(self.dlnk*np.arange(self.nbin-self.ktab1.size+1,dtype=float))
                self.ktab2 = np.sqrt(self.kbin2[1:]*self.kbin2[:-1])
                self.kbin = np.concatenate((self.kbin1,np.delete(self.kbin2,0)))
                self.ktab = np.concatenate((self.ktab1,self.ktab2))
            else:
                if self.verbose:
                    self.print_this('... not enough dynamic range: using linear bins throughout',self.logfile)
                self.dk = (self.kNy-self.kmin)/(self.nbin)
                self.kbin = self.kmin + self.dk*np.arange(self.nbin+1,dtype=float)
                self.ktab = 0.5*(self.kbin[1:]+self.kbin[:-1])

        self.k2_compare = (self.kbin/self.Delta_k)**2

        if self.verbose:
            self.print_this('... creating mesh',self.logfile)

        self.krange = np.arange(self.grid,dtype=int)
        self.K1,self.K2,self.K3 = np.meshgrid(self.krange,self.krange,self.krange,
                                              sparse=True,indexing='ij')

        # map upper half axes to negative values
        self.K1[self.K1 >= self.grid/2] = self.K1[self.K1 >= self.grid/2] - self.grid
        self.K2[self.K2 >= self.grid/2] = self.K2[self.K2 >= self.grid/2] - self.grid
        self.K3[self.K3 >= self.grid/2] = self.K3[self.K3 >= self.grid/2] - self.grid

        # Correction for windows
        # NGP : prod sinc(Lk_i/2Ng)
        #     = prod np.sinc(K[i](2pi/L)(L/Ng)/(2pi))
        #     = prod np.sinc(K[i]/Ng)
        # CIC : NGP^2
        self.NGP_corr = (np.sinc(1.0*self.K1/self.grid)
                         *np.sinc(1.0*self.K2/self.grid)
                         *np.sinc(1.0*self.K3/self.grid))
        # self.CIC_corr = self.NGP_corr**2
        # shape (~grid,~grid,~grid)

        self.KK = self.K1**2 + self.K2**2 + self.K3**2
        # shape (~grid,~grid,~grid)

        if self.verbose:
            self.print_this('... setting up index arrays',self.logfile)

        self.IND = [[]]
        for k in range(self.nbin-1):
            self.IND.append([])
            
        self.I_SIZE = np.zeros(self.nbin,dtype=int)
        for k in range(self.nbin):
            cond = ((self.KK >= self.k2_compare[k]) & (self.KK < self.k2_compare[k+1]))
            self.IND[k] = np.where(cond)
            self.I_SIZE[k] = self.IND[k][0].size
            del cond
            gc.collect()
        self.I_SIZE_MAX = np.max(self.I_SIZE)

        if self.verbose:
            self.print_this('... done with setup',self.logfile)
    ###############################################


    ###############################################
    def density_field(self,pos,contrast=True,interlace=False):
        """ Compute density field from particle locations using CIC.
            (See Kravtsov's notes at
             http://background.uchicago.edu/~whu/Courses/Ast321_11/pm.pdf
             for a nice exposition.)
            Assumes pos is numpy array with pos.shape = (3,ndata)
            Assumes position range along each dimension is (0,Lbox)
            Returns density (contrast) [shape (self.grid,self.grid,self.grid)]
        """

        if pos.shape[0] != 3:
            raise TypeError("Inconsistent data dimensions!! Try again.")

        if np.any(pos) < 0.0:
            raise ValueError("Positions should be in range (0,Lbox) along each dimension!!")

        ndata = 1*pos.shape[1]

        if self.verbose:
            self.print_this('Computing density field...',self.logfile)

        if self.verbose:
            self.print_this('... updating density',self.logfile)
        density = np.zeros(self.grid**3)
        # using np.empty would lead to 'density' being updated across multiple calls!

        c_update_density = self.lib_cic.update_density
        c_update_density.restype = None
        c_update_density.argtypes = [c_long,c_int,ndpointer(c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(c_double, flags="C_CONTIGUOUS"),
                                     c_double,c_double]

        c_update_density(c_long(ndata),c_int(self.grid),density,
                         pos[0].astype('float64'),
                         pos[1].astype('float64'),
                         pos[2].astype('float64'),
                         self.cell_size,c_double(0.0))

        density = np.reshape(density,(self.grid,self.grid,self.grid),order='F')

        if interlace:
            if self.verbose:
                self.print_this('... interlacing (NEEDS TESTING!!)',self.logfile)
            density2 = np.zeros(self.grid**3)
            c_update_density(c_long(ndata),c_int(self.grid),density2,
                             pos[0].astype('float64'),
                             pos[1].astype('float64'),
                             pos[2].astype('float64'),
                             self.cell_size,c_double(1.0))
            density2 = np.reshape(density2,(self.grid,self.grid,self.grid),order='F')

            density += density2            
            del density2
            gc.collect()
            
            density /= 2.0

        if contrast:
            density = density*self.grid**3/(1.0*ndata + self.TINY) - 1.0

        gc.collect()

        return density
    ###############################################


    ###############################################
    def coarsen(self,data3d,coarsen_by):
        """ Coarse-grain data on 3d grid by an integer. 
            Expects data3d of shape (NG,NG,NG) where NG is a multiple of coarsen_by,
            and integer coarsen_by.
            See https://stackoverflow.com/questions/25173979/aggregate-numpy-array-by-summing/25175460#25175460
            Returns coarse grained data of shape (NC,NC,NC) where NC = NG/coarsen_by.
        """
        if len(data3d.shape) != 3:
            raise TypeError("Wrong data shape detected in PowerSpectrum.coarsen()!")
        NG = data3d.shape[0]
        if (NG % coarsen_by) != 0:
            raise ValueError("Incompatible coarsening factor detected in PowerSpectrum.coarsen()!")
        NC = NG/coarsen_by
        data = np.reshape(data3d,(NC,coarsen_by,NC,coarsen_by,NC,coarsen_by))
        data = np.sum(data,axis=(1,3,5))/coarsen_by**3
        gc.collect()
        return data
    ###############################################


    ###############################################
    def fourier_transform_density(self,density,CIC=True):
        """ Apply Fourier transform to CIC density and correct for CIC window.
            Assumes density.shape = (self.grid,self.grid,self.grid); density is real.
            CIC controls whether or not to deconvolve kernel (default True).
            Returns Fourier-transformed density of same shape. 
        """

        FTdensity = fft.ifftn(density)
        if CIC:
            if self.verbose:
                self.print_this('... deconvolving CIC filter',self.logfile)
            FTdensity /= self.NGP_corr**2
        else:
            if self.verbose:
                self.print_this('... CIC filter will not be deconvolved',self.logfile)

        return FTdensity
    ###############################################


    ###############################################
    def tidal_tensor_field(self,input_array,R_smooth,input_is_density=False,input_is_FTdensity=False,CIC=True):
        """ Use FFT to compute 6 independent components psi_{ij} (1<=i<=j<=3) of tidal tensor 
            at each grid location. 

            Assumes density.shape = (self.grid,self.grid,self.grid); density is real.
            If input is positions, then density is computed using self.density_field

            R_smooth = Gaussian smoothing radius in units of Lbox.

            Returns psi_11,psi_22,psi_33,psi_12,psi_13,psi_23 [shape (self.grid,self.grid,self.grid)].
        """

        if not input_is_FTdensity:
            density = input_array if input_is_density else self.density_field(input_array)
            if self.verbose:
                self.print_this('... Fourier transforming density',self.logfile)
            FTdensity = self.fourier_transform_density(density,CIC=CIC)
        else:
            FTdensity = 1.0*input_array

        if self.verbose:
            self.print_this("... applying Gaussian smoothing with radius {0:.2e}*Lbox".format(R_smooth),self.logfile)
        # # note k = (2pi/L)K, so k^2 R^2 / 2 = 2 pi^2 K^2 (R/L)^2
        # kernel = np.exp(-2*np.pi**2*self.KK*R_smooth**2)
        # Oliver points out that it's better to sample in real space and FT that.
        # W(x) = exp(-x^2/(2R^2))/(2pi)^(3/2)/(R/Dx)^3#
        # R = R_s*Lbox; x^2 = KK*Dx^2 (reuse KK) : (-L/2..0..L/2)^2
        # (x/R)^2 = KK*(Dx/L)^2/R_s^2 = KK / GRID^2 / R_s^2
        kernel = fft.ifftn(np.exp(-0.5*self.KK/(self.grid*R_smooth)**2)/(2*np.pi)**1.5/(R_smooth)**3).real # ifftn has 1/GRID^3 in it
        kernel /= kernel[0,0,0]
        FTdensity *= kernel
        del kernel
        gc.collect()
        
        if self.verbose:
            self.print_this('... calculating tidal tensor',self.logfile)

        sinK1 = np.sin(self.grid_inv*self.K1)
        sinK2 = np.sin(self.grid_inv*self.K2)
        sinK3 = np.sin(self.grid_inv*self.K3)
        sinKK = sinK1**2 + sinK2**2 + sinK3**2
        # shape (~grid,~grid,~grid)
        
        if self.verbose:
            self.print_this('... ... psi11',self.logfile)
        psi11 = fft.fftn(sinK1**2/(sinKK + self.TINY)*FTdensity)
        if self.verbose:
            self.print_this('... ... psi22',self.logfile)
        psi22 = fft.fftn(sinK2**2/(sinKK + self.TINY)*FTdensity)
        if self.verbose:
            self.print_this('... ... psi33',self.logfile)
        psi33 = fft.fftn(sinK3**2/(sinKK + self.TINY)*FTdensity)
        if self.verbose:
            self.print_this('... ... psi12',self.logfile)
        psi12 = fft.fftn(sinK1*sinK2/(sinKK + self.TINY)*FTdensity)
        if self.verbose:
            self.print_this('... ... psi13',self.logfile)
        psi13 = fft.fftn(sinK1*sinK3/(sinKK + self.TINY)*FTdensity)
        if self.verbose:
            self.print_this('... ... psi23',self.logfile)
        psi23 = fft.fftn(sinK2*sinK3/(sinKK + self.TINY)*FTdensity)

        if self.verbose:
            self.print_this('... done',self.logfile)
        del FTdensity,sinK1,sinK2,sinK3,sinKK
        gc.collect()

        return psi11,psi22,psi33,psi12,psi13,psi23
    ###############################################


    ###############################################
    def density_hessian_field(self,input_array,R_smooth,input_is_density=False,input_is_FTdensity=False,CIC=True):
        """ Use FFT to compute 6 independent components H_{ij} (1<=i<=j<=3) of density Hessian
            at each grid location. 

            Assumes density.shape = (self.grid,self.grid,self.grid); density is real.
            If input is positions, then density is computed using self.density_field

            R_smooth = Gaussian smoothing radius in units of Lbox.

            Returns H_11,H_22,H_33,H_12,H_13,H_23 [shape (self.grid,self.grid,self.grid)].
        """

        if not input_is_FTdensity:
            density = input_array if input_is_density else self.density_field(input_array)
            if self.verbose:
                self.print_this('... Fourier transforming density',self.logfile)
            FTdensity = self.fourier_transform_density(density,CIC=CIC)
        else:
            FTdensity = input_array

        if self.verbose:
            self.print_this("... applying Gaussian smoothing with radius {0:.2e}*Lbox".format(R_smooth),self.logfile)
        # # note k = (2pi/L)K, so k^2 R^2 / 2 = 2 pi^2 K^2 (R/L)^2
        # kernel = np.exp(-2*np.pi**2*self.KK*R_smooth**2)
        # Oliver points out that it's better to sample in real space and FT that.
        # W(x) = exp(-x^2/(2R^2))/(2pi)^(3/2)/(R/Dx)^3#
        # R = R_s*Lbox; x^2 = KK*Dx^2 (reuse KK) : (-L/2..0..L/2)^2
        # (x/R)^2 = KK*(Dx/L)^2/R_s^2 = KK / GRID^2 / R_s^2
        kernel = fft.ifftn(np.exp(-0.5*self.KK/(self.grid*R_smooth)**2)/(2*np.pi)**1.5/(R_smooth)**3).real # ifftn has 1/GRID^3 in it
        kernel /= kernel[0,0,0]
        FTdensity *= kernel
        del kernel
        gc.collect()
        
        if self.verbose:
            self.print_this('... calculating density Hessian',self.logfile)

        sinK1 = np.sin(self.grid_inv*self.K1)
        sinK2 = np.sin(self.grid_inv*self.K2)
        sinK3 = np.sin(self.grid_inv*self.K3)
        # shape (~grid,~grid,~grid)
        
        if self.verbose:
            self.print_this('... ... H11',self.logfile)
        H11 = fft.fftn(-sinK1**2*FTdensity)
        if self.verbose:
            self.print_this('... ... H22',self.logfile)
        H22 = fft.fftn(-sinK2**2*FTdensity)
        if self.verbose:
            self.print_this('... ... H33',self.logfile)
        H33 = fft.fftn(-sinK3**2*FTdensity)
        if self.verbose:
            self.print_this('... ... H12',self.logfile)
        H12 = fft.fftn(-sinK1*sinK2*FTdensity)
        if self.verbose:
            self.print_this('... ... H13',self.logfile)
        H13 = fft.fftn(-sinK1*sinK3*FTdensity)
        if self.verbose:
            self.print_this('... ... H23',self.logfile)
        H23 = fft.fftn(-sinK2*sinK3*FTdensity)

        del sinK1,sinK2,sinK3
        gc.collect()

        return H11,H22,H33,H12,H13,H23
    ###############################################
        

    ###############################################
    def Pk_grid(self,input_array,input_array2=None,input_is_density=False,input_is_FTdensity=False,CIC=True):
        """ Use FFT to compute Fourier transform of density and its power spectrum. 

            Assumes density.shape = (self.grid,self.grid,self.grid); density is real.
            If input is positions, then density is computed using self.density_field

            Returns P(k) = |FTdensity|^2 or Re(FTdensity.FTdensity2*) on self.ktab.
        """

        cross = False if input_array2 is None else True

        # density = input_array if input_is_density else self.density_field(input_array)
        # if self.logfile==None: print '... Fourier transforming field 1'
        # else: writelog(self.logfile,'... Fourier transforming field 1\n')
        # FTdensity = self.fourier_transform_density(density)

        if not input_is_FTdensity:
            density = input_array if input_is_density else self.density_field(input_array)
            if self.verbose:
                self.print_this('... Fourier transforming density',self.logfile)
            FTdensity = self.fourier_transform_density(density,CIC=CIC)
        else:
            FTdensity = input_array

        density_k = np.zeros((self.nbin,self.I_SIZE_MAX),dtype=complex)
        for k in range(self.nbin):
            density_k[k,:self.I_SIZE[k]] = FTdensity[self.IND[k]]
        del FTdensity
        gc.collect()

        if cross:
            if not input_is_FTdensity:
                density2 = input_array2 if input_is_density else self.density_field(input_array2)
                if self.verbose:
                    self.print_this('... Fourier transforming density 2',self.logfile)
                FTdensity2 = np.conjugate(self.fourier_transform_density(density2,CIC=CIC))
            else:
                FTdensity2 = np.conjugate(input_array2)

            # density2 = input_array2 if input_is_density else self.density_field(input_array2)
            # if self.verbose:
            #     if self.logfile==None: print '... Fourier transforming field 2'
            #     else: writelog(self.logfile,'... Fourier transforming field 2\n')
            # FTdensity2 = np.conjugate(self.fourier_transform_density(density2))

            density2_k = np.zeros((self.nbin,self.I_SIZE_MAX),dtype=complex)
            for k in range(self.nbin):
                density2_k[k,:self.I_SIZE[k]] = FTdensity2[self.IND[k]]
            del FTdensity2
            gc.collect()

        if self.verbose:
            self.print_this('... calculating P(k)',self.logfile)
        Pk = np.zeros(self.nbin,dtype=float)

        Pk = (np.sum(np.absolute(density_k)**2,axis=1)
              if not cross else
              np.sum(np.real(density_k*density2_k),axis=1))

        Pk *= self.Lbox**3/(self.I_SIZE + self.TINY)

        del density_k
        if cross:
            del density2_k
        gc.collect()
        
        return Pk
    ###############################################


    ############################################################
    def halobyhalo_b1(self,input_array,hpos,kmax=0.1,ret_Pk=False,input_is_density=False,input_is_FTdensity=False,CIC=True):
        """ Halo-by-halo linear bias. Given input_array containing either
             matter density estimate of shape (grid,grid,grid) [input_is_density=True] 
             or dark matter positions of shape (3,Npart) [input_is_density=False] 
             and halo positions hpos of shape (Ntrc,3), estimates linear bias b1 for each halo.
             kmax: largest k value (h/Mpc) to use in calculating b1 (default 0.1 h/Mpc)
             Returns 1. b1_trc: 1 array of shape (Ntrc,)
                     2. b1_k,ktab[krange]: 2 arrays of shape (krange.size,)
                     3. (optionally) Pk_matter,ktab: 2 arrays of shape (ktab.size,)
        """

        if self.verbose:
            self.print_this("... Halo-by-halo linear bias",self.logfile)

        if len(hpos.shape) != 2:
            raise TypeError("Incompatible shape for position data. Need (Ntrc,3), detected (" 
                            + ','.join(['%d' % (i,) for i in hpos.shape]) +').')

        if hpos.shape[1] != 3:
            dim = 'dimension'
            if hpos.shape[1] > 1:
                dim += 's'
            raise TypeError("Only 3-d data sets supported. Detected {0:d} ".format(hpos.shape[1]) + dim + '.')

        if input_is_density | input_is_FTdensity:
            if len(input_array.shape) != 3:
                raise TypeError("Incompatible shape for input_array when input_is_density=True."
                                +" Need (grid,grid,grid), detected (" 
                                + ','.join(['%d' % (i,) for i in input_array.shape]) +').')
            if input_array.shape[0] != self.grid:
                raise TypeError("Incompatible shape for input_array when input_is_density=True."
                                +" Need ({0:d},{0:d},{0:d}), detected (".format(self.grid) 
                                + ','.join(['%d' % (i,) for i in input_array.shape]) +').')
            if self.verbose:
                self.print_this("... ... using supplied density field",self.logfile)
            if input_is_FTdensity:
                delta_matter = input_array.copy()
            else:
                delta_grid = input_array.copy()
        else:
            if (len(input_array.shape) != 2) | (input_array.shape[0]!=3):
                raise TypeError("Incompatible shape for input_array when input_is_density=True."
                                +" Need (Npart,3), detected (" 
                                + ','.join(['%d' % (i,) for i in input_array.shape]) +').')
            if self.verbose:
                self.print_this("... ... calculating CIC density field",self.logfile)
            delta_grid = self.density_field(input_array)

        KMAX = np.min([np.where(self.ktab <= kmax)[0][-1],self.nbin/2])

        Ntrc = hpos.shape[0]

        hpos_grid = hpos*self.grid/self.Lbox
        hgrid = np.fmod(hpos_grid,self.grid).astype('int')
        # shape(Ntrc,3)
        # hgrid contains integer vectors identifying the cell (floor) corresponding to each halo: used by CIC

        if not input_is_FTdensity:
            if self.verbose:
                self.print_this("... ... Fourier transforming matter density",self.logfile)
            delta_matter = self.fourier_transform_density(delta_grid,CIC=CIC)
            del delta_grid

        Pk_matter = self.Pk_grid(delta_matter,input_is_FTdensity=1,CIC=CIC)
            
        cond_posPk = (Pk_matter > 0.0)
        ind_posPk = np.where(cond_posPk)[0]
        if ind_posPk.size:
            KMIN = ind_posPk[0]
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
            cond = ((self.KK >= self.k2_compare[krange[k]]) 
                    & (self.KK < self.k2_compare[krange[k]+1]))
            ind[k] = np.where(cond) # indices of cells contributing to bin
            i_size[k] = ind[k][0].size # number of cells contributing to bin
            k1_array[k] = self.K1[ind[k][0],:,:].flatten()*self.grid_inv # 2pi/self.grid is for later convenience
            k2_array[k] = self.K2[:,ind[k][1],:].flatten()*self.grid_inv
            k3_array[k] = self.K3[:,:,ind[k][2]].flatten()*self.grid_inv
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
        cp1 = (hgrid + 1) % self.grid

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

        if self.verbose:
            self.print_this("... ... reporting mode-weighted sum of b1(k)",self.logfile)
        wts_k = 1.0*nmodes_k*(Pk_matter[krange]+self.TINY) # see notes for multiplication by P_mm
        wts_k = wts_k/np.sum(wts_k)
        b1 = np.sum(wts_k*bias_k,axis=1)
            
        bias_k = np.mean(bias_k,axis=0)

        return ((b1,bias_k,self.ktab[krange]) 
                if not ret_Pk else 
                (b1,bias_k,self.ktab[krange],Pk_matter,self.ktab))

    ############################################################
