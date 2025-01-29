#!/usr/bin/python

import paths 
from constants import TINY

import sys
sys.path.append(paths.python_path)
import numpy as ny
import scipy.spatial as syspat
import scipy.fftpack as fft
from utilities import quit_,writelog
from multiprocessing import Pool
import gc

##//////////////////////////////////////////////////////////////////////
## Various routines for correlation function measurements
##//////////////////////////////////////////////////////////////////////


from numpy.ctypeslib import ndpointer
from ctypes import *
lib_cic = cdll.LoadLibrary(paths.python_path+'cic_update.so')

class PowerSpectrum:
    """ Power spectrum of particles/density field in cubic box. """
    
    def __init__(self,dim=3,grid=256,Lbox=200.0,lgbin=1,nbin=30,KSEP=3,NFAC=10,logfile=None,verbose=True):
        """ Initialise the following:
            -- dim: number of dimensions (default 3)
            -- grid: resolution at which to compute density in case of particles
            -- Lbox: box size along one dimension
            -- lgbin: use logarithmic (1) or linear+log (0) binning
            -- nbin: number of bins in k
            -- KSEP: dynamic range separator for linear+log binning.
            -- NFAC: factor to separate number of bins for linear+log binning.
            Additionally defines the variables/arrays:
            cell_size,Delta_k,kmin,kmax,kbin,ktab,(dlnk or dk)
            Methods: 
            density_field,fourier_transform_density,
            tidal_tensor_field,density_hessian_field,density_field_slow,Pk_grid            
            """
        self.dim = dim
        self.lgbin = lgbin
        self.nbin = nbin
        self.Lbox = Lbox
        self.grid = grid
        self.logfile = logfile
        self.verbose = verbose
        self.dynran = ny.log2(self.grid/2) # dynamic range from fundamental mode to Nyquist frequency
        self.NFAC = NFAC
        self.KSEP = KSEP        

        if self.verbose:
            if self.logfile==None: print 'Setting up k-space...\n'
            else: writelog(self.logfile,'Setting up k-space...\n')

        if self.dim != 3:
            quit_("Only 3-d implemented so far. Sorry!")
            
        self.cell_size = self.Lbox/self.grid
        self.Delta_k = 2*ny.pi/self.Lbox

        self.kmin = 1.0*self.Delta_k
        self.kNy = ny.pi/self.cell_size # don't change this!
        if self.lgbin:
            if self.verbose:
                if self.logfile==None: print '... using log binning\n'
                else: writelog(self.logfile,'... using log binning\n')
            self.dlnk = ny.log(self.kNy/self.kmin)/(self.nbin)
            self.kbin = self.kmin*ny.exp(self.dlnk*ny.arange(self.nbin+1,dtype=float))
            self.ktab = ny.sqrt(self.kbin[1:]*self.kbin[:-1])
        else:
            if self.dynran > self.KSEP: 
                self.KSEP = int(self.dynran/2) 
            if self.KSEP > 0:
                if self.verbose:
                    if self.logfile==None: print '... using linear+log binning\n'
                    else: writelog(self.logfile,'... using linear+log binning\n')
                self.dk = (self.kmin*(2**self.KSEP - 1))/(self.nbin/self.NFAC)
                self.kbin1 = self.kmin + self.dk*ny.arange(self.nbin/self.NFAC+1,dtype=float)
                self.ktab1 = 0.5*(self.kbin1[1:]+self.kbin1[:-1])
                self.dlnk = ny.log(self.kNy/(self.kmin*2**self.KSEP))/(self.nbin-self.ktab1.size)
                self.kbin2 = self.kbin1[-1]*ny.exp(self.dlnk*ny.arange(self.nbin-self.ktab1.size+1,dtype=float))
                self.ktab2 = ny.sqrt(self.kbin2[1:]*self.kbin2[:-1])
                self.kbin = ny.concatenate((self.kbin1,ny.delete(self.kbin2,0)))
                self.ktab = ny.concatenate((self.ktab1,self.ktab2))
            else:
                if self.verbose:
                    if self.logfile==None: print '... not enough dynamic range: using linear bins throughout\n'
                    else: writelog(self.logfile,'... not enough dynamic range: using linear bins throughout\n')
                self.dk = (self.kNy-self.kmin)/(self.nbin)
                self.kbin = self.kmin + self.dk*ny.arange(self.nbin+1,dtype=float)
                self.ktab = 0.5*(self.kbin[1:]+self.kbin[:-1])

        self.k2_compare = (self.kbin/self.Delta_k)**2

        if self.verbose:
            if self.logfile==None: print '... creating mesh\n'
            else: writelog(self.logfile,'... creating mesh\n')

        self.krange = ny.arange(self.grid,dtype=int)
        self.K1,self.K2,self.K3 = ny.meshgrid(self.krange,self.krange,self.krange,
                                              sparse=True,indexing='ij')

        # map upper half axes to negative values
        self.K1[self.K1 >= self.grid/2] = self.K1[self.K1 >= self.grid/2] - self.grid
        self.K2[self.K2 >= self.grid/2] = self.K2[self.K2 >= self.grid/2] - self.grid
        self.K3[self.K3 >= self.grid/2] = self.K3[self.K3 >= self.grid/2] - self.grid

        # Correction for windows
        # NGP : prod sinc(Lk_i/2Ng)
        #     = prod ny.sinc(K[i](2pi/L)(L/Ng)/(2pi))
        #     = prod ny.sinc(K[i]/Ng)
        # CIC : NGP^2
        self.NGP_corr = (ny.sinc(1.0*self.K1/self.grid)
                         *ny.sinc(1.0*self.K2/self.grid)
                         *ny.sinc(1.0*self.K3/self.grid))
        # self.CIC_corr = self.NGP_corr**2
        # shape (~grid,~grid,~grid)

        self.KK = self.K1**2 + self.K2**2 + self.K3**2
        # shape (~grid,~grid,~grid)

        if self.verbose:
            if self.logfile==None: print '... setting up index arrays\n'
            else: writelog(self.logfile,'... setting up index arrays\n')

        self.IND = [[]]
        for k in range(self.nbin-1):
            self.IND.append([])
            
        self.I_SIZE = ny.zeros(self.nbin,dtype=int)
        for k in range(self.nbin):
            cond = ((self.KK >= self.k2_compare[k]) & (self.KK < self.k2_compare[k+1]))
            self.IND[k] = ny.where(cond)
            self.I_SIZE[k] = self.IND[k][0].size
            del cond
            gc.collect()
        self.I_SIZE_MAX = ny.max(self.I_SIZE)

        if self.verbose:
            if self.logfile==None: print '... done with setup\n'
            else: writelog(self.logfile,'... done with setup\n')


    def density_field(self,pos,contrast=True,interlace=False):
        """ Compute density field from particle locations using CIC.
            (See Kravtsov's notes at
             http://background.uchicago.edu/~whu/Courses/Ast321_11/pm.pdf
             for a nice exposition.)
            Assumes pos is numpy array with pos.shape = (dim,ndata)
            Assumes position range along each dimension is (0,Lbox)
            Returns density (contrast) [shape (self.grid,self.grid,self.grid)]
        """

        if pos.shape[0] != self.dim:
            quit_("Inconsistent data dimensions!! Try again.")

        if ny.any(pos) < 0.0:
            quit_("Positions should be in range (0,Lbox) along each dimension!!")

        ndata = 1*pos.shape[1]

        if self.verbose:
            if self.logfile==None: print 'Computing density field...'
            else: writelog(self.logfile,'Computing density field...\n')

        if self.verbose:
            if self.logfile==None: print '... updating density'
            else: writelog(self.logfile,'... updating density\n')
        density = ny.zeros(self.grid**3)
        # using ny.empty would lead to 'density' being updated across multiple calls!

        c_update_density = lib_cic.update_density
        c_update_density.restype = None
        c_update_density.argtypes = [c_int,c_int,ndpointer(c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(c_double, flags="C_CONTIGUOUS"),
                                     ndpointer(c_double, flags="C_CONTIGUOUS"),
                                     c_double,c_double]

        c_update_density(c_int(ndata),c_int(self.grid),density,
                         pos[0].astype('float64'),
                         pos[1].astype('float64'),
                         pos[2].astype('float64'),
                         self.cell_size,c_double(0.0))

        density = ny.reshape(density,(self.grid,self.grid,self.grid),order='F')

        if interlace:
            if self.verbose:
                if self.logfile==None: print '... interlacing (NEEDS TESTING!!)'
                else: writelog(self.logfile,'... interlacing (NEEDS TESTING!!)\n')
            density2 = ny.zeros(self.grid**3)
            c_update_density(c_int(ndata),c_int(self.grid),density2,
                             pos[0].astype('float64'),
                             pos[1].astype('float64'),
                             pos[2].astype('float64'),
                             self.cell_size,c_double(1.0))
            density2 = ny.reshape(density2,(self.grid,self.grid,self.grid),order='F')

            density += density2            
            del density2
            gc.collect()
            
            density /= 2.0

        if contrast:
            density = density*self.grid**3/(1.0*ndata + TINY) - 1.0

        gc.collect()

        return density


    def coarsen(self,data3d,coarsen_by):
        """ Coarse-grain data on 3d grid by an integer. 
            Expects data3d of shape (NG,NG,NG) where NG is a multiple of coarsen_by,
            and integer coarsen_by.
            See https://stackoverflow.com/questions/25173979/aggregate-numpy-array-by-summing/25175460#25175460
            Returns coarse grained data of shape (NC,NC,NC) where NC = NG/coarsen_by.
        """
        if len(data3d.shape) != 3:
            quit_("Wrong data shape detected in PowerSpectrum.coarsen()!")
        NG = data3d.shape[0]
        if (NG % coarsen_by) != 0:
            quit_("Incompatible coarsening factor detected in PowerSpectrum.coarsen()!")
        NC = NG/coarsen_by
        data = ny.reshape(data3d,(NC,coarsen_by,NC,coarsen_by,NC,coarsen_by))
        data = ny.sum(data,axis=(1,3,5))/coarsen_by**3
        gc.collect()
        return data


    def fourier_transform_density(self,density,CIC=True):
        """ Apply Fourier transform to CIC density and correct for CIC window.
            Assumes density.shape = (self.grid,self.grid,self.grid); density is real.
            CIC controls whether or not to deconvolve kernel (default True).
            Returns Fourier-transformed density of same shape. 
        """

        FTdensity = fft.ifftn(density)
        if CIC:
            if self.verbose:
                if self.logfile==None: print '... ... deconvolving CIC filter'
                else: writelog(self.logfile,'... deconvolving CIC filter\n')
            FTdensity /= self.NGP_corr**2
        else:
            if self.verbose:
                if self.logfile==None: print '... ... CIC filter will not be deconvolved'
                else: writelog(self.logfile,'... CIC filter will not be deconvolved\n')

        return FTdensity


    def tidal_tensor_field(self,input_array,R_smooth,input_is_density=0,input_is_FTdensity=0,CIC=True):
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
                if self.logfile==None: 
                    print '... Fourier transforming density'
                else: 
                    writelog(self.logfile,'... Fourier transforming density\n')
            FTdensity = self.fourier_transform_density(density,CIC=CIC)
        else:
            FTdensity = 1.0*input_array

        if self.verbose:
            if self.logfile==None: print "... applying Gaussian smoothing with radius {0:.2e}*Lbox".format(R_smooth)
            else: writelog(self.logfile,"... applying Gaussian smoothing with radius {0:.2e}*Lbox\n".format(R_smooth))
        # # note k = (2pi/L)K, so k^2 R^2 / 2 = 2 pi^2 K^2 (R/L)^2
        # kernel = ny.exp(-2*ny.pi**2*self.KK*R_smooth**2)
        # Oliver points out that it's better to sample in real space and FT that.
        # W(x) = exp(-x^2/(2R^2))/(2pi)^(3/2)/(R/Dx)^3#
        # R = R_s*Lbox; x^2 = KK*Dx^2 (reuse KK) : (-L/2..0..L/2)^2
        # (x/R)^2 = KK*(Dx/L)^2/R_s^2 = KK / GRID^2 / R_s^2
        kernel = fft.ifftn(ny.exp(-0.5*self.KK/(self.grid*R_smooth)**2)/(2*ny.pi)**1.5/(R_smooth)**3).real # ifftn has 1/GRID^3 in it
        kernel /= kernel[0,0,0]
        FTdensity *= kernel
        del kernel
        gc.collect()
        
        if self.verbose:
            if self.logfile==None: print '... calculating tidal tensor'
            else: writelog(self.logfile,'... calculating tidal tensor\n')

        sinK1 = ny.sin(2*ny.pi*self.K1/self.grid)
        sinK2 = ny.sin(2*ny.pi*self.K2/self.grid)
        sinK3 = ny.sin(2*ny.pi*self.K3/self.grid)
        sinKK = sinK1**2 + sinK2**2 + sinK3**2
        # shape (~grid,~grid,~grid)
        
        if self.verbose:
            if self.logfile==None: print '... ... psi11'
            else: writelog(self.logfile,'... ... psi11\n')
        psi11 = fft.fftn(sinK1**2/(sinKK + TINY)*FTdensity)
        if self.verbose:
            if self.logfile==None: print '... ... psi22'
            else: writelog(self.logfile,'... ... psi22\n')
        psi22 = fft.fftn(sinK2**2/(sinKK + TINY)*FTdensity)
        if self.verbose:
            if self.logfile==None: print '... ... psi33'
            else: writelog(self.logfile,'... ... psi33\n')
        psi33 = fft.fftn(sinK3**2/(sinKK + TINY)*FTdensity)
        gc.collect()
        if self.verbose:
            if self.logfile==None: print '... ... psi12'
            else: writelog(self.logfile,'... ... psi12\n')
        psi12 = fft.fftn(sinK1*sinK2/(sinKK + TINY)*FTdensity)
        if self.verbose:
            if self.logfile==None: print '... ... psi13'
            else: writelog(self.logfile,'... ... psi13\n')
        psi13 = fft.fftn(sinK1*sinK3/(sinKK + TINY)*FTdensity)
        if self.verbose:
            if self.logfile==None: print '... ... psi23'
            else: writelog(self.logfile,'... ... psi23\n')
        psi23 = fft.fftn(sinK2*sinK3/(sinKK + TINY)*FTdensity)

        if self.verbose:
            if self.logfile==None: print '... done'
            else: writelog(self.logfile,'... done\n')
        del FTdensity,sinK1,sinK2,sinK3,sinKK
        gc.collect()

        return psi11,psi22,psi33,psi12,psi13,psi23


    def density_hessian_field(self,input_array,R_smooth,input_is_density=0,input_is_FTdensity=0,CIC=True):
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
                if self.logfile==None: 
                    print '... Fourier transforming density'
                else: 
                    writelog(self.logfile,'... Fourier transforming density\n')
            FTdensity = self.fourier_transform_density(density,CIC=CIC)
        else:
            FTdensity = input_array

        if self.verbose:
            if self.logfile==None: print "... applying Gaussian smoothing with radius {0:.2e}*Lbox".format(R_smooth)
            else: writelog(self.logfile,"... applying Gaussian smoothing with radius {0:.2e}*Lbox\n".format(R_smooth))
        # # note k = (2pi/L)K, so k^2 R^2 / 2 = 2 pi^2 K^2 (R/L)^2
        # kernel = ny.exp(-2*ny.pi**2*self.KK*R_smooth**2)
        # Oliver points out that it's better to sample in real space and FT that.
        # W(x) = exp(-x^2/(2R^2))/(2pi)^(3/2)/(R/Dx)^3#
        # R = R_s*Lbox; x^2 = KK*Dx^2 (reuse KK) : (-L/2..0..L/2)^2
        # (x/R)^2 = KK*(Dx/L)^2/R_s^2 = KK / GRID^2 / R_s^2
        kernel = fft.ifftn(ny.exp(-0.5*self.KK/(self.grid*R_smooth)**2)/(2*ny.pi)**1.5/(R_smooth)**3).real # ifftn has 1/GRID^3 in it
        kernel /= kernel[0,0,0]
        FTdensity *= kernel
        del kernel
        gc.collect()
        
        if self.verbose:
            if self.logfile==None: print '... calculating density Hessian'
            else: writelog(self.logfile,'... calculating density Hessian\n')

        sinK1 = ny.sin(2*ny.pi*self.K1/self.grid)
        sinK2 = ny.sin(2*ny.pi*self.K2/self.grid)
        sinK3 = ny.sin(2*ny.pi*self.K3/self.grid)
        # shape (~grid,~grid,~grid)
        
        if self.verbose:
            if self.logfile==None: print '... ... H11'
            else: writelog(self.logfile,'... ... H11\n')
        H11 = fft.fftn(-sinK1**2*FTdensity)
        if self.verbose:
            if self.logfile==None: print '... ... H22'
            else: writelog(self.logfile,'... ... H22\n')
        H22 = fft.fftn(-sinK2**2*FTdensity)
        if self.verbose:
            if self.logfile==None: print '... ... H33'
            else: writelog(self.logfile,'... ... H33\n')
        H33 = fft.fftn(-sinK3**2*FTdensity)
        if self.verbose:
            if self.logfile==None: print '... ... H12'
            else: writelog(self.logfile,'... ... H12\n')
        H12 = fft.fftn(-sinK1*sinK2*FTdensity)
        if self.verbose:
            if self.logfile==None: print '... ... H13'
            else: writelog(self.logfile,'... ... H13\n')
        H13 = fft.fftn(-sinK1*sinK3*FTdensity)
        if self.verbose:
            if self.logfile==None: print '... ... H23'
            else: writelog(self.logfile,'... ... H23\n')
        H23 = fft.fftn(-sinK2*sinK3*FTdensity)

        del sinK1,sinK2,sinK3
        gc.collect()

        return H11,H22,H33,H12,H13,H23
        

    def Pk_grid(self,input_array,input_array2=None,input_is_density=0,input_is_FTdensity=0,CIC=True):
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
                if self.logfile==None: 
                    print '... Fourier transforming density'
                else: 
                    writelog(self.logfile,'... Fourier transforming density\n')
            FTdensity = self.fourier_transform_density(density,CIC=CIC)
        else:
            FTdensity = input_array

        density_k = ny.zeros((self.nbin,self.I_SIZE_MAX),dtype=complex)
        for k in range(self.nbin):
            density_k[k,:self.I_SIZE[k]] = FTdensity[self.IND[k]]
        del FTdensity
        gc.collect()

        if cross:
            if not input_is_FTdensity:
                density2 = input_array2 if input_is_density else self.density_field(input_array2)
                if self.verbose:
                    if self.logfile==None: 
                        print '... Fourier transforming density 2'
                    else: 
                        writelog(self.logfile,'... Fourier transforming density 2\n')
                FTdensity2 = ny.conjugate(self.fourier_transform_density(density2,CIC=CIC))
            else:
                FTdensity2 = ny.conjugate(input_array2)

            # density2 = input_array2 if input_is_density else self.density_field(input_array2)
            # if self.verbose:
            #     if self.logfile==None: print '... Fourier transforming field 2'
            #     else: writelog(self.logfile,'... Fourier transforming field 2\n')
            # FTdensity2 = ny.conjugate(self.fourier_transform_density(density2))

            density2_k = ny.zeros((self.nbin,self.I_SIZE_MAX),dtype=complex)
            for k in range(self.nbin):
                density2_k[k,:self.I_SIZE[k]] = FTdensity2[self.IND[k]]
            del FTdensity2
            gc.collect()

        if self.verbose:
            if self.logfile==None: print '... calculating P(k)'
            else: writelog(self.logfile,'... calculating P(k)\n')
        Pk = ny.zeros(self.nbin,dtype=float)

        Pk = (ny.sum(ny.absolute(density_k)**2,axis=1)
              if not cross else
              ny.sum(ny.real(density_k*density2_k),axis=1))

        Pk *= self.Lbox**3/(self.I_SIZE + TINY)

        del density_k
        if cross:
            del density2_k
        gc.collect()
        
        return Pk



    # def Pk_grid(self,input_array,input_array2=None,input_is_density=0,input_is_FTdensity=0):
    #     """ Use FFT to compute Fourier transform of density and its power spectrum. 

    #         Assumes density.shape = (self.grid,self.grid,self.grid); density is real.
    #         If input is positions, then density is computed using self.density_field

    #         Returns P(k) = |FTdensity|^2 or Re(FTdensity.FTdensity2*) on self.ktab.
    #     """

    #     cross = False if input_array2 is None else True

    #     density = input_array if input_is_density else self.density_field(input_array)

    #     if self.logfile==None: print '... Fourier transforming field 1'
    #     else: writelog(self.logfile,'... Fourier transforming field 1\n')
    #     FTdensity = self.fourier_transform_density(density)
    #     density_k = ny.zeros((self.nbin,self.I_SIZE_MAX),dtype=complex)
    #     for k in range(self.nbin):
    #         density_k[k,:self.I_SIZE[k]] = FTdensity[self.IND[k]]
    #     del FTdensity
    #     gc.collect()

    #     if cross:
    #         density2 = input_array2 if input_is_density else self.density_field(input_array2)

    #         if self.logfile==None: print '... Fourier transforming field 2'
    #         else: writelog(self.logfile,'... Fourier transforming field 2\n')
    #         FTdensity2 = ny.conjugate(self.fourier_transform_density(density2))
    #         density2_k = ny.zeros((self.nbin,self.I_SIZE_MAX),dtype=complex)
    #         for k in range(self.nbin):
    #             density2_k[k,:self.I_SIZE[k]] = FTdensity2[self.IND[k]]
    #         del FTdensity2
    #         gc.collect()

    #     if self.logfile==None: print '... calculating P(k)'
    #     else: writelog(self.logfile,'... calculating P(k)\n')
    #     Pk = ny.zeros(self.nbin,dtype=float)

    #     Pk = (ny.sum(ny.absolute(density_k)**2,axis=1)
    #           if not cross else
    #           ny.sum(ny.real(density_k*density2_k),axis=1))

    #     Pk *= self.Lbox**3/(self.I_SIZE + TINY)

    #     del density_k
    #     if cross:
    #         del density2_k
    #     gc.collect()
        
    #     return Pk


    def density_field_slow(self,pos,contrast=True):
        """ Compute density field from particle locations using CIC.
            (See http://background.uchicago.edu/~whu/Courses/Ast321_11/pm.pdf
             for a nice exposition.)
            Assumes pos is numpy array with pos.shape = (dim,ndata)
            Assumes position range along each dimension is (0,Lbox)
            Returns density (contrast) [shape (self.grid,self.grid,self.grid)]
        """

        if pos.shape[0] != self.dim:
            quit_("Inconsistent data dimensions!! Try again.")

        if ny.any(pos) < 0.0:
            quit_("Positions should be in range (0,Lbox) along each dimension!!")

        if self.verbose:
            if self.logfile==None: print 'Computing density field...'
            else: writelog(self.logfile,'Computing density field...\n')

        ndata = 1*pos.shape[1]
        density = ny.zeros((self.grid,self.grid,self.grid),dtype=float)

        # Cell to which each particle belongs
        # shape (ndata,dim); type int
        cells = ny.floor(pos.T/self.cell_size).astype(int) % self.grid
        cp1 = (cells + 1) % self.grid

        cic_d = ny.zeros((self.dim,ndata),dtype=float)
        for d in range(self.dim):
            temp_pos = pos[d]/self.cell_size
            cic_d[d] = ny.fabs(temp_pos - ny.floor(temp_pos))# - 0.5) # incorrect subtraction pointed out by sujatha r. changed 15 Oct 2018
            del temp_pos
            gc.collect()

        cic_d = cic_d.T                
        cic_t = 1.0 - cic_d
        # shape (ndata,dim)

        if self.verbose:
            if self.logfile==None: print '... calculating cic weights'
            else: writelog(self.logfile,'... calculating cic weights\n')

        cic_ttt = cic_t[:,0]*cic_t[:,1]*cic_t[:,2]
        cic_dtt = cic_d[:,0]*cic_t[:,1]*cic_t[:,2]
        cic_tdt = cic_t[:,0]*cic_d[:,1]*cic_t[:,2]
        cic_ttd = cic_t[:,0]*cic_t[:,1]*cic_d[:,2]
        cic_ddt = cic_d[:,0]*cic_d[:,1]*cic_t[:,2]
        cic_dtd = cic_d[:,0]*cic_t[:,1]*cic_d[:,2]
        cic_tdd = cic_t[:,0]*cic_d[:,1]*cic_d[:,2]
        cic_ddd = cic_d[:,0]*cic_d[:,1]*cic_d[:,2]

        del cic_d,cic_t
        gc.collect()

        #######################################
        print '... looping over particles'
        for n in xrange(ndata):
            c0 = cells[n,0]
            c1 = cells[n,1]
            c2 = cells[n,2]
            cp10 = cp1[n,0]
            cp11 = cp1[n,1]
            cp12 = cp1[n,2]
            
            density[c0,c1,c2] += cic_ttt[n]
            density[cp10,c1,c2] += cic_dtt[n]
            density[c0,cp11,c2] += cic_tdt[n]
            density[c0,c1,cp12] += cic_ttd[n]
            density[cp10,cp11,c2] += cic_ddt[n]
            density[cp10,c1,cp12] += cic_dtd[n]
            density[c0,cp11,cp12] += cic_tdd[n]
            density[cp10,cp11,cp12] += cic_ddd[n]
            
            # status_bar(n,ndata,text='of particles done')
        #######################################

        if contrast:
            density = density*self.grid**3/(1.0*ndata + TINY) - 1.0

        del cic_ttt,cic_dtt,cic_tdt,cic_ttd,cic_ddt,cic_dtd,cic_tdd,cic_ddd
        del cells,cp1

        return density


    def Pk_grid_old(self,input_array,input_array2=None,input_is_density=0):
        """ Use FFT to compute Fourier transform of density and its power spectrum. 

            Assumes density.shape = (self.grid,self.grid,self.grid); density is real.
            If input is positions, then density is computed using self.density_field

            Returns P(k) = |FTdensity|^2 or Re(FTdensity.FTdensity2*) on self.ktab.
        """

        cross = False if input_array2 is None else True

        density = input_array if input_is_density else self.density_field(input_array)
        if self.verbose:
            if self.logfile==None: print '... Fourier transforming field 1'
            else: writelog(self.logfile,'... Fourier transforming field 1\n')
        FTdensity = self.fourier_transform_density(density)
        if cross:
            density2 = input_array2 if input_is_density else self.density_field(input_array2)
            if self.verbose:
                if self.logfile==None: print '... Fourier transforming field 2'
                else: writelog(self.logfile,'... Fourier transforming field 2\n')
            FTdensity2 = ny.conjugate(self.fourier_transform_density(density2))

        if self.verbose:
            if self.logfile==None: print '... calculating P(k)'
            else: writelog(self.logfile,'... calculating P(k)\n')
        Pk = ny.zeros(self.nbin,dtype=float)

        kmesh = []
        for n1 in range(self.grid/2+1):
            for n2 in range(n1,self.grid/2+1):
                for n3 in range(n2,self.grid/2+1):
                    if n1**2 + n2**2 + n3**2 <= (self.grid/2)**2:
                        kmesh.append((n1,n2,n3))
        kmesh = ny.array(kmesh)
        # shape (~grid,3)

        for b in range(self.nbin-1,-1,-1):
            cond = (ny.sum(kmesh**2,axis=1) <= self.k2_compare[b])
            ind = ny.where(~cond)[0]
            if ind.size:
                for p in range(3):
                    if not cross:
                        Pk[b] += ny.mean(ny.absolute(FTdensity[kmesh[ind,p%3],
                                                               kmesh[ind,(p+1)%3],
                                                               kmesh[ind,(p+2)%3]])**2)
                    else:
                        Pk[b] += ny.mean(ny.real(FTdensity[kmesh[ind,p%3],
                                                           kmesh[ind,(p+1)%3],
                                                           kmesh[ind,(p+2)%3]]
                                                 *FTdensity2[kmesh[ind,p%3],
                                                             kmesh[ind,(p+1)%3],
                                                             kmesh[ind,(p+2)%3]]))

                    if not cross:
                        Pk[b] += ny.mean(ny.absolute(FTdensity[kmesh[ind,(p+1)%3],
                                                               kmesh[ind,p%3],
                                                               kmesh[ind,(p+2)%3]])**2)
                    else:
                        Pk[b] += ny.mean(ny.real(FTdensity[kmesh[ind,(p+1)%3],
                                                           kmesh[ind,p%3],
                                                           kmesh[ind,(p+2)%3]]
                                                 *FTdensity2[kmesh[ind,(p+1)%3],
                                                             kmesh[ind,p%3],
                                                             kmesh[ind,(p+2)%3]]))

            kmesh = kmesh[cond]
            del cond,ind
            gc.collect()

        del FTdensity,kmesh
        if cross:
            del FTdensity2
        gc.collect()

        Pk *= (self.Lbox**3/6.0) # 6 to account for permutations above
        
        return Pk

#/////////////////////////////////////////


def unwrap_auto_CF_workhorse(*arg,**kwarg):
    return TwoPointCorrelationFunction.auto_CF_workhorse(*arg,**kwarg)

def unwrap_DD_pbc_workhorse(*arg,**kwarg):
    return TwoPointCorrelationFunction.DD_pbc_workhorse(*arg,**kwarg)


class TwoPointCorrelationFunction(object):
    """ 2-point (cross-)correlation functions in bins of separation. 
        Assumes cubic box throughout.
    """

    def __init__(self,lgbin=1,rmin=1e-2,rmax=3e1,nbin=16,ranwts=None,Lbox=1.0,nproc=4,
                 proj=0,pimax=100.0,los=2,ran_ratio=1.173,dp_RR=0.2,ran_min=12,verbose=False):
        """ Initialise the following:
            -- lgbin    : use logarithmic (1) or linear (0) binning
            -- rmin,rmax: min,max separations in r (or rp)
            -- nbin     : number of bins in r (or rp).
            -- ranwts   : if not None, expect (NG,NG,NG)-shaped array of probabilities
                          on a grid: weights for trimming random locations; 
                          useful for non-trivial geometries. 
            -- Lbox     : Box size in same units as rmin,rmax,pimax.
            -- proj     : if 1, calculate projected correlation function wp(rp), else xi(r)
            -- pimax    : max separation in pi (bins in pi will be rp-dependent)
            -- los      : line-of-sight direction, default z-axis
            -- ran_ratio: factor such that int(ndata*ran_ratio) = number of randoms
                                                                  per catalog
            -- dp_RR    : fractional tolerance for RR at smallest separation
            -- ran_min  : minimum number of randoms per galaxy
             
            Methods: 
            pair_counts,auto_CF_workhorse,auto_CF,auto_CF_parallel,cross_CF,auto_CF_std
            Convenience functions: 
            get_cond,face_pair_counts,side_pair_counts,peanohilbert_boundary.
            """
        self.lgbin = lgbin
        self.ran_ratio = ran_ratio
        self.dp_RR = dp_RR
        self.rmin = rmin
        self.rmax = rmax
        self.nbin = nbin
        self.verbose = verbose
        self.Lbox = Lbox
        self.RAN_MIN = ran_min
        self.NPROC = nproc

        self.ranwts = ranwts
        if self.ranwts is not None:
            self.meanranwts = ny.sum(self.ranwts[self.ranwts >= 0.0])/self.ranwts.shape[0]**3
        else:
            self.meanranwts = 1.0

        if self.lgbin==1:
            self.dlnr = ny.log(self.rmax/self.rmin)/(self.nbin-1)
            self.rbin = self.rmin*ny.exp(self.dlnr*ny.arange(self.nbin,dtype=float))
            self.rmid = ny.sqrt(self.rbin[1:]*self.rbin[:-1])
        else:
            self.dr = (self.rmax-self.rmin)/(self.nbin-1)
            self.rbin = self.rmin + self.dr*ny.arange(self.nbin,dtype=float)
            self.rmid = 0.5*(self.rbin[1:]+self.rbin[:-1])

        self.lnrbin = ny.log(self.rbin)

        self.proj = proj
        if self.proj:
            self.pimin = 0.0
            self.pimax = pimax
            self.npibin_lo = 16 # 16
            self.npibin_hi = 51 # 51
            self.npibin = self.npibin_lo + self.npibin_hi - 1

            self.pibin = ny.zeros((self.nbin-1,self.npibin),dtype=float)
            self.pimid_lo = ny.zeros((self.nbin-1,self.npibin_lo-1),dtype=float)
            self.dpi_lo = ny.zeros(self.nbin-1,dtype=float)
            self.pimid_hi = ny.zeros((self.nbin-1,self.npibin_hi-1),dtype=float)
            self.dpi_hi = ny.zeros(self.nbin-1,dtype=float)

            for rp in range(self.nbin-1):
                self.dpi_lo[rp] = (self.rbin[rp]-self.pimin)/(self.npibin_lo-1)
                self.pibin[rp][:self.npibin_lo] = self.pimin + self.dpi_lo[rp]*ny.arange(self.npibin_lo,dtype=float)
                self.pimid_lo[rp] = 0.5*(self.pibin[rp][1:self.npibin_lo]+self.pibin[rp][:self.npibin_lo-1])
                
                self.dpi_hi[rp] = (self.pimax-self.rbin[rp])/(self.npibin_hi-1)
                self.pibin[rp][self.npibin_lo-1:] = (self.rbin[rp] 
                                                     + self.dpi_hi[rp]*ny.arange(self.npibin_hi,dtype=float))
                self.pimid_hi[rp] = 0.5*(self.pibin[rp][self.npibin_lo:]+self.pibin[rp][self.npibin_lo-1:-1])

            self.los = los
            self.non_los = ny.where(ny.arange(3,dtype=int) != self.los)[0]


        self.depth = 2 # For creating subvolumes
        self.nsub = self.depth**3

        self.axis = ['x','y','z']

    def pair_counts(self,pos_1,pos_2,cond_1,cond_2):
        """ Unnormalised binned pair counts 
            between two data sets (can be the same data set).
            pos_j should have shape (3,ndata_j) and
            cond_j should be a boolean array of shape (ndata_j,), where j=1,2.
            Returns vector of length nbin-1 or (nbin-1,nPibin-1) or (1,), type int 
        """

        if ny.any(cond_1) & ny.any(cond_2):
            if not self.proj:
                cum_counts = ny.zeros(self.nbin,dtype=int)
                tree_1 = syspat.cKDTree(pos_1.T[cond_1])
                tree_2 = syspat.cKDTree(pos_2.T[cond_2])
                for r in range(self.nbin):
                    ind = tree_1.query_ball_tree(tree_2,self.rbin[r])
                    # ind is list of size len(data1)
                    # each element j is list of nbrs of data1[j]
                    nbr_counts = ny.asarray(map(len,ind)) 
                    # nbr_counts element j is number of neighbours of data1[j]
                    cum_counts[r] = ny.sum(nbr_counts)
                    # status_bar(r,self.nbin)
                    ############
                gc.collect()
                bin_counts = ny.diff(cum_counts)
            else:
                bin_counts = ny.zeros((self.nbin-1,self.npibin-1),dtype=int)                
                ########################
                pos_rp_1 = pos_1[self.non_los]
                pos_rp_2 = pos_2[self.non_los]
                tree_rp_1 = syspat.cKDTree(pos_rp_1.T[cond_1]) # tested changing leafsize: 
                tree_rp_2 = syspat.cKDTree(pos_rp_2.T[cond_2]) # no effect at all on computation time
                ndata_1 = tree_rp_1.n
                del pos_rp_1,pos_rp_2
                gc.collect()
                ########################
                pos_pi_1 = pos_1[self.los][cond_1]
                pos_pi_2 = pos_2[self.los][cond_2] 
                ########################
                ind_rp = ny.array(tree_rp_1.query_ball_tree(tree_rp_2,self.rbin[0]))
                rp_len = ny.array([len(ind_rp[j]) for j in range(ndata_1)])
                # ind_rp is array of size len(data1)
                # each element j is list of nbrs of data1[j] in data2
                for rp in range(1,self.nbin):
                    ########################
                    ind_rpDrp = ny.array(tree_rp_1.query_ball_tree(tree_rp_2,self.rbin[rp]))
                    rpDrp_len = ny.array([len(ind_rpDrp[j]) for j in range(ndata_1)])
                    # Outer code segment gives w(theta)
                    # bin_counts[rp-1,0] += ny.sum(rpDrp_len-rp_len)

                    cond_nonzero = (rpDrp_len > rp_len)
                    nonzero_ind = ny.where(cond_nonzero)[0]                    
                    for j in nonzero_ind:
                        ind_Drp = ny.setdiff1d(ind_rpDrp[j],ind_rp[j],assume_unique=True)
                        ########################
                        # ind_Drp is array of indices of data2 that are rp-nbrs of data1[j]
                        # Use these for pi measurement
                        ########################
                        if ind_Drp.size: # only these data1 indices are relevant
                            delta_pi = ny.fabs(pos_pi_2[ind_Drp] - pos_pi_1[j]) 
                            counts = ny.array([ny.where((delta_pi > self.pibin[rp-1,pi]) 
                                                        & (delta_pi <= self.pibin[rp-1,pi+1]))[0].size
                                               for pi in range(self.npibin-1)])
                            # which of these are also pi-nbrs.
                            bin_counts[rp-1] += counts
                            # We only care about number of (rp,pi)-nbrs, not identities
                        del ind_Drp
                        ########################                                            
                    del nonzero_ind

                    ind_rp = ind_rpDrp
                    rp_len = rpDrp_len
                    gc.collect()
                    ########################
               
                del tree_rp_1,tree_rp_2,ind_rp
                gc.collect()
            ###################
        else:
            bin_counts = 0

        return bin_counts 


    def auto_CF_workhorse(self,pos_data,ls=1,seed=None,dd=0):
        """ Memory efficient auto-correlation of data points. 
            Ignores periodicity.
            Subdivides full volume into 8 subvolumes. 
            Ignores purely vertex-sharing pairs of subvolumes which contribute negligibly.
            Time scales eventually like nran^1.5.
            Requires ~10x less memory than std calc of auto_CF_std(). 
            Faster than std calc for nran > 50k/200k in 3-d, system+rmax-dependent.
            Assumes pos_data.shape() = (3,ndata).
            Returns DD or [RR and/or DR].
        """

        boundary,ph_ind = self.peanohilbert_boundary()

        ndata = pos_data.shape[1]

        if dd:
            DD = (ny.zeros((self.nbin-1,self.npibin-1),dtype=float) 
                  if self.proj else 
                  ny.zeros(self.nbin-1,dtype=float))
        else:
            ny.random.seed(seed)
            nran = int(self.ran_ratio*ndata)            
            pos_ran = ny.random.rand(3,nran)
            if self.ranwts is not None:
                NGRID = self.ranwts.shape[0] # expecting ranwts.shape = (NGRID,NGRID,NGRID)
                ix,iy,iz = ((NGRID*pos_ran) % NGRID).astype(int)
                uni = ny.random.rand(nran)
                pos_ran = pos_ran.T[uni < self.ranwts[ix,iy,iz]].T
                nran = pos_ran.shape[1]
                del ix,iy,iz,uni
                gc.collect()
            pos_ran *= self.Lbox
    
            RR = (ny.zeros((self.nbin-1,self.npibin-1),dtype=float) 
                  if self.proj else 
                  ny.zeros(self.nbin-1,dtype=float))
            if ls:
                DR = 1.0*RR


        for s in range(self.nsub):
            gc.collect()
            bdry = boundary[s]
            cond_data = ((pos_data[0] > bdry['xmin']) & (pos_data[0] <= bdry['xmax']) &
                         (pos_data[1] > bdry['ymin']) & (pos_data[1] <= bdry['ymax']) & 
                         (pos_data[2] > bdry['zmin']) & (pos_data[2] <= bdry['zmax'])) 
            if dd:
                DD = DD + 1.0*self.pair_counts(pos_data,pos_data,cond_data,cond_data)
                del cond_data
            else:
                cond_ran = ((pos_ran[0] > bdry['xmin']) & (pos_ran[0] <= bdry['xmax']) &
                            (pos_ran[1] > bdry['ymin']) & (pos_ran[1] <= bdry['ymax']) & 
                            (pos_ran[2] > bdry['zmin']) & (pos_ran[2] <= bdry['zmax'])) 
                RR = RR + 1.0*self.pair_counts(pos_ran,pos_ran,cond_ran,cond_ran)
                if ls:
                    DR = DR + 0.5*(self.pair_counts(pos_data,pos_ran,cond_data,cond_ran) 
                                   + self.pair_counts(pos_ran,pos_data,cond_ran,cond_data))
                del cond_ran,cond_data

            if self.verbose:
                print "... {0:d} of {1:d} main subvolumes done".format(s+1,self.nsub)
            
        if self.verbose:
            print "... main subvolumes done"

        n_obvious = self.nsub-1
        for s in range(n_obvious):
            gc.collect()
            dph_ind = ph_ind[s+1]-ph_ind[s]
            ax_fc = ny.where(dph_ind != 0)[0][0] 
            ot_ax = ny.where(dph_ind == 0)[0]
            
            s_hi = s+1 if dph_ind[ax_fc]==1 else s
            s_lo = s if dph_ind[ax_fc]==1 else s+1
            
            bdry_hi = boundary[s_hi]
            bdry_lo = boundary[s_lo]

            if dd:
                pc_DD = self.face_pair_counts(pos_data,0.0,ax_fc,
                                              bdry_hi,bdry_lo,ls,dd)
                DD = DD + pc_DD
            else:
                pc_RR,pc_DR = self.face_pair_counts(pos_data,pos_ran,ax_fc,
                                                    bdry_hi,bdry_lo,ls,dd)
                RR = RR + pc_RR
                if ls:
                    DR = DR + pc_DR

            if self.verbose:
                print "... {0:d} of {1:d} obvious face-sharing pairs done".format(s+1,n_obvious)
            
        if self.verbose:
            print "... obvious face-sharing pairs done"

        # Following assumes depth=2
        s_hi_list = [7,6,5]
        s_lo_list = [0,1,2]
        for r in range(len(s_hi_list)):
            s_hi = s_hi_list[r]
            s_lo = s_lo_list[r]
            bdry_hi = boundary[s_hi]
            bdry_lo = boundary[s_lo]
                
            if dd:
                pc_DD = self.face_pair_counts(pos_data,0.0,0,
                                              bdry_hi,bdry_lo,ls,dd)
                DD = DD + pc_DD
            else:
                pc_RR,pc_DR = self.face_pair_counts(pos_data,pos_ran,0,
                                                    bdry_hi,bdry_lo,ls,dd)
                RR = RR + pc_RR
                if ls:
                    DR = DR + pc_DR
                
            if self.verbose:
                print "... {0:d} of {1:d} other face-sharing pairs done".format(r+1,5)
            #############################
        s_hi_list = [3,4]
        s_lo_list = [0,7]
        for r in range(len(s_hi_list)):
            s_hi = s_hi_list[r]
            s_lo = s_lo_list[r]
            bdry_hi = boundary[s_hi]
            bdry_lo = boundary[s_lo]
                
            if dd:
                pc_DD = self.face_pair_counts(pos_data,0.0,2,
                                              bdry_hi,bdry_lo,ls,dd)
                DD = DD + pc_DD
            else:
                pc_RR,pc_DR = self.face_pair_counts(pos_data,pos_ran,2,
                                                    bdry_hi,bdry_lo,ls,dd)
                RR = RR + pc_RR
                if ls:
                    DR = DR + pc_DR
                
            if self.verbose:
                print "... {0:d} of {1:d} other face-sharing pairs done".format(r+4,5)
            
        if self.verbose:
            print "... all other face-sharing pairs done"

        # Shared sides (only cross talk needed)
        kmax = 3
        for j in range(5):
        #######
            for k in range(kmax):
                s1,s2 = j,j+2*(k+1)
                bdry1 = boundary[s1]
                bdry2 = boundary[s2]
                ind1 = ph_ind[s1]
                ind2 = ph_ind[s2]
                if dd:
                    pc_DD = self.side_pair_counts(pos_data,0.0,
                                                  bdry1,ind1,bdry2,ind2,ls,dd)                    
                    DD = DD + pc_DD
                else:
                    pc_RR,pc_DR = self.side_pair_counts(pos_data,pos_ran,
                                                        bdry1,ind1,bdry2,ind2,ls,dd)
                    
                    RR = RR + pc_RR
                    if ls:
                        DR = DR + pc_DR
                #######
                if (j+1)%2 == 0:
                    kmax -= 1
            #############################
        if self.verbose:
            print "... all side-sharing pairs done"

        #############################
        # Note multiplication by Lbox**3. Matched by missing Lbox**3 in self.RR_theory()
        if dd:
            DD = DD/ndata/(ndata-1)*self.Lbox**3
            return DD
        else:
            RR = RR/nran/(nran-1)*self.Lbox**3
            DR = DR/ndata/nran*self.Lbox**3 if ls else 0.0
            return RR,DR


    def auto_CF(self,pos_data,ls=1,seed=None):
        """ Convenience function that calls auto_CF_workhorse()
            appropriately for 3-d or projected correlation functions.
            Ignores periodicity.
            Assumes pos_data.shape() = (3,ndata).
            Returns Landy-Szalay (ls=1) or Peebles-Hauser (ls=0) estimator.
        """
        # if self.proj:
        #     avg_RR_min = (2*ny.pi*(self.rmin/self.Lbox)**2*self.dlnr*(self.pimax-self.pimin)/self.Lbox
        #                   if self.lgbin else
        #                   2*ny.pi*(self.rmin/self.Lbox)*self.dr/self.Lbox*(self.pimax-self.pimin)/self.Lbox)
        # else:
        avg_RR_min = (4*ny.pi*(self.rmin/self.Lbox)**3*self.dlnr
                      if self.lgbin else
                      4*ny.pi*(self.rmin/self.Lbox)**2*self.dr/self.Lbox)
        N_ran = int(self.ran_ratio*pos_data.shape[1])
        if self.ranwts is not None:
            N_ran *= self.meanranwts
        avg_RR_min *= N_ran*(N_ran+1)
        N_ran_cat = ny.max([self.RAN_MIN,1+int((1.0/avg_RR_min + 4.0/N_ran)/self.dp_RR**2)]) 
        # to get (100*dp_RR)% error on smallest RR assuming Poisson stats.
        # min value 10 gives [var(RR)+4var(DR)]/var(DD) <~ 0.11
        print N_ran,N_ran_cat

        # !! CAREFUL WITH seed BELOW !!
        DD = self.auto_CF_workhorse(pos_data,ls=ls,dd=1)
        RR,DR = self.auto_CF_workhorse(pos_data,ls=ls,seed=seed,dd=0)
        for n in range(1,N_ran_cat):
            SEED = seed+n if seed is not None else seed
            temp_RR,temp_DR = self.auto_CF_workhorse(pos_data,ls=ls,seed=SEED,dd=0)
            RR += temp_RR
            if ls:
                DR += temp_DR
            # if self.verbose:
            # status_bar(n,N_ran_cat)
            
        RR = RR/(1.0*N_ran_cat)
        DR = DR/(1.0*N_ran_cat)

        cf = DD/(RR + TINY) - 1.0
        if ls:
            cf = cf - 2*(DR/(RR + TINY) - 1.0)

        if self.proj:
            cf = 2*(ny.trapz(cf[:,:self.npibin_lo-1],x=self.pimid_lo,axis=1)
                    + ny.trapz(cf[:,self.npibin_lo-1:],x=self.pimid_hi,axis=1)#)
                    + 0.5*((self.pimid_hi[:,0]-self.pimid_lo[:,-1])
                           *(cf[:,self.npibin_lo-2] + cf[:,self.npibin_lo-1])))

        return cf


    def auto_CF_parallel(self,pos_data,ls=1,seed=42):
        """ Parallelised convenience function that calls auto_CF_workhorse()
            appropriately for 3-d or projected correlation functions.
            Ignores periodicity.
            Assumes pos_data.shape() = (3,ndata).
            Returns Landy-Szalay (ls=1) or Peebles-Hauser (ls=0) estimator.
        """
        avg_RR_min = (4*ny.pi*(self.rmin/self.Lbox)**3*self.dlnr
                      if self.lgbin else
                      4*ny.pi*(self.rmin/self.Lbox)**2*self.dr/self.Lbox)
        N_ran = int(self.ran_ratio*pos_data.shape[1])
        if self.ranwts is not None:
            N_ran *= self.meanranwts
        avg_RR_min *= N_ran*(N_ran+1)
        N_ran_cat = ny.max([self.RAN_MIN,1+int((1.0/avg_RR_min + 4.0/N_ran)/self.dp_RR**2)]) 
        # to get dp_RR% error on smallest RR assuming Poisson stats.
        # min value 10 gives [var(RR)+4var(DR)]/var(DD) <~ 0.11
        print N_ran,N_ran_cat

        # !! CAREFUL WITH seed BELOW !!
        DD = self.auto_CF_workhorse(pos_data,ls=ls,dd=1)
        RR,DR = self.auto_CF_workhorse(pos_data,ls=ls,seed=seed,dd=0)
        pool = Pool(processes=self.NPROC)
        results = [pool.apply_async(unwrap_auto_CF_workhorse,(self,pos_data,ls,seed+n,0)) 
                   for n in range(N_ran_cat-1)]

        for n in range(N_ran_cat-1):
            temp_RR,temp_DR = results[n].get()
            RR += temp_RR
            if ls:
                DR += temp_DR
            # if self.verbose:
            # status_bar(n,N_ran_cat)
        pool.close()
        
        RR = RR/(1.0*N_ran_cat)
        DR = DR/(1.0*N_ran_cat)

        cf = DD/(RR + TINY) - 1.0
        if ls:
            cf = cf - 2*(DR/(RR + TINY) - 1.0)

        if self.proj:
            cf = 2*(ny.trapz(cf[:,:self.npibin_lo-1],x=self.pimid_lo,axis=1)
                    + ny.trapz(cf[:,self.npibin_lo-1:],x=self.pimid_hi,axis=1)#)
                    + 0.5*((self.pimid_hi[:,0]-self.pimid_lo[:,-1])
                           *(cf[:,self.npibin_lo-2] + cf[:,self.npibin_lo-1])))

        return cf


    def get_cond(self,pos,ax_fc,bdry,hi):
        """ Get condition to select data in chosen slice of chosen cell."""

        ax_min = self.axis[ax_fc]+'min'
        ax_max = self.axis[ax_fc]+'max'

        ot_ax = ny.where(ny.arange(3,dtype=int) != ax_fc)[0]
        ot_min = [self.axis[ot_ax[a]]+'min' for a in range(len(ot_ax))]
        ot_max = [self.axis[ot_ax[a]]+'max' for a in range(len(ot_ax))]

        cond = (((pos[ax_fc] > bdry[ax_min]) & (pos[ax_fc] <= bdry[ax_min] + self.rmax))
                if hi else
                ((pos[ax_fc] > bdry[ax_max] - self.rmax) & (pos[ax_fc] <= bdry[ax_max])))

        cond = cond & (pos[ot_ax[0]] > bdry[ot_min[0]]) & (pos[ot_ax[0]] <= bdry[ot_max[0]])
        cond = cond & (pos[ot_ax[1]] > bdry[ot_min[1]]) & (pos[ot_ax[1]] <= bdry[ot_max[1]])

        return cond


    def face_pair_counts(self,pos_data,pos_ran,ax_fc,bdry_hi,bdry_lo,ls,dd):
        """ Convenience function for pair counts across faces."""

        c_data_hi = self.get_cond(pos_data,ax_fc,bdry_hi,1)
        c_data_lo = self.get_cond(pos_data,ax_fc,bdry_lo,0)
        if dd:
            pc_DD = (ny.zeros((self.nbin-1,self.npibin-1),dtype=float) 
                     if self.proj else 
                     ny.zeros(self.nbin-1,dtype=float))
            pc_DD = pc_DD + 1.0*(self.pair_counts(pos_data,pos_data,c_data_hi,c_data_lo) 
                                 + self.pair_counts(pos_data,pos_data,c_data_lo,c_data_hi))
            return pc_DD
        else:
            c_ran_hi = self.get_cond(pos_ran,ax_fc,bdry_hi,1)
            c_ran_lo = self.get_cond(pos_ran,ax_fc,bdry_lo,0)
            pc_RR = (ny.zeros((self.nbin-1,self.npibin-1),dtype=float) 
                     if self.proj else 
                     ny.zeros(self.nbin-1,dtype=float))
            pc_DR = 1.0*pc_RR

            pc_RR = pc_RR + 1.0*(self.pair_counts(pos_ran,pos_ran,c_ran_hi,c_ran_lo) 
                                 + self.pair_counts(pos_ran,pos_ran,c_ran_lo,c_ran_hi))
            if ls:
                pc_DR = (pc_DR 
                         + 0.5*(self.pair_counts(pos_data,pos_ran,c_data_hi,c_ran_lo) 
                                + self.pair_counts(pos_ran,pos_data,c_ran_lo,c_data_hi) )
                         + 0.5*(self.pair_counts(pos_data,pos_ran,c_data_lo,c_ran_hi) 
                                + self.pair_counts(pos_ran,pos_data,c_ran_hi,c_data_lo)))
            return pc_RR,pc_DR


    def side_pair_counts(self,pos_data,pos_ran,bdry1,ind1,bdry2,ind2,ls,dd):
        """ Convenience function for pair counts across cells with shared sides.
            Cells with shared faces not counted.
            Only relevant for 3-d."""

        ax_fc = ny.where(ind1 != ind2)[0] # non-shared axes: define faces
        hi1 = ny.zeros(2,dtype=bool)
        for a in range(2):
            if ind1[ax_fc[a]] > ind2[ax_fc[a]]:
                hi1[a] = 1
        # E.g., hi1[0] = 1 means, for cell 1, 1st face is high
        # This guarantees that for cell 2, 1st face will be low.
        # Hence,
        hi2 = ~hi1

        c_data_11 = self.get_cond(pos_data,ax_fc[0],bdry1,hi1[0])
        c_data_12 = self.get_cond(pos_data,ax_fc[1],bdry1,hi1[1])
        c_data_21 = self.get_cond(pos_data,ax_fc[0],bdry2,hi2[0])
        c_data_22 = self.get_cond(pos_data,ax_fc[1],bdry2,hi2[1])
        c_data_1 = c_data_11 & c_data_12
        c_data_2 = c_data_21 & c_data_22

        if dd:
            pc_DD = (ny.zeros((self.nbin-1,self.npibin-1),dtype=float) 
                     if self.proj else 
                     ny.zeros(self.nbin-1,dtype=float))
            pc_DD = pc_DD + 1.0*(self.pair_counts(pos_data,pos_data,c_data_1,c_data_2) 
                                 + self.pair_counts(pos_data,pos_data,c_data_2,c_data_1))
            return pc_DD
        else:
            c_ran_11 = self.get_cond(pos_ran,ax_fc[0],bdry1,hi1[0])
            c_ran_12 = self.get_cond(pos_ran,ax_fc[1],bdry1,hi1[1])
            c_ran_21 = self.get_cond(pos_ran,ax_fc[0],bdry2,hi2[0])
            c_ran_22 = self.get_cond(pos_ran,ax_fc[1],bdry2,hi2[1])
            c_ran_1 = c_ran_11 & c_ran_12
            c_ran_2 = c_ran_21 & c_ran_22
            
            pc_RR = (ny.zeros((self.nbin-1,self.npibin-1),dtype=float) 
                     if self.proj else 
                     ny.zeros(self.nbin-1,dtype=float))
            pc_DR = 1.0*pc_RR

            pc_RR = pc_RR + 1.0*(self.pair_counts(pos_ran,pos_ran,c_ran_1,c_ran_2) 
                                 + self.pair_counts(pos_ran,pos_ran,c_ran_2,c_ran_1))
            if ls:
                pc_DR = (pc_DR 
                         + 0.5*(self.pair_counts(pos_data,pos_ran,c_data_1,c_ran_2) 
                                + self.pair_counts(pos_ran,pos_data,c_ran_2,c_data_1))
                         + 0.5*(self.pair_counts(pos_data,pos_ran,c_data_2,c_ran_1) 
                                + self.pair_counts(pos_ran,pos_data,c_ran_1,c_data_2)))
            return pc_RR,pc_DR

        
    def peanohilbert_boundary(self):
        """ Create Peano-Hilbert indexed boundaries for sub-volumes.
            Used for fast CF calculations. Assumes depth = 2 for now."""

        Lsub = 1.0*self.Lbox/self.depth
        boundary = (ny.zeros(self.nsub,dtype=[('xmin',float),('xmax',float),
                                         ('ymin',float),('ymax',float),
                                         ('zmin',float),('zmax',float)]))

        ind = ny.zeros((self.nsub,3),dtype=int)

        for k in range(4):
            ind[k + 4,0] = 1
        for k in range(2):
            ind[4*k+1,1] = 1
            ind[4*k+2,1] = 1
        for k in range(4):
            ind[k+2,2] = 1
                
        ind = ind.T
        boundary['xmin'] = Lsub*ind[0]
        boundary['xmax'] = boundary['xmin'] + Lsub
        boundary['ymin'] = Lsub*ind[1]
        boundary['ymax'] = boundary['ymin'] + Lsub
        boundary['zmin'] = Lsub*ind[2]
        boundary['zmax'] = boundary['zmin'] + Lsub

        return boundary,ind.T


    def DD_only(self,pos_data):
        """ DD calculation.
            Assumes pos_data.shape() = (3,ndata).
            Returns DD.
        """
        ndata = pos_data.shape[1]

        cond_data = ny.ones(ndata,dtype=bool)

        cf = 1.0*self.pair_counts(pos_data,pos_data,cond_data,cond_data)/ndata/(ndata-1)*self.Lbox**3

        return cf


    def DD_pbc_workhorse(self,pos_data,i,ndata,pbc=1):
        """ Brute force (non-tree) DD calculation workhorse.
            Assumes pos_data.shape() = (3,ndata).
            Returns pair counts for ith object.
        """
        if pbc:
            dx = ny.fabs(pos[0][i]-pos[0][i+1:])
            dy = ny.fabs(pos[1][i]-pos[1][i+1:])
            dz = ny.fabs(pos[2][i]-pos[2][i+1:])
            dist2 = ((ny.minimum(dx,self.Lbox-dx))**2
                     +(ny.minimum(dy,self.Lbox-dy))**2
                     +(ny.minimum(dz,self.Lbox-dz))**2)
        else:
            dx2 = (pos[0][i]-pos[0][i+1:])**2
            dy2 = (pos[1][i]-pos[1][i+1:])**2
            dz2 = (pos[2][i]-pos[2][i+1:])**2
            dist2 = dx2 + dy2 + dz2
        lndist = 0.5*ny.log(dist2)
        freq,lnr_edgi = ny.histogram(lndist,bins=self.lnrbin)
        # status_bar(i,ndata)
        return freq


    def DD_pbc_parallel(self,pos_data,pbc=1):
        """ Brute force (non-tree) parallelised DD calculation.
            Assumes pos_data.shape() = (3,ndata).
            Returns DD.
        """
        ndata = pos_data.shape[1]

        freq_DD = ny.zeros(self.nbin,dtype=int)
        pool = Pool(processes=self.NPROC)
        results = [pool.apply_async(unwrap_DD_pbc_workhorse,(pos_data,i,ndata,pbc)) 
                   for i in range(ndata-1)]
        freq_temp = ny.asarray([results[i].get()
                                for i in range(ndata-1)])
        pool.close()
        pool.join()
        freq_DD = ny.sum(freq_temp,axis=0)/(0.5*ndata*(ndata-1))*self.Lbox**3

        return freq_DD/(0.5*ndata*(ndata-1))*self.Lbox**3

    def RR_theory(self):
        return 4*ny.pi/3*(self.rbin[1:]**3-self.rbin[:-1]**3)


    def auto_CF_std(self,pos_data,ls=1,seed=None,ran_factor=10):
        """ Auto-correlation of data points.
            Very memory-inefficient compared to auto_CF(), but much faster for small data sets.
            Assumes pos_data.shape() = (3,ndata).
            Assumes square/cubic box of side Lbox. (Ignores periodicity for now.)
            Returns Landy-Szalay (ls=1) or Peebles-Hauser (ls=0) estimator.
        """
        ndata = pos_data.shape[1]
        nran = int(ran_factor*ndata)

        ny.random.seed(seed)
        pos_ran = ny.random.rand(3,nran)
        if self.ranwts is not None:
            NGRID = self.ranwts.shape[0] # expecting ranwts.shape = (NGRID,NGRID,NGRID)
            ix,iy,iz = ((NGRID*pos_ran) % NGRID).astype(int)
            uni = ny.random.rand(nran)
            pos_ran = pos_ran.T[uni < self.ranwts[ix,iy,iz]].T
            nran = pos_ran.shape[1]
            del ix,iy,iz,uni
            gc.collect()
        pos_ran *= self.Lbox
        print nran

        cond_data = ny.ones(ndata,dtype=bool)
        cond_ran = ny.ones(nran,dtype=bool)

        cf = ny.zeros(self.nbin-1,dtype=float)

        print '... RR'
        RR = 1.0*self.pair_counts(pos_ran,pos_ran,cond_ran,cond_ran)/nran/(nran-1)
        print '... DD'
        DD = 1.0*self.pair_counts(pos_data,pos_data,cond_data,cond_data)/ndata/(ndata-1)
        cf = DD/(RR + TINY) - 1.0

        if ls:
            print '... DR'
            DR = 1.0*self.pair_counts(pos_data,pos_ran,cond_data,cond_ran)/ndata/nran
            cf = cf - 2*(DR/(RR + TINY) - 1.0)

        del cond_data,cond_ran,pos_ran
        gc.collect()

        return cf


    def cross_CF(self,pos_data1,pos_data2,ls=1,seed=None,ran_factor=10):
        """ Cross-correlation of 2 data sets.
            Currently only this memory-inefficient version is implemented.
            Assumes pos_data.shape() = (3,ndata).
            Assumes square/cubic box of side Lbox. (Ignores periodicity for now.)
            Returns Landy-Szalay (ls=1) or Peebles-Hauser (ls=0) estimator.
        """
        ndata1 = len(pos_data1.T)
        ndata2 = len(pos_data2.T)
        nran = ran_factor*ny.max([ndata1,ndata2])
        
        ny.random.seed(seed)
        pos_ran1 = self.Lbox*ny.random.rand(nran,3)
        pos_ran2 = self.Lbox*ny.random.rand(nran,3)

        tree_data1 = syspat.cKDTree(pos_data1.T)
        tree_data2 = syspat.cKDTree(pos_data2.T)
        tree_ran1 = syspat.cKDTree(pos_ran1)
        tree_ran2 = syspat.cKDTree(pos_ran2)

        cf = ny.zeros(self.nbin-1,dtype=float)

        R1R2 = 1.0*self.pair_counts(tree_ran1,tree_ran2)/nran**2
        D1D2 = 1.0*self.pair_counts(tree_data1,tree_data2)/ndata1/ndata2

        cf = D1D2/(R1R2 + TINY) - 1.0

        if ls:
            D1R2 = 1.0*self.pair_counts(tree_data1,tree_ran2)/ndata1/nran
            D2R1 = 1.0*self.pair_counts(tree_data2,tree_ran1)/ndata2/nran
            cf = cf - (D1R2 + D2R1)/(R1R2 + TINY) + 2.0

        return cf

#/////////////////////////////////////////
#/////////////////////////////////////////

if __name__ == "__main__":

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from time import time
    from multiprocessing import Pool

    NPROC = 16

##########################


# class PowerSpectrum:
#     """ Power spectrum of particles/density field in cubic box. """

#     def __init__(self,dim=3,grid=256,Lbox=200.0,lgbin=1,nbin=30,logfile=None):
#         """ Initialise the following:
#             -- dim: number of dimensions (default 3)
#             -- grid: resolution at which to compute density in case of particles
#             -- Lbox: box size along one dimension
#             -- lgbin: use logarithmic (1) or linear (0) binning
#             -- nbin: number of bins in k
#             Additionally defines the variables/arrays:
#             cell_size,Delta_k,kmin,kmax,kbin,ktab,(dlnk or dk)
#             Methods: 
#             density_field,Pk_grid            
#             """
#         self.dim = dim
#         self.lgbin = lgbin
#         self.nbin = nbin
#         self.Lbox = Lbox
#         self.grid = grid
#         self.logfile = logfile

#         if self.dim != 3:
#             quit_("Only 3-d implemented so far. Sorry!")

#         self.cell_size = self.Lbox/self.grid
#         self.Delta_k = 2*ny.pi/self.Lbox

#         self.kmin = 1.0*self.Delta_k
#         self.kNy = ny.pi/self.cell_size # don't change this!
#         if self.lgbin:
#             self.dlnk = ny.log(self.kNy/self.kmin)/(self.nbin)
#             self.kbin = self.kmin*ny.exp(self.dlnk*ny.arange(self.nbin+1,dtype=float))
#             self.ktab = ny.sqrt(self.kbin[1:]*self.kbin[:-1])
#         else:
#             self.dk = (self.kNy-self.kmin)/(self.nbin)
#             self.kbin = self.kmin + self.dk*ny.arange(self.nbin+1,dtype=float)
#             self.ktab = 0.5*(self.kbin[1:]+self.kbin[:-1])

#         self.k2_compare = (self.kbin/self.Delta_k)**2


#     def density_field(self,pos):
#         """ Compute density field from particle locations using CIC.
#             (See http://background.uchicago.edu/~whu/Courses/Ast321_11/pm.pdf
#              for a nice exposition.)
#             Assumes pos is numpy array with pos.shape = (dim,ndata)
#             Assumes position range along each dimension is (0,Lbox)
#             Returns density contrast [shape (self.grid,self.grid,self.grid)]
#         """

#         if pos.shape[0] != self.dim:
#             quit_("Inconsistent data dimensions!! Try again.")

#         if ny.any(pos) < 0.0:
#             quit_("Positions should be in range (0,Lbox) along each dimension!!")

#         ndata = 1*pos.shape[1]
#         density = ny.zeros((self.grid,self.grid,self.grid),dtype=float)
#         if self.logfile==None: print 'Computing density field...'
#         else: writelog(self.logfile,'Computing density field...\n')

#         # Cell to which each particle belongs
#         # shape (ndata,dim); type int
#         cells = ny.floor(pos.T/self.cell_size).astype(int) % self.grid
#         cp1 = (cells + 1) % self.grid

#         cic_d = ny.zeros((self.dim,ndata),dtype=float)
#         for d in range(self.dim):
#             temp_pos = pos[d]/self.cell_size
#             cic_d[d] = ny.fabs(temp_pos - ny.floor(temp_pos) - 0.5)
#             del temp_pos
#             gc.collect()

#         cic_d = cic_d.T                
#         cic_t = 1.0 - cic_d
#         # shape (ndata,dim)

#         if self.logfile==None: print '... looping over particles'
#         else: writelog(self.logfile,'... looping over particles\n')
#         for n in xrange(ndata):
#             c0 = cells[n,0]
#             c1 = cells[n,1]
#             c2 = cells[n,2]
#             cp10 = cp1[n,0]
#             cp11 = cp1[n,1]
#             cp12 = cp1[n,2]

#             density[c0,c1,c2] += cic_t[n,0]*cic_t[n,1]*cic_t[n,2]            
#             density[cp10,c1,c2] += cic_d[n,0]*cic_t[n,1]*cic_t[n,2]
#             density[c0,cp11,c2] += cic_t[n,0]*cic_d[n,1]*cic_t[n,2]
#             density[c0,c1,cp12] += cic_t[n,0]*cic_t[n,1]*cic_d[n,2]
#             density[cp10,cp11,c2] += cic_d[n,0]*cic_d[n,1]*cic_t[n,2]
#             density[cp10,c1,cp12] += cic_d[n,0]*cic_t[n,1]*cic_d[n,2]
#             density[c0,cp11,cp12] += cic_t[n,0]*cic_d[n,1]*cic_d[n,2]            
#             density[cp10,cp11,cp12] += cic_d[n,0]*cic_d[n,1]*cic_d[n,2]
            
#             if ((n+1) % int(ndata/10.) == 0):
#                 if self.logfile==None: print "... {0:.1f}% of particles done".format(100.0*(n+1.)/ndata)
#                 else: writelog(self.logfile,"... {0:.1f}% of particles done\n".format(100.0*(n+1.)/ndata))

#         density = density*self.grid**3/(1.0*ndata) - 1.0

#         del cic_d,cic_t

#         return density


#     def Pk_grid(self,input_array,input_is_density=0):
#         """ Use FFT to compute Fourier transform of density and its power spectrum. 

#             Assumes density.shape = (self.grid,self.grid,self.grid); density is real.
#             If input is positions, then density is computed using self.density_field

#             Returns P(k) = |FTdensity|^2 on self.ktab.
#         """

#         if self.logfile==None: print '... Fourier transforming'
#         else: writelog(self.logfile,'... Fourier transforming\n')
#         if not input_is_density:
#             density = self.density_field(input_array)
#             FTdensity = fft.ifftn(density)
#         else:
#             FTdensity = fft.ifftn(input_array)

#         if self.logfile==None: print '... calculating P(k)'
#         else: writelog(self.logfile,'... calculating P(k)\n')
#         Pk = ny.zeros(self.nbin,dtype=float)

#         kmesh = []
#         for n1 in range(self.grid/2+1):
#             for n2 in range(n1,self.grid/2+1):
#                 for n3 in range(n2,self.grid/2+1):
#                     if n1**2 + n2**2 + n3**2 <= (self.grid/2)**2:
#                         kmesh.append((n1,n2,n3))
#         kmesh = ny.array(kmesh)
#         # shape (~grid,3)

#         for b in range(self.nbin-1,-1,-1):
#             cond = (ny.sum(kmesh**2,axis=1) <= self.k2_compare[b])
#             ind = ny.where(~cond)[0]
#             if ind.size:
#                 # Correct for CIC window: prod sinc(Lk_i/2Ng)^2 
#                 #                         = prod ny.sinc(kmesh[i](2pi/L)(L/Ng)/(2pi))^2
#                 #                         = prod ny.sinc(kmesh[i]/Ng)^2
#                 for p in range(3):
#                     cic_corr = (ny.sinc(1.0*kmesh[ind,p%3]/self.grid)
#                                 *ny.sinc(1.0*kmesh[ind,(p+1)%3]/self.grid)
#                                 *ny.sinc(1.0*kmesh[ind,(p+2)%3]/self.grid))**2

#                     Pk[b] += ny.mean(ny.absolute(FTdensity[kmesh[ind,p%3],
#                                                            kmesh[ind,(p+1)%3],
#                                                            kmesh[ind,(p+2)%3]]
#                                                  /cic_corr)**2)

#                     cic_corr = (ny.sinc(1.0*kmesh[ind,(p+1)%3]/self.grid)
#                                 *ny.sinc(1.0*kmesh[ind,p%3]/self.grid)
#                                 *ny.sinc(1.0*kmesh[ind,(p+2)%3]/self.grid))**2

#                     Pk[b] += ny.mean(ny.absolute(FTdensity[kmesh[ind,(p+1)%3],
#                                                            kmesh[ind,p%3],
#                                                            kmesh[ind,(p+2)%3]]
#                                                  /cic_corr)**2)
#             kmesh = kmesh[cond]
#             del cond,ind
#             gc.collect()

#         del FTdensity,kmesh
#         gc.collect()

#         Pk *= (self.Lbox**3/6.0) # 6 to account for permutations above
        
#         return Pk
