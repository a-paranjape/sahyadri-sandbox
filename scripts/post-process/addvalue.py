import os
import numpy as np
from scipy import linalg
import gc
from readers import HaloReader
from correlations import PowerSpectrum
import h5py

class AddValue(HaloReader):
    """ Generate and write value added catalog (.vahc file). """
    def __init__(self,sim_stem='scm1024',real=1,snap=200,kmax=0.1,massdef='m200b',
                 density=None,hpos=None,halos=None,ps=None,input_is_FTdensity=True,
                 grid=512,CIC=True,logfile=None,verbose=True):
        HaloReader.__init__(self,sim_stem=sim_stem,real=real,snap=snap,logfile=logfile,verbose=verbose)
        self.kmax = kmax
        self.massdef = massdef
        self.grid = grid
        self.Npmin = self.calc_Npmin_default(self.grid)
        self.mmin = self.mpart*self.Npmin
        self.density = density
        self.hpos = hpos
        self.halos = halos
        self.CIC = CIC
        self.ps = ps if ps is not None else PowerSpectrum(grid=self.grid,Lbox=self.Lbox,logfile=self.logfile,verbose=self.verbose)
        if (ps is not None) & (self.grid != self.ps.grid):
            raise TypeError('Incompatible grids detected in AddValue().')
        self.iFTd = input_is_FTdensity

        self.Nhalo = self.halos.size
        self.hpos_grid = self.hpos*self.grid/self.Lbox
        self.hgrid_round = np.fmod(np.round(self.hpos_grid),self.grid).astype('int')
        # shape(Nhalo,3)
        # hgrid_round contains integer vectors identifying the cell (round) corresponding to each halo: equiv to NGP

        # size of grid on which to interpolate smoothing scales
        self.N_RG = 30 # 60 is fine for ~1-2% accuracy
        self.OM_DEF = 0.276         # default Om for RG calculations
        self.M_NORM = 6.4173e13     # normalising mass scale for RG calculations
        self.RG_2R200b_NORM = 0.894 # RGeff(2R200b) at normalising mass scale and default Om.

        self.RGmin = 0.325*(self.Lbox/300.0)*(self.NENCL_2R200B/8.0)**(1/3.)*(512.0/self.grid) 
        # min defined by enclosed cells
        self.RGmax = 5*self.RG_2R200b_NORM*((self.MHALO_MAX/self.M_NORM)*(self.OM_DEF/self.Om))**(1/3.) 
        # max defined by 10*R200b of largest allowed halo (encloses 8x)
        self.RGtab = np.logspace(np.log10(self.RGmin),np.log10(self.RGmax),self.N_RG)

        self.smoothing_scales = np.concatenate((self.RGtab,np.array([2.0,3.0,5.0])))/self.Lbox
        self.N_scales = self.smoothing_scales.size
        self.N_scalestrings = len(self.scale_strings)
        self.NTRC_SPLIT = 80000 # for sample size larger than this, split b1 calculation into chunks in AddValue()
    ###############################################
                

    ###############################################
    def add_value(self,write_vahc=False):
        """ Given matter (Fourier) density field, halo positions and halo properties, 
            along with instance powspec of PowerSpectrum(),
            calculate array of value-added quantities and optionally write .vahc file. 
        """
        va_props = np.zeros(self.Nhalo,dtype=self.vadtypelist)
        va_props['ID'] = self.halos['ID']

        va_props['b1'] = self.calc_b1()

        out = self.calc_tensor(tensor='tidal')
        for s in self.scale_strings:
            va_props['lam1_'+s] = out['lam1_'+s] 
            va_props['lam2_'+s] = out['lam2_'+s] 
            va_props['lam3_'+s] = out['lam3_'+s]
            
        out = self.calc_tensor(tensor='hessian')
        for s in self.scale_strings[-2:]:
            va_props['lamH1_'+s] = out['lamH1_'+s] 
            va_props['lamH2_'+s] = out['lamH2_'+s] 
            va_props['lamH3_'+s] = out['lamH3_'+s]
            
        del out
        gc.collect()

        if write_vahc:
            va_cat = self.sim_stem + '/r{0:d}/out_{1:d}.vahc'.format(self.real,self.snap)
            if self.verbose:
                self.print_this('Writing to file: '+ va_cat,self.logfile)
            va_cat = self.halo_path + va_cat
            header_string =  "# Value added halo catalog. All quantities computed on {0:d}^3 grid\n".format(self.grid)
            header_string += '# haloID'
            for s in self.scale_strings:
                header_string += ' lam1_'+s+' lam2_'+s+' lam3_'+s
            for s in self.scale_strings[-2:]:
                header_string += ' lamH1_'+s+' lamH2_'+s+' lamH3_'+s
            header_string += ' b1' # deleted old 'b1', renamed old 'b1wtd' as new 'b1'
            header_string += '\n'
            f = open(va_cat,'w')
            f.write(header_string)
            f.close()
            self.write_structured(va_cat,va_props)
        
        return va_props
    ###############################################


    ###############################################
    def calc_b1(self):
        out = np.zeros(self.Nhalo)
        RANGE_MAX = np.max([1,np.rint(self.Nhalo/self.NTRC_SPLIT*(self.kmax - self.ps.Delta_k)/(0.1 - 2*np.pi/300)).astype(int)])
        if RANGE_MAX == 1:
            b1_tmp,b1_k,ktab_trunc = self.ps.halobyhalo_b1(self.density,self.hpos,kmax=self.kmax,
                                                           input_is_density=True,input_is_FTdensity=self.iFTd,CIC=self.CIC) 
            out[:] = b1_tmp 
            del b1_tmp
        else:
            if self.verbose:
                self.print_this("... ... splitting tracers into {0:d} ranges".format(RANGE_MAX),self.logfile)
            for r in range(RANGE_MAX):
                sl = np.s_[r*self.Nhalo/RANGE_MAX:(r+1)*self.Nhalo/RANGE_MAX]
                b1_tmp,b1_k,ktab_trunc = self.ps.halobyhalo_b1(self.density,self.hpos[sl],kmax=self.kmax,
                                                               input_is_density=True,input_is_FTdensity=self.iFTd,CIC=self.CIC)
                out[sl] = b1_tmp 
                del b1_tmp
                if self.verbose:
                    self.print_this("... ... ... range {0:d} of {1:d} done".format(r+1,RANGE_MAX),self.logfile)
        gc.collect()
        return out
    ###############################################

    
    ###############################################
    def calc_tensor(self,tensor='tidal'):
        if tensor not in ['tidal','hessian']:
            raise ValueError("kwarg tensor should be one of ['tidal','hessian'] in calc_tensor().")
        allowed_h = np.where(self.halos[self.massdef] > self.mmin)[0]
        N_allowed = allowed_h.size

        N_scales_this = self.N_scales if tensor == 'tidal' else 2
        smoothing_scales_this = self.smoothing_scales if tensor == 'tidal' else self.smoothing_scales[-2:]
        scale_strings_this = self.scale_strings if tensor == 'tidal' else self.scale_strings[-2:]
        tens_eigvals = np.zeros((self.Nhalo,N_scales_this,3),dtype=float)
        
        if self.verbose:
            self.print_this("... ignoring halos smaller than mmin = {0:.3e}Msun/h for tidal tensor".format(self.mmin),self.logfile)
            self.print_this("... ... {0:d} halos will be used".format(N_allowed),self.logfile)

        env_str = 'tidal environment' if tensor == 'tidal' else '(dimensionless) density Hessian'
        if self.verbose:
            self.print_this("... calculating "+env_str+" for {0:d} of {1:d} halos".format(N_allowed,self.Nhalo),self.logfile)

        env_fun = self.ps.tidal_tensor_field if tensor == 'tidal' else self.ps.density_hessian_field
        for s in range(N_scales_this):
            Rs = smoothing_scales_this[s]
            self.print_this("... ... Rs = {0:.2f}Mpc/h".format(Rs*self.Lbox),self.logfile)
            tens11,tens22,tens33,tens12,tens13,tens23 = env_fun(self.density,Rs,input_is_FTdensity=self.iFTd,CIC=self.CIC)

            self.print_this('... ... looping over halos',self.logfile)
            cnt = 0
            for h in allowed_h:
                hcell = self.hgrid_round[h]

                # consider using interpolation over nbrs below
                tens11_h = tens11[hcell[0],hcell[1],hcell[2]].real
                tens22_h = tens22[hcell[0],hcell[1],hcell[2]].real
                tens33_h = tens33[hcell[0],hcell[1],hcell[2]].real
                tens12_h = tens12[hcell[0],hcell[1],hcell[2]].real
                tens13_h = tens13[hcell[0],hcell[1],hcell[2]].real
                tens23_h = tens23[hcell[0],hcell[1],hcell[2]].real

                tens_mat = np.matrix(np.zeros((3,3),dtype=float))
                tens_mat[np.diag_indices(3)]    = np.array([tens11_h,tens22_h,tens33_h])
                tens_mat[np.triu_indices(3,1)]  = np.array([tens12_h,tens13_h,tens23_h])
                tens_mat[np.tril_indices(3,-1)] = np.array([tens12_h,tens13_h,tens23_h])

                tens_eigvals[h,s] = linalg.eigvalsh(tens_mat)
                if self.verbose:
                    self.status_bar(cnt,N_allowed)
                cnt += 1

            del tens11,tens22,tens33,tens12,tens13,tens23
            gc.collect()


        tens_eigvals = np.transpose(tens_eigvals,(2,0,1))
        # shape (3,Nhalo,N_scales)
        
        tens_eigvals_final = np.zeros((len(scale_strings_this),3,self.Nhalo),dtype=float)

        last_start = -3 if tensor == 'tidal' else -2
        for s in range(last_start,0):
            tens_eigvals_final[s] = tens_eigvals[:,:,s].copy()

        if tensor == 'tidal':
            if self.verbose:
                self.print_this('... interpolating to Nx R200b for all halos',self.logfile)
            for s in range(self.N_scalestrings-3):
                if self.verbose:
                    self.print_this("... ... {0:d}x R200b".format(2*(s+1)),self.logfile)
                RGeff = (s+1)*self.RG_2R200b_NORM*((self.halos[self.massdef]/self.M_NORM)*(self.OM_DEF/self.Om))**(1/3.) 
                # works because s=0,1,2,3 so s+1=1x,2x,3x,4x base value
                self.print_this('... ... ... looping over halos',self.logfile)
                cnt = 0
                for h in allowed_h:
                    for eig in range(3):
                        tens_eigvals_final[s,eig,h] = np.interp(np.log10(RGeff[h]),np.log10(self.RGtab),tens_eigvals[eig,h,:self.N_RG])
                    if self.verbose:
                        self.status_bar(cnt,N_allowed)
                    cnt += 1

                
        out_dtype = []
        tstr = 'lam'
        if tensor == 'hessian':
            tstr += 'H'
        for s in scale_strings_this:
            out_dtype.extend([(tstr+'1_'+s,'f'),(tstr+'2_'+s,'f'),(tstr+'3_'+s,'f')])
        out = np.zeros(self.Nhalo,dtype=out_dtype)
            
        for s in range(len(scale_strings_this)):
            out[tstr+'1_'+scale_strings_this[s]] = tens_eigvals_final[s,0]
            out[tstr+'2_'+scale_strings_this[s]] = tens_eigvals_final[s,1]
            out[tstr+'3_'+scale_strings_this[s]] = tens_eigvals_final[s,2]

        del tens_eigvals,tens_eigvals_final
        gc.collect()
        
        return out
    ###############################################
