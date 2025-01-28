import sys
import numpy as np
from time import time
import gc
import socket
# import multiprocessing as mp



#######################################################################
class Paths(object):
    """ Paths for various local directories. """
    def __init__(self):
        self.home_path = '/home/aseem/iucaa/Sahyadri/sahyadri-sandbox/' #'/mnt/home/faculty/caseem/'
        self.scratch_path = self.home_path + 'Test/' # '/scratch/aseem/'
        self.config_path = self.home_path + 'Test/' # 'config/'
        if(socket.gethostname()=='Shadabs-MacBook-Pro.local'):
            self.home_path='/Users/shadab/Documents/Projects/sahyadri-sandbox/'
        elif(socket.gethostname()=='pawna'):
            self.home_path='/user/shadab/Projects/sahyadri-sandbox/'
            self.scratch_path = '/storage/shadab/' # '/scratch/aseem/'
            self.config_path = self.home_path + 'config/' # 'config/'

        self.python_path = self.home_path + 'scripts/post-process/'

        self.sim_path = self.scratch_path + 'sims/'
        self.halo_path = self.scratch_path + 'halos/'
        self.gal_path = self.scratch_path + 'galaxies/'
        self.config_transfer_path = self.config_path + 'transfer/'
        self.config_sim_path = self.config_path + 'sims/'
        self.config_halo_path = self.config_path + 'halos/'
#######################################################################

        
#######################################################################
class Constants(object):
    """ Useful constants. """
    def __init__(self):

        # constants of nature
        self.speed_of_light = 2.99792458e5 # c in km/s
        self.c_by_H0 = 0.01*self.speed_of_light # present Hubble radius in Mpc/h
        self.H0inv = 9.784619421 # 1/H0 in Gyr/h. 
        self.rhoc = 2.775366e11 # present critical density in (Msun/h)/(Mpc/h)^3; value from PDG2012
        self.dcsph = 1.686

        # useful conversions
        self.PI = 3.14159265359
        self.full_sky = 4*self.PI*(180./self.PI)**2 # square degrees in the full sky.. ~= 41253
        self.Mpc_per_km = 3.2408e-20 # Mpc per km
        self.yr_per_s  = 3.171e-8    # years per second

        # utility numerical values
        self.TINY = 1e-15
        self.NOTSOTINY = 1e-8

        # no. of physical processors available
        # self.NPROC = np.max([1,mp.cpu_count()//2])
#######################################################################


#######################################################################
class Utilities(object):
    """ Useful general-purpose functions. """
    ############################################################
    def __init__(self):
        self.select_these = np.vectorize(self.select_these_scalar)
        self.select_not_these = np.vectorize(self.select_not_these_scalar)
    ############################################################

    ############################################################
    def heaviside(self,x):
        return 0.5*(np.sign(x)+1)
    ############################################################

    ############################################################
    def wpercentile(self,data,weights=None,percentile=50.0):
        """ Weighted percentiles of data set (flattened by default).
             If weights is None, calculates usual percentile using numpy.percentile().
             If weights is not None, should be of same shape as data containing weights (needn't be normalised).
             Default percentile is 50 (median), controlled by percentile kwarg.
             Returns (weighted) percentile of data, without interpolation.
        """
        data_flat = data.flatten()
        if weights is None:
            out = np.percentile(data_flat,percentile)
        else:
            if weights.shape !=  data.shape:
                raise TypeError('Incompatible weights and data detected in wpercentiles()')
            wts = weights.flatten()

            sorter = np.argsort(data_flat)
            sorted_data = data_flat[sorter]
            sorted_wts = wts[sorter]

            cumsum = np.cumsum(sorted_wts)
            cutoff = 0.01*percentile*np.sum(sorted_wts)
            out = sorted_data[cumsum > cutoff][0]
            del wts,sorter,sorted_data,sorted_wts,cumsum,cutoff

        del data_flat
        gc.collect()

        return out
    ############################################################

    
    ############################################################
    def gen_latin_hypercube(self,Nsamp=10,dim=2,symmetric=True,param_mins=None,param_maxs=None,
                            rng=None):
        """ Generate Latin hypercube sample (symmetric by default). 
             Either param_mins and param_maxs should both be None or both be array-like of shape (dim,). 
            -- rng: either None or instance of numpy.random.RandomState(). Default None.
             Code from FirefoxMetzger's answer at
             https://codereview.stackexchange.com/questions/223569/generating-latin-hypercube-samples-with-numpy
             See https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.49.7292&rep=rep1&type=pdf
             for some ideas reg utility of symmetric Latin hypercubes.
             Returns array of shape (Nsamp,dim) with values in range (0,1) or respective minimum to maximum values.
        """
        if (param_mins is not None): 
            if (param_maxs is None):
                raise TypeError("param_mins and param_maxs should both be None or array-like of shape (dim,)")
            if len(param_mins) != dim:
                raise TypeError("len(param_mins) should be equal to dim")
        if (param_maxs is not None) :
            if(param_mins is None):
                raise TypeError("param_mins and param_maxs should both be None or array-like of shape (dim,)")
            if len(param_maxs) != dim:
                raise TypeError("len(param_maxs) should be equal to dim")

        rng_use = rng if rng is not None else np.random.RandomState()

        if symmetric:
            available_indices = [set(range(Nsamp)) for _ in range(dim)]
            samples = []

            # if Nsamp is odd, we have to choose the midpoint as a sample
            if Nsamp % 2 != 0:
                k = Nsamp//2
                samples.append([k] * dim)
                for idx in available_indices:
                    idx.remove(k)

            # sample symmetrical pairs
            for _ in range(Nsamp//2):
                sample1 = list()
                sample2 = list()

                for idx in available_indices:
                    k = rng_use.choice(np.array(list(idx)),size=1,replace=False)[0]# random.sample(idx, 1)[0]
                    sample1.append(k)
                    sample2.append(Nsamp-1-k)
                    idx.remove(k)
                    idx.remove(Nsamp-1-k)

                samples.append(sample1)
                samples.append(sample2)

            samples = np.array(samples)/(1.0*Nsamp)
        else:
            samples = np.array([rng_use.permutation(Nsamp) for i in range(dim)])/(1.0*Nsamp)
            samples = samples.T

        if (param_mins is not None):
            for d in range(dim):
                samples[:,d] *= (param_maxs[d] - param_mins[d])
                samples[:,d] += param_mins[d]

        return samples
    ############################################################


    ############################################################
    def time_this(self,start_time,logfile=None):
        totsec = time() - start_time
        minutes = int(totsec/60)
        seconds = totsec - 60*minutes
        self.print_this("{0:d} min {1:.2f} seconds\n".format(minutes,seconds),logfile)
        return
    ############################################################

    ############################################################
    def derivative(self,f_pp,f_p,f0,f_m,f_mm,h,order=1):
        """ Return 5-point estimate of 1st or 2nd order derivatives. Error in each case is O(h^4). """
        out = np.zeros_like(f0)
        if order==1:
            out = -f_pp + 8*f_p - 8*f_m + f_mm
        elif order==2:
            out = -f_pp + 16*f_p - 30*f0 + 16*f_m - f_mm
            out /= h
        else:
            raise Exception('Only order = 1 and 2 supported.')
        out /= (12*h)
        return out
    ############################################################

    ############################################################
    def write_to_file(self,filestring,seq):
        """ Opens filestring for appending and writes tab-separated list seq to it. """
        with open(filestring,'a') as f:
            s = "{0:.6e}".format(seq[0])
            for i in range(1,len(seq)):
                s += "\t" + "{0:.6e}".format(seq[i])
            s += "\n"
            f.write(s)
        return
    ############################################################


    ############################################################
    def write_structured(self,filestring,recarray,dlmt=' '):
        """ Opens filestring for appending and writes 
        rows of structured array recarray to it.
        """
        with open(filestring,'a') as f:
            for row in recarray:
                f.write(dlmt.join([str(item) for item in row]))
                f.write('\n')
        return
    ############################################################


    ############################################################
    def writelog(self,logfile,strng,overwrite=False):
        """ Convenience function for pipe-safety. """
        app_str = 'w' if overwrite else 'a'
        with open(logfile,app_str) as g:
            g.write(strng)
        return
    ############################################################


    ############################################################
    def print_this(self,print_string,logfile,overwrite=False):
        """ Convenience function for printing to logfile or stdout."""
        if logfile is not None:
            self.writelog(logfile,print_string+'\n',overwrite=overwrite)
        else:
            print(print_string)
        return
    ############################################################


    ############################################################
    def status_bar(self,n,ntot,freq=100,text='done'):
        """ Print status bar with user-defined text and frequency. """
        if freq > ntot:
            freq = ntot
        if ((n+1) % int(1.0*ntot/freq) == 0):
            frac = (n+1.)/ntot
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %.f%% " % ('.'*int(frac*20),100*frac) + text)
            sys.stdout.flush()
        if n==ntot-1: self.print_this('',None)
        return
    ############################################################

    ############################################################
    def select_these_scalar(self,elmt,wanted_elmt):
        """ Select elements from elmt which belong to wanted_elmt.
             For fast evaluation with large arrays, ensure wanted_elmt is
             a set by applying set() to the array.
             Returns boolean array of shape elmt.
        """
        return elmt in wanted_elmt
    ############################################################

    ############################################################
    def select_not_these_scalar(self,elmt,wanted_elmt):
        """ Select elements from elmt which *do not* belong to wanted_elmt.
             For fast evaluation with large arrays, ensure wanted_elmt is
             a set by applying set() to the array.
             Returns boolean array of shape elmt.
        """
        return elmt not in wanted_elmt
    ############################################################

#######################################################################
