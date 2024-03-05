import numpy as np
import pandas as pd
import gc


########################################################
# Reader for (ROCKSTAR) halo catalog. 
########################################################

class HaloReader(object,Paths):
    """ Reader for (ROCKSTAR) halo catalog. """

    def __init__(self,sim_stem='scm1024',logfile=None):

        Paths.__init__(self)
        self.sim_stem = sim_stem
        self.logfile = logfile

        self.HALO_STORAGE_BASE1 = halo_path

        self.halodatatype = con.dict_halo_data_type if not self.TREES else con.dict_halo_data_trees
        self.halodatanames = con.names_halo_data_type if not self.TREES else con.names_halo_data_trees
        self.vadatatype = con.dict_halo_data_vac
        self.vadatanames = con.names_halo_data_vac


        # dictionary of sig8 values
        self.SIG8_DICT = {'scm1024':0.811,'su1024/delta0.0':0.811,'su512/delta0.0':0.811,'scmL1024':0.811,
                          'bdm_cdm1024':0.815,'bdm_zs1e5f0.51024':0.814195126,
                          'wdm0.0keV512':0.829,'wdm0.2keV512':0.809481520,
                          'wscm0.4keV1024':0.8094514213,'bolshoi':0.82}
        self.sig8 = self.SIG8_DICT[sim_stem]


        self.NS_DICT = {'scm1024':0.961,'su1024/delta0.0':0.961,'su512/delta0.0':0.961,'scmL1024':0.961,
                          'bdm_cdm1024':0.0677,'bdm_zs1e5f0.51024':0.9677,
                          'wdm0.0keV512':0.96,'wdm0.2keV512':0.96,
                          'wscm0.4keV1024':0.961,'bolshoi':0.95}
        self.ns = self.NS_DICT[sim_stem]

        if self.TREES:
            self.scalefile = halo_path + sim_stem + '/scales.txt'
            self.SCALES = ny.loadtxt(self.scalefile,dtype=[('snapnum','i'),('scale','f')])
            self.REDSHIFT = 1.0/self.SCALES['scale'] - 1.0


    def read_this(self,real,snap,VA=False,verbose=True,GRID=512):
        """ Read (value added) halo catalog for snapshot number 'snap' from realisation 'real'.
            Returns structured array.
        """

        halo_storage_path = self.HALO_STORAGE_BASE1

        halocat = self.sim_stem + '/r'+str(real)+'/' + 'out_' + str(snap)
        if not VA:
            if self.TREES:
                halocat += '.trees'
            else:
                halocat += '.parents'
        else:
            if GRID != 512:
                halocat += '_'+str(GRID)
            halocat += '.vahc'

        if verbose:
            print_string = '... using file: '+ halocat
            if self.logfile: writelog(self.logfile,print_string+'\n')                
            else: print print_string
        halocat = halo_storage_path + halocat

        hdtype = self.halodatatype if not VA else self.vadatatype
        hnames = self.halodatanames if not VA else self.vadatanames
        halos = pd.read_csv(halocat,dtype=hdtype,names=hnames,
                            comment='#',delim_whitespace=True,header=None).to_records()

        return halos



class Paths(object):
    """ Paths for various local directories. """
    def __init__(self):
        self.home_path = '/home/aseem/iucaa/Sahyadri/sahyadri-sandbox/' #'/mnt/home/faculty/caseem/'
        self.python_path = self.home_path + 'scripts/post-process/'

        self.scratch_path = self.home_path + 'Test/' # '/scratch/aseem/'
        self.sim_path = scratch_path + 'sims/'
        self.halo_path = scratch_path + 'halos/'
        self.gal_path = scratch_path + 'galaxies/'

        self.config_path = self.home_path + 'Test/' # 'config/'
        self.config_transfer_path = self.config_path + 'transfer/'
        self.config_sim_path = self.config_path + 'sims/'
        self.config_halo_path = self.config_path + 'halos/'
