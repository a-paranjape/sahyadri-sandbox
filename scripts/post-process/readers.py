import os
import numpy as np
import pandas as pd
import gc
from utilities import Utilities,Paths
import h5py

########################################################
# Reader for HDF5 Gadget-4 snapshot. 
########################################################
class SnapshotReader(Utilities,Paths):
    """ Reader for HDF5 Gadget-4 snapshot. Reads single snapshot for one particle type. """
    ###############################################
    def __init__(self,sim_stem='scm1024',real=1,snap=200,ptype=1,logfile=None,verbose=True):
        # code below inspired by Pylians (https://github.com/franciscovillaescusa/Pylians3/blob/master/library/readgadget.py)
        # and simplified to focus on Gadget-4 HDF5 snapshots.
        
        Paths.__init__(self)
        Utilities.__init__(self)
        
        self.sim_stem = sim_stem
        self.real = real
        self.snap = snap
        self.ptype = ptype # 1 for DM
        self.logfile = logfile
        self.verbose = verbose

        self.snapshot_file = self.sim_stem + '/r'+str(self.real) + '/snapshot_{0:03d}'.format(self.snap) 
        if os.path.exists(self.sim_path + self.snapshot_file+'.hdf5'):
            self.snapshot_file += '.hdf5' 
        elif os.path.exists(self.sim_path + self.snapshot_file+'.0.hdf5'):
            self.snapshot_file += '.0.hdf5'
        else:
            raise Exception('File not found!')
        
        if self.verbose:
            self.print_this('Snapshot Reader:\n... preparing to read file: '+self.snapshot_file,self.logfile)
            
        self.snapshot_file = self.sim_path + self.snapshot_file

        f = h5py.File(self.snapshot_file,'r')
        self.scale    = f['Header'].attrs[u'Time']
        self.redshift = f['Header'].attrs[u'Redshift']
        self.npart_this    = (f['Header'].attrs[u'NumPart_ThisFile']).astype(np.int64)
        self.npart    = (f['Header'].attrs[u'NumPart_Total']).astype(np.int64)
        self.nFile  = int(f['Header'].attrs[u'NumFilesPerSnapshot'])
        self.massarr  = f['Header'].attrs[u'MassTable']
        self.Lbox  = f['Header'].attrs[u'BoxSize']

        self.npart_this = self.npart_this[self.ptype]
        self.npart = self.npart[self.ptype]
        self.mpart = self.massarr[self.ptype]*1e10 # Msun/h

        self.Om = f['Parameters'].attrs[u'Omega0']
        self.Olam = f['Parameters'].attrs[u'OmegaLambda']
        self.hubble = f['Parameters'].attrs[u'HubbleParam']
        
        if self.verbose:
            self.print_this('... loaded header and parameters',self.logfile)

        f.close()
    ###############################################

    ###############################################
    def read_block(self,block='pos',down_to=0,seed=None):
        """ Read positions, velocities or IDs of one complete snapshot. """
        if block not in ['pos','vel','ids']:
            raise ValueError("block should be one of ['pos','vel','ids'] in read_block().")

        prefix = 'PartType%d/'%self.ptype
        if block == 'pos':
            suffix = 'Coordinates'
        elif block == 'vel':
            suffix = 'Velocities'
        elif block == 'ids':
            suffix = 'ParticleIDs'
            
        if self.verbose:
            self.print_this('... reading '+suffix,self.logfile)

        f = h5py.File(self.snapshot_file,'r')
        out_part = f[prefix+suffix][:]
        f.close()
        if self.nFile > 1:
            # NEEDS TESTING
            out = np.zeros(self.npart,dtype=out_part.dtype) if block=='ids' else np.zeros((self.npart,3),dtype=out_part.dtype)
            out[:self.npart_this] = out_part 
            shift = self.npart_this 
            for i in range(1,self.nFile):
                filename = self.snapshot_file[:-6]+str(i)+'.hdf5'
                f = h5py.File(self.snapshot_file,'r')
                np_this = (f['Header'].attrs[u'NumPart_ThisFile']).astype(np.int64)
                out[shift:shift+np_this] = f[prefix+suffix][:]
                f.close()
                shift += np_this
        else:
            out = out_part.copy()

        del out_part

        if down_to**3 > self.npart:
            if self.verbose:
                self.print_this('Not enough particles! Using original sample...',self.logfile)
            down_to = 0
            
        if down_to > 0:
            if self.verbose:
                self.print_this('... downsampling to {0:d}^3'.format(down_to),self.logfile)
            rng = np.random.RandomState(seed)
            n_sample = int(down_to**3)
            
            ind = rng.choice(self.npart,size=n_sample,replace=False)
            out = out[ind]
            del ind
            
        gc.collect()
        
        return out if block == 'ids' else out.T # notice transpose
    ###############################################

########################################################
# Reader for (ROCKSTAR) halo catalog. 
########################################################
class HaloReader(SnapshotReader):
    """ Reader for (ROCKSTAR) halo catalog. """
    ###############################################
    def __init__(self,sim_stem='scm1024',real=1,snap=200,logfile=None,verbose=True):

        SnapshotReader.__init__(self,sim_stem=sim_stem,real=real,snap=snap,logfile=logfile,verbose=verbose)

        self.halocat_stem = self.sim_stem + '/r'+str(self.real)+'/' + 'out_' + str(self.snap)        

        self.halodatatype = {'Scale':float,'ID':'int64','descScale':float,'descID':'int64','numProg':int,
                             'pid':int,'upid':int,'descpid':int,'phantom':int,'sam_mvir':float,
                             'mbnd_vir':float,'rvir':float,'rs':float,'vrms':float,'mmp':int,
                             'ScaleLastMM':float,'vmax':float,
                             'x':float,'y':float,'z':float,'vx':float,'vy':float,'vz':float,
                             'Jx':float,'Jy':float,'Jz':float,'spin':float,
                             'BreadthFirstID':'int64','DepthFirstID':'int64','TreeRootID':'int64','OrigHaloID':'int64',
                             'snapnum':int,'NextCoprogDepthFirstID':'int64','LastProgDepthFirstID':'int64',
                             'LastMainLeafDepthFirstID':float,'TidalForce':float,'TidalID':float, # new cols
                             'rs_klypin':float,
                             'mvir':float,'m200b':float,'m200c':float,'mCustom2':float,'mCustom':float,
                             'Xoff':float,'Voff':float,'spin_bullock':float,
                             'b_to_a':float,'c_to_a':float,'Ax':float,'Ay':float,'Az':float,
                             'b_to_a_500c':float,'c_to_a_500c':float,'Ax_500c':float,'Ay_500c':float,'Az_500c':float,
                             'TbyU':float,'Mpe_Behroozi':float,'Mpe_Diemer':float,'halfmassradius':float,
                             'macc':float,'mpeak':float,'vacc':float,'vpeak':float,'halfmassscale':float,
                             'accrate_inst':float,'accrate_100Myr':float,'accrate_1tdyn':float,
                             'accrate_2tdyn':float,'accrate_mpeak':float,
                             'acc_logvmax_inst':float,'acc_logvmax_1tdyn':float, # new cols
                             'mpeakscale':float,
                             'accscale':float,'firstaccscale':float,'firstaccmvir':float,
                             'firstaccvmax':float,'vmaxAtmpeak':float,
                             'TidalForce_tdyn':float, 'log(vmax/vmax_max(tdyn,tmpeak))':float,
                             'time_futmerg':float,'futmerg_mmpid':float,'spin_mpeakscale':float # new cols
                             }
        
        self.halodatanames = ['Scale','ID','descScale','descID','numProg',
                              'pid','upid','descpid','phantom','sam_mvir',
                              'mbnd_vir','rvir','rs','vrms','mmp',
                              'ScaleLastMM','vmax',
                              'x','y','z','vx','vy','vz',
                              'Jx','Jy','Jz','spin',
                              'BreadthFirstID','DepthFirstID','TreeRootID','OrigHaloID',
                              'snapnum','NextCoprogDepthFirstID','LastProgDepthFirstID',
                              'LastMainLeafDepthFirstID','TidalForce','TidalID', # new cols
                              'rs_klypin',
                              'mvir','m200b','m200c','mCustom2','mCustom',
                              'Xoff','Voff','spin_bullock',
                              'b_to_a','c_to_a','Ax','Ay','Az',
                              'b_to_a_500c','c_to_a_500c','Ax_500c','Ay_500c','Az_500c',
                              'TbyU','Mpe_Behroozi','Mpe_Diemer','halfmassradius',
                              'macc','mpeak','vacc','vpeak','halfmassscale',
                              'accrate_inst','accrate_100Myr','accrate_1tdyn',
                              'accrate_2tdyn','accrate_mpeak',
                              'acc_logvmax_inst','acc_logvmax_1tdyn', # new cols
                              'mpeakscale',
                              'accscale','firstaccscale','firstaccmvir',
                              'firstaccvmax','vmaxAtmpeak',
                              'TidalForce_tdyn', 'log(vmax/vmax_max(tdyn,tmpeak))','time_futmerg','futmerg_mmpid','spin_mpeakscale' # new cols
                              ]
        
        self.vadatatype = {'ID':'int64',
                           'lam1_R2R200b':float,'lam2_R2R200b':float,'lam3_R2R200b':float,
                           'lam1_R4R200b':float,'lam2_R4R200b':float,'lam3_R4R200b':float,
                           'lam1_R6R200b':float,'lam2_R6R200b':float,'lam3_R6R200b':float,
                           'lam1_R8R200b':float,'lam2_R8R200b':float,'lam3_R8R200b':float,
                           'lam1_R2Mpch':float,'lam2_R2Mpch':float,'lam3_R2Mpch':float,
                           'lam1_R3Mpch':float,'lam2_R3Mpch':float,'lam3_R3Mpch':float,
                           'lam1_R5Mpch':float,'lam2_R5Mpch':float,'lam3_R5Mpch':float,
                           'lamH1_R3Mpch':float,'lamH2_R3Mpch':float,'lamH3_R3Mpch':float,
                           'lamH1_R5Mpch':float,'lamH2_R5Mpch':float,'lamH3_R5Mpch':float,
                           'b1':float,'b1wtd':float}
        self.vadatanames = ['ID',
                            'lam1_R2R200b','lam2_R2R200b','lam3_R2R200b',
                            'lam1_R4R200b','lam2_R4R200b','lam3_R4R200b',
                            'lam1_R6R200b','lam2_R6R200b','lam3_R6R200b',
                            'lam1_R8R200b','lam2_R8R200b','lam3_R8R200b',
                            'lam1_R2Mpch','lam2_R2Mpch','lam3_R2Mpch',
                            'lam1_R3Mpch','lam2_R3Mpch','lam3_R3Mpch',
                            'lam1_R5Mpch','lam2_R5Mpch','lam3_R5Mpch',
                            'lamH1_R3Mpch','lamH2_R3Mpch','lamH3_R3Mpch',
                            'lamH1_R5Mpch','lamH2_R5Mpch','lamH3_R5Mpch',
                            'b1','b1wtd']

        self.scalefile = self.halo_path + sim_stem + '/scales.txt'
        self.SCALES = np.loadtxt(self.scalefile,dtype=[('snapnum','i'),('scale','f')])
        self.REDSHIFT = 1.0/self.SCALES['scale'] - 1.0
    ###############################################


    ###############################################
    def read_this(self,va=False):
        """ Read (value added) halo catalog for snapshot number 'snap' from realisation 'real'.
            Returns structured array.
        """
        if not va:
            halocat = self.halocat_stem + '.trees'
        else:
            halocat = self.halocat_stem + '.vahc'

        if self.verbose:
            self.print_this('... using file: '+ halocat,self.logfile)
        halocat = self.halo_path + halocat

        hdtype = self.halodatatype if not va else self.vadatatype
        hnames = self.halodatanames if not va else self.vadatanames
        halos = pd.read_csv(halocat,dtype=hdtype,names=hnames,comment='#',delim_whitespace=True,header=None).to_records()

        return halos
    ###############################################

    
    ###############################################
    def prep_halos(self,va=False,QE=0.5,massdef='mvir',Npmin=100,keep_subhalos=False):
        """ Reads halo (+ vahc) catalogs for given realisation and snapshot. 
             Cleans catalog by selecting relaxed objects in range max(0,1-QE) <= 2T/|U| <= 1+QE 
             where QE > 0 (default QE=0.5; Bett+07).
             Selects objects with at least Npmin particles for given massdef.
             Optionally removes subhalos (set keep_subhalos=False).
             Returns array of shape (3,Ndata) for positions (Mpc/h); structured array(s) for full halo properties (+ vahc).
             Halos will be sorted by (increasing) massdef.
        """ 

        if self.verbose:
            self.print_this("... preparing halo data",self.logfile)
        halos = self.read_this()
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

        pos = np.array([halos['x'],halos['y'],halos['z']])
        # if (self.RSD) & (halos.size > 0):
        #     if self.verbose:
        #         self.print_this("... ... applying redshift space displacement",self.logfile)
        #     pos[2] = pos[2] + 0.01*halos['vz']*(1+self.redshift)/self.EHub(self.redshift)
        if va:
            vahc = self.read_this(va=True)
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

        return (pos.T,halos,vahc) if va else (pos.T,halos)
    ###############################################
