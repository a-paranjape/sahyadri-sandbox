import os
import numpy as np
import pandas as pd
import gc
from utilities import Utilities,Constants,Paths
import h5py

########################################################
# Reader for HDF5 Gadget-4 snapshot. 
########################################################
class SnapshotReader(Utilities,Paths):
    """ Reader for HDF5 Gadget-4 snapshot. Reads single snapshot for one particle type. """
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

    def read_block(self,block='pos'):
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
        gc.collect()
            
        return out

########################################################
# Reader for (ROCKSTAR) halo catalog. 
########################################################
class HaloReader(Utilities,Paths):
    """ Reader for (ROCKSTAR) halo catalog. """

    def __init__(self,sim_stem='scm1024',logfile=None,verbose=True):

        Paths.__init__(self)
        Utilities.__init__(self)
        
        self.sim_stem = sim_stem
        self.logfile = logfile
        self.verbose = verbose

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


    def read_this(self,real,snap,VA=False,GRID=512):
        """ Read (value added) halo catalog for snapshot number 'snap' from realisation 'real'.
            Returns structured array.
        """

        halocat = self.sim_stem + '/r'+str(real)+'/' + 'out_' + str(snap)
        if not VA:
            halocat += '.trees'
        else:
            if GRID != 512:
                halocat += '_'+str(GRID)
            halocat += '.vahc'

        if self.verbose:
            print_string = '... using file: '+ halocat
            self.print_this(print_string,self.logfile)
        halocat = self.halo_path + halocat

        hdtype = self.halodatatype if not VA else self.vadatatype
        hnames = self.halodatanames if not VA else self.vadatanames
        halos = pd.read_csv(halocat,dtype=hdtype,names=hnames,
                            comment='#',delim_whitespace=True,header=None).to_records()

        return halos
