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
    def __init__(self,sim_stem='scm1024',real=1,snap=200,ptype=1,logfile=None,verbose=True,read_header=True):
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
            raise Exception('File not found!\n sim_path=%s \nfile_name=%s  [.hdf5 or .0.hdf5]'%(
                self.sim_path,self.snapshot_file))

        if self.verbose:
            self.print_this('Snapshot Reader:\n... preparing to read file: '+self.snapshot_file,self.logfile)
            
        self.snapshot_file = self.sim_path + self.snapshot_file

        #read header only if this is set to true, just so that even if snapshot is not available then we should be able to use this function for rest of the file path defintions
        self.read_header=read_header
        if(self.read_header):
            self.read_snapshot_header()

    def read_snapshot_header(self):
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
        
        self.halodatanames = list(self.halodatatype.keys())
        
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
                           'b1':float} # deleted old 'b1', renamed old 'b1wtd' as new 'b1'
        
        self.vadatanames = list(self.vadatatype.keys())

        # needed for AddValue
        self.vadtypelist = [('ID','i8'),
                            ('lam1_R2R200b','f'),('lam2_R2R200b','f'),('lam3_R2R200b','f'),
                            ('lam1_R4R200b','f'),('lam2_R4R200b','f'),('lam3_R4R200b','f'),
                            ('lam1_R6R200b','f'),('lam2_R6R200b','f'),('lam3_R6R200b','f'),
                            ('lam1_R8R200b','f'),('lam2_R8R200b','f'),('lam3_R8R200b','f'),
                            ('lam1_R2Mpch','f'),('lam2_R2Mpch','f'),('lam3_R2Mpch','f'),
                            ('lam1_R3Mpch','f'),('lam2_R3Mpch','f'),('lam3_R3Mpch','f'),
                            ('lam1_R5Mpch','f'),('lam2_R5Mpch','f'),('lam3_R5Mpch','f'),
                            ('lamH1_R3Mpch','f'),('lamH2_R3Mpch','f'),('lamH3_R3Mpch','f'),
                            ('lamH1_R5Mpch','f'),('lamH2_R5Mpch','f'),('lamH3_R5Mpch','f'),
                            ('b1','f')] # deleted old 'b1', renamed old 'b1wtd' as new 'b1'
        
        self.scale_strings = ['R2R200b','R4R200b','R6R200b','R8R200b','R2Mpch','R3Mpch','R5Mpch']

        self.scalefile = self.halo_path + sim_stem + '/scales.txt'
        self.SCALES = np.loadtxt(self.scalefile,dtype=[('snapnum','i'),('scale','f')])
        self.REDSHIFT = 1.0/self.SCALES['scale'] - 1.0
        
        # number of grid cells enclosed inside 2*R_200b. needed for calc_Npmin_default()
        self.NENCL_2R200B = 0.2 # 0.2
        self.MHALO_MAX = 3e15
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
    def convert_halos_fits(self):
        #example of how to execute this function to convert to halo catalog
        #import the library
        import ConvertToFITS as Cfits

        #input directory for halo catalog
        #self.halocat_stem = sr.sim_stem + '/r'+str(sr.real)+'/' + 'out_' + str(sr.snap)
        indir=self.halo_path+self.sim_stem+ '/r'+str(self.real)+'/'
        #root name for the halo catalog assuming in input directory .trees and .vahc files
        rootin='out_' + str(self.snap)

        #output directory: set this to None if want to use input directory for output
        outdir=None
        #outdir=None

        #list of output to be written
        #basic: file contain only the most used information about the halos
        #extended: file contains all information in .trees except what is included in basic
        #vahc: write the tidal information from vahc
        #To do: include the tidal ranks
        #you can chose any combination of output_type and corresponding files will be written
        list_output_type=['basic','extended','vahc']
        Cfits.convert_fits(indir=indir,rootin=rootin,outdir=outdir,list_output_type=list_output_type)
    ###############################################

    
    ###############################################
    def prep_halos(self,va=False,QE=0.5,massdef='mvir',Npmin=100,keep_subhalos=False):
        """ Reads halo (+ vahc) catalogs for given realisation and snapshot. 
             Cleans catalog by selecting relaxed objects in range max(0,1-QE) <= 2T/|U| <= 1+QE 
             where QE > 0 (default QE=0.5; Bett+07). Use QE=None to skip cleaning.
             Selects objects with at least Npmin particles for given massdef.
             Optionally removes subhalos (set keep_subhalos=False).
             Returns array of shape (Ndata,3) for positions (Mpc/h); structured array(s) for full halo properties (+ vahc).
             Halos will be sorted by (increasing) massdef.
        """ 

        if self.verbose:
            self.print_this("... preparing halo data",self.logfile)
        halos = self.read_this()
        Nhalos_all = halos.size
        mmin = self.mpart*Npmin
        cond_clean = (halos[massdef] >= mmin)
        if QE is not None:
            TbyU_max = 0.5*(1+QE)
            TbyU_min = np.max([0.0,0.5*(1-QE)])
            cond_clean = cond_clean & ((halos['TbyU'] < TbyU_max) & (TbyU_min < halos['TbyU']))
        if not keep_subhalos:
            cond_clean = cond_clean & (halos['pid'] == -1)
        if self.verbose:
            self.print_this("... ... using mass definition " + massdef + " > {0:.3e} Msun/h".format(mmin),self.logfile)
            if QE is not None:
                self.print_this("... ... only relaxed objects retained with {0:.2f} < 2T/|U| < {1:.2f}"
                                .format(2*TbyU_min,2*TbyU_max),self.logfile)
            else:
                self.print_this("... ... skipping relaxation cleaning",self.logfile)
            if not keep_subhalos:
                self.print_this("... ... discarding subhalos",self.logfile)

        halos = halos[cond_clean]
        if self.verbose:
            self.print_this("... ... kept {0:d} of {1:d} objects in catalog".format(halos.size,Nhalos_all),self.logfile)

        hpos = np.array([halos['x'],halos['y'],halos['z']])
        # if (self.RSD) & (halos.size > 0):
        #     if self.verbose:
        #         self.print_this("... ... applying redshift space displacement",self.logfile)
        #     hpos[2] = hpos[2] + 0.01*halos['vz']*(1+self.redshift)/self.EHub(self.redshift)
        if va:
            vahc = self.read_this(va=True)
            vahc = vahc[cond_clean]

        hpos = hpos % self.Lbox
        hpos = hpos.T  # shape (Ntrc,3)

        del cond_clean
        gc.collect()

        if self.verbose:
            self.print_this("... ... sorting by "+massdef,self.logfile)
        sorter = halos[massdef].argsort()
        halos = halos[sorter]
        hpos = hpos[sorter]
        if va:
            vahc = vahc[sorter]

        del sorter
        gc.collect()

        return (hpos,halos,vahc) if va else (hpos,halos)
    ###############################################

    
    ###############################################
    def calc_Npmin_default(self,grid):
        Npmin = 200*self.NENCL_2R200B*(self.npart/1024**3.)*(512./grid)**3
        return Npmin
    ###############################################
