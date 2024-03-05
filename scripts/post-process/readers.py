import numpy as np
import pandas as pd
import gc
from utilities import Utilities,Constants,Paths

########################################################
# Reader for (ROCKSTAR) halo catalog. 
########################################################

class HaloReader(object,Paths,Utilities):
    """ Reader for (ROCKSTAR) halo catalog. """

    def __init__(self,sim_stem='scm1024',logfile=None,verbose=True):

        Paths.__init__(self)
        Utilities.__init__(self)
        
        self.sim_stem = sim_stem
        self.logfile = logfile
        self.verbose = verbose

        self.halodatatype = {'Scale':float,'ID':long,'descScale':float,'descID':long,'numProg':int,
                             'pid':int,'upid':int,'descpid':int,'phantom':int,'sam_mvir':float,
                             'mbnd_vir':float,'rvir':float,'rs':float,'vrms':float,'mmp':int,
                             'ScaleLastMM':float,'vmax':float,
                             'x':float,'y':float,'z':float,'vx':float,'vy':float,'vz':float,
                             'Jx':float,'Jy':float,'Jz':float,'spin':float,
                             'BreadthFirstID':long,'DepthFirstID':long,'TreeRootID':long,'OrigHaloID':long,
                             'snapnum':int,'NextCoprogDepthFirstID':long,'LastProgDepthFirstID':long,
                             'rs_klypin':float,
                             'mvir':float,'m200b':float,'m200c':float,'mCustom2':float,'mCustom':float,
                             'Xoff':float,'Voff':float,'spin_bullock':float,
                             'b_to_a':float,'c_to_a':float,'Ax':float,'Ay':float,'Az':float,
                             'b_to_a_500c':float,'c_to_a_500c':float,'Ax_500c':float,'Ay_500c':float,'Az_500c':float,
                             'TbyU':float,'Mpe_Behroozi':float,'Mpe_Diemer':float,'halfmassradius':float,
                             'macc':float,'mpeak':float,'vacc':float,'vpeak':float,'halfmassscale':float,
                             'accrate_inst':float,'accrate_100Myr':float,'accrate_1tdyn':float,
                             'accrate_2tdyn':float,'accrate_mpeak':float,'mpeakscale':float,
                             'accscale':float,'firstaccscale':float,'firstaccmvir':float,
                             'firstaccvmax':float,'vmaxAtmpeak':float}
        
        self.halodatanames = ['Scale','ID','descScale','descID','numProg',
                              'pid','upid','descpid','phantom','sam_mvir',
                              'mbnd_vir','rvir','rs','vrms','mmp',
                              'ScaleLastMM','vmax',
                              'x','y','z','vx','vy','vz',
                              'Jx','Jy','Jz','spin',
                              'BreadthFirstID','DepthFirstID','TreeRootID','OrigHaloID',
                              'snapnum','NextCoprogDepthFirstID','LastProgDepthFirstID',
                              'rs_klypin',
                              'mvir','m200b','m200c','mCustom2','mCustom',
                              'Xoff','Voff','spin_bullock',
                              'b_to_a','c_to_a','Ax','Ay','Az',
                              'b_to_a_500c','c_to_a_500c','Ax_500c','Ay_500c','Az_500c',
                              'TbyU','Mpe_Behroozi','Mpe_Diemer','halfmassradius',
                              'macc','mpeak','vacc','vpeak','halfmassscale',
                              'accrate_inst','accrate_100Myr','accrate_1tdyn',
                              'accrate_2tdyn','accrate_mpeak','mpeakscale',
                              'accscale','firstaccscale','firstaccmvir',
                              'firstaccvmax','vmaxAtmpeak']
        
        self.vadatatype = {'ID':long,
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
