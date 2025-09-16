import os
import numpy as np
import pandas as pd
import gc
from utilities import Utilities,Paths
import h5py

#These libraries are needed to use fits files
import ConvertToFITS as Cfits
import fitsio as F
import sys
import json

import time

def ltime(date=True):
    tm=time.localtime()
    datestr=str(tm.tm_mday)+'/'+str(tm.tm_mon)+'/'+str(tm.tm_year)
    tmstr=str(tm.tm_hour)+':'+str(tm.tm_min)+':'+str(tm.tm_sec)

    if(date):
        tmstr=datestr+' '+tmstr

    return tmstr

########################################################
# Reader for HDF5 Gadget-4 snapshot. 
########################################################
class SnapshotReader(Utilities,Paths):
    """ Reader for HDF5 Gadget-4 snapshot. Reads single snapshot for one particle type. """
    ###############################################
    def __init__(self,sim_stem='scm1024',real=1,snap=200,ptype=1,logfile=None,verbose=True,read_header=True,use_compressed=False,subsamples=[]):
        '''use_compressed will read using compressed files, in this case one could give a series of subsamples to accumulate all the file for outputs. This should match the subsamples given while compressing'''
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
        self.use_compressed=use_compressed
        self.subsamples=subsamples
        self.compressed_fileroot='comp_snapshot_{0:03d}'.format(self.snap) 
        
        self.snapshot_file =  self.sim_stem + '/r'+str(self.real) + '/snapshot_{0:03d}'.format(self.snap)
        self.snapdir =  self.sim_stem + '/r'+str(self.real) + '/snapdir_{0:03d}'.format(self.snap)
        #read header only if this is set to true, just so that even if snapshot is not available then we should be able to use this function for rest of the file path defintions
        self.read_header=read_header
        if(self.read_header):
            self.read_snapshot_header()

    def read_snapshot_header(self):
        if os.path.exists(self.snapshot_file+'.hdf5'):
            self.snapshot_ext = '.hdf5' 
        elif os.path.exists(self.snapshot_file+'.0.hdf5'):
            self.snapshot_ext = '.0.hdf5'
        elif(os.path.exists(self.snapdir)):
            if(os.path.exists(self.snapdir+'/snapshot_{0:03d}.hdf5'.format(self.snap))):
                self.snapshot_file=self.snapdir+'/snapshot_{0:03d}'.format(self.snap)
                self.snapshot_ext = '.hdf5'
            elif(os.path.exists(self.snapdir+'/snapshot_{0:03d}.0.hdf5'.format(self.snap))):
                self.snapshot_file=self.snapdir+'/snapshot_{0:03d}'.format(self.snap)
                self.snapshot_ext = '.0.hdf5'

        else:
            #raise Exception('File not found!\n sim_path=%s \nfile_name=%s  [.hdf5 or .0.hdf5]'%(
                #self.sim_path,self.snapshot_file))
            raise Exception('File not found!\n sim_path=%s'%self.snapdir+'snapshot file:%s'%self.snapshot_file)

        if self.verbose:
            self.print_this('Snapshot Reader:\n... preparing to read file: '+self.snapshot_file,self.logfile)
            
        if(self.use_compressed):
            header_info=self.read_compressed_header()
            self.compression_dic=header_info['Header']['compression_dic']
            self.subsamples=self.compression_dic['subsamples']

        else:
            f = h5py.File(self.snapshot_file+self.snapshot_ext,'r')
            header_info={}
            for group_name in ['Config', 'Header', 'Parameters']:
                if group_name in f:
                    group = f[group_name]
                    header_info[group_name] = {}
                    # Read attributes
                    for attr_name, attr_value in group.attrs.items():
                        header_info[group_name][attr_name] = self.json_serializable(attr_value)
            f.close()

        self.scale    = header_info['Header']['Time']#f['Header'].attrs[u'Time']
        self.redshift = header_info['Header']['Redshift']
        self.npart_this    = np.array(header_info['Header']['NumPart_ThisFile']).astype(np.int64)
        self.npart    = np.array(header_info['Header']['NumPart_Total']).astype(np.int64)
        self.nFile  = int(header_info['Header']['NumFilesPerSnapshot'])
        self.massarr  = header_info['Header']['MassTable']
        self.Lbox  = header_info['Header']['BoxSize']

        self.npart_this = self.npart_this[self.ptype]
        self.npart = self.npart[self.ptype]
        self.mpart = self.massarr[self.ptype]*1e10 # Msun/h

        self.Om = header_info['Parameters']['Omega0']
        self.Olam = header_info['Parameters']['OmegaLambda']
        self.hubble = header_info['Parameters']['HubbleParam']
        
        if self.verbose:
            self.print_this('... loaded header and parameters',self.logfile)

    ###############################################

    ###############################################
    def read_block(self,block='pos',down_to=0,seed=None,subsamples=[],raw_nfile=[]):
        """ Read positions, velocities or IDs of one complete snapshot.
        if use_compressed =True then you can use subsamples list to select which subsample to read, by default it will read all the subsamples
        if use_compressed=False and snapshot is in multiple files then you can use raw_nfile to chose which files to read, by default it will read all file. This is relevant only for simulations with multiple file per snapshot"""
        if block not in ['pos','vel','ids','potential']:
            raise ValueError("block should be one of ['pos','vel','ids'] in read_block().")
        
        if(self.use_compressed):
            transform_quant={'pos':'positions','vel':'velocities','potential':'potentials','ids':'ids'}
            if(subsamples==[]):
                subsamples=self.subsamples
            res_dic=self.load_compressed(subsamples,load_quant=[transform_quant[block]])
            return res_dic[transform_quant[block]]

        prefix = 'PartType%d/'%self.ptype
        if block == 'pos':
            suffix = 'Coordinates'
        elif block == 'vel':
            suffix = 'Velocities'
        elif block == 'ids':
            suffix = 'ParticleIDs'
        elif block == 'potential':
            suffix = 'Potential'
            
        if self.verbose:
            self.print_this('... reading '+suffix,self.logfile)

        if(self.nFile==1):
            fnum_list=[0]
        elif(raw_nfile ==[]):
            fnum_list=np.arange(0,self.nFile)
        else:
            fnum_list=raw_nfile



        for ii, ifile in enumerate(fnum_list):
            if(self.nFile==1):
                filename = self.snapshot_file+self.snapshot_ext
            else:
                filename = self.snapshot_file+'.'+str(ifile)+'.hdf5'
            f = h5py.File(filename,'r')
            np_arr_this = (f['Header'].attrs[u'NumPart_ThisFile']).astype(np.int64)
            #select the ptype
            np_this=np_arr_this[self.ptype]
            out_this=f[prefix+suffix][:]

            if(ii==0):
                out=out_this
            elif(block in ['ids','potential']):
                out=np.append(out,out_this)
            else:
                out=np.row_stack([out,out_this])

        return out if block in ['ids','potential'] else out.T # notice transpose

        f = h5py.File(self.snapshot_file+self.snapshot_ext,'r')
        out_part = f[prefix+suffix][:]
        f.close()
        if self.nFile > 1:
            # NEEDS TESTING
            out = np.zeros(self.npart,dtype=out_part.dtype) if block in ['ids','potential'] else np.zeros((self.npart,3),dtype=out_part.dtype)
            np_this=out_part.shape[0]
            out[:np_this] = out_part 
            shift = np_this
            for i in range(1,self.nFile):
                filename = self.snapshot_file+'.'+str(i)+'.hdf5'
                f = h5py.File(filename,'r')
                np_arr_this = (f['Header'].attrs[u'NumPart_ThisFile']).astype(np.int64)
                #select the ptype
                np_this=np_arr_this[self.ptype]
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
        
        return out if block in ['ids','potential'] else out.T # notice transpose

    def compress_snapshot(self, subsamples,ref_snapshot=200,quant_write=['positions','ids','velocities','potentials'],optimized=True):
        """
        Compress the snapshot data, create subsamples, and save header information.
        
        Args:
        subsamples (list): List of subsample sizes to create.
        compressed_fileroot (str): Root name for the compressed files, defined in the main init
        
        Returns:
        None
        """
        import Nbody_Compression as NbodyCompress 
        if self.verbose:
            self.print_this(f'{ltime()} Begin Compressing snapshot data and saving header...', self.logfile)

        # Ensure header is read
        if not hasattr(self, 'scale'):
            self.read_snapshot_header()

        # Prepare the full path for compressed files
        compressed_path = self.sim_path +  os.path.join(self.sim_stem, f'r{self.real}', 'compressed')
        os.makedirs(compressed_path, exist_ok=True)
        full_fileroot = os.path.join(compressed_path, self.compressed_fileroot)
        

        # Compression related setting
        Ndata=self.npart
        Lbox=self.Lbox; Ngrid=int(2*Lbox); 
        vmax=5000.0; pmax=1e5
        full_count_bit_depth=12
        little_endian=True


        # Create JSON file with full snapshot information
        json_filename = f"{full_fileroot}_info.json"
        if(os.path.isfile(json_filename)):
            self.print_this(f'{ltime()} Snapshot information already exists: {json_filename}', self.logfile)
        else:
            compression_dic={'subsamples':subsamples,'Ngrid':Ngrid,
                    'Lbox':Lbox,'vmax':vmax,'pmax':pmax,'full_count_bit_depth':full_count_bit_depth}
            json_data=self.hdf5_to_json(json_filename,compression_dic)
            if self.verbose:
                self.print_this(f'{ltime()} Snapshot information saved to: {json_filename}', self.logfile)



        # Create subsamples
        subsample_indices = NbodyCompress.create_subsamples(Ndata, subsamples)
        self.print_this(f'{ltime()} Created Subsample indices',self.logfile)
        #for ii, indices in enumerate(subsample_indices):
        #    self.print_this(f'{ltime()} sub {subsamples[ii]}, {indices.size} {indices.min()} {indices.max()}',self.logfile)

       
        transform_quant={'positions':'pos','velocities':'vel','potentials':'potential','ids':'ids'}
        sort_indices_dic={}
        for ii, indices in enumerate(subsample_indices):
            sort_indices_dic[ii]=None
        for qq,quant in enumerate(quant_write): 
            # Read positions, velocities, and potential
            original_quant = self.read_block(transform_quant[quant])
            if(quant not in ['potentials']):
                original_quant= original_quant.T
            tmp_shape=original_quant.shape
            self.print_this(f'{ltime()} Loaded raw values of {quant}, array shape {tmp_shape}',self.logfile)


            # Compress and save each subsample
            for ii, indices in enumerate(subsample_indices):
                #file_prefix=f"{full_fileroot}_subsample{subsamples[ii]}"
                subsample_dir=f"{compressed_path}/subsample{subsamples[ii]}/"
                os.makedirs(subsample_dir, exist_ok=True)
                file_prefix=f"{subsample_dir}/{self.compressed_fileroot}_subsample{subsamples[ii]}"
                this_fname=NbodyCompress.compressed_filename(file_prefix, quant)
                if(os.path.isfile(this_fname)):
                    self.print_this(f'{ii} Compressed file already exists! {this_fname}', self.logfile)
                    continue
                
                    
                attribute_dic={'Ndata':indices.size,'L':Lbox,'Ngrid':Ngrid,'ref_snapshot':ref_snapshot,
                         'vmax':vmax,'pmax':pmax,'subsamp_percent':subsamples[ii],
                         'little_endian':little_endian,'byteorder':sys.byteorder}

                chunk_size=int(0.1*indices.size)
                if(chunk_size%2==1):
                    chunk_size=chunk_size+1

                if(quant=='positions'):
                    if(not optimized):
                        compressed_quant,full_counts_dic, sort_indices_dic[ii]=NbodyCompress.compress_nbody_data(original_quant[indices],
                          Lbox, Ngrid, vmax, pmax,quant=quant,sort_indices=sort_indices_dic[ii],
                          full_count_bit_depth=full_count_bit_depth,little_endian=little_endian,npart=self.npart)
                    else:
                        self.print_this(f'Using optimized version of compression for positions chunk_size={chunk_size}',self.logfile)
                        compressed_quant,full_counts_dic, sort_indices_dic[ii]=NbodyCompress.compress_nbody_data_optimized(original_quant[indices],
                          Lbox, Ngrid, vmax, pmax,quant=quant,sort_indices=sort_indices_dic[ii],
                          full_count_bit_depth=full_count_bit_depth,little_endian=little_endian,npart=self.npart,chunk_size=chunk_size)
                    attribute_dic['full_count_bit_depth']=full_count_bit_depth
                    attribute_dic['full_count_dtype']= '%s'%full_counts_dic['dtype']
                else:
                    if(not optimized):
                        compressed_quant=NbodyCompress.compress_nbody_data(original_quant[indices],
                          Lbox, Ngrid, vmax, pmax,quant=quant,sort_indices=sort_indices_dic[ii],
                          full_count_bit_depth=full_count_bit_depth,little_endian=little_endian,npart=self.npart)
                    else:
                        self.print_this(f'Using optimized version of compression for {quant} chunk_size={chunk_size}',self.logfile)
                        compressed_quant=NbodyCompress.compress_nbody_data_optimized(original_quant[indices],
                          Lbox, Ngrid, vmax, pmax,quant=quant,sort_indices=sort_indices_dic[ii],
                          full_count_bit_depth=full_count_bit_depth,little_endian=little_endian,npart=self.npart,chunk_size=chunk_size)
                    full_counts_dic=None

                NbodyCompress.save_compressed_data(file_prefix, compressed_quant, attribute_dic,
                                              full_counts_dic=full_counts_dic, quant=quant,verbose=False)

                if self.verbose:
                    this_fname=NbodyCompress.compressed_filename(file_prefix, quant)
                    self.print_this(f'{ltime()} sub-{ii} Compression complete {this_fname}', self.logfile)


        return
        #end of compression function


    def load_compressed(self, subsamples,load_quant=['positions','velocities','potentials','ids']):
        """
        Load and decompress snapshot data.
        
        Args:
        compressed_fileroot (str): Root name of the compressed files, defined in the init
        subsamples list of (int): The subsample size to load.
        
        Returns:
        tuple: (positions, velocities, potential)
        """
        import Nbody_Compression as NbodyCompress 
        if self.verbose:
            self.print_this(f'Loading compressed data for subsamples {subsamples}...', self.logfile)


        # Prepare the full path for compressed files
        compressed_path = self.sim_path + os.path.join(self.sim_stem, f'r{self.real}', 'compressed')
        full_fileroot = os.path.join(compressed_path, self.compressed_fileroot)

        decomp_dic={}
        # Call your decompression function
        for qq,quant in enumerate(load_quant):
            for ii, indices in enumerate(subsamples):
                subsample_dir=f"{compressed_path}/subsample{subsamples[ii]}/"
                this_root=f"{subsample_dir}/{self.compressed_fileroot}_subsample{subsamples[ii]}"
                if(qq==0 and ii==0):
                    attribute_dic = NbodyCompress.load_compressed_data(this_root,load_quant=['attributes'])
                tmp_dic = NbodyCompress.decompress_nbody_data(this_root,subsamples[ii], attribute_dic,load_quant=[quant])
                if(ii==0):
                    decomp_dic[quant]= tmp_dic[quant]
                else:
                    if(quant in ['ids','potentials']):
                        decomp_dic[quant]=np.append(decomp_dic[quant],tmp_dic[quant])
                    else:
                        decomp_dic[quant]=np.row_stack([decomp_dic[quant],tmp_dic[quant]])


        if self.verbose:
            self.print_this('Decompression complete.', self.logfile)

        return decomp_dic


    def json_serializable(self, obj):
        """Helper function to make an object JSON serializable."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        elif isinstance(obj, np.bytes_):
            return obj.decode('utf-8')
        else:
            return str(obj)

    def hdf5_to_json(self, output_filename,compression_dic):
        """
        Read Config, Header, and Parameters groups from the HDF5 file and write them to a JSON file.
        
        Args:
        output_filename (str): Name of the output JSON file.
        
        Returns:
        None
        """
        if self.verbose:
            self.print_this(f'Reading HDF5 groups and writing to JSON: {output_filename}', self.logfile)

        json_data = {}

        with h5py.File(self.snapshot_file+self.snapshot_ext, 'r') as f:
            for group_name in ['Config', 'Header', 'Parameters']:
                if group_name in f:
                    group = f[group_name]
                    json_data[group_name] = {}
                    
                    # Read attributes
                    for attr_name, attr_value in group.attrs.items():
                        json_data[group_name][attr_name] = self.json_serializable(attr_value)
                    
                    # Read datasets if any
                    for dset_name, dset in group.items():
                        if isinstance(dset, h5py.Dataset):
                            data = dset[:]
                            json_data[group_name][dset_name] = self.json_serializable(data)
        #Also add the subsamples in the header
        json_data['Header']['subsamples']= subsamples
        json_data['Header']['compression_dic']= compression_dic
        # Write to JSON file
        with open(output_filename, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        if self.verbose:
            self.print_this(f'JSON file created: {output_filename}', self.logfile)

        return json_data

    def read_compressed_header(self):
        """
        Read the header information from the compressed snapshot.
        
        Args:
        compressed_fileroot (str): Root name of the compressed files.
        
        Returns:
        dict: Header information
        """
        compressed_path = self.sim_path + os.path.join(self.sim_stem, f'r{self.real}', 'compressed')
        header_file = os.path.join(compressed_path, f"{self.compressed_fileroot}_info.json")
        
        if not os.path.exists(header_file):
            raise FileNotFoundError(f"Header file not found: {header_file}")

        with open(header_file, 'r') as f:
            header_info = json.load(f)

        if self.verbose:
            self.print_this(f'Header information loaded from: {header_file}', self.logfile)

        return header_info 
    ###############################################

########################################################
# Reader for (ROCKSTAR) halo catalog. 
########################################################
class HaloReader(SnapshotReader):
    """ Reader for (ROCKSTAR) halo catalog. """
    ###############################################
    def __init__(self,sim_stem='scm1024',real=1,snap=200,logfile=None,verbose=True,read_header=True):

        SnapshotReader.__init__(self,sim_stem=sim_stem,real=real,snap=snap,logfile=logfile,verbose=verbose,read_header=read_header)

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
       
        #in the file header this ID is called haloID, it will be good to be consistent, the fits file uses the file header
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

        #This and above vadatatype seems to be same
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
    def convert_halos_fits(self, list_output_type=['basic','extended','vahc']):
        '''This reads the raw halos and convert them in fits.gz
        we write the .trees in two files called basic and extended and vahc in a seperate file
        '''

        #example of how to execute this function to convert to halo catalog

        #input directory for halo catalog
        #self.halocat_stem = sr.sim_stem + '/r'+str(sr.real)+'/' + 'out_' + str(sr.snap)
        indir=self.halo_path+self.sim_stem+ '/r'+str(self.real)+'/'
        #root name for the halo catalog assuming in input directory .trees and .vahc files
        rootin='out_' + str(self.snap)

        #output directory: set this to None if want to use input directory for output
        outdir=indir+'compressed/'
        #outdir=None

        fits_exists=False
        for oo,out_type in enumerate(list_output_type):
            outfile='%s%s_%s.fits.gz'%(outdir,rootin,out_type)
            if(os.path.isfile(outfile)):
                print('!Warning: Fits file exists',out_type,outfile)
                fits_exists=True


        if(fits_exists):
            print('Since some of the fits file exists not doing anything')
            print('Please clean the output folder and then re-run, if needed')
            return
        else:
            #list of output to be written
            #basic: file contain only the most used information about the halos
            #extended: file contains all information in .trees except what is included in basic
            #vahc: write the tidal information from vahc
            #To do: include the tidal ranks
            #you can chose any combination of output_type and corresponding files will be written
            Cfits.convert_fits(indir=indir,rootin=rootin,outdir=outdir,list_output_type=list_output_type)
            return
    ###############################################

    
    ###############################################
    def prep_halos(self,va=False,ext=False,QE=0.5,massdef='mvir',Npmin=100,keep_subhalos=False,use_fits=False, sorthalos=True, sortdef=None):
        """ Reads halo (+ vahc) (+ext propperties only for fits) catalogs for given realisation and snapshot. 
             Cleans catalog by selecting relaxed objects in range max(0,1-QE) <= 2T/|U| <= 1+QE 
             where QE > 0 (default QE=0.5; Bett+07). Use QE=None to skip cleaning.
             Selects objects with at least Npmin particles for given massdef.
             Optionally removes subhalos (set keep_subhalos=False).
             Returns array of shape (Ndata,3) for positions (Mpc/h); structured array(s) for full halo properties (+ vahc).
             Halos will be sorted by (increasing) massdef.
             use_fits: True then load information from the .fits.gz file otherwise load from the .tree/.vahc files
             sorthalos: if True, sorts the halos based on sortdef. otherwise keeps the ordering like the input file. Default is True
             sortdef: the property of the halo based on which they are sorted. If none, taken to be massdef
        """
        if sortdef is None:
            sortdef=massdef

        if(use_fits):
            return self.prep_halos_fits(va=va,ext=ext,QE=QE,massdef=massdef,Npmin=Npmin,keep_subhalos=keep_subhalos, sorthalos=sorthalos, sortdef=sortdef)

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

        
        if (sorthalos is True):
            if self.verbose:
                self.print_this("... ... sorting by "+sortdef,self.logfile)
            sorter = halos[sortdef].argsort()
            halos = halos[sorter]
            hpos = hpos[sorter]
            if va:
                vahc = vahc[sorter]

            del sorter
        gc.collect()

        return (hpos,halos,vahc) if va else (hpos,halos)
    ###############################################
    def fits_basic_exits(self):
        '''Tests if the basic fits file already exists of not'''
        halo_basic=self.halo_path + self.halocat_stem + '_basic.fits.gz'
        return os.path.isfile(halo_basic)


    ###############################################
    #prepare halos with fits
    def prep_halos_fits(self,va=False,ext=False,QE=0.5,massdef='mvir',Npmin=100,keep_subhalos=False, sorthalos=True, sortdef=None):
            """ Reads halo (+ vahc using va=True) (+extended properties using ext=True) catalogs for given realisation and snapshot. 
                 Cleans catalog by selecting relaxed objects in range max(0,1-QE) <= 2T/|U| <= 1+QE 
                 where QE > 0 (default QE=0.5; Bett+07). Use QE=None to skip cleaning.
                 Selects objects with at least Npmin particles for given massdef.
                 Optionally removes subhalos (set keep_subhalos=False).
                 Returns array of shape (Ndata,3) for positions (Mpc/h); structured array(s) for full halo properties (+ vahc).
                 Halos will be sorted by (increasing) massdef.
                 sorthalos: sorts halos according to massdef if set to True. Default is True
                 sortdef: the property of the halo based on which they are sorted. If none, taken to be massdef

            """

            #make sure mass definition follow the same convention as in the header/fits
            if(massdef[0]=='m'):
                massdef='M'+massdef[1:]

            if self.verbose:
                self.print_this("... preparing halo data",self.logfile)

            #open the fits handle for basic and extended
            halo_basic=self.halo_path + self.halocat_stem + '_basic.fits.gz'
            fbasic=F.FITS(halo_basic)

            Nhalos_all = fbasic[1][massdef][:].size
            mmin = self.mpart*Npmin
            cond_clean = (fbasic[1][massdef][:] >= mmin)
            if QE is not None:
                TbyU_max = 0.5*(1+QE)
                TbyU_min = np.max([0.0,0.5*(1-QE)])
                cond_clean = cond_clean & ((fbasic[1]['T/|U|'][:] < TbyU_max) & (TbyU_min < fbasic[1]['T/|U|'][:]))
            if not keep_subhalos:
                cond_clean = cond_clean & (fbasic[1]['pid'][:] == -1)
            if self.verbose:
                self.print_this("... ... using mass definition " + massdef + " > {0:.3e} Msun/h".format(mmin),self.logfile)
                if QE is not None:
                    self.print_this("... ... only relaxed objects retained with {0:.2f} < 2T/|U| < {1:.2f}"
                                    .format(2*TbyU_min,2*TbyU_max),self.logfile)
                else:
                    self.print_this("... ... skipping relaxation cleaning",self.logfile)
                if not keep_subhalos:
                    self.print_this("... ... discarding subhalos",self.logfile)

            index_clean=np.where(cond_clean)[0]
            #if ext is true then load all the record in the extended file as well
            if ext:
                ext_fits=self.halo_path + self.halocat_stem + '_extended.fits.gz'
                with F.FITS(ext_fits) as fext:
                    halos = Cfits.fits_to_record([fbasic,fext],index_clean,list_sel_col=None)
            else:
                halos=Cfits.fits_to_record(fbasic,index_clean,list_sel_col=None)

            if self.verbose:
                self.print_this("... ... kept {0:d} of {1:d} objects in catalog".format(halos.size,Nhalos_all),self.logfile)


            hpos = np.array([fbasic[1]['x'][index_clean],fbasic[1]['y'][index_clean],fbasic[1]['z'][index_clean]])

            # if (self.RSD) & (halos.size > 0):
            #     if self.verbose:
            #         self.print_this("... ... applying redshift space displacement",self.logfile)
            #     hpos[2] = hpos[2] + 0.01*halos['vz']*(1+self.redshift)/self.EHub(self.redshift)
            if va:
                vahc_fits=self.halo_path + self.halocat_stem + '_vahc.fits.gz'
                with F.FITS(vahc_fits) as fvahc:
                    vahc = Cfits.fits_to_record(fvahc,index_clean,list_sel_col=None)


            hpos = hpos % self.Lbox
            hpos = hpos.T  # shape (Ntrc,3)

            del cond_clean
            gc.collect()

            if (sorthalos is True):
                if self.verbose:
                    self.print_this("... ... sorting by "+sortdef,self.logfile)
                sorter = halos[sortdef].argsort()
                halos = halos[sorter]
                hpos = hpos[sorter]
                if va:
                    vahc = vahc[sorter]

                del sorter
            gc.collect()
            fbasic.close()

            return (hpos,halos,vahc) if va else (hpos,halos)
    ###############################################


    ###############################################
    def calc_Npmin_default(self,grid):
        Npmin = 200*self.NENCL_2R200B*(self.npart/1024**3.)*(512./grid)**3
        return Npmin
    ###############################################
