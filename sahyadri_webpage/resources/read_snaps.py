import numpy as np
import h5py
import time
import sys, os


def decompress_12bit(compressed, max_value, output_size, allow_negative=True, little_endian=True,input_dtype='float'):
    '''Deccompresses values from 8 bit unint into 15 bit uint combine 1.5 element inot a 12 bit integer
    if input_dtype='integer is given then assumes integer and max value is not used for
    this case if allow_negative=True then the rand is -2047 - 2047 otherwise 0-4095
    '''
    # Allocate output array
    unpacked = np.zeros(output_size, dtype=np.uint16)
    # Unpack 12-bit values from 8-bit array
    if little_endian:
        unpacked[0::2] = (compressed[0::3].astype(np.uint16) |
                          ((compressed[1::3].astype(np.uint16) & 0x0F) << 8))  # Reconstruct even indices
        unpacked[1::2] = (((compressed[1::3].astype(np.uint16) & 0xF0) >> 4) |
                          (compressed[2::3].astype(np.uint16) << 4))  # Reconstruct odd indices
    else:  # big endian
        unpacked[0::2] = (compressed[0::3].astype(np.uint16) << 4 |
                          ((compressed[1::3].astype(np.uint16) & 0xF0) >> 4))  # Reconstruct even indices
        unpacked[1::2] = ((compressed[1::3].astype(np.uint16) & 0x0F) << 8 |
                          compressed[2::3].astype(np.uint16))  # Reconstruct odd indices

    if(input_dtype=='integer'):
        if allow_negative:
            # Scale back to [-max_value, max_value]
            return unpacked - 2047 
        else:
            # Scale back to [0, max_value]
            return unpacked 
    else:
        if allow_negative:
            # Scale back to [-max_value, max_value]
            return (unpacked.astype(float) / 2047- 1) * max_value
        else:
            # Scale back to [0, max_value]
            return unpacked.astype(float) / 4095 * max_value



def decompress_integer_array(packed, overflow_indices, overflow_values, original_dtype, bit_depth, original_size, little_endian=True):
    if bit_depth in (8, 16, 32, 64):
        # Use native types for these bit depths
        unpacked = packed.astype(original_dtype)
    else:
        # Decompress with the default uint16 type
        unpacked = decompress_12bit(packed, None, original_size, allow_negative=False, little_endian=little_endian, input_dtype='integer')
        #
        # Check if any overflow values exceed uint16 max (65535)
        # or if the original dtype can handle larger values
        if(overflow_values.size==0):
            msg='No overflow values'
        elif (np.any(overflow_values >= 65536) or
            np.dtype(original_dtype).itemsize > np.dtype(np.uint16).itemsize):
            # Convert to a type that can handle larger values before restoring overflows
            if np.issubdtype(original_dtype, np.integer):
                # If original type was integer, use appropriate integer type
                if np.max(overflow_values) <= np.iinfo(np.uint32).max:
                    unpacked = unpacked.astype(np.uint32)
                else:
                    unpacked = unpacked.astype(np.uint64)
            else:
                # If original type was float or other, use the original type
                unpacked = unpacked.astype(original_dtype)
    #
    # Restore overflow values
    unpacked[overflow_indices] = overflow_values
    #
    # Final conversion to original dtype if needed
    if unpacked.dtype != original_dtype:
        unpacked = unpacked.astype(original_dtype)
    #
    return unpacked


def grid_index_to_3d(index, Ngrid):
    z = index // (Ngrid * Ngrid)
    y = (index % (Ngrid * Ngrid)) // Ngrid
    x = index % Ngrid
    return np.column_stack([x, y, z])


def decompress_nbody_data(file_root,subsample,attribute_dic,load_quant=[]):#compressed_dic,attribute_dic):
    little_endian=attribute_dic['little_endian']
    assert sys.byteorder==attribute_dic['byteorder']
    res_dic={}

    Ngrid=attribute_dic['Ngrid']; L=attribute_dic['L']

    grid_size = L/Ngrid
    
  
    if('positions' in load_quant):
        compressed_dic = load_compressed_data(file_root,load_quant=['positions'])
        #decompress full counts
        if (attribute_dic['full_count_dtype']=='int32'):    
            dtype_full_count=np.int32
        elif (attribute_dic['full_count_dtype']=='int64'):    
            dtype_full_count=np.int64

        Ngrid3=np.power(attribute_dic['Ngrid'],3)
        full_counts=decompress_integer_array(compressed_dic['full_counts_packed'],
            compressed_dic['full_counts_overflow_indices'],compressed_dic['full_counts_overflow_values'],
            dtype_full_count,attribute_dic['full_count_bit_depth'],Ngrid3)

        # clean up memory used for full count
        compressed_dic['full_counts_packed']=None
        compressed_dic['full_counts_overflow_indices']=None
        compressed_dic['full_counts_overflow_values']=None
 
        # Reconstruct single indices
        single_indices = np.repeat(np.arange(Ngrid**3), full_counts)
    
        # Convert single indices back to 3D grid indices
        grid_indices = grid_index_to_3d(single_indices, attribute_dic['Ngrid'])
    
        single_indices=None

        # Decompress positions
        normalized_diffs = compressed_dic['positions'].astype(float) / 255
        #transformed_diffs = compressed_positions.astype(float) / 255
        #normalized_diffs = inverse_nonlinear_transform(transformed_diffs)
        position_diffs = normalized_diffs * grid_size
        res_dic['positions'] = grid_indices * grid_size + position_diffs
    
        normalized_diffs=None
        position_diffs=None
        grid_indices=None
        compressed_dic['positions']=None
        compressed_dic=None


    if('velocities' in load_quant):
        compressed_dic = load_compressed_data(file_root, load_quant=['velocities'])
        output_size=int(compressed_dic['velocities'].shape[0]/ 1.5)
        vmax=attribute_dic['vmax']
        # Decompress velocities
        dvx = decompress_12bit(compressed_dic['velocities'][:,0],vmax,output_size, allow_negative=True, little_endian=little_endian)
        dvy = decompress_12bit(compressed_dic['velocities'][:,1],vmax,output_size, allow_negative=True, little_endian=little_endian)
        dvz = decompress_12bit(compressed_dic['velocities'][:,2],vmax,output_size, allow_negative=True, little_endian=little_endian)
        res_dic['velocities']=np.column_stack([dvx,dvy,dvz])
        #decompressed_velocities = decompress_12bit(compressed_velocities, vmax)
        compressed_dic['velocities']=None
    
    
    # Decompress potential
    if('potentials' in load_quant):
        compressed_dic = load_compressed_data(file_root, load_quant=['potentials'])
        output_size=int(compressed_dic['potentials'].shape[0]/ 1.5)
        res_dic['potentials'] = decompress_12bit(compressed_dic['potentials'], attribute_dic['pmax'],output_size, 
               allow_negative=True, little_endian=little_endian)
        #decompressed_potential = decompress_12bit(compressed_potential, pmax)
        compressed_dic['potentials']=None    
    
    # Decompress ids
    if('ids' in load_quant):
        compressed_dic = load_compressed_data(file_root, load_quant=['ids'])
        #output_size=int(compressed_dic['ids'].shape[0]/ 1.5)
        #res_dic['potentials'] = decompress_12bit(compressed_dic['potentials'], attribute_dic['pmax'],output_size, 
        #       allow_negative=True, little_endian=little_endian)
        #decompressed_potential = decompress_12bit(compressed_potential, pmax)
        res_dic['ids']=compressed_dic['ids']

    return res_dic

def compressed_filename(filename_prefix, quant):
    return f"{filename_prefix}_{quant}.h5"

def print_this(print_string,logfile,overwrite=False):
    """ Convenience function for printing to logfile or stdout."""
    if logfile is not None:
        writelog(logfile,print_string+'\n',overwrite=overwrite)
    else:
        print(print_string)
    return


def load_compressed_data(filename_prefix, load_quant=['positions','velocities','potentials','attributes','ids']):

    fname_dic={'positions':compressed_filename(filename_prefix,'positions'),
                'velocities':compressed_filename(filename_prefix,'velocities'),
                'potentials':compressed_filename(filename_prefix,'potential'),
                'ids':compressed_filename(filename_prefix,'ids'),
              }

    out_dic={}

    if('attributes' in load_quant):
        with h5py.File(fname_dic['positions'],'r') as f:
            for tt,tkey in enumerate(f.attrs.keys()):
                out_dic[tkey]=f.attrs[tkey]
    # Load positions
    if('positions' in load_quant):
        with h5py.File(fname_dic['positions'],'r') as f:
            out_dic['full_counts_packed'] = f['full_counts'][:]
            out_dic['full_counts_overflow_indices'] = f['full_counts_overflow_indices'][:]
            out_dic['full_counts_overflow_values'] = f['full_counts_overflow_values'][:]
            out_dic['positions'] = f['compressed_positions'][:]

    
    if('velocities' in load_quant):
        # Load velocities
        with h5py.File(fname_dic['velocities'],'r') as f:
            out_dic['velocities'] = f['compressed_velocities'][:]
            #for tt,tkey in enumerate(f.attrs.keys()):
            #    if(attribute_dic[tkey]!=f.attrs[tkey]):
            #        print('Warning: Attribute mismatch: %s'%(tkey))
            #        print('In Positions file(%s): ',tkey,attribute[tkey])
            #        print('In Velocities file: ',tkey,f.attrs[tkey])

    
    if('potentials' in load_quant):
        # Load potential
        with h5py.File(fname_dic['potentials'],'r') as f:
            out_dic['potentials'] = f['compressed_potential'][:]

    if('ids' in load_quant):
        # Load ids
        with h5py.File(fname_dic['ids'],'r') as f:
            out_dic['ids'] = f['compressed_ids'][:]

    
    return out_dic

#############################################################################################################
def load_compressed(basedir="./", sim_stem="sahyadri/default2048/", real=1, snap=100, subsamples=[1],load_quant=['positions','velocities','potentials','ids'], verbose=True):
        """
        Load and decompress snapshot data.

        Args:
        basedir: directory into which Sahyadri data is downloaded (default "./" )
        sim_stem: simulation name (default "sahyadri/default2048/" )
        real: simulation realization (default 1)
        subsamples list of (int): The subsample size to load
        load_quant: list of quantities to load

        Returns:
        Dictionary with keys load_quant
        """
        if verbose:
            print(f'Loading compressed data for subsamples {subsamples}...')


        # Prepare the full path for compressed files
        sim_path = basedir + "sims/"
        compressed_path = sim_path + os.path.join(sim_stem, f'r{real}', 'compressed')
        compressed_fileroot="comp_snapshot_%d"%snap

        decomp_dic={}
        # Call your decompression function
        for qq,quant in enumerate(load_quant):
            for ii, indices in enumerate(subsamples):
                subsample_dir=f"{compressed_path}/subsample{subsamples[ii]}/"
                this_root=f"{subsample_dir}/{compressed_fileroot}_subsample{subsamples[ii]}"
                if(qq==0 and ii==0):
                    attribute_dic = load_compressed_data(this_root,load_quant=['attributes'])
                tmp_dic = decompress_nbody_data(this_root,subsamples[ii], attribute_dic,load_quant=[quant])
                if(ii==0):
                    decomp_dic[quant]= tmp_dic[quant]
                else:
                    if(quant in ['ids','potentials']):
                        decomp_dic[quant]=np.append(decomp_dic[quant],tmp_dic[quant])
                    else:
                        decomp_dic[quant]=np.row_stack([decomp_dic[quant],tmp_dic[quant]])


        if verbose:
            print('Decompression complete.')

        return decomp_dic


