import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
import compression12bit as c12b
#import compress_integer_array as c12int
import sys

def nonlinear_transform(x, alpha=0.99):
    return np.tanh(x / alpha) / np.tanh(1 / alpha)

def inverse_nonlinear_transform(y, alpha=0.99):
    return alpha * np.arctanh(y * np.tanh(1 / alpha))

def grid_index_to_3d(index, Ngrid):
    z = index // (Ngrid * Ngrid)
    y = (index % (Ngrid * Ngrid)) // Ngrid
    x = index % Ngrid
    return np.column_stack([x, y, z])

def compress_12bit(data, max_value):
    scaled = np.clip(data / max_value, -1, 1)
    return np.clip(np.round((scaled + 1) * 2047), 0, 4095).astype(np.uint16)

def decompress_12bit(compressed, max_value):
    return (compressed.astype(float) / 2047 - 1) * max_value


def create_subsamples(N, subsamples):
    """
    Create random subsamples that add up to 100% of the data, 
    with an even number of elements in each subsample.
    
    Args:
    N (int): Total number of elements (must be even)
    subsamples (list): List of percentages for each subsample
    
    Returns:
    list: List of arrays containing indices for each subsample
    """
    if sum(subsamples) != 100:
        raise ValueError("Subsamples must add up to 100")
    
    if( len(np.unique(subsamples))!=len(subsamples)):
        raise ValueError(f"Subsamples must have unique values and add up to 100: {subsamples}")

    if N % 2 != 0:
        raise ValueError("N must be even")

    indices = np.arange(N)
    np.random.shuffle(indices)

    subsample_indices = []
    start = 0
    remaining_elements = N

    for i, sample_percent in enumerate(subsamples):
        if i == len(subsamples) - 1:
            # Last subsample gets all remaining elements
            end = N
        else:
            # Calculate the number of elements for this subsample
            elements = int(round(N * sample_percent / 100))
            # Ensure even number of elements
            elements = elements if elements % 2 == 0 else elements + 1
            # Adjust for remaining elements
            elements = min(elements, remaining_elements)
            end = start + elements

        subsample_indices.append(indices[start:end])
        #print(f"Subsample {i+1}: {sample_percent}%, Elements: {end-start} {start} {end}")

        start = end
        remaining_elements -= (end - start)

    return subsample_indices

def Get_sorted_indices(positions,L, Ngrid):
    grid_size = L / Ngrid
    grid_indices = np.floor(positions / grid_size).astype(int)

    # Convert 3D grid indices to single index
    single_indices = grid_indices[:, 0] + Ngrid * (grid_indices[:, 1] + Ngrid * grid_indices[:, 2])

    # Sort data based on single index
    sort_indices = np.argsort(single_indices)

    sorted_single_indices = single_indices[sort_indices]

    return sorted_single_indices, sort_indices


def compress_nbody_data(original_quant, L, Ngrid, vmax, pmax,quant='positions',sort_indices=None,full_count_bit_depth=12,little_endian=True):
    if (quant in ['positions', 'velocities'] and original_quant.shape[1]!= 3):
        raise ValueError("Input positions and velocities should have shape (Ndata, 3)")
   
    if(sort_indices is None and quant!='positions'):
        raise ValueError('You must provide the sort_indices if input quant is not positions')
    
    if(sort_indices is not None):
        if(sort_indices.size != original_quant.shape[0]): 
            raise ValueError('The size of sort_indices and input array rows must match')
    grid_size = L / Ngrid
    
    if(sort_indices is None and quant=='positions'):
        for ii in range(0,3): #To take care of the co-ordinates exactly Lbox
            original_quant[:,ii]=original_quant[:,ii]%L
 
        sorted_single_indices, sort_indices= Get_sorted_indices(original_quant,L, Ngrid)
        # Sort data based on single index
        sorted_positions = original_quant[sort_indices]

        # Count objects in each grid cell
        unique_indices, index_bins = np.unique(sorted_single_indices, return_index=True)
        counts = np.diff(np.append(index_bins, len(sorted_single_indices)))
    
        # Create full counts array including empty cells
        full_counts = np.zeros(Ngrid**3, dtype=int)
        full_counts[unique_indices] = counts
        #compress the full counts
        #full_counts=c12int.pack_12bit(full_counts)
        full_counts_dic={}
        full_counts_dic['packed'], full_counts_dic['overflow_indices'], full_counts_dic['overflow_values'],full_counts_dic['dtype'],full_counts_dic['bit_depth']=c12b.compress_integer_array(full_counts, full_count_bit_depth)
    
        # Compress positions
        grid_corners = grid_index_to_3d(sorted_single_indices, Ngrid) * grid_size
        position_diffs = sorted_positions - grid_corners
        normalized_diffs = position_diffs / grid_size
        compressed_positions = np.clip(np.round(normalized_diffs * 255), 0, 255).astype(np.uint8)
        #transformed_diffs = nonlinear_transform(normalized_diffs)
        return compressed_positions, full_counts_dic, sort_indices
    elif(quant=='velocities'): 
        # Sort data based on single index
        sorted_velocities = original_quant[sort_indices]
        # Compress velocities
        cvx, _ = c12b.compress_12bit(sorted_velocities[:,0], vmax, allow_negative=True, little_endian=little_endian)
        cvy, _ = c12b.compress_12bit(sorted_velocities[:,1], vmax, allow_negative=True, little_endian=little_endian)
        cvz, _ = c12b.compress_12bit(sorted_velocities[:,2], vmax, allow_negative=True, little_endian=little_endian)
        compressed_velocities=np.column_stack([cvx,cvy,cvz])
        #compressed_velocities = compress_12bit(sorted_velocities, vmax)
        return compressed_velocities
    elif(quant=='potentials'):
        sorted_potential = original_quant[sort_indices]
        # Compress potential
        compressed_potential, _ = c12b.compress_12bit(sorted_potential, pmax, allow_negative=True, little_endian=little_endian)
        #compressed_potential = compress_12bit(sorted_potential, pmax)
        return compressed_potential
    elif(quant=='ids'):
        sorted_ids = original_quant[sort_indices]
        if(sorted_ids.size<np.power(1024,3)):
            ids_bit_depth=32
        else:
            ids_bit_depth=64
        packed_ids, overflow_indices, overflow_values,ids_dtype,ids_bit_depth=c12b.compress_integer_array(sorted_ids, ids_bit_depth)
        return packed_ids
    
 

def decompress_nbody_data(file_root,subsample,attribute_dic,load_quant=[]):#compressed_dic,attribute_dic):
    little_endian=attribute_dic['little_endian']
    assert sys.byteorder==attribute_dic['byteorder']
    res_dic={}

    Ngrid=attribute_dic['Ngrid']; L=attribute_dic['L']

    grid_size = L/Ngrid
    
  
    if('positions' in load_quant):
        compressed_dic = load_compressed_data(file_root,load_quant=['positions'])
        #decompress full counts
        #full_counts=c12int.unpack_12bit(compressed_dic['full_counts'])
        if (attribute_dic['full_count_dtype']=='int32'):    
            dtype_full_count=np.int32
        elif (attribute_dic['full_count_dtype']=='int64'):    
            dtype_full_count=np.int64

        Ngrid3=np.power(attribute_dic['Ngrid'],3)
        full_counts=c12b.decompress_integer_array(compressed_dic['full_counts_packed'],
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
        dvx = c12b.decompress_12bit(compressed_dic['velocities'][:,0],vmax,output_size, allow_negative=True, little_endian=little_endian)
        dvy = c12b.decompress_12bit(compressed_dic['velocities'][:,1],vmax,output_size, allow_negative=True, little_endian=little_endian)
        dvz = c12b.decompress_12bit(compressed_dic['velocities'][:,2],vmax,output_size, allow_negative=True, little_endian=little_endian)
        res_dic['velocities']=np.column_stack([dvx,dvy,dvz])
        #decompressed_velocities = decompress_12bit(compressed_velocities, vmax)
        compressed_dic['velocities']=None
    
    
    # Decompress potential
    if('potentials' in load_quant):
        compressed_dic = load_compressed_data(file_root, load_quant=['potentials'])
        output_size=int(compressed_dic['potentials'].shape[0]/ 1.5)
        res_dic['potentials'] = c12b.decompress_12bit(compressed_dic['potentials'], attribute_dic['pmax'],output_size, 
               allow_negative=True, little_endian=little_endian)
        #decompressed_potential = decompress_12bit(compressed_potential, pmax)
        compressed_dic['potentials']=None    
    
    # Decompress ids
    if('ids' in load_quant):
        compressed_dic = load_compressed_data(file_root, load_quant=['ids'])
        #output_size=int(compressed_dic['ids'].shape[0]/ 1.5)
        #res_dic['potentials'] = c12b.decompress_12bit(compressed_dic['potentials'], attribute_dic['pmax'],output_size, 
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


def save_compressed_data(filename_prefix, compressed_data, attribute_dic, full_counts_dic=None, quant='positions',verbose=False):
   
    if(full_counts_dic is None and quant=='positions'):
        raise ValueError("full_counts_dic must not be None for positions to be saved")

 
    if(quant=='positions'):# Save positions
        with h5py.File(compressed_filename(filename_prefix,'positions'),'w') as f:
            f.create_dataset('compressed_positions', data=compressed_data, compression="gzip")
            #save three componenet of full count
            f.create_dataset('full_counts', data=full_counts_dic['packed'], compression="gzip")
            f.create_dataset('full_counts_overflow_indices', data=full_counts_dic['overflow_indices'], compression="gzip")
            f.create_dataset('full_counts_overflow_values', data=full_counts_dic['overflow_values'], compression="gzip")
            for tt,tkey in enumerate(attribute_dic.keys()):
                f.attrs[tkey]=attribute_dic[tkey]
            #f.attrs['L'] = L
            #f.attrs['Ngrid'] = Ngrid

    if(quant=='velocities'):# Save positions
        # Save velocities
        with h5py.File(compressed_filename(filename_prefix,'velocities'),'w') as f:
            f.create_dataset('compressed_velocities', data=compressed_data, compression="gzip")
            for tt,tkey in enumerate(attribute_dic.keys()):
                f.attrs[tkey]=attribute_dic[tkey]
            #f.attrs['vmax'] = vmax
    
    if(quant=='potentials'):# Save positions
        # Save potential
        with h5py.File(compressed_filename(filename_prefix,'potential'),'w') as f:
            f.create_dataset('compressed_potential', data=compressed_data, compression="gzip")
            for tt,tkey in enumerate(attribute_dic.keys()):
                f.attrs[tkey]=attribute_dic[tkey]
            #f.attrs['pmax'] = pmax
    
    if(quant=='ids'):# Save positions
        # Save potential
        with h5py.File(compressed_filename(filename_prefix,'ids'),'w') as f:
            f.create_dataset('compressed_ids', data=compressed_data, compression="gzip")
            for tt,tkey in enumerate(attribute_dic.keys()):
                f.attrs[tkey]=attribute_dic[tkey]
            #f.attrs['pmax'] = pmax

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

def test_compression(Ndata=100000, L=100.0, Ngrid=32, vmax=6000.0, pmax=1e5, subsamples=[1,4,5,20,30,40],
      full_count_bit_depth=12,little_endian=True):
    # Generate random data
    positions = np.random.rand(Ndata, 3) * L
    velocities = (np.random.rand(Ndata, 3) * 2 - 1) * vmax
    potential = (np.random.rand(Ndata) * 2 - 1) * pmax
    


    # Create subsamples
    subsample_indices = create_subsamples(Ndata, subsamples)

    # Compress and save each subsample
    for i, indices in enumerate(subsample_indices):
        subsample_positions = positions[indices]
        subsample_velocities = velocities[indices]
        subsample_potential = potential[indices]

        compressed_positions, compressed_velocities, compressed_potential, full_counts_dic = compress_nbody_data(
            subsample_positions, subsample_velocities, subsample_potential, L, Ngrid, vmax, pmax,
            full_count_bit_depth=full_count_bit_depth,little_endian=little_endian
        )

        attribute_dic={'Ndata':indices.size,'L':L,'Ngrid':Ngrid,'vmax':vmax,'pmax':pmax,'subsamp_percent':subsamples[i],
                 'full_count_bit_depth':full_count_bit_depth,'full_count_dtype': '%s'%full_counts_dic['dtype'],
                 'little_endian':little_endian,'byteorder':sys.byteorder}
        print('oring: packed',full_counts_dic['packed'][:10])
        print('oring: ov ind',full_counts_dic['overflow_indices'][:10])
        print('oring: ov value',full_counts_dic['overflow_values'][:10])
        save_compressed_data(f"compressed_nbody_subsample_{subsamples[i]}", 
                             compressed_positions, compressed_velocities, compressed_potential, 
                             full_counts_dic, L, Ngrid, vmax, pmax, i,attribute_dic)

    # Test decompression for one subsample (e.g., the first one)
    test_subsample_index = 0
    compressed_dic, attribute_dic = load_compressed_data(f"compressed_nbody_subsample_{subsamples[test_subsample_index]}", test_subsample_index)
    decompressed_positions, decompressed_velocities, decompressed_potential = decompress_nbody_data(compressed_dic,attribute_dic)

    sorted_single_indices, sort_indices= Get_sorted_indices(positions[subsample_indices[test_subsample_index]],L, Ngrid)
    
    # Calculate errors for the test subsample
    original_positions = positions[subsample_indices[test_subsample_index]]
    original_velocities = velocities[subsample_indices[test_subsample_index]]
    original_potential = potential[subsample_indices[test_subsample_index]]
    

    position_errors = np.abs(original_positions[sort_indices,:] - decompressed_positions)
    velocity_errors = np.abs(original_velocities[sort_indices,:] - decompressed_velocities)
    potential_errors = np.abs(original_potential[sort_indices] - decompressed_potential)

    # Print statistics
    print(f"Total number of data points: {Ndata}")
    print(f"Subsamples: {subsamples}")
    print(f"Test subsample: {subsamples[test_subsample_index]}% ({len(subsample_indices[test_subsample_index])} points)")
    print(f"Position - Max error: {np.max(position_errors):.6f}, Mean error: {np.mean(position_errors):.6f}")
    print(f"Velocity - Max error: {np.max(velocity_errors):.6f}, Mean error: {np.mean(velocity_errors):.6f}")
    print(f"Potential - Max error: {np.max(potential_errors):.6f}, Mean error: {np.mean(potential_errors):.6f}")
    
    # Calculate compression ratio for the test subsample
    original_size = original_positions.nbytes + original_velocities.nbytes + original_potential.nbytes
    compressed_size = (compressed_positions.nbytes + compressed_velocities.nbytes + 
                       compressed_potential.nbytes)# + full_counts.nbytes)
    print('original_size:',original_size/(1024*1024))
    print('compressed_size:',compressed_size/(1024*1024))
    print(f"Compression ratio for test subsample: {original_size / compressed_size:.2f}")

    # Plot error distributions for the test subsample
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, errors, title in zip([ax1, ax2, ax3], 
                                 [position_errors.flatten(), velocity_errors.flatten(), potential_errors.flatten()],
                                 ['Position', 'Velocity', 'Potential']):
        emin, emax = np.log10(errors.min()), np.log10(errors.max())
        bins = np.logspace(emin, emax, 50)
        ax.hist(errors, bins=bins)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'{title} Error Distribution')
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'error_distributions_subsample_{subsamples[test_subsample_index]}.png')
    plt.close()

if __name__ == "__main__":
    #test_compression(Ndata=1000000, L=200.0, Ngrid=200, vmax=6000.0, pmax=1e5, subsamples=[1,4,5,20,30,40])
    test_compression(Ndata=10000000, L=20.0, Ngrid=20, vmax=6000.0, pmax=1e5, subsamples=[100])
