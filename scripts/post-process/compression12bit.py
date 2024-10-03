import numpy as np

def compress_12bit(data, max_value, allow_negative=True, little_endian=True,input_dtype='float'):
    '''compresses values to 12 bit precision and store them in a unint8 array spliting each element in 1.5 element of array
    if input_dtype='integer is given then assumes integer and max value is not used for
    this case if allow_negative=True then the rand is -2047 - 2047 otherwise 0-4095
    '''
    if data.size % 2 != 0:
        raise ValueError(f"Array size must be even {data.size}")

    if(input_dtype=='integer'):
        if allow_negative:
            # Scale from [-max_value, max_value] to [0, 4095]
            scaled = np.clip(data + 2047, 0, 4095)
        else:
            # Scale from [0, max_value] to [0, 4095]
            scaled = np.clip(data, 0, 4095)
    else:            
        if allow_negative:
            # Scale from [-max_value, max_value] to [0, 4095]
            scaled = np.clip((data / max_value + 1) * 2047.5, 0, 4095)
        else:
            # Scale from [0, max_value] to [0, 4095]
            scaled = np.clip(data / max_value * 4095, 0, 4095)

    quantized = np.round(scaled).astype(np.uint16)

    # Allocate output array (3 bytes for every 2 values)
    packed = np.zeros(int(np.ceil(data.size * 1.5)), dtype=np.uint8)

    # Pack 12-bit values into 8-bit array
    if little_endian:
        packed[0::3] = quantized[0::2] & 0xFF  # Lower 8 bits of even indices
        packed[1::3] = ((quantized[0::2] >> 8) & 0x0F) | ((quantized[1::2] & 0x0F) << 4)  # Upper 4 bits of even indices and lower 4 bits of odd indices
        packed[2::3] = quantized[1::2] >> 4  # Upper 8 bits of odd indices
    else:  # big endian
        packed[0::3] = quantized[0::2] >> 4  # Upper 8 bits of even indices
        packed[1::3] = ((quantized[0::2] & 0x0F) << 4) | ((quantized[1::2] >> 8) & 0x0F)  # Lower 4 bits of even indices and upper 4 bits of odd indices
        packed[2::3] = quantized[1::2] & 0xFF  # Lower 8 bits of odd indices

    return packed, quantized

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
            return (unpacked.astype(float) / 2047.5 - 1) * max_value
        else:
            # Scale back to [0, max_value]
            return unpacked.astype(float) / 4095 * max_value



def compress_integer_array(arr, bit_depth=None,little_endian=True,possible_bit_depth= [8,12,16,32,64]):
    arr = np.asarray(arr)

    if arr.size % 2 != 0:
        raise ValueError(f"Array size must be even {arr.size}")

    if bit_depth is None:
        bit_depth = find_optimal_bit_depth(arr, overflow_threshold)
    
    if bit_depth not in possible_bit_depth:
        raise ValueError(f"Bit depth must be {possible_bit_depth}")

    max_value = (1 << bit_depth) - 1

    # Find overflow values
    overflow_mask = arr > max_value
    overflow_indices = np.nonzero(overflow_mask)[0]
    overflow_values = arr[overflow_mask]

    # Compress values
    compressed = np.minimum(arr, max_value)

    # Pack bits
    if bit_depth in (8, 16, 32, 64):
        # Use native types for these bit depths
        dtype = f'uint{bit_depth}'
        packed = compressed.astype(dtype)
    else:
        packed , _ = compress_12bit(compressed, None, allow_negative=False, little_endian=little_endian,input_dtype='integer')
        #packed=pack_12bit(compressed)

    return packed, overflow_indices, overflow_values, arr.dtype, bit_depth
    
def decompress_integer_array(packed, overflow_indices, overflow_values, original_dtype, bit_depth, original_size,little_endian=True):
    if bit_depth in (8, 16, 32, 64):
        # Use native types for these bit depths
        unpacked = packed.astype(original_dtype)
    else:
        #unpacked = unpack_12bit(packed)
        unpacked = decompress_12bit(packed, None, original_size, allow_negative=False, little_endian=little_endian,input_dtype='integer')
    
    # Restore overflow values
    unpacked[overflow_indices] = overflow_values

    return unpacked

def find_optimal_bit_depth(arr, overflow_threshold):
    possible_bit_depths = [8, 12, 16, 32, 64]
    arr_max = np.max(arr)

    optimal_bit_depth = None
    min_total_size = float('inf')

    for bit_depth in possible_bit_depths:
        max_value = (1 << bit_depth) - 1
        overflow_count = np.sum(arr > max_value)
        overflow_fraction = overflow_count / len(arr)

        if overflow_fraction <= overflow_threshold:
            packed_size = calculate_packed_size(len(arr), bit_depth)
            overflow_size = overflow_count * (arr.itemsize + 4)  # 4 bytes for index
            total_size = packed_size + overflow_size

            if total_size < min_total_size:
                min_total_size = total_size
                optimal_bit_depth = bit_depth

        if bit_depth >= arr_max:
            break

    return optimal_bit_depth or 64  # Default to 64 if no suitable bit depth found

def calculate_packed_size(array_length, bit_depth):
    if bit_depth in (8, 16, 32, 64):
        return array_length * (bit_depth // 8)
    elif bit_depth == 12:
        return (array_length * 12 + 7) // 8  # Round up to nearest byte
    else:
        raise ValueError("Unsupported bit depth")




# Test function (same as before)
def test_12bit_compression(size=1000000):
    np.random.seed(42)
    
    # Test with negative numbers
    original_data_neg = np.random.uniform(-100, 100, size)
    max_value_neg = np.max(np.abs(original_data_neg))

    compressed_neg, quantized_neg = compress_12bit(original_data_neg, max_value_neg, allow_negative=True)
    decompressed_neg = decompress_12bit(compressed_neg, max_value_neg, original_data_neg.size, allow_negative=True)

    # Test with only positive numbers
    original_data_pos = np.random.uniform(0, 100, size)
    max_value_pos = np.max(original_data_pos)

    compressed_pos, quantized_pos = compress_12bit(original_data_pos, max_value_pos, allow_negative=False)
    decompressed_pos = decompress_12bit(compressed_pos, max_value_pos, original_data_pos.size, allow_negative=False)

    # Detailed debugging output
    print("Debugging output:")
    for i in range(10):
        print(f"\nIndex {i}:")
        print(f"  Negative case:")
        print(f"    Original: {original_data_neg[i]:.6f}")
        print(f"    Quantized: {quantized_neg[i]}")
        print(f"    Decompressed: {decompressed_neg[i]:.6f}")
        print(f"    Error: {abs(original_data_neg[i] - decompressed_neg[i]):.6f}")
        print(f"  Positive case:")
        print(f"    Original: {original_data_pos[i]:.6f}")
        print(f"    Quantized: {quantized_pos[i]}")
        print(f"    Decompressed: {decompressed_pos[i]:.6f}")
        print(f"    Error: {abs(original_data_pos[i] - decompressed_pos[i]):.6f}")

    # Check compression ratios and accuracy (same as before)
    # ...

def test_compression_decompression(bit_depth, size=10000):
    np.random.seed(42)
    input_arr = np.random.randint(0, 1 << (bit_depth + 2), size=size, dtype=np.int64)

    packed, overflow_indices, overflow_values, original_dtype, bit_depth = compress_integer_array(input_arr, bit_depth=bit_depth)
    decompressed = decompress_integer_array(packed, overflow_indices, overflow_values, original_dtype, bit_depth, input_arr.size)

    assert np.array_equal(decompressed, input_arr), f"Test failed for {bit_depth}-bit depth"
    print(f"Test passed for {bit_depth}-bit depth")

    # Calculate compression ratio
    original_size = input_arr.nbytes
    compressed_size = packed.nbytes + overflow_indices.nbytes + overflow_values.nbytes
    compression_ratio = original_size / compressed_size
    print(f"Compression ratio for {bit_depth}-bit depth: {compression_ratio:.2f}")



# Run the test
if __name__ == "__main__":
    #test_12bit_compression()
    for bit_depth in [None,8, 12, 16, 24, 32]:
        test_compression_decompression(bit_depth)
