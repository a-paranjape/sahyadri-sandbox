import numpy as np
from readers import HaloReader,SnapshotReader
from correlations import PowerSpectrum
from time import time
import gc


import h5py

import socket
print(socket.gethostname())

import time

def ltime(date=True):
    tm=time.localtime()
    datestr=str(tm.tm_mday)+'/'+str(tm.tm_mon)+'/'+str(tm.tm_year)
    tmstr=str(tm.tm_hour)+':'+str(tm.tm_min)+':'+str(tm.tm_sec)

    if(date):
        tmstr=datestr+' '+tmstr

    return tmstr

if(False):#Sinhagad
    sim_stem='/mnt/home/project/chpc2501005/shadab/test_sinhagad/'
    snap_list = [0]
    real = 1
    grid = 256
    downsample=0
    Npmin = 30

    Seed = 42
else:
    #sahyadri sim on pegasus
    sim_stem='/mnt/home/project/chpc2501005/data/sims/sahyadri/default2048/'
    snap_list = [72]
    real = 1
    grid = 2048
    downsample=0
    Npmin = 30

    Seed = 42
    
    
for ss,snap in enumerate(snap_list):
    print('%s Begin snap=%d'%(ltime(),snap))
    sr = SnapshotReader(sim_stem=sim_stem,real=real,snap=snap,read_header=True,logfile='compress.log')

    # The compressed file will be split in several percentage subsample
    # One if free to choose this setting with following requirement
    # a) The sum of the subsamples must add to 100
    # b) The elements must be unique that is same subsample cannot be created twice because the file neme convention

    #subsamples=[1,3,6,10,30,50]
    subsamples=[1,3,5,9,10,12,15,20,25]
    quant_write=['positions','ids','velocities']#,'potentials']

    # read the header
    sr.read_snapshot_header()

    # The directory in the output directory to store compressed output
    compressed_fileroot='compressed'

    # This call reads the original snapshot and write the compressed file
    sr.compress_snapshot( subsamples,quant_write=quant_write)
