#!/usr/bin/python

from paths import *

import sys
sys.path.append(python_path)
import numpy as ny
from utilities import quit_,writelog

import struct
import gc


########################################################
# Reader for GADGET-2 snapshot. 
########################################################

class Gadget2Reader(object):
    """ Reader for Gadget-2 snapshot. """

    def __init__(self,ptype=1,read_pos=True,read_vel=True,read_ID=False,logfile=None,verbose=True):
        
        self.ptype = ptype # 1 for CDM
        self.read_pos = read_pos
        self.read_vel = read_vel
        self.read_ID = read_ID
        self.logfile = logfile
        self.verbose = verbose

        self.sz_int = struct.calcsize('i')
        self.sz_flt = struct.calcsize('f')
        self.sz_dbl = struct.calcsize('d')
        if((self.sz_int != 4) or (self.sz_flt != 4) or (self.sz_dbl != 8)):
            quit_("Problem with data-type sizes.")

        self.npart = ny.zeros(6,dtype='i')
        self.massarr = ny.zeros(6,dtype='d')
        self.snap_time = ny.float64(0.0)
        self.redshift = ny.float64(0.0)
        self.flagSFR = ny.int32(0)
        self.flagFBk = ny.int32(0)
        self.npart_tot = ny.zeros(6,dtype='i')
        self.flagCool = ny.int32(0)
        self.nFile = ny.int32(0)
        self.Lbox = ny.float64(0.0)
        self.Om = ny.float64(0.0)
        self.OLam = ny.float64(0.0)
        self.hubble = ny.float64(0.0)
        self.bytesleft = (256 - 6*self.sz_int - 6*self.sz_dbl 
                          - self.sz_dbl - self.sz_dbl 
                          - self.sz_int - self.sz_int - 6*self.sz_int
                          - self.sz_int - self.sz_int
                          - self.sz_dbl - self.sz_dbl - self.sz_dbl - self.sz_dbl)
        self.la = ny.zeros(self.bytesleft/self.sz_int,dtype='i')


    def read_this(self,snapshot):
        """ Read snapshot and store required data in memory. Hard-coded for DM-only sims. 
            Returns pos [shape (3,npart)], vel [shape (3,npart)], ids [shape (npart)]
            if specified, else single zeroes.
        """
        ########################################################
        # Header variables
        ########################################################

        if self.verbose:
            if self.logfile==None: print "Reading file "+snapshot
            else: writelog(self.logfile, "Reading file "+snapshot+'\n')
        pos = 0.0
        vel = 0.0
        ids = 0
        with open(snapshot,"rb") as f:
            ########################################################
            # Read header
            ########################################################
            if self.logfile==None: print "... header"
            else: writelog(self.logfile,"... header\n")
            recl = struct.unpack('i',f.read(4))[0]
            self.npart = ny.fromfile(f,dtype='i',count=6)
            self.massarr = ny.fromfile(f,dtype='d',count=6)
            self.snap_time = struct.unpack('d',f.read(self.sz_dbl))[0]
            self.redshift = struct.unpack('d',f.read(self.sz_dbl))[0]
            self.flagSFR = struct.unpack('i',f.read(self.sz_int))[0]
            self.flagFBk = struct.unpack('i',f.read(self.sz_int))[0]
            self.npart_tot = ny.fromfile(f,dtype='i',count=6)
            self.flagCool = struct.unpack('i',f.read(self.sz_int))[0]
            self.nFile = struct.unpack('i',f.read(self.sz_int))[0]
            self.Lbox = struct.unpack('d',f.read(self.sz_dbl))[0]
            self.Om = struct.unpack('d',f.read(self.sz_dbl))[0]
            self.OLam = struct.unpack('d',f.read(self.sz_dbl))[0]
            self.hubble = struct.unpack('d',f.read(self.sz_dbl))[0]
            self.la = ny.fromfile(f,dtype='i',count=self.bytesleft/self.sz_int)
            endrec = struct.unpack('i',f.read(4))[0]

            ntot_file = ny.sum(self.npart)

            if self.read_pos:
                ########################################################
                # Read positions
                ########################################################
                if self.verbose:
                    if self.logfile==None: print "... positions"
                    else: writelog(self.logfile,"... positions\n")
                pos = ny.zeros(3*ntot_file,dtype='f')
                recl = struct.unpack('i',f.read(4))[0]
                pos = ny.fromfile(f,dtype='f',count=3*ntot_file)
                pos = ny.reshape(pos,(-1,3))
                endrec = struct.unpack('i',f.read(4))[0]    
                pos = pos.T
            else:
                if self.verbose:
                    if self.logfile==None: print '... skipping positions'
                    else: writelog(self.logfile,'... skipping positions\n')

            if self.read_vel:
                ########################################################
                # Read velocities
                ########################################################
                if self.verbose:
                    if self.logfile==None: print "... velocities"
                    else: writelog(self.logfile,"... velocities\n")
                vel = ny.zeros(3*ntot_file,dtype='f')
                recl = struct.unpack('i',f.read(4))[0]
                vel = ny.fromfile(f,dtype='f',count=3*ntot_file)
                vel = ny.reshape(vel,(-1,3))
                endrec = struct.unpack('i',f.read(4))[0]    
                vel = vel.T
            else:
                if self.verbose:
                    if self.logfile==None: print '... skipping velocities'
                    else: writelog(self.logfile,'... skipping velocities\n')

            if self.read_ID:
                ########################################################
                # Read particle IDs
                ########################################################
                if self.verbose:
                    if self.logfile==None: print "... particle IDs"
                    else: writelog(self.logfile,"... particle IDs\n")
                ids = ny.zeros(ntot_file,dtype='i')
                recl = struct.unpack('i',f.read(4))[0]
                ids = ny.fromfile(f,dtype='i',count=ntot_file)
                endrec = struct.unpack('i',f.read(4))[0]    
            else:
                if self.verbose:
                    if self.logfile==None: print '... skipping particle IDs'
                    else: writelog(self.logfile,'... skipping particle IDs\n')

        gc.collect()
        return pos,vel,ids
        ################################################################

    def downsample(self,snapshot,down_to=128,seed=None):
        """ Read snapshot and return randomly downsampled data. """
        
        pos,vel,ids = self.read_this(snapshot)

        if down_to**3 > self.npart[self.ptype]:
            if self.verbose:
                if self.logfile==None: print 'Not enough particles! Using original sample...'
                else: writelog(self.logfile,'Not enough particles! Using original sample...\n')
            down_to = int(self.npart[self.ptype]**(1/3.))

        N_SAMPLE = int(down_to**3)

        ny.random.seed(seed)
        ind = ny.random.choice(self.npart[self.ptype],size=N_SAMPLE,replace=False)
        pos = pos.T[ind].T
        if self.read_vel:
            vel = vel.T[ind].T
        if self.read_ID:
            ids = ids[ind]

        return pos,vel,ids

########################################################################
