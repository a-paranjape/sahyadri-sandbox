#!/usr/bin/python

import paths 
import constants as con

import sys
sys.path.append(paths.python_path)
import numpy as ny
from halo_reader import HaloReader
from utilities import write_to_file,writelog,quit_,time_this_log,massfunc_hist1d
from time import time

########################################################
# Read halo catalog from ROCKSTAR 
# and generate mass function. 
########################################################

if(len(sys.argv)==11):
    sim_stem = sys.argv[1]
    snapnum = int(sys.argv[2])
    real = int(sys.argv[3])
    Lbox = float(sys.argv[4])
    Om = float(sys.argv[5])
    hubble = float(sys.argv[6])
    Npside = int(sys.argv[7])
    Npmin = int(sys.argv[8])
    lgmmax = float(sys.argv[9])
    TREES = int(sys.argv[10])
else: 
    sim_stem = raw_input("Specify path in folder " + paths.halo_path + " (e.g., `su128/delta0.0' or 'scm1024'): ")
    snapnum = int(raw_input("Specify snapshot (e.g., 0-200): "))
    real = int(raw_input("Specify realisation (e.g., 1-10): "))
    Lbox = float(raw_input("Specify box length in Mpc/h: "))
    Om = float(raw_input("Specify Omega_m: "))
    hubble = float(raw_input("Specify hubble: "))
    Npside = int(raw_input("Specify number of particles on a side (e.g., 512): "))
    Npmin = int(raw_input("Specify minimum number of particles per halo: "))
    lgmmax = float(raw_input("Specify lgm_max for histogram: "))
    TREES = int(raw_input("Specify whether (1) or not (0) merger trees available: "))

start_time = time()

logfile = paths.halo_path + sim_stem + '/' + 'r'+str(real)+'/'
logfile += 'mf_' + str(snapnum) +'.log'
f = open(logfile,'w')
f.close()
writelog(logfile," Calculating mass functions for sim: "+sim_stem+"\n")

Vbox = Lbox**3

tags = ['mvir','m200b','m200c','mCustom2','mCustom'] 
# tags = ['mvir','m200b','m200c'] 
ntags = len(tags)
mass_string = ''
for t in tags[:-1]: mass_string += t+','
mass_string += tags[-1]

mmin = Om*con.rhoc*Vbox*Npmin/Npside**3

lgmmin = ny.log10(mmin)
dlgm = 0.2
dlnm = dlgm*ny.log(10)
nlgm = int((lgmmax-lgmmin)/dlgm)
mbins = ny.logspace(lgmmin,lgmmax,nlgm+1)
mcenter = ny.sqrt(mbins[1:]*mbins[:-1])

dndlnm = ny.zeros((ntags,nlgm),dtype=float)

########################################################
# Read data 
########################################################
writelog(logfile,'Reading and binning data...'+'\n')
hr = HaloReader(sim_stem=sim_stem,TREES=TREES,logfile=logfile)
halos = hr.read_this(real,snapnum)

QE = 0.5 # 0 < QE < 1
writelog(logfile,"Halos will satisfy {0:.2f} < 2T/|U| < {1:.2f}\n".format(1.-QE,1.+QE))
cond_QE = (2*halos['TbyU'] < (1+QE)) & ((1-QE) < 2*halos['TbyU'])
halos = halos[cond_QE]

Nhalo_all = halos.size
halos = halos[halos['pid']==-1]
Nhalo_parents = halos.size
writelog(logfile,"... satellite fraction = {0:.5f}\n".format(1 - 1.0*Nhalo_parents/Nhalo_all))
for t in range(ntags):
    dndlnm[t],temp = ny.histogram(halos[tags[t]],bins=mbins,density=False)
    dndlnm[t] = dndlnm[t]/dlnm/Vbox
    if t==0:
        writelog(logfile,"Nhalos: direct = {0:d}; integrated = {1:.1f}\n"
                 .format(Nhalo_parents,Vbox*dlnm*ny.sum(dndlnm[t])))

outfile  ='r'+str(real)+'/mf_' + str(snapnum) +'_r'+str(real)+'.txt'
writelog(logfile,'Writing to file: '+ outfile+'\n')
outfile = paths.halo_path + sim_stem + '/' + outfile
fh = open(outfile,'w')
fh.write("#\n# Mass functions for " + sim_stem + '/'+'r'+str(real)+'/out_' + str(snapnum)+"\n")
fh.write("# This file contains dn/dlnm (h/Mpc)^3 for various mass definitions.\n")
fh.write("#\n# mass (Msun/h) | dndlnm["+mass_string+"]\n")
fh.close()
for m in range(nlgm):
    mlist = [mcenter[m]]
    for t in range(ntags):
        mlist.append(dndlnm[t,m])
    write_to_file(outfile,mlist)
########################################################

time_this_log(start_time,logfile)
