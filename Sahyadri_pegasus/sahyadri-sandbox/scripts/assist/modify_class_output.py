import numpy as np
import sys

print('Modifying CLASS output for use in NGEN-IC')
if (len(sys.argv) == 2):
    infile = sys.argv[1]
else:
    infile = input("Specify input filename: ")

print('... reading file: ' + infile)
data = np.loadtxt(infile).T
if data.shape[0] != 2:
    raise TypeError('Need exactly 2 columns in input file of modify_class_output.py')

data_out = np.zeros_like(data)
data_out[0] = np.log10(data[0])
data_out[1] = np.log10(data[0]**3*data[1]/(2*np.pi**2))

outfile = infile[:-3]+'txt'
print('... writing to file: ' + outfile)
np.savetxt(outfile,data_out.T,fmt='%.8e')
print('...done')
