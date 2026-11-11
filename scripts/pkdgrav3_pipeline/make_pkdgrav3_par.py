#!/usr/bin/env python3
"""Write the pkdgrav3 parameter file of a pipeline run from the variables of its config (environment).

usage: make_pkdgrav3_par.py <out.par>   (called by job_sim.pbs after sourcing the config)
IC_FILE set (non-empty): read the ICs from that file instead of generating them (Z_START must be its redshift).
One output per entry of OUT_Z; STEPS[i] steps from the previous output (or Z_START) to OUT_Z[i].
"""
import os
import sys

E = os.environ
zs = [float(z) for z in E['OUT_Z'].split()]
steps = [int(s) for s in E['STEPS'].split()]
if len(zs) != len(steps) or sorted(zs, reverse=True) != zs or zs[0] >= float(E['Z_START']):
    sys.exit('OUT_Z must decrease, start below Z_START, and match STEPS in length')
sub = [float(p) for p in E['SUBSAMPLES'].split(',')]
if abs(sum(sub) - 100) > 1e-9:
    sys.exit('SUBSAMPLES must sum to 100')
lines = [
    '# written by sahyadri-sandbox scripts/pkdgrav3_pipeline/make_pkdgrav3_par.py from %s' % E['CONF'],
    'from accuracy import classic_theta_switch, classic_replicas_switch',
    'achOutName      = "snap"',
    'dBoxSize        = %s' % E['LBOX'],
    'nSteps          = %s' % steps,
    'dRedTo          = %s' % zs,
    'iOutInterval    = 1000000',     # outputs only at the dRedTo boundaries
    'iCheckInterval  = 0',
    'h               = %s' % E['HUBBLE'],
    'dOmega0         = %s' % E['OMEGA_M'],
    'dLambda         = %s' % E['OMEGA_L'],
    'bComove         = True',
    'bPeriodic       = True',
    'bEwald          = True',
    'bEpsAccStep     = True',
    'bNewKDK         = True',
    'dEta            = 0.3',
    'dTheta          = classic_theta_switch()',
    'nReplicas       = classic_replicas_switch()',
    'bMemUnordered   = False',       # ordered: particle ID = Lagrangian lattice index
    'bMemParticleID  = False',
    'bDoDensity      = False',
    'bOverwrite      = True',
    'bVDetails       = True',
    'bCompressedOutput = True',
    'lstCompSubsamples = %s' % sub,
    'achSimName      = "%s"' % E['STEM'],
    'achSimSet       = "Sahyadri"',
]
if E.get('IC_FILE', ''):
    # read the initial conditions from a file (e.g. Gadget-4 NGenIC ICs converted by
    # gadget_pkdgrav3_compare/gadget4_ic_to_gadget2.py); the starting time comes from the file. pkdgrav3 generates
    # ICs whenever nGrid is given, so none of the IC-generation parameters may appear.
    # pkdgrav3 derives its physical units (KmPerSecUnit, box size in the HDF5/.pkc headers, potential in (km/s)^2)
    # only when nGrid is set (units.cxx); otherwise they are all 1. Give dKpcUnit and dMsolUnit explicitly, with the
    # values of the nGrid branch (G = 1, rho_crit = 3/(8 pi), L = 1; constants of units.h).
    KPCCM, MSOLG, GCGS = 3.085678e21, 1.99e33, 6.67e-8
    kpc = float(E['LBOX']) * 1e3 / float(E['HUBBLE'])
    H = 100.0 * float(E['HUBBLE']) * 1e5 / (1e3 * KPCCM)
    msol = (kpc * KPCCM)**3 / MSOLG * 3.0 * H**2 / (8.0 * 3.141592653589793 * GCGS)
    lines += ['achInFile       = "%s"' % E['IC_FILE'], 'dKpcUnit        = %.17g' % kpc, 'dMsolUnit       = %.17g' % msol]
else:
    lines += ['achTfFile       = "%s"' % E['TF_FILE'], 'nGrid           = %d' % int(E['NPART']), 'iLPT            = 2',
              'iSeed           = %d' % int(E['SEED']), 'dRedFrom        = %s' % E['Z_START'],
              'dSigma8         = %s' % E['SIGMA8'], 'dSpectral       = %s' % E['NS']]
if float(E.get('OMEGA_DE', 0) or 0) != 0 or float(E.get('W0', -1)) != -1 or float(E.get('WA', 0)) != 0:
    lines += ['dOmegaDE        = %s' % E.get('OMEGA_DE', 0), 'w0              = %s' % E.get('W0', -1),
              'wa              = %s' % E.get('WA', 0)]
if E.get('ALSO_HDF5', '0') == '1':
    lines += ['bHDF5           = True', 'bCompAlsoStandard = True']
for l in E.get('EXTRA_PAR', '').split(';'):
    if l.strip():
        lines.append(l.strip())
open(sys.argv[1], 'w').write('\n'.join(lines) + '\n')
print('wrote %s: %d outputs at z = %s, %d steps' % (sys.argv[1], len(zs), zs, sum(steps)))
