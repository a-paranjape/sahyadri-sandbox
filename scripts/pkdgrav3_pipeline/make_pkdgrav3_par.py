#!/usr/bin/env python3
"""Write the pkdgrav3 parameter file of a pipeline run from the variables of its config (environment).

usage: make_pkdgrav3_par.py <out.par>   (called by job_sim.pbs after sourcing the config)
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
    'achTfFile       = "%s"' % E['TF_FILE'],
    'nGrid           = %d' % int(E['NPART']),
    'dBoxSize        = %s' % E['LBOX'],
    'iLPT            = 2',
    'iSeed           = %d' % int(E['SEED']),
    'dRedFrom        = %s' % E['Z_START'],
    'nSteps          = %s' % steps,
    'dRedTo          = %s' % zs,
    'iOutInterval    = 1000000',     # outputs only at the dRedTo boundaries
    'iCheckInterval  = 0',
    'h               = %s' % E['HUBBLE'],
    'dOmega0         = %s' % E['OMEGA_M'],
    'dLambda         = %s' % E['OMEGA_L'],
    'dSigma8         = %s' % E['SIGMA8'],
    'dSpectral       = %s' % E['NS'],
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
