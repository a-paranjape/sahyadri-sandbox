#!/usr/bin/env python3
"""Write a Gadget-4 parameter file from config/sims/run.param.template with the same substitutions as
scripts/submit_CiPod.sh (Sahyadri production), using the variables of the comparison config (environment).

usage: make_gadget4_param.py ics|run <out.param>
  ics  NGenIC build, RestartFlag 6: writes <GADGET_DIR>/ics/snapshot_ics_000.hdf5 (CLASS P(k) at Z_START,
       PowerSpectrumType 2, no renormalisation, PrimordialIndex 1: the tilt is in the CLASS spectrum)
  run  non-NGenIC build: reads that IC file (ICFormat 3), outputs at OUTPUTS_TXT (+ TimeMax) to <GADGET_DIR>
"""
import os
import sys

E = os.environ
mode, out = sys.argv[1], sys.argv[2]
assert mode in ('ics', 'run')
gdir = E['GADGET_DIR']
om, ok = float(E['OMEGA_M']), float(E.get('OMEGA_K', 0) or 0)
set_ = {
    'OutputDir': gdir + '/ics' if mode == 'ics' else gdir,
    'OutputListFilename': E['OUTPUTS_TXT'],
    'TimeBegin': '%.8g' % (1.0 / (1.0 + float(E['Z_START']))),
    'TimeMax': '1.0',
    'Omega0': E['OMEGA_M'],
    'OmegaLambda': '%.8f' % (1 - om - ok),     # as submit_CiPod.sh: 1 - OMEGA_M - OMEGA_K
    'OmegaBaryon': E['OMEGA_B'],
    'HubbleParam': E['HUBBLE'],
    'BoxSize': E['LBOX'],
    'NumFilesPerSnapshot': '1',
    'MaxFilesWithConcurrentIO': '1',
    'SofteningComovingClass0': E['FORCE_RES'],
    'SofteningMaxPhysClass0': E['FORCE_RES'],
    'MaxMemSize': E.get('G4_MAXMEM', '4000'),
}
ngenic = ['NSample', 'GridSize', 'Seed', 'SphereMode', 'PowerSpectrumType', 'ReNormalizeInputSpectrum',
          'PrimordialIndex', 'ShapeGamma', 'Sigma8', 'PowerSpectrumFile', 'InputSpectrum_UnitLength_in_cm']
if mode == 'ics':
    set_.update({'InitCondFile': gdir + '/ics/unused', 'NSample': E['NPART'], 'GridSize': E['NPART'],
                 'Seed': E['SEED'], 'SphereMode': '0', 'PowerSpectrumType': '2', 'ReNormalizeInputSpectrum': '0',
                 'PrimordialIndex': '1.0', 'PowerSpectrumFile': E['CLASS_PK']})
else:
    set_.update({'InitCondFile': gdir + '/ics/snapshot_ics_000', 'ICFormat': '3'})
lines, seen = [], set()
for l in open(os.path.join(E['SANDBOX'], 'config/sims/run.param.template')):
    w = l.split()
    if w and w[0] in ngenic and mode == 'run':
        continue                                  # as submit_CiPod.sh with NGENIC=0: no NGenIC parameters
    if w and w[0] in set_:
        l = '%-32s %s\n' % (w[0], set_[w[0]])
        seen.add(w[0])
    lines.append(l)
missing = set(set_) - seen
assert not missing, missing
open(out, 'w').write(''.join(lines))
print('wrote %s (%s)' % (out, mode))
