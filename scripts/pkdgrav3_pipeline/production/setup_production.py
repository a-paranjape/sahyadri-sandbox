#!/usr/bin/env python3
"""
setup_production.py: set up Sahyadri production runs (pkdgrav3 -> Rockstar -> consistent-trees -> .vahc -> FITS)
on pawna. It only creates directories and writes configuration files and a submit script; nothing is submitted.
Check the files, adjust them if needed, then run the submit script it prints.

Layout (ROOT default /storage/$USER/sahyadri_sims; SIM = Sahyadri_L<L>N<N>; COSMO = <cosmology><N>, e.g. default1024):
  ROOT/sims/SIM/COSMO/            cosmology.txt, outputs_a.txt, transfer_pkdgrav3.dat, class_pk_z<zstart>.dat
  ROOT/sims/SIM/COSMO/r<R>/       run.conf (pipeline), p.par (pkdgrav3), output_steps.txt, submit.sh, README.txt,
                                  logs/ (job logs), checkpoints/, compressed/ (snap.<step>.s<pct>.<w>.pkc)
  ROOT/halos/SIM/COSMO/r<R>/      rockstar_input.cfg (Rockstar), README.txt; later out_<snap>.trees, .vahc, .tens,
                                  trees/, compressed/*.fits.gz
  ROOT/halos/SIM/COSMO/scales.txt written by the ctrees stage

Simulation settings:
  cosmology    cosmologies/<name>.cosmo (default: the Sahyadri default, as pegasus production)
  ICs          pkdgrav3 2LPT at z = Z_START (49) from the CLASS P(k) at Z_START that pegasus production gave
               NGenIC: transfer_pkdgrav3.dat = sqrt(P/k^ns), with dSigma8 chosen so that pkdgrav3 reproduces that
               P(k) at Z_START (pkdgrav3 scales dSigma8 by its own growth D(a_start)/D(1))
  gravity      as /storage/swarajit/pkdgrav3_sims/pkdgrav3_n256_gadget4ic_v4_corrected/examples/cosmology_test.par:
               dSoft = 1/60 of the mean particle spacing (comoving), dEta = 0.3 with bEpsAccStep (adaptive steps),
               dTheta = classic_theta_switch(), nReplicas = classic_replicas_switch(), Ewald, bNewKDK
  time steps   top-level steps of about the same size in a as that run (STEP_DA = 4.6154e-4 =
               (1 - 1/13)/2000), i.e. round(delta a / STEP_DA) steps between outputs; ~2120 in total
  outputs      the scale factors in sahyadri_outputs_a.txt (101, z = 11.98 ... 0; the last is z = 0 exactly)
  potential    stored (dCompPotStep = 100 (km/s)^2) in every snapshot at z <= 0.5
  checkpoints  every CHECK_INTERVAL steps to r<R>/checkpoints; pkdgrav3 stops cleanly 1 h before the PBS wall
               time (iWallRunTime); continue with `submit.sh restart`

usage: setup_production.py [--root DIR] [--L 200] [--N 1024] [--cosmo default] [--real 1|1,2|1-5]
                           [--codes DIR] [--pkd-io-root DIR] [--step-da 4.6154e-4] [--check-interval 250] [--force]
Run with the pipeline's Python (numpy, scipy), e.g. /storage/shadab/pipeline_test/codes/venv/bin/python.
"""
import argparse
import datetime
import math
import os
import shutil
import sys
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)
SANDBOX = os.path.dirname(os.path.dirname(PIPE))
SUBSAMPLES = '1,3,5,9,10,12,15,20,25'


def read_cosmo(name):
    path = os.path.join(HERE, 'cosmologies', name + '.cosmo')
    if not os.path.isfile(path):
        sys.exit('no cosmology file %s (available: %s)' % (path, ', '.join(
            f[:-6] for f in sorted(os.listdir(os.path.join(HERE, 'cosmologies'))) if f.endswith('.cosmo'))))
    c = {}
    for line in open(path):
        line = line.split('#', 1)[0].strip() if not line.lstrip().startswith('COMMENT') else line.strip()
        if '=' in line:
            k, v = line.split('=', 1)
            c[k.strip()] = v.strip().strip('"')
    c['PK_FILE'] = os.path.join(HERE, 'cosmologies', c['PK_FILE'])
    c['file'] = path
    return c


def reals_of(s):
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def seed_of(r):
    for line in open(os.path.join(SANDBOX, 'config', 'sims', 'seeds.txt')):
        f = line.split()
        if len(f) >= 2 and int(f[0]) == r:
            return int(f[1])
    sys.exit('no seed for realisation %d in config/sims/seeds.txt' % r)


def growth(a, om, ok):
    """Linear growth D(a) (unnormalised) for matter + curvature + Lambda, no radiation: D ~ H int da/(a H)^3."""
    ol = 1.0 - om - ok
    E = lambda x: math.sqrt(om / x**3 + ok / x**2 + ol)
    return E(a) * quad(lambda x: 1.0 / (x * E(x))**3, 0.0, a, epsabs=0, epsrel=1e-12, limit=500)[0]


def ic_spectrum(c):
    """pkdgrav3 transfer table and dSigma8 that reproduce the CLASS P(k) at Z_START.

    pkdgrav3 (ic/ic.cxx PowerTransfer): P(k) = norm k^ns T(k)^2 with ln T a natural cubic spline in ln k, and
    norm = (dSigma8 D(a)/D(1))^2 / var8, var8 its integral over the table's k range. So T = sqrt(P_class / k^ns)
    and dSigma8 = sigma8(P_class at Z_START, same k range and spline) x D(1)/D(a_start)."""
    k, P = np.loadtxt(c['PK_FILE'], comments='#', unpack=True)
    ns = float(c['NS'])
    T = np.sqrt(P / k**ns)
    lnk = np.log(k)
    sp = CubicSpline(lnk, np.log(P), bc_type='natural')   # = 2 spline(ln T) + ns ln k (splines are linear)

    def integrand(lk):
        kk = math.exp(lk)
        x = 8.0 * kk
        w = 3.0 * (math.sin(x) - x * math.cos(x)) / x**3
        return math.exp(sp(lk)) * kk**3 * w * w / (2 * math.pi**2)
    with warnings.catch_warnings():  # the oscillating tail (k > 10 h/Mpc) contributes < 1e-8; same truncation as pkdgrav3
        warnings.simplefilter('ignore')
        var = sum(quad(integrand, lnk[i], lnk[i + 1], epsabs=0, epsrel=1e-10, limit=1000)[0] for i in range(len(k) - 1))
    s8_start = math.sqrt(var)
    om, ok = float(c['OMEGA_M']), float(c.get('OMEGA_K', 0))
    a0 = 1.0 / (1.0 + float(c['Z_START']))
    s8 = s8_start * growth(1.0, om, ok) / growth(a0, om, ok)
    return k, T, s8_start, s8


def steps_plan(a_out, a_start, step_da):
    nst, steps, prev, s = [], [], a_start, 0
    for a in a_out:
        n = max(1, int(round((a - prev) / step_da)))
        nst.append(n)
        s += n
        steps.append(s)
        prev = a
    return nst, steps


def resources(N):
    """Default PBS resources by size (edit run.conf to change). Measured: 512^3 sim 25 GB / 80 min on 64 cores for 60
    steps; Rockstar ~2x its particle array (32 B/particle); .vahc PP_NPROC x 26 GB at grid 512."""
    if N <= 256:
        return dict(SIM=(64, '120gb', '24:00:00'), RS=(16, '64gb', '12:00:00'), CT=(8, '32gb', '12:00:00'),
                    PP=(16, '160gb', '24:00:00'), FITS=(8, '32gb', '06:00:00'), RS_BLOCKS=8, RS_WRITERS=16, PP_NPROC=4)
    if N <= 512:
        return dict(SIM=(96, '200gb', '72:00:00'), RS=(32, '128gb', '24:00:00'), CT=(16, '64gb', '24:00:00'),
                    PP=(32, '300gb', '48:00:00'), FITS=(16, '64gb', '12:00:00'), RS_BLOCKS=16, RS_WRITERS=32, PP_NPROC=8)
    return dict(SIM=(192, '700gb', '168:00:00'), RS=(64, '400gb', '96:00:00'), CT=(16, '200gb', '48:00:00'),
                PP=(32, '400gb', '72:00:00'), FITS=(16, '200gb', '24:00:00'), RS_BLOCKS=32, RS_WRITERS=64, PP_NPROC=8)


def fmt_l(L):
    return ('%d' % L) if float(L).is_integer() else ('%g' % L).replace('.', 'p')


def write(path, text, force, mode=0o644):
    if os.path.exists(path) and not force:
        sys.exit('%s exists; use --force to overwrite the configuration files' % path)
    with open(path, 'w') as f:
        f.write(text)
    os.chmod(path, mode)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='/storage/%s/sahyadri_sims' % os.environ.get('USER', 'shadab'))
    ap.add_argument('--L', type=float, default=200.0, help='box size [Mpc/h]')
    ap.add_argument('--N', type=int, default=1024, help='particles per side')
    ap.add_argument('--cosmo', default='default', help='cosmologies/<name>.cosmo')
    ap.add_argument('--real', default='1', help='realisations, e.g. 1 or 1,2 or 1-5 (seeds: config/sims/seeds.txt)')
    ap.add_argument('--codes', default='/storage/shadab/pipeline_test/codes',
                    help='Rockstar, consistent-trees, venv (setup_pawna.sh)')
    ap.add_argument('--pkd-io-root', default='/storage/shadab/pkdgrav3_io', help='pkdgrav3 build')
    ap.add_argument('--outputs', default=os.path.join(HERE, 'sahyadri_outputs_a.txt'))
    ap.add_argument('--step-da', type=float, default=(1 - 1 / 13.) / 2000,
                    help='top-level step size in a (default: that of the 256^3 reference run)')
    ap.add_argument('--soft-frac', type=float, default=1 / 60., help='softening / mean particle spacing')
    ap.add_argument('--pot-zmax', type=float, default=0.5, help='store the potential at z <= this')
    ap.add_argument('--pot-step', type=float, default=100.0, help='potential step [(km/s)^2]')
    ap.add_argument('--check-interval', type=int, default=250, help='steps between checkpoints')
    ap.add_argument('--force', action='store_true', help='overwrite existing configuration files')
    a = ap.parse_args()

    c = read_cosmo(a.cosmo)
    L, N = a.L, a.N
    sim = 'Sahyadri_L%sN%d' % (fmt_l(L), N)
    cname = '%s%d' % (a.cosmo, N)
    stem = '%s/%s' % (sim, cname)
    om, ok = float(c['OMEGA_M']), float(c.get('OMEGA_K', 0))
    ol = 1.0 - om - ok
    zstart = float(c['Z_START'])
    a_start = 1.0 / (1.0 + zstart)
    a_out = np.loadtxt(a.outputs, comments='#', ndmin=1)
    if abs(a_out[-1] - 1.0) < 1e-9:
        a_out[-1] = 1.0
    if np.any(np.diff(a_out) <= 0) or a_out[0] <= a_start:
        sys.exit('output scale factors must increase and start after a_start = %g' % a_start)
    z_out = [0.0 if x == 1.0 else 1.0 / x - 1.0 for x in a_out]
    nst, steps = steps_plan(a_out, a_start, a.step_da)
    soft_box = a.soft_frac / N
    force_res = L * soft_box
    k, T, s8_start, s8 = ic_spectrum(c)
    res = resources(N)
    wall_h = int(res['SIM'][2].split(':')[0])
    now = datetime.datetime.now().isoformat(timespec='seconds')

    cdir = os.path.join(a.root, 'sims', sim, cname)
    os.makedirs(cdir, exist_ok=True)
    pk_copy = os.path.join(cdir, 'class_pk_z%g.dat' % zstart)
    tf = os.path.join(cdir, 'transfer_pkdgrav3.dat')
    shutil.copyfile(c['PK_FILE'], pk_copy)
    with open(tf, 'w') as f:
        f.write('# pkdgrav3 transfer table for %s: k [h/Mpc], T = sqrt(P_CLASS(k, z=%g) / k^ns), ns = %s\n'
                '# from %s; use with dSpectral = %s, dSigma8 = %.8f\n' % (cname, zstart, c['NS'], c['PK_FILE'], c['NS'], s8))
        for kk, tt in zip(k, T):
            f.write('%.12e %.12e\n' % (kk, tt))
    with open(os.path.join(cdir, 'outputs_a.txt'), 'w') as f:
        f.write('# output scale factors (from %s)\n' % a.outputs)
        f.write(''.join('%.16e\n' % x for x in a_out))
    write(os.path.join(cdir, 'cosmology.txt'), (
        '# %s, written by setup_production.py %s from %s\n' % (cname, now, c['file']) +
        ''.join('%s=%s\n' % (kk, c[kk]) for kk in ('OMEGA_M', 'OMEGA_B', 'OMEGA_K', 'HUBBLE', 'NS', 'AS', 'Z_START')) +
        'OMEGA_L=%.7f\nSIGMA8_ZSTART=%.8f   # of the CLASS P(k) at Z_START\n'
        'SIGMA8_LINEAR_Z0=%.8f   # the same, scaled to z = 0 with the LCDM growth factor (= pkdgrav3 dSigma8)\n'
        % (ol, s8_start, s8)), True)

    print('%s: %d outputs (z = %.3f ... 0), %d top-level steps from z = %g; softening %.6g Mpc/h (1/%g of the spacing);'
          ' sigma8 = %.5f' % (stem, len(a_out), z_out[0], steps[-1], zstart, force_res, 1 / a.soft_frac, s8))
    for r in reals_of(a.real):
        sdir = os.path.join(a.root, 'sims', sim, cname, 'r%d' % r)
        hdir = os.path.join(a.root, 'halos', sim, cname, 'r%d' % r)
        if os.path.isfile(os.path.join(sdir, 'snapnames.txt')) and not a.force:
            sys.exit('%s already has outputs (snapnames.txt); not touching it' % sdir)
        for d in (sdir, os.path.join(sdir, 'compressed'), os.path.join(sdir, 'checkpoints'), os.path.join(sdir, 'logs'), hdir):
            os.makedirs(d, exist_ok=True)
        seed = seed_of(r)
        name = '%s/r%d' % (stem, r)
        # ------------------------------------------------------------ pkdgrav3 parameter file
        par = '\n'.join([
            '# pkdgrav3 parameter file of the Sahyadri run %s (seed %d), written by setup_production.py %s' % (name, seed, now),
            'from accuracy import classic_theta_switch, classic_replicas_switch',
            'achOutName      = "snap"',
            'achIoPath       = "compressed/{name}.{step:05d}{type}"     # snapshots (.pkc) in compressed/',
            'achCheckpointPath = "checkpoints/{name}.{step:05d}{type}"',
            '',
            '# initial conditions: 2LPT at z = %g with the CLASS P(k) of %s at that redshift' % (zstart, c['file']),
            '# (pegasus production gave the same table to NGenIC); dSigma8 makes pkdgrav3 reproduce it at z = %g' % zstart,
            'achTfFile       = "%s"' % tf,
            'nGrid           = %d' % N,
            'dBoxSize        = %s' % ('%g' % L),
            'iLPT            = 2',
            'iSeed           = %d' % seed,
            'dRedFrom        = %g' % zstart,
            '',
            '# cosmology: %s' % c.get('COMMENT', a.cosmo),
            'h               = %s' % c['HUBBLE'],
            'dOmega0         = %s' % c['OMEGA_M'],
            'dLambda         = %.7f' % ol,
            'dSpectral       = %s' % c['NS'],
            'dSigma8         = %.8f     # linear sigma8 at z = 0 (sigma8 of the CLASS P(k) at z = %g: %.8f)' % (s8, zstart, s8_start),
            '',
            '# gravity: as the 256^3 reference run (swarajit pkdgrav3_n256_gadget4ic_v4_corrected/examples/cosmology_test.par)',
            'dSoft           = %.10g   # box units = %.6g Mpc/h comoving = 1/%g of the mean spacing' % (soft_box, force_res, 1 / a.soft_frac),
            'bComove         = True',
            'bPeriodic       = True',
            'bEwald          = True',
            'bEpsAccStep     = True',
            'dEta            = 0.3',
            'dTheta          = classic_theta_switch()',
            'nReplicas       = classic_replicas_switch()',
            'bNewKDK         = True',
            '',
            '# %d outputs at the scale factors of %s; top-level steps of ~%.4g in a' % (len(a_out), a.outputs, a.step_da),
            'nSteps          = %s' % nst,
            'dRedTo          = [%s]' % ', '.join('%.10g' % z for z in z_out),
            'iOutInterval    = 1000000      # outputs only at the dRedTo boundaries',
            'iCheckInterval  = %d' % a.check_interval,
            'iWallRunTime    = %d         # minutes: PBS wall time %d h minus 1 h' % (wall_h * 60 - 60, wall_h),
            '',
            '# memory layout: ordered, particle ID = Lagrangian lattice index (no bMemParticleID)',
            'bMemUnordered   = False',
            'bMemParticleID  = False',
            'bDoDensity      = False',
            'bOverwrite      = True',
            'bVDetails       = True',
            '',
            '# compressed snapshots (.pkc), subsamples; potential at z <= %g' % a.pot_zmax,
            'bCompressedOutput   = True',
            'lstCompSubsamples   = [%s]' % SUBSAMPLES.replace(',', ', '),
            'dCompPotStep        = %g' % a.pot_step,
            'dCompPotMaxRedshift = %g' % a.pot_zmax,
            'achSimName      = "%s"' % name,
            'achSimSet       = "Sahyadri"',
            'achSimComment   = "%s"' % c.get('COMMENT', a.cosmo),
        ]) + '\n'
        write(os.path.join(sdir, 'p.par'), par, a.force)
        with open(os.path.join(sdir, 'output_steps.txt'), 'w') as f:
            f.write('# step  a  z  (planned outputs; Rockstar snapshot number = line number - 1)\n')
            for s, x, z in zip(steps, a_out, z_out):
                f.write('%05d %.10f %.6f\n' % (s, x, z))
        # ------------------------------------------------------------ Rockstar configuration
        rs = '\n'.join([
            '# Rockstar input for %s, written by setup_production.py %s (used by job_rockstar.pbs via RS_CFG)' % (name, now),
            'FILE_FORMAT = "PKC"',
            'INBASE = "%s/compressed"' % sdir,
            'FILENAME = "snap.<snap>.<block>"',
            'SNAPSHOT_NAMES = "%s/snapnames.txt"' % sdir,
            'NUM_SNAPS = %d' % len(a_out),
            'STARTING_SNAP = 0',
            'NUM_BLOCKS = %d' % res['RS_BLOCKS'],
            'NUM_WRITERS = %d' % res['RS_WRITERS'],
            'FORK_READERS_FROM_WRITERS = 1',
            'FORK_PROCESSORS_PER_MACHINE = %d' % res['RS_WRITERS'],
            'PARALLEL_IO = 1',
            'PARALLEL_IO_SERVER_ADDRESS = "localhost"',
            'OUTBASE = "%s"' % hdir,
            'PERIODIC = 1',
            'FORCE_RES = %.8g' % force_res,
            'PKC_SUBSAMPLES = ""',
            '# as pegasus production (config/halos/sahyadri/rockstar_default2048_r1.cfg)',
            'TEMPORAL_HALO_FINDING = 1',
            'STRICT_SO_MASSES = 0',
            'MIN_HALO_OUTPUT_SIZE = 30',
            'MIN_HALO_PARTICLES = 20',
            'UNBOUND_THRESHOLD = 0.5',
            'OVERLAP_LENGTH = 3',
        ]) + '\n'
        write(os.path.join(hdir, 'rockstar_input.cfg'), rs, a.force)
        # ------------------------------------------------------------ pipeline configuration
        R = res
        conf = '''# Sahyadri pipeline configuration of %(name)s, written by setup_production.py %(now)s.
# Sourced by every job (scripts/pkdgrav3_pipeline/pipeline_env.sh). Edit as needed before submitting.
SANDBOX=%(sandbox)s
PKD_IO_ROOT=%(pkd)s
CODES=%(codes)s
SCRATCH=%(root)s                    # sims/<STEM>/r<REAL>, halos/<STEM>/r<REAL>
LOGS=%(sdir)s/logs

STEM=%(stem)s
REAL=%(r)d
NPART=%(N)d
LBOX=%(L)s
PKC_SUBDIR=compressed                             # .pkc files in <sim dir>/compressed
PKD_PAR=%(sdir)s/p.par               # pkdgrav3 parameter file (written by setup_production.py)
OUT_Z="%(nout)d outputs, see output_steps.txt"    # informational; the outputs are in p.par
SUBSAMPLES="%(sub)s"

# resources (one pawna node: 192 cores, 755 GB; workq maximum wall time 168 h)
SIM_NCPU=%(sim_n)d;  SIM_MEM=%(sim_m)s;  SIM_WALL=%(sim_w)s    # iWallRunTime in p.par = SIM_WALL - 1 h
RS_NCPU=%(rs_n)d;    RS_MEM=%(rs_m)s;   RS_WALL=%(rs_w)s
CT_NCPU=%(ct_n)d;    CT_MEM=%(ct_m)s;   CT_WALL=%(ct_w)s
PP_NCPU=%(pp_n)d;    PP_MEM=%(pp_m)s;   PP_WALL=%(pp_w)s
FITS_NCPU=%(fi_n)d;  FITS_MEM=%(fi_m)s; FITS_WALL=%(fi_w)s

# halo finding: Rockstar reads the .pkc files; full configuration in the halo directory
RS_CFG=%(hdir)s/rockstar_input.cfg
FORCE_RES=%(fr).8g                       # = pkdgrav3 softening (Mpc/h comoving)
RS_NUM_BLOCKS=%(rsb)d; RS_WRITERS=%(rsw)d          # informational: the values used are in RS_CFG
RS_PKC_SUBSAMPLES=""
MIN_HALO_OUTPUT_SIZE=30; MIN_HALO_PARTICLES=20; UNBOUND_THRESHOLD=0.5

# merger trees
CLEAN_ROCKSTAR=1        # clean_trees.pl: keep out_<snap>.trees, logs/; delete out_*.list, halos_*, hlists/, outputs/

# value-added catalogs (.vahc, .tens): density on PP_GRID^3 from all particles, built block by block
PP_GRID=512             # production grid; memory ~26 GB per process
PP_DOWN_TO=0            # 0 = all particles (no downsampling)
PP_TENS=1
PP_NPROC=%(ppn)d             # snapshots in parallel

# FITS
FITS_NPROC=16
FITS_DELETE_ASCII=0     # 1: delete the ASCII catalogs after verified FITS (check the log first)
''' % dict(name=name, now=now, sandbox=SANDBOX, pkd=a.pkd_io_root, codes=a.codes, root=a.root, sdir=sdir, hdir=hdir,
           stem=stem, r=r, N=N, L='%g' % L, nout=len(a_out), sub=SUBSAMPLES,
           sim_n=R['SIM'][0], sim_m=R['SIM'][1], sim_w=R['SIM'][2], rs_n=R['RS'][0], rs_m=R['RS'][1], rs_w=R['RS'][2],
           ct_n=R['CT'][0], ct_m=R['CT'][1], ct_w=R['CT'][2], pp_n=R['PP'][0], pp_m=R['PP'][1], pp_w=R['PP'][2],
           fi_n=R['FITS'][0], fi_m=R['FITS'][1], fi_w=R['FITS'][2], fr=force_res, rsb=R['RS_BLOCKS'],
           rsw=R['RS_WRITERS'], ppn=R['PP_NPROC'])
        write(os.path.join(sdir, 'run.conf'), conf, a.force)
        sub = '''#!/bin/bash
# Submit %(name)s (written by setup_production.py %(now)s). Check run.conf, p.par and
# %(hdir)s/rockstar_input.cfg first.
#   ./submit.sh            whole chain: sim -> rockstar -> ctrees -> vahc -> fits (each after the previous succeeded)
#   ./submit.sh dry        print the qsub commands only
#   ./submit.sh restart    continue the simulation from the newest checkpoint, then the rest of the chain
#   ./submit.sh halos      rockstar -> ctrees -> vahc -> fits (simulation finished)
#   ./submit.sh <options>  any submit_pipeline.sh options, e.g. --stages vahc,fits
SUB=%(pipe)s/submit_pipeline.sh
CONF=%(sdir)s/run.conf
case "${1:-all}" in
  all)     exec "$SUB" "$CONF" ;;
  dry)     exec "$SUB" "$CONF" --dry-run ;;
  restart) exec "$SUB" "$CONF" --restart latest ;;
  halos)   exec "$SUB" "$CONF" --stages rockstar,ctrees,vahc,fits ;;
  *)       exec "$SUB" "$CONF" "$@" ;;
esac
''' % dict(name=name, now=now, hdir=hdir, pipe=PIPE, sdir=sdir)
        write(os.path.join(sdir, 'submit.sh'), sub, a.force, 0o755)
        readme = '''%(name)s: Sahyadri pkdgrav3 run (written by setup_production.py %(now)s)
  run.conf          pipeline configuration (paths, resources, halo and post-processing settings)
  p.par             pkdgrav3 parameter file (cosmology, ICs, gravity, outputs, potential, checkpoints)
  output_steps.txt  planned outputs: pkdgrav3 step, a, z (Rockstar snapshot <snap> = row <snap>)
  submit.sh         submission (./submit.sh dry first)
  compressed/       snapshots snap.<step>.s<pct>.<writer>.pkc (after the run; snapnames.txt lists the outputs)
  checkpoints/      pkdgrav3 checkpoints;  logs/  job logs (<stage>.<run>.<job>.log, peak memory in each)
  ../               cosmology.txt, outputs_a.txt, transfer_pkdgrav3.dat, class_pk_z*.dat (shared by realisations)
Halos: %(hdir)s (rockstar_input.cfg; later out_<snap>.trees, .vahc, .tens, trees/, compressed/*.fits.gz)
Guide: %(sandbox)s/docs/pkdgrav3_pipeline_pawna.md, section "Production runs"
''' % dict(name=name, now=now, hdir=hdir, sandbox=SANDBOX)
        write(os.path.join(sdir, 'README.txt'), readme, True)
        write(os.path.join(hdir, 'README.txt'), '%s halos: Rockstar input rockstar_input.cfg; simulation and pipeline '
              'configuration in %s (run.conf, p.par)\n' % (name, sdir), True)
        print('  r%d (seed %d): %s\n        halos: %s' % (r, seed, sdir, hdir))
    print('''
Nothing was submitted. Check (and edit if needed) for each realisation:
  r<R>/run.conf  r<R>/p.par  halos/.../r<R>/rockstar_input.cfg   (and %s/cosmology.txt)
then submit with   <sim dir>/submit.sh dry   and   <sim dir>/submit.sh''' % cdir)
    if N > 512:
        print('NOTE: N = %d > 512: runs above 512^3 need approval before submission.' % N)
    if N >= 2048:
        print('NOTE: 2048^3 needs ~750 GB on one node and staged I/O (pkdgrav3 tools/staging/run_staged.pbs); '
              'job_sim.pbs writes straight to /storage.')


if __name__ == '__main__':
    main()
