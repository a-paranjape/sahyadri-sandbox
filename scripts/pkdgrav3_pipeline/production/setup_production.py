#!/usr/bin/env python3
"""
setup_production.py: set up Sahyadri production runs (pkdgrav3 -> Rockstar -> consistent-trees -> .vahc -> FITS)
on pawna. It only creates directories and writes configuration files and a submit script; nothing is submitted.
Check the files, adjust them if needed, then run the submit script it prints.

Layout (ROOT = the working area while the run is computed, default /user/$USER/sahyadri_scratch (home: ~10x the
write speed of /storage); ARCHIVE = the long-term copy of the kept data, default /storage/$USER/sahyadri_sims, same
layout, read-only once checked (retention.py, archive_run.py); SIM = Sahyadri_L<L>N<N>; COSMO = <cosmology><N>):
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
  memory       bMemIntegerPosition = True (positions as 32-bit integers, 36 B per particle; ~0.78 x the memory of
               doubles; 512^3 test 24 Nov 2026: P(k) within 1e-3 to k = 4 h/Mpc, halo mass function within 0.3 %);
               --no-int-pos for doubles. Checkpoints keep the layout: restarts use the setting they were made with
  checkpoints  every CHECK_INTERVAL steps to r<R>/checkpoints (newest 2 kept); pkdgrav3 stops cleanly 1-4 h before
               the PBS wall time (iWallRunTime); continue with `submit.sh restart`
  I/O          staged (pkdgrav3 tools/staging/run_staged.pbs): snapshots and checkpoints go to the node's local SSD
               and are copied to NFS while the simulation runs (SIM_STAGING=1 in run.conf)

  --schedule [P]  also register every realisation with the run manager (tools/sahyadri_manager, must be running)
                  and put its simulation in the manager's queue with priority P (default 0; lower runs first); the
                  manager submits them within max_sim_running / max_sim_queued
usage: setup_production.py [--schedule [P]] [--root DIR] [--archive DIR|''] [--keep-full-z "0 0.05 ... 4.9089"] [--keep-pct 9] [--L 200] [--N 1024] [--cosmo default|a,b|var_*|all] [--real 1|1,2|1-5]
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
    """Read ``cosmologies/<name>.cosmo`` (``KEY = value`` lines; COMMENT lines kept whole) into a dict.

    Exits with the list of available cosmologies if the file does not exist.
    """
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
    """Expand a realisation list like ``'1-3,7'`` to ``[1, 2, 3, 7]``."""
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def seed_of(r):
    """Initial-condition seed of realisation ``r`` from ``config/sims/seeds.txt`` (exits if missing)."""
    for line in open(os.path.join(SANDBOX, 'config', 'sims', 'seeds.txt')):
        f = line.split()
        if len(f) >= 2 and int(f[0]) == r:
            return int(f[1])
    sys.exit('no seed for realisation %d in config/sims/seeds.txt' % r)


def dark_energy(c):
    """(w0, wa) of a cosmology file; (-1, 0) is a cosmological constant."""
    return float(c.get('W0', -1)), float(c.get('WA', 0))


def growth(a, c):
    """Linear growth D(a) for matter + curvature + (w0, wa) dark energy, no radiation, as pkdgrav3 integrates it
    (H from csmExp2Hub; growth ODE of csmComoveGrowth). Same function as the P(k) checks (class_pk.growth)."""
    sys.path.insert(0, PIPE)
    from class_pk import growth as growth_ode
    w0, wa = dark_energy(c)
    return float(growth_ode([a], float(c['OMEGA_M']), float(c.get('OMEGA_K', 0)), w0, wa)[0])


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
        """Integrand of sigma_8^2 in ln k: P(k) k^3 W(8k)^2 / (2 pi^2), top-hat window of 8 Mpc/h."""
        kk = math.exp(lk)
        x = 8.0 * kk
        w = 3.0 * (math.sin(x) - x * math.cos(x)) / x**3
        return math.exp(sp(lk)) * kk**3 * w * w / (2 * math.pi**2)
    with warnings.catch_warnings():  # the oscillating tail (k > 10 h/Mpc) contributes < 1e-8; same truncation as pkdgrav3
        warnings.simplefilter('ignore')
        var = sum(quad(integrand, lnk[i], lnk[i + 1], epsabs=0, epsrel=1e-10, limit=1000)[0] for i in range(len(k) - 1))
    s8_start = math.sqrt(var)
    a0 = 1.0 / (1.0 + float(c['Z_START']))
    s8 = s8_start * growth(1.0, c) / growth(a0, c)
    return k, T, s8_start, s8


def steps_plan(a_out, a_start, step_da):
    """Number of pkdgrav3 steps per output interval and the cumulative step of each output.

    ``a_out``: scale factors of the outputs; ``a_start``: initial a; ``step_da``: target step in a (at least one
    step per interval). Returns ``(steps per interval, step number of each output)``.
    """
    nst, steps, prev, s = [], [], a_start, 0
    for a in a_out:
        n = max(1, int(round((a - prev) / step_da)))
        nst.append(n)
        s += n
        steps.append(s)
        prev = a
    return nst, steps


def rockstar_mem(N):
    """PBS memory for Rockstar: 1.7 x its particle array (32 B per particle) + 8 GB, rounded up to 16 GB.
    Measured peak 1.75-1.9 x the array at 512^3 (16-64 writers, two z~0 snapshots, pawna job 2536781; 7.1-8.5 GB in
    all 11 batches of L200N512, z = 12 to 0, no trend with z): particles, their overlap copies (+26% at 16 writers,
    +40% at 64) and an 8-byte FoF index per particle, plus the previous snapshot's halos. At 2048^3 (64 writers,
    32 readers, pawna, L200N2048 r3-r5 batches of snapshots 0-34, 18-19 Nov 2026): 422-430 GB = 1.56 x the array.
    128^3: 16gb, 512^3: 16gb, 1024^3: 80gb, 2048^3: 480gb (was 2.2 x: 624gb; at 480gb a 250 GB vahc job fits on the
    same 755 GB pawna node)."""
    gb = 1.7 * 32.0 * float(N)**3 / 1e9 + 8.0
    return '%dgb' % (16 * int(np.ceil(gb / 16.0)))


def verify_resources(N, site, queue):
    """Stage verify (verify_run.py; + KEPT_FILES.json checksums on swarm/pegasus): (cores, mem, wall, pkc_check at once).
    pkc_check is CPU-bound (decompression); measured 20 Nov 2026 on pegasus (debug node, 32 cores, Lustre, r2 2048^3
    snapshots of 54-59 GB): 1 snapshot x 8 threads 1127 s, 1 x 32 threads 350 s, 4 x 8 threads 795 s for 4,
    2 x 16 threads 297 s for 2 (best: 13 kept snapshots ~35 min); 1.1-1.2 GB per pkc_check. SHA-256: 1.1 GB/s one
    file at a time, 4.8 GB/s with 8 threads (archive_run.py's code), ~3.6 TB kept at 2048^3 -> ~13 min. The halo checks
    read headers and binary ID columns: minutes. Up to 512^3 the old defaults (verify measured 490 s at 512^3)."""
    if N <= 512:
        return 8, '32gb', '04:00:00' if queue == 'debug' else '12:00:00', 4
    # pawna: snapshots on NFS, read speed not measured -> longer wall; pegasus/swarm ~1 h expected
    wall = '04:00:00' if queue == 'debug' else ('12:00:00' if site == 'pawna' else '06:00:00')
    return 32, '16gb', wall, 2


def parquet_resources(N):
    """Stage parquet (fits_to_parquet.py, streams in row blocks: ~1 GB per process at most). Measured on pawna, L500N512:
    13.7 GB of FITS in 156 s on 8 processes (631 CPU-s; the tree file alone 155 s, one process). The halo rows grow
    ~8x per factor 8 in particles at fixed box; the tree file is converted by one process (gzip is sequential)."""
    if N <= 512:
        return 8, '16gb', '02:00:00'
    # 2048^3 (profile 20 Nov 2026, swarm, one process: 35 MB/s of .fits.gz): the merged tree file (~225 GB) took
    # ~1.8 h in one process; threaded per file (f0c8c84, swarm compute-1-8, Gadget 2048^3 tree file 1.61 GB gz:
    # 59.6 s -> 21.9 s, same bytes) ~40 min. The ~400 halo files (~260 GB) on the other 7. Up to 1.7 GB per process
    # measured (8 processes ~14 GB) -> 24gb.
    if N <= 1024:
        return 8, '24gb', '06:00:00'
    return 8, '24gb', '12:00:00'


def resources(N):
    """Default PBS resources by size (edit run.conf to change). Measured: 512^3 sim 25 GB / 80 min on 64 cores for 60
    steps; Rockstar see rockstar_mem (one writer per core: 64 writers 2.9x faster than 16 at 512^3 with the output
    on the node's disk); .vahc PP_NPROC x ~13 GB at grid 512 (26 GB with PP_FAST=0)."""
    rsm = rockstar_mem(N)
    if N <= 256:
        return dict(SIM=(64, '120gb', '24:00:00'), RS=(16, rsm, '12:00:00'), CT=(2, '16gb', '12:00:00'),
                    PP=(16, '160gb', '24:00:00'), FITS=(8, '32gb', '06:00:00'), RS_BLOCKS=8, RS_WRITERS=16, PP_NPROC=4)
    if N <= 512:
        return dict(SIM=(96, '200gb', '72:00:00'), RS=(64, rsm, '24:00:00'), CT=(2, '32gb', '24:00:00'),
                    PP=(32, '300gb', '48:00:00'), FITS=(16, '64gb', '12:00:00'), RS_BLOCKS=16, RS_WRITERS=64, PP_NPROC=8)
    # consistent-trees is serial (one process plus a forked writer): 2 cores, so it can share a node.
    # Memory 0.8 GB at 512^3 (6 snapshots), grows with the halos per snapshot: ~8 GB at 1024^3, ~60 GB at 2048^3.
    # 1024^3 and up: consistent-trees with CT_BOX_DIVISIONS=5 (125 tree files, halo_trees_to_catalog.pl one process
    # per file up to 20); vahc on grid 1024, 3 snapshots at a time on half a node (measured on swarm, grid 1024,
    # 3 x 10 threads: 93 GB per process, 240 GB per job; 2048^3 adds ~20 GB per process for the halos)
    ct = (20, '32gb', '48:00:00') if N <= 1024 else (20, '128gb', '168:00:00')
    return dict(SIM=(192, '700gb', '168:00:00'), RS=(64, rsm, '96:00:00'), CT=ct,
                PP=(96, '370gb', '72:00:00'), FITS=(16, '160gb', '24:00:00'), RS_BLOCKS=32, RS_WRITERS=64, PP_NPROC=3)


PEG_HOME = '/mnt/home/project/calam.shadab/sahyadri_pipeline'
# per site: defaults of --root, --archive, --codes, --pkd-io-root, --queue (installs: install_production.sh). A run of another site than pawna is archived
# on pawna (REMOTE_ARCHIVE=1): the manager on pawna pulls its kept files from the site's data node.
SITES = dict(
    pawna=dict(root='/user/{user}/sahyadri_scratch', archive='/storage/{user}/sahyadri_sims',
               codes='/user/{user}/sahyadri_prod/codes', pkd='/user/{user}/sahyadri_prod/pkdgrav3_io', queue='workq'),
    pegasus=dict(root='/data/project/hpc2502016/shadab/sahyadri_scratch', archive='/storage/shadab/sahyadri_sims',
                 codes=PEG_HOME + '/codes', pkd=PEG_HOME + '/pkdgrav3_io', queue='project'),
    # swarm (TIFR): install_production.sh swarm; Lustre /user1 for everything (no staging: ~1.1 GB/s per node)
    swarm=dict(root='/user1/{user}/sahyadri_scratch', archive='/storage/shadab/sahyadri_sims',
               codes='/user1/{user}/sahyadri_pipeline/codes', pkd='/user1/{user}/sahyadri_pipeline/pkdgrav3_io',
               queue='workq'),
    # pride (TIFR theory): install_production.sh pride; the same Lustre /user1 as swarm (own scratch directory);
    # Rockstar, consistent-trees and the venv are built again there (codes), pkdgrav3 for Haswell
    pride=dict(root='/user1/{user}/sahyadri_scratch_pride', archive='/storage/shadab/sahyadri_sims',
               codes='/user1/{user}/sahyadri_pride/codes', pkd='/user1/{user}/sahyadri_pride/pkdgrav3_io',
               queue='workq'),
    # leap (TIFR theory): the same Lustre /user1 (own scratch directory); pkdgrav3 from leap/build_c7.sh (conda
    # environment for glibc 2.17, skylake-avx512). The simulation stage needs only the pipeline's Python (numpy, scipy,
    # pkcreader), and pride's venv runs on leap; Rockstar, consistent-trees and fitsio of pride's codes need a newer
    # glibc, so the halo stages are not installed on leap yet (run them on pride or swarm: same /user1)
    leap=dict(root='/user1/{user}/sahyadri_scratch_leap', archive='/storage/shadab/sahyadri_sims',
              codes='/user1/{user}/sahyadri_pride/codes', pkd='/user1/{user}/sahyadri_c7/pkdgrav3_io',
              queue='workq'),
)
SWARM_WALL = '72:00:00'   # every swarm queue
PRIDE_WALL = '72:00:00'   # pride: one queue (workq), 72 h at most
PRIDE_NODE_MEM = '240gb'  # pride nodes: 252 GB free in PBS on 13 of 16 (one 189 GB, one 220 GB)
LEAP_WALL = '72:00:00'    # leap: workq, 72 h (resources_default.walltime); 340 running cores per user
LEAP_NODE_MEM = '175gb'   # leap nodes: 187 GB in PBS (196302716 kb)
# leap nodes with an ACTIVE hfi1 fabric port that were up on 22 Nov 2026 (compute-4-3: port DOWN; 4-7: node down,
# port active before; 4-1, 4-10: offline): a multi-node simulation names its nodes (SIM_SELECT), since PBS cannot
# leave one node out. The first SIM_NODES are used; if one goes down, edit SIM_SELECT in run.conf (a restart reads it)
LEAP_FABRIC_NODES = ['compute-4-%d' % i for i in (2, 4, 5, 6, 8, 9, 11, 12)]
# pegasus queues (qstat -Qf, 13 Nov 2026): project (-P PEGASUS_PROJECT, the project's budget; 96 h used here),
# regular (user budget, 48 h at most), debug (4 h, tests); budgets in core-hours (amgr checkbalance)
PEGASUS_PROJECT = 'hpc2502016'
# pegasus GPU nodes gpu001-004: one Tesla P100 each, 32 Skylake cores, 385608 MB (PBS); amgr charges a GPU like a core
GPU_NODE_MEM = '360gb'
RS_NODE_MEM = '360gb'    # Rockstar memory per pegasus node (Skylake nodes: 385 GB, ~370 GB usable)
RS_MARGIN = 1.1          # Rockstar over several nodes: memory asked for = 1.1 x rockstar_mem, spread over the nodes
PEGASUS_WALL = dict(project='96:00:00', regular='48:00:00', debug='04:00:00')


def resources_pegasus(N, queue):
    """pegasus: 32-core nodes (385 GB Skylake hpc001-060; 514 GB Ice Lake / Sapphire Rapids hpc061-100, debug queue
    on hpc081+), no node-local staging. Simulation with 2 MPI ranks per node (16 threads each), as pkdgrav3's pegasus
    tests; 3 nodes from 1024^3. Wall times: 96 h (project queue), at most 4 h in the debug queue."""
    rsm = rockstar_mem(N)
    if N <= 256:
        r = dict(SIM=(32, '100gb', '24:00:00', 1), RS=(16, rsm, '12:00:00'), CT=(2, '16gb', '12:00:00'),
                 PP=(16, '160gb', '24:00:00'), FITS=(8, '32gb', '06:00:00'), RS_BLOCKS=8, RS_WRITERS=16, PP_NPROC=4)
    elif N <= 512:
        r = dict(SIM=(32, '400gb', '96:00:00', 1), RS=(32, rsm, '24:00:00'), CT=(2, '32gb', '24:00:00'),
                 PP=(32, '300gb', '48:00:00'), FITS=(16, '64gb', '12:00:00'), RS_BLOCKS=16, RS_WRITERS=32, PP_NPROC=8)
    else:
        # vahc: one node (32 cores, 385 GB): grid 1024, 3 snapshots x ~10 threads (~290-340 GB at 2048^3)
        r = dict(SIM=(32, '480gb', '96:00:00', 3), RS=(32, rsm, '96:00:00'), CT=(20, '128gb', '96:00:00'),
                 PP=(32, '360gb', '96:00:00'), FITS=(16, '160gb', '24:00:00'), RS_BLOCKS=32, RS_WRITERS=32, PP_NPROC=3)
    # Rockstar over several nodes when one node cannot hold it (RS_NODES, job_rockstar.pbs): the fewest nodes that hold
    # 1.1 x rockstar_mem (which already has a margin: 2.2 x the particle array against 1.75-1.9 x measured; the 10 %
    # covers unequal shares of the writers) at most RS_NODE_MEM each; 2048^3: 480 GB -> 2 nodes x 264 GB
    r['RS'] = r['RS'] + (1,)
    need = RS_MARGIN * int(rsm[:-2])
    if need > int(RS_NODE_MEM[:-2]):
        nodes = int(np.ceil(need / int(RS_NODE_MEM[:-2])))
        r['RS'] = (r['RS'][0], '%dgb' % int(np.ceil(need / nodes)), r['RS'][2], nodes)
    cap = PEGASUS_WALL.get(queue, '96:00:00')
    for k in ('SIM', 'RS', 'CT', 'PP', 'FITS'):
        r[k] = (r[k][0], r[k][1], min(r[k][2], cap)) + tuple(r[k][3:])
    return r


def resources_swarm(N):
    """swarm: 12 nodes x 64 Ice Lake cores (Xeon 8358), 504 GB, shared between jobs; 72 h at most; Lustre /user1.
    Simulation with 2 MPI ranks per node; one node (32 cores, free cores of a shared node) up to 512^3, a whole node
    at 1024^3, 2 whole nodes at 2048^3 (~785 GB). Rockstar over shared nodes where one cannot hold it (RS_NODES,
    RS_PLACE=scatter): 2048^3 -> 2 nodes. Measured (17 Nov 2026, 512^3 restart at z = 0.06): 24.5 s/step on 32 cores,
    11.9 s on 2 x 32 (2.06x); Rockstar on 2 shared nodes as fast as on one, memory split evenly."""
    rsm = rockstar_mem(N)
    w = SWARM_WALL
    if N <= 256:
        r = dict(SIM=(32, '60gb', w, 1), RS=(16, rsm, '12:00:00'), CT=(2, '16gb', '12:00:00'),
                 PP=(16, '160gb', '24:00:00'), FITS=(8, '32gb', '06:00:00'), RS_BLOCKS=8, RS_WRITERS=16, PP_NPROC=4)
    elif N <= 512:
        r = dict(SIM=(32, '100gb', w, 1), RS=(32, rsm, '24:00:00'), CT=(2, '32gb', '24:00:00'),
                 PP=(32, '300gb', '48:00:00'), FITS=(16, '64gb', '12:00:00'), RS_BLOCKS=16, RS_WRITERS=32, PP_NPROC=8)
    else:
        sim = (64, '200gb', w, 1) if N <= 1024 else (64, '480gb', w, 2)
        # vahc 32 cores: swarm allows 310 running cores per user and two 2048^3 simulations take 256
        r = dict(SIM=sim, RS=(32, rsm, w), CT=(20, '128gb', w), PP=(32, '400gb', w), FITS=(16, '160gb', '24:00:00'),
                 RS_BLOCKS=32, RS_WRITERS=32, PP_NPROC=3)
    r['RS'] = r['RS'] + (1,)
    need = RS_MARGIN * int(rsm[:-2])
    if need > int(RS_NODE_MEM[:-2]):
        nodes = int(np.ceil(need / int(RS_NODE_MEM[:-2])))
        r['RS'] = (r['RS'][0], '%dgb' % int(np.ceil(need / nodes)), r['RS'][2], nodes)
    return r


def resources_pride(N):
    """pride: 16 nodes x 20 Haswell cores (2 x Xeon E5-2660 v3, AVX2), 189-252 GB, 1 Gb/s Ethernet (no InfiniBand),
    72 h; Lustre /user1 shared with swarm; ~390 GB node-local disk (/, /tmp, /var/tmp). Simulation on one node up to
    1024^3 (~200 GB); 2048^3 would need 5 nodes over 1 Gb/s Ethernet (untested, not recommended). vahc grid 1024
    (~97 GB per snapshot process) 2 snapshots at a time; Rockstar over shared nodes of at most PRIDE_NODE_MEM."""
    rsm = rockstar_mem(N)
    w = PRIDE_WALL
    if N <= 256:
        # Rockstar writers (= RS_NCPU): a product of three factors > 1 for periodic boundaries (12 = 2x2x3, 20 = 2x2x5;
        # 10 = 2x1x5 failed: 128^3 test 764005)
        r = dict(SIM=(20, '60gb', w, 1), RS=(12, rsm, '12:00:00'), CT=(2, '16gb', '12:00:00'),
                 PP=(16, '160gb', '24:00:00'), FITS=(8, '32gb', '06:00:00'), RS_BLOCKS=8, RS_WRITERS=12, PP_NPROC=4)
    elif N <= 512:
        r = dict(SIM=(20, '100gb', w, 1), RS=(20, rsm, '24:00:00'), CT=(2, '32gb', '24:00:00'),
                 PP=(20, '220gb', '48:00:00'), FITS=(16, '64gb', '12:00:00'), RS_BLOCKS=16, RS_WRITERS=20, PP_NPROC=6)
    else:
        sim = (20, '230gb', w, 1) if N <= 1024 else (20, '230gb', w, 5)
        r = dict(SIM=sim, RS=(20, rsm, w), CT=(20, '128gb', w), PP=(20, '245gb', w), FITS=(16, '160gb', '24:00:00'),
                 RS_BLOCKS=32, RS_WRITERS=20, PP_NPROC=2)
    r['RS'] = r['RS'] + (1,)
    need = RS_MARGIN * int(rsm[:-2])
    if need > int(PRIDE_NODE_MEM[:-2]):
        nodes = int(np.ceil(need / int(PRIDE_NODE_MEM[:-2])))
        r['RS'] = (r['RS'][0], '%dgb' % int(np.ceil(need / nodes)), r['RS'][2], nodes)
    return r



def resources_leap(N):
    """leap: 12 nodes x 32 Skylake cores (2 x Xeon Gold 6142), 187 GB, 100 Gb/s hfi1 fabric; 72 h; Lustre /user1 shared
    with swarm; 24 GB node-local disk (no staging, no local FITS temporary files). Simulation with 2 MPI ranks per
    node over the fabric (env_leap.sh); 2048^3 on 6 nodes (expected peak ~750 GB: 125 GB per node; 175gb asked).
    Measured (20-21 Nov 2026, 512^3 restart at z = 0.06, 2 x 16 threads per node): 34.3 s/step on 1 node, 20.0 on 2,
    10.5 on 4, 7.8 on 6 (swarm: 24.5 on 1 x 32 cores, 11.9 on 2 x 32). Halo stages: not installed on leap (SITES)."""
    rsm = rockstar_mem(N)
    w = LEAP_WALL
    if N <= 256:
        r = dict(SIM=(32, '60gb', w, 1), RS=(16, rsm, '12:00:00'), CT=(2, '16gb', '12:00:00'),
                 PP=(16, '160gb', '24:00:00'), FITS=(8, '32gb', '06:00:00'), RS_BLOCKS=8, RS_WRITERS=16, PP_NPROC=4)
    elif N <= 512:
        r = dict(SIM=(32, '100gb', w, 1), RS=(32, rsm, '24:00:00'), CT=(2, '32gb', '24:00:00'),
                 PP=(32, '170gb', '48:00:00'), FITS=(16, '64gb', '12:00:00'), RS_BLOCKS=16, RS_WRITERS=32, PP_NPROC=4)
    else:
        sim = (32, '175gb', w, 2) if N <= 1024 else (32, LEAP_NODE_MEM, w, 6)
        # vahc grid 1024 (~97 GB per snapshot process): one snapshot at a time per 187 GB node
        r = dict(SIM=sim, RS=(32, rsm, w), CT=(20, '128gb', w), PP=(32, '175gb', w), FITS=(16, '160gb', '24:00:00'),
                 RS_BLOCKS=32, RS_WRITERS=32, PP_NPROC=1)
    r['RS'] = r['RS'] + (1,)
    need = RS_MARGIN * int(rsm[:-2])
    if need > int(LEAP_NODE_MEM[:-2]):
        nodes = int(np.ceil(need / int(LEAP_NODE_MEM[:-2])))
        r['RS'] = (r['RS'][0], '%dgb' % int(np.ceil(need / nodes)), r['RS'][2], nodes)
    return r


def leap_select(R):
    """SIM_SELECT for a multi-node simulation on leap: the first SIM_NODES nodes of LEAP_FABRIC_NODES, named."""
    n, ncpu, mem = R['SIM'][3], R['SIM'][0], R['SIM'][1]
    if n > len(LEAP_FABRIC_NODES):
        sys.exit('leap: %d simulation nodes asked for, %d have a working fabric port' % (n, len(LEAP_FABRIC_NODES)))
    return '+'.join('1:host=%s:ncpus=%d:mpiprocs=2:mem=%s' % (h, ncpu, mem) for h in LEAP_FABRIC_NODES[:n])


def fmt_l(L):
    """Box size for run names: ``200`` for 200.0, ``12p5`` for 12.5."""
    return ('%d' % L) if float(L).is_integer() else ('%g' % L).replace('.', 'p')


def same_content(path, text):
    """path exists and holds text, apart from '#' comment lines (the set-up time)."""
    if not os.path.isfile(path):
        return False
    strip = lambda lines: [x for x in lines if not x.lstrip().startswith('#')]   # noqa: E731
    with open(path) as f:
        return strip(f.read().splitlines()) == strip(text.splitlines())


def write_shared(path, text):
    """A file of the set-up directory, shared by its realisations (archived with each run): left untouched (and its
    mtime with it) when only its comments would change, so setting up another realisation does not make the kept
    files of the runs already verified or archived look changed."""
    if same_content(path, text):
        return False
    with open(path + '.tmp', 'w') as f:
        f.write(text)
    os.chmod(path + '.tmp', 0o644)
    os.replace(path + '.tmp', path)
    return True


KEEP_FULL_Z_PROD = '0 0.05 0.1 0.15 0.2 0.3 0.5 0.7137 0.8 1.0 1.9821 2.9563 4.9089'


def write(path, text, force, mode=0o644):
    """Write ``text`` to ``path`` with permissions ``mode``; exits if it exists unless ``force``."""
    if os.path.exists(path) and not force:
        sys.exit('%s exists; use --force to overwrite the configuration files' % path)
    with open(path, 'w') as f:
        f.write(text)
    os.chmod(path, mode)


def site_block(a, R):
    """run.conf lines that depend on the site."""
    if a.site in ('swarm', 'pride', 'leap'):
        return '\n'.join([
            'SITE=%s                 # computed on %s; archived on pawna (the manager pulls the kept files)' % (
                a.site, a.site),
            'REMOTE_ARCHIVE=1        # verify also writes KEPT_FILES.json (checksums for the pull)',
            'PBS_QUEUE=%s' % a.queue,
            'PBS_PROJECT=            # %s has no core-hour budgets' % a.site,
            # whole nodes only where the simulation needs most of their memory (2048^3); else shared like Rockstar
            'SIM_NODES=%d; SIM_MPIPROCS=2; SIM_PLACE=%s   # SIM_NCPU cores per node; 2 MPI ranks per node' % (
                R['SIM'][3], ('scatter:excl' if int(R['SIM'][1][:-2]) > 300 or a.site == 'leap' else 'scatter')
                if R['SIM'][3] > 1 else 'free'),
            'SIM_ENV=%s/scripts/pkdgrav3_pipeline/%s/env_%s.sh   # pkdgrav3 run-time environment (conda MPI, ssh '
            'without the conda library path)' % (SANDBOX, a.site, a.site),
            'SIM_GPU=0',
            'SIM_STAGING=0           # snapshots and checkpoints written straight to SCRATCH (Lustre, ~1.1 GB/s per node)',
            'RS_PLACE=scatter        # a multi-node Rockstar job shares the nodes with other jobs (%s nodes are shared)'
            % a.site]
            + (['SIM_SELECT=%s   # the nodes, named: only these have a working fabric port (setup_production.py '
                'LEAP_FABRIC_NODES; compute-4-3: port down)' % leap_select(R)]
               if a.site == 'leap' and R['SIM'][3] > 1 else []))
    if a.site == 'pawna':
        return ('PBS_QUEUE=%s\n'
                'SIM_STAGING=1           # snapshots and checkpoints written to the node\'s SSD, copied to NFS during the run'
                % a.queue)
    return '\n'.join([
        'SITE=%s                 # computed on %s; archived on pawna (the manager pulls the kept files)' % (a.site, a.site),
        'REMOTE_ARCHIVE=1        # verify also writes KEPT_FILES.json (checksums for the pull)',
        'PBS_QUEUE=%s' % a.queue,
        'PBS_PROJECT=%s           # qsub -P: charged to this project\'s budget (empty: the user\'s budget)' % (
            PEGASUS_PROJECT if a.queue == 'project' else ''),
        'SIM_NODES=%d; SIM_MPIPROCS=2; SIM_PLACE=%s   # SIM_NCPU cores per node; 2 MPI ranks per node' % (
            R['SIM'][3], 'scatter:excl' if R['SIM'][3] > 1 else 'free'),
        'SIM_ENV=%s/pkdgrav3/tools/pegasus/env_pegasus.sh   # pkdgrav3 run-time environment (conda MPI)' % a.pkd_io_root,
        'SIM_GPU=%d               # 1: GPU nodes (ngpus=1 per node) and the CUDA build $PKD_IO_ROOT/build_gpu' % (1 if a.sim_gpu else 0),
        'SIM_STAGING=0           # snapshots and checkpoints written straight to SCRATCH (Lustre)']
                     + (['XC_WALL=02:00:00; PQ_WALL=02:00:00   # debug queue: at most 4 h (VF_WALL below)'] if a.queue == 'debug' else []))


def schedule_run(conf, name, priority):
    """Register `conf` with the running run manager and queue its simulation (through the manager's socket)."""
    sys.path.insert(0, os.path.join(SANDBOX, 'tools', 'sahyadri_manager'))
    import manager
    if not os.path.exists(manager.SOCK):
        sys.exit('the run manager is not running (%s): start tools/sahyadri_manager/start_manager.sh, then '
                 'manager.py add <run.conf> and manager.py schedule %s' % (manager.SOCK, name))
    for action in (dict(action='add_run', conf=conf, paused=False), dict(action='schedule', run=name, priority=priority)):
        if manager.call(action) != 0:
            sys.exit('manager refused %s for %s' % (action['action'], conf))


def main():
    """Command line (argparse, see ``--help``): set up the directories and configuration files of production
    runs for the given box, particle number, cosmologies and realisations on a site.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--site', default='pawna', choices=sorted(SITES),
                    help='where the run is computed (pegasus: run this script on pegasus; the archive stays on pawna)')
    ap.add_argument('--queue', default=None, help='PBS queue (default: workq on pawna, project on pegasus)')
    ap.add_argument('--root', default=None, help='working area (SCRATCH) while the run is computed (site default)')
    ap.add_argument('--archive', default=None,
                    help="long-term copy of the kept data, on pawna ('' = no archive stage; site default)")
    # the production retention (2048^3 runs r2-r5, r10; the 512^3 runs): 13 full snapshots, 9 % (1+3+5) of the others
    ap.add_argument('--keep-full-z', default=KEEP_FULL_Z_PROD, help='full snapshots kept: outputs nearest these z')
    ap.add_argument('--keep-pct', type=float, default=9, help='percent of the particles kept of the other snapshots')
    ap.add_argument('--L', type=float, default=200.0, help='box size [Mpc/h]')
    ap.add_argument('--N', type=int, default=1024, help='particles per side')
    ap.add_argument('--cosmo', default='default',
                    help="cosmologies/<name>.cosmo; several: comma separated and/or shell patterns ('var_*'), or 'all'")
    ap.add_argument('--real', default='1', help='realisations, e.g. 1 or 1,2 or 1-5 (seeds: config/sims/seeds.txt)')
    ap.add_argument('--codes', default=None, help='Rockstar, consistent-trees, venv (setup_pawna.sh; site default)')
    ap.add_argument('--pkd-io-root', default=None, help='pkdgrav3 build (site default)')
    ap.add_argument('--outputs', default=os.path.join(HERE, 'sahyadri_outputs_a.txt'))
    ap.add_argument('--step-da', type=float, default=(1 - 1 / 13.) / 2000,
                    help='top-level step size in a (default: that of the 256^3 reference run)')
    ap.add_argument('--soft-frac', type=float, default=1 / 60., help='softening / mean particle spacing')
    ap.add_argument('--pot-zmax', type=float, default=0.5, help='store the potential at z <= this')
    ap.add_argument('--pot-step', type=float, default=100.0, help='potential step [(km/s)^2]')
    ap.add_argument('--check-interval', type=int, default=None,
                    help='steps between checkpoints (default: 250 on pawna; 100 on swarm and pegasus, where a 2048^3 job '
                         'does ~3-7 steps/h and 250 steps are more than one wall time)')
    ap.add_argument('--sim-nodes', type=int, default=None,
                    help='pegasus: nodes of the simulation (default: 1 up to 512^3, 3 above); memory per node then '
                         '1.5 x the expected peak (747 GiB at 2048^3, scaled as N^3) / nodes, at most 480 GB')
    ap.add_argument('--sim-gpu', action='store_true',
                    help='pegasus: run the simulation on the GPU nodes (one P100 each) with the CUDA build '
                         '$PKD_IO_ROOT/build_gpu (build_pkdgrav3_gpu_pegasus.sh); nodes default 3 from 2048^3 (memory), '
                         '1 below; memory per node at most %s' % GPU_NODE_MEM)
    ap.add_argument('--rs-nodes', type=int, default=None,
                    help='pegasus: Rockstar nodes (default: the fewest for 1.1 x its memory estimate at most %s each; 2 '
                         'at 2048^3); memory per node then 1.1 x the estimate / nodes' % RS_NODE_MEM)
    ap.add_argument('--rs-batch', type=int, choices=(0, 1), default=1,
                    help='1: Rockstar in batches while the simulation runs; 0: one Rockstar job after it')
    ap.add_argument('--no-int-pos', action='store_true',
                    help='positions as doubles (48 B per particle) instead of bMemIntegerPosition (36 B, the default)')
    ap.add_argument('--force', action='store_true', help='overwrite existing configuration files')
    ap.add_argument('--schedule', nargs='?', const=0, type=int, default=None, metavar='PRIORITY',
                    help='register with the run manager and queue the simulations (priority, default 0)')
    a = ap.parse_args()
    S = SITES[a.site]
    if a.check_interval is None:
        a.check_interval = 250 if a.site == 'pawna' else 100
    user = os.environ.get('USER', 'shadab')
    for k, sk in (('root', 'root'), ('archive', 'archive'), ('codes', 'codes'), ('pkd_io_root', 'pkd'), ('queue', 'queue')):
        if getattr(a, k) is None:
            setattr(a, k, S[sk].format(user=user))
    if a.site != 'pawna' and a.schedule is not None:
        sys.exit('--schedule registers with the manager on pawna; for a %s run register it there: '
                 'manager.py add %s:<run.conf> (printed below)' % (a.site, a.site))
    sys.path.insert(0, PIPE)
    import retention
    try:   # the kept percentage must be a sum of the leading subsample files
        retention.kept_subsamples(dict(SUBSAMPLES=SUBSAMPLES, KEEP_SUBSAMPLE_PCT=str(a.keep_pct)))
    except ValueError as e:
        sys.exit(str(e))
    names = cosmo_names(a.cosmo)
    print('cosmologies: %s; realisations %s' % (', '.join(names), a.real))
    for cn in names:
        setup_cosmo(a, cn)


def cosmo_names(spec):
    """'default', 'a,b', 'var_*', 'all' -> cosmology names (cosmologies/<name>.cosmo), in the order given."""
    import fnmatch
    have = sorted(f[:-6] for f in os.listdir(os.path.join(HERE, 'cosmologies')) if f.endswith('.cosmo'))
    out = []
    for part in [p.strip() for p in spec.split(',') if p.strip()]:
        hit = have if part == 'all' else ([part] if part in have else fnmatch.filter(have, part))
        if not hit:
            sys.exit('no cosmology %s (available: %s)' % (part, ', '.join(have)))
        out += [h for h in hit if h not in out]
    return out


def setup_cosmo(a, cosmo):
    """Set up every realisation of cosmology ``cosmo`` for the parsed arguments ``a``.

    For each realisation writes, in the simulation directory: ``p.par`` (pkdgrav3 parameters), ``output_steps.txt``
    (step, a, z of each output), ``run.conf`` (pipeline configuration), ``submit.sh``, ``postprocess.sh`` and
    ``README.txt``; in the halo directory: ``rockstar_input.cfg`` and ``README.txt``. Also derives the run's
    memory estimate, retention and resource requests (see the module docstring).
    """
    a.cosmo = cosmo
    c = read_cosmo(a.cosmo)
    L, N = a.L, a.N
    sim = 'Sahyadri_L%sN%d' % (fmt_l(L), N)
    cname = '%s%d' % (a.cosmo, N)
    stem = '%s/%s' % (sim, cname)
    om, ok = float(c['OMEGA_M']), float(c.get('OMEGA_K', 0))
    ol = 1.0 - om - ok          # dark energy: Lambda, or (w0, wa) fluid with the same density today
    w0, wa = dark_energy(c)
    lcdm = (w0, wa) == (-1.0, 0.0)
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
    res = (resources(N) if a.site == 'pawna' else resources_swarm(N) if a.site == 'swarm'
           else resources_pride(N) if a.site == 'pride' else resources_leap(N) if a.site == 'leap'
           else resources_pegasus(N, a.queue))
    if a.sim_gpu and a.site != 'pegasus':
        sys.exit('--sim-gpu: only pegasus has GPU nodes')
    # pkdgrav3 memory at 2048^3 on one pawna node (191 threads, r5 job 2539063): 830 GiB = RSS 735 + swap 95 GiB
    # (particles 457 GiB at 48 B, trees ~334 GiB at 200 B per node, ~40 GiB other). bMemIntegerPosition: 36 B per
    # particle and 160 B per tree node -> ~650 GiB (0.78 x); 512^3 test (pawna node3, 64 threads, /usr/bin/time):
    # peak RSS 25.0 -> 22.3 GiB, P(k) within 1e-3 up to k = 4 h/Mpc and halo mass function within 0.3 % (24 Nov 2026)
    peak_gb = 830 * 1.0737 * (N / 2048.) ** 3 * (1.0 if a.no_int_pos else 0.78)
    if a.sim_gpu and not a.sim_nodes:
        # the fewest GPU nodes (~370 GB usable each) holding 1.25 x the peak: 1 up to ~1520^3, 2 to ~1920^3, 3 at 2048^3
        # (integer positions; doubles: 1 up to ~1400^3, 2 to ~1760^3, 3 to ~2020^3, 4 at 2048^3)
        a.sim_nodes = max(1, int(np.ceil(1.25 * peak_gb / int(GPU_NODE_MEM[:-2]))))
    if a.sim_nodes:
        if a.site == 'pawna':
            sys.exit('--sim-nodes: pawna simulations run on one node')
        cap = int(GPU_NODE_MEM[:-2]) if a.sim_gpu else 480
        if peak_gb * 1.05 / a.sim_nodes > cap:
            sys.exit('%d %snodes hold at most %d GB each, the simulation needs ~%.0f GB: use more nodes'
                     % (a.sim_nodes, 'GPU ' if a.sim_gpu else '', cap, peak_gb))
        mem = int(min(cap, max(100, 1.5 * peak_gb / a.sim_nodes)))
        res['SIM'] = (res['SIM'][0], '%dgb' % mem, res['SIM'][2], a.sim_nodes)
    if a.rs_nodes:
        if a.site == 'pawna':
            sys.exit('--rs-nodes: pawna Rockstar runs on one node')
        need = RS_MARGIN * int(rockstar_mem(N)[:-2])
        res['RS'] = (res['RS'][0], '%dgb' % int(np.ceil(need / a.rs_nodes)), res['RS'][2], a.rs_nodes)
        if need / a.rs_nodes > int(RS_NODE_MEM[:-2]):
            sys.exit('--rs-nodes %d: %.0f GB per node, more than %s' % (a.rs_nodes, need / a.rs_nodes, RS_NODE_MEM))
    wall_h = int(res['SIM'][2].split(':')[0])
    # pkdgrav3 stops this long before the PBS wall time: the staging drain must still copy the last checkpoint
    # (36 B/particle with integer positions, 48 with doubles; ~52 GB at 1024^3 with doubles, ~35 min at the ~25 MB/s
    # of /storage) and snapshot to NFS
    margin = 60 if N <= 512 else (120 if N <= 1024 else 240)
    if a.site != 'pawna':   # no staging: checkpoints go straight to Lustre (~412 GB at 2048^3, minutes)
        margin = min(margin, 90)
    now = datetime.datetime.now().isoformat(timespec='seconds')

    cdir = os.path.join(a.root, 'sims', sim, cname)
    os.makedirs(cdir, exist_ok=True)
    pk_copy = os.path.join(cdir, 'class_pk_z%g.dat' % zstart)
    tf = os.path.join(cdir, 'transfer_pkdgrav3.dat')
    with open(c['PK_FILE']) as f:
        write_shared(pk_copy, f.read())
    write_shared(tf, '# pkdgrav3 transfer table for %s: k [h/Mpc], T = sqrt(P_CLASS(k, z=%g) / k^ns), ns = %s\n'
                 '# from %s; use with dSpectral = %s, dSigma8 = %.8f\n' % (cname, zstart, c['NS'], c['PK_FILE'], c['NS'], s8)
                 + ''.join('%.12e %.12e\n' % (kk, tt) for kk, tt in zip(k, T)))
    write_shared(os.path.join(cdir, 'outputs_a.txt'), '# output scale factors (from %s)\n' % a.outputs
                 + ''.join('%.16e\n' % x for x in a_out))
    write_shared(os.path.join(cdir, 'cosmology.txt'), (
        '# %s, written by setup_production.py %s from %s\n' % (cname, now, c['file']) +
        ''.join('%s=%s\n' % (kk, c[kk]) for kk in ('OMEGA_M', 'OMEGA_B', 'OMEGA_K', 'HUBBLE', 'NS', 'AS', 'Z_START')) +
        ('OMEGA_L=%.7f\n' % ol if lcdm else 'OMEGA_L=0\nOMEGA_DE=%.7f\n' % ol) + 'W0=%r\nWA=%r\n' % (w0, wa) +
        'SIGMA8_ZSTART=%.8f   # of the CLASS P(k) at Z_START\n'
        'SIGMA8_LINEAR_Z0=%.8f   # the same, scaled to z = 0 with the no-radiation growth factor (= pkdgrav3 dSigma8)\n'
        % (s8_start, s8)))

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
            '# snapshots (.pkc) in compressed/, checkpoints in checkpoints/. With SIM_STAGING=1 (run.conf) the job',
            '# overrides both with the node-local staging directory; the drain copies the files to these directories.',
            'achIoPath       = "compressed/{name}.{step:05d}{type}"',
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
        ] + (['dLambda         = %.7f' % ol] if lcdm else [
            'dLambda         = 0',
            'dOmegaDE        = %.7f     # (w0, wa) dark energy: w(a) = w0 + wa (1 - a)' % ol,
            'w0              = %r' % w0,
            'wa              = %r' % wa]) + [
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
            'iWallRunTime    = %d         # minutes: PBS wall time %d h minus %d min for the last copies' % (wall_h * 60 - margin, wall_h, margin),
            '',
            '# memory layout: ordered, particle ID = Lagrangian lattice index (no bMemParticleID)',
            'bMemUnordered   = False',
            'bMemParticleID  = False',
            ] + ([] if a.no_int_pos else [
            '# positions as 32-bit integers (steps of 2^-31 box), 36 B per particle instead of 48 and 160 B per tree node',
            '# instead of 200: ~0.78 x the memory. Checkpoints keep this layout: a restart must use the same setting',
            'bMemIntegerPosition = True',
            ]) + [
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
SCRATCH=%(root)s                    # working area: sims/<STEM>/r<REAL>, halos/<STEM>/r<REAL>, state/<STEM>/r<REAL>
ARCHIVE=%(archive)s                    # long-term copy of the kept data (stage archive); empty = none
LOGS=%(sdir)s/logs

# retention (retention.py; `retention.py run.conf` prints the resolved plan): what ARCHIVE keeps. The rest stays in
# SCRATCH until the archive is checked and the deletion is approved in the manager (tools/sahyadri_manager).
KEEP_FULL_Z="%(keepz)s"   # full snapshots: the planned output nearest each redshift
KEEP_FULL_SNAPS=""                # ... or Rockstar snapshot numbers (overrides KEEP_FULL_Z)
KEEP_SUBSAMPLE_PCT=%(keeppct)g             # %% of the particles of every other snapshot: a sum of the leading SUBSAMPLES (1,4,9,18,...)
KEEP_HALO_TABLES="basic,extended,tens,vahc"   # FITS halo tables, every snapshot
KEEP_TREES=1                      # merger-tree FITS
KEEP_CHECKPOINT=0                 # 1: also the newest checkpoint
ARCHIVE_FILE_MODE=0444; ARCHIVE_DIR_MODE=0555   # read-only for everyone once checked (0440/0550: group only)

# manager (tools/sahyadri_manager): Rockstar in batches while the simulation runs
RS_BATCH=%(rs_batch)d              # 0: one Rockstar job after the simulation
RS_BATCH_MIN=10         # snapshots per batch while the simulation runs (the last batch takes what is left)

STEM=%(stem)s
REAL=%(r)d
NPART=%(N)d
LBOX=%(L)s
PKC_SUBDIR=compressed                             # .pkc files in <sim dir>/compressed
PKD_PAR=%(sdir)s/p.par               # pkdgrav3 parameter file (written by setup_production.py)
OUT_Z="%(nout)d outputs, see output_steps.txt"    # informational; the outputs are in p.par
SUBSAMPLES="%(sub)s"

# resources (one pawna node: 192 cores, 755 GB; workq maximum wall time 168 h)
SIM_NCPU=%(sim_n)d;  SIM_MEM=%(sim_m)s;  SIM_WALL=%(sim_w)s    # iWallRunTime in p.par = SIM_WALL - %(margin)d min
%(site_block)s
SIM_KEEPCHK=2           # keep the newest 2 complete checkpoints in checkpoints/ (0 = all)
SIM_CHK_BPP=%(chk_bpp)d          # checkpoint bytes per particle (36: bMemIntegerPosition in p.par, 48: doubles); manager space estimates
SIM_CHECK_INTERVAL=%(chk_int)d   # steps between checkpoints (iCheckInterval in p.par; with SIM_STAGING=0 also set on every restart)
RS_NCPU=%(rs_n)d;    RS_MEM=%(rs_m)s;   RS_WALL=%(rs_w)s
RS_NODES=%(rs_nodes)d              # Rockstar nodes: RS_NCPU cores, RS_MEM and RS_NCPU writers on each (server on the first, a client per node)
CT_NCPU=%(ct_n)d;    CT_MEM=%(ct_m)s;   CT_WALL=%(ct_w)s
CT_BOX_DIVISIONS=%(ctdiv)s      # consistent-trees BOX_DIVISIONS (empty: 1, one tree file); the FITS stage merges the files into one
PP_NCPU=%(pp_n)d;    PP_MEM=%(pp_m)s;   PP_WALL=%(pp_w)s
FITS_NCPU=%(fi_n)d;  FITS_MEM=%(fi_m)s; FITS_WALL=%(fi_w)s
PQ_NCPU=%(pq_n)d;    PQ_MEM=%(pq_m)s;   PQ_WALL=%(pq_w)s
VF_NCPU=%(vf_n)d;    VF_MEM=%(vf_m)s;   VF_WALL=%(vf_w)s
VF_PKC_NPROC=%(vf_p)d          # verify: kept full snapshots checked by pkc_check at once (VF_NCPU threads shared)

# halo finding: Rockstar reads the .pkc files; full configuration in the halo directory
RS_CFG=%(hdir)s/rockstar_input.cfg
FORCE_RES=%(fr).8g                       # = pkdgrav3 softening (Mpc/h comoving)
RS_NUM_BLOCKS=%(rsb)d; RS_WRITERS=%(rsw)d          # readers from RS_CFG; writers = RS_NCPU (RS_WRITERS only with RS_WRITERS_FIXED=1)
RS_LOCAL_OUT=%(rs_local)d          # 1: Rockstar writes to the node's disk, catalogs copied here (0: write here directly; pegasus: Lustre)
RS_PKC_SUBSAMPLES=""
MIN_HALO_OUTPUT_SIZE=30; MIN_HALO_PARTICLES=20; UNBOUND_THRESHOLD=0.5

# merger trees
CLEAN_ROCKSTAR=1        # clean_trees.pl: keep out_<snap>.trees, logs/; delete out_*.list, halos_*, hlists/, outputs/

# value-added catalogs (.vahc, .tens): density on PP_GRID^3 from all particles, built block by block
PP_GRID=%(ppgrid)d            # min(N, 1024); memory per process: grid 512 ~13 GB, grid 1024 ~93 GB (PP_FAST=1, measured)
PP_BINARY=1             # .vahc/.tens/.vpot binary (halo_table_io.py; same header lines and values as the text)
PP_DOWN_TO=0            # 0 = all particles (no downsampling)
PP_TENS=1
PP_FAST=1               # 1: vahc_fast.py (identical .vahc/.tens to AddValue, ~12x faster); 0: original AddValue
PP_POT=1                # 1 (needs PP_FAST): out_<snap>.vpot, mean particle potential around halos (snapshots with potential)
PP_NPROC=%(ppn)d             # snapshots in parallel
PP_PK=1                 # matter + halo-mass-bin P(k) and bias of every output from the same grid (out_<snap>.pk),
                        # CLASS P(k) at every output (halos/.../pk/), checked against each other in stage verify
PK_MBINS=100,300,1000,3000,10000   # halo mass bin edges in particles (host halos, Mvir); last bin open
PK_KLIN=0.1             # k [h/Mpc] of the linear-scale checks and the large-scale bias
PK_ABS_TOL=0.02; PK_GROWTH_TOL=0.03   # verify: |P/P_ref - 1| <= 4 sigma + PK_ABS_TOL; growth ratio within PK_GROWTH_TOL
PK_DELTA2_LIN=0.01      # growth checked only on bins with Delta^2_lin(k, z) <= this (mode coupling of the few low-k modes)

# FITS
FITS_NPROC=16
FITS_CHECK_NPROC=4      # snapshots checked at once (2048^3: ~15-30 GB each) plus the tree check
FITS_DELETE_ASCII=0     # keep 0 with the manager: verify reads the .vahc/.tens/.vpot catalogues (deleted from SCRATCH after the archive)

# Parquet copy of the FITS tables and trees (stage parquet after fits: <halo dir>/parquet/, same names; archived and
# verified like the FITS). 0: no Parquet (stage skipped, nothing archived; the directory can then be deleted)
PARQUET=1
''' % dict(name=name, now=now, sandbox=SANDBOX, pkd=a.pkd_io_root, codes=a.codes, root=a.root, sdir=sdir, hdir=hdir,
           archive=a.archive, keepz=a.keep_full_z, keeppct=a.keep_pct,
           stem=stem, r=r, N=N, L='%g' % L, nout=len(a_out), sub=SUBSAMPLES,
           sim_n=R['SIM'][0], sim_m=R['SIM'][1], sim_w=R['SIM'][2], rs_n=R['RS'][0], rs_m=R['RS'][1], rs_w=R['RS'][2], rs_nodes=(R['RS'][3] if len(R['RS']) > 3 else 1),
           ct_n=R['CT'][0], ct_m=R['CT'][1], ct_w=R['CT'][2], pp_n=R['PP'][0], pp_m=R['PP'][1], pp_w=R['PP'][2],
           site_block=site_block(a, R), rs_local=1 if a.site == 'pawna' else 0, margin=margin, chk_int=a.check_interval, chk_bpp=48 if a.no_int_pos else 36, rs_batch=a.rs_batch, fi_n=R['FITS'][0], fi_m=R['FITS'][1], fi_w=R['FITS'][2], vf_n=verify_resources(N, a.site, a.queue)[0], vf_m=verify_resources(N, a.site, a.queue)[1], vf_w=verify_resources(N, a.site, a.queue)[2], vf_p=verify_resources(N, a.site, a.queue)[3], pq_n=parquet_resources(N)[0], pq_m=parquet_resources(N)[1],
           pq_w=parquet_resources(N)[2], fr=force_res, rsb=R['RS_BLOCKS'],
           rsw=R['RS_WRITERS'], ppn=R['PP_NPROC'], ppgrid=min(N, 1024), ctdiv='5' if N >= 1024 else '')
        write(os.path.join(sdir, 'run.conf'), conf, a.force)
        sub = '''#!/bin/bash
# Submit %(name)s (written by setup_production.py %(now)s). Check run.conf, p.par and
# %(hdir)s/rockstar_input.cfg first.
#   ./submit.sh            whole chain: sim -> rockstar -> ctrees -> vahc -> fits -> parquet (each after the previous succeeded)
#   ./submit.sh dry        print the qsub commands only
#   ./submit.sh restart    continue the simulation from the newest checkpoint, then the rest of the chain
#   ./submit.sh halos      rockstar -> ctrees -> vahc -> fits -> parquet (simulation finished)
#   ./submit.sh <options>  any submit_pipeline.sh options, e.g. --stages vahc,fits
SUB=%(pipe)s/submit_pipeline.sh
CONF=%(sdir)s/run.conf
case "${1:-all}" in
  all)     exec "$SUB" "$CONF" ;;
  dry)     exec "$SUB" "$CONF" --dry-run ;;
  restart) exec "$SUB" "$CONF" --restart latest ;;
  halos)   exec "$SUB" "$CONF" --stages rockstar,ctrees,vahc,fits,parquet ;;
  *)       exec "$SUB" "$CONF" "$@" ;;
esac
''' % dict(name=name, now=now, hdir=hdir, pipe=PIPE, sdir=sdir)
        write(os.path.join(sdir, 'submit.sh'), sub, a.force, 0o755)
        pp = '''#!/bin/bash
# Post-processing of %(name)s beyond the pipeline (written by setup_production.py %(now)s). Run it when wanted,
# after the pipeline (verify) has finished and before the archive (or from the run manager page, Post-processing):
#   ./postprocess.sh vpot    halo potential profiles out_<snap>.vpot (outputs that store the potential, z <= 0.5),
#                            then the FITS tables of those outputs again with the phi_* columns, then verify; only
#                            needed when the vahc stage ran with PP_POT=0 (PP_POT=1 writes .vpot in the same pass)
#   ./postprocess.sh xcorr   halo-halo potential cross-correlation at every output with .vpot, then verify
#                            (XC_* in run.conf, job_xcorr.pbs)
#   ./postprocess.sh vprof   stacked volume-weighted potential profiles around halos (xcorr mass bins) at every
#                            output that stores the potential, then verify (VPROF_* in run.conf, job_vprof.pbs)
#   ./postprocess.sh all     vpot, fits, xcorr, vprof, verify
#   ./postprocess.sh dry ... print the qsub commands only
# The products are kept files (FITS, xcorr/): after the archive the archive is not updated by this script.
SUB=%(pipe)s/submit_pipeline.sh
CONF=%(sdir)s/run.conf
DRY=; [ "${1:-}" = dry ] && { DRY=--dry-run; shift; }
case "${1:-}" in
  vpot)  exec "$SUB" "$CONF" --stages vpot,fits,verify $DRY ;;
  xcorr) exec "$SUB" "$CONF" --stages xcorr,verify $DRY ;;
  vprof) exec "$SUB" "$CONF" --stages vprof,verify $DRY ;;
  all)   exec "$SUB" "$CONF" --stages vpot,fits,xcorr,vprof,verify $DRY ;;
  *)     sed -n '2,14p' "$0"; exit 1 ;;
esac
''' % dict(name=name, now=now, pipe=PIPE, sdir=sdir)
        write(os.path.join(sdir, 'postprocess.sh'), pp, a.force, 0o755)
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
        if a.schedule is not None:
            schedule_run(os.path.join(sdir, 'run.conf'), name, a.schedule)
        elif a.site != 'pawna':
            print('        register on pawna: tools/sahyadri_manager/manager.py add %s:%s/run.conf; then schedule %s:%s'
                  % (a.site, sdir, a.site, name))
    print('''
Nothing was submitted. Check (and edit if needed) for each realisation:
  r<R>/run.conf  r<R>/p.par  halos/.../r<R>/rockstar_input.cfg   (and %s/cosmology.txt)
then submit with   <sim dir>/submit.sh dry   and   <sim dir>/submit.sh''' % cdir)
    if N > 512:
        print('NOTE: N = %d > 512: runs above 512^3 need approval before submission.' % N)
    if N >= 2048:
        print('NOTE: 2048^3 needs ~750 GB on one node (peak RSS 747 GiB measured) and ~8-10 days: raise SIM_WALL '
              'beyond the 168 h workq limit or plan restarts (submit.sh restart).')


if __name__ == '__main__':
    main()
