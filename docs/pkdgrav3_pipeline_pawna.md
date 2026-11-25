# Sahyadri pipeline with pkdgrav3 on pawna

How to run the full chain on pawna: **simulation (pkdgrav3, compressed `.pkc` snapshots) → halos (Rockstar,
reading `.pkc` directly) → merger trees (consistent-trees) → value-added catalogs (`.vahc`, `.tens`) → FITS**.
The scripts are in [`scripts/pkdgrav3_pipeline/`](../scripts/pkdgrav3_pipeline/). One config file describes a
run; `submit_pipeline.sh` submits the stages as a chain of PBS jobs.

Branch: `rockstar-pkc-reader` (not merged into `master`). A LaTeX report with all tests, bugs, plots and
resource measurements is in [`docs/report/`](report/) (`pkdgrav3_pipeline_report.pdf`; figures made by
`make_figures.ipynb` from `data/`). Status of the tests is in
[section 9](#9-what-was-tested); problems and fixes in [section 10](#10-troubleshooting).

**Production runs** (Sahyadri layout under `/storage/shadab/sahyadri_sims`): [section 11](#11-production-runs-setup_productionpy).

**Contents:** [Quick start](#0-quick-start-1283-test-about-1-h) ·
[1 Layout](#1-directory-layout-of-a-run) · [2 Setup](#2-one-time-setup) · [3 Configure](#3-configure-a-run) ·
[4 Submit](#4-submit) · [5 Stages](#5-the-stages) · [6 postprocess.py](#6-other-post-processing-postprocesspy) ·
[7 Fixes](#7-fixes-made-while-setting-this-up-2026-11-11) · [8 Larger runs](#8-larger-runs-resources-and-pegasus) ·
[9 Tests](#9-what-was-tested) · [10 Troubleshooting](#10-troubleshooting) · [11 Production runs](#11-production-runs-setup_productionpy)

```
 job_sim.pbs        job_rockstar.pbs       job_ctrees.pbs            job_vahc.pbs            job_fits.pbs
 pkdgrav3   ──────► Rockstar (PKC) ──────► consistent-trees ───────► run_addvalue.py ──────► compress_halo_outputs.py
 snap.<step>.s*.pkc out_<snap>.list        out_<snap>.trees          out_<snap>.vahc/.tens   compressed/*.fits.gz
 snapnames.txt      halos_<snap>.*         trees/, ../scales.txt                             (+ check_compressed_halos.py)
        └──────────── particles (.pkc) are read again by job_vahc ──────────┘
```

## 0. Quick start (128^3 test, about 1 h)

On the pawna login node, as your own user (replace the paths if you keep things elsewhere):

```bash
# 1. pkdgrav3 with .pkc output (once; 20-40 min)        -> /storage/$USER/pkdgrav3_io
git clone -b main https://github.com/shadaba/pkdgrav3.git /tmp/$USER-pkd-setup
bash /tmp/$USER-pkd-setup/tools/pawna/setup_pawna.sh /storage/$USER/pkdgrav3_io

# 2. this repository, Rockstar, consistent-trees, Python venv (once; ~40 min, mostly pip)
mkdir -p /storage/$USER/pipeline_test && cd /storage/$USER/pipeline_test
git clone -b rockstar-pkc-reader https://github.com/a-paranjape/sahyadri-sandbox.git
bash sahyadri-sandbox/scripts/pkdgrav3_pipeline/setup_pawna.sh $PWD/codes /storage/$USER/pkdgrav3_io

# 3. one config per run: the 128^3 test config already points to these paths
cp sahyadri-sandbox/scripts/pkdgrav3_pipeline/example_test128.conf test128.conf
sahyadri-sandbox/scripts/pkdgrav3_pipeline/submit_pipeline.sh test128.conf --dry-run   # look first
sahyadri-sandbox/scripts/pkdgrav3_pipeline/submit_pipeline.sh test128.conf             # 5 chained jobs

# 4. follow it
qstat -u $USER
tail -f data/logs/*.pkdtest_lcdm128_r1.*.log
ls data/halos/pkdtest/lcdm128/r1/compressed            # the result: FITS catalogs + trees
```

For a production-size run start from `example_512.conf` (section 3) instead.

## 1. Directory layout of a run

Everything follows the layout that `scripts/post-process/readers.py` expects, under `SCRATCH` of the config:

| Path | Written by | Contents |
|---|---|---|
| `$SCRATCH/sims/<STEM>/r<REAL>/` | sim | `p.par`, `snap.<step>.s<pct>.<writer>.pkc` (one set per output), `snap.<step>.done`, `snapnames.txt`, `run.log`, `snap.log` |
| `$SCRATCH/halos/<STEM>/r<REAL>/` | rockstar, ctrees, vahc, fits | Rockstar `out_<snap>.list`, `halos_<snap>.*` (deleted by the ctrees stage unless `CLEAN_ROCKSTAR=0`); then `out_<snap>.trees`, `trees/`, `logs/` (rockstar.cfg, merger_tree.cfg, logs), `out_<snap>.vahc`, `out_<snap>.tens`, `compressed/` |
| `$SCRATCH/halos/<STEM>/scales.txt` | ctrees | `snap scale` of every output (read by `HaloReader`) |
| `$LOGS/` | every job | `<stage>.<STEM>_r<REAL>.<jobid>.log`, `chain.*.txt` |

`snapnames.txt` maps Rockstar's snapshot number to pkdgrav3's step: line `<snap>` (0-based) is the
zero-padded step of that output. Rockstar (`SNAPSHOT_NAMES`) and `readers.py` both use it.

## 2. One-time setup

### 2a. pkdgrav3

Our pkdgrav3 fork (`https://github.com/shadaba/pkdgrav3`, branch `main`) has the compressed `.pkc` output.
Build it as described in pkdgrav3's `docs/running_on_pawna.md` (one command, 20-40 min):

```bash
git clone -b main https://github.com/shadaba/pkdgrav3.git /tmp/$USER-pkd-setup
bash /tmp/$USER-pkd-setup/tools/pawna/setup_pawna.sh /storage/$USER/pkdgrav3_io
```

This gives `/storage/$USER/pkdgrav3_io/{pkdgrav3,build/pkdgrav3,build_snapcomp,venv}`. Dependencies from the
system: GCC 11, CMake 3.22, OpenMPI 5 (module), HDF5 1.10, GSL, FFTW 3, Boost, Python 3.10. zstd, cpptrace
and libdwarf are downloaded by CMake. `build_snapcomp/` also holds `pkc_check`, `pkc_to_gadget` and the
zstd library that Rockstar links against.

`pkc_to_gadget` with `--blocks`, `--from-hdf5` and version-3 headers (used to validate the Rockstar reader
against Gadget input, not by the pipeline) is in pkdgrav3 `main` since 2026-11-11 (merge `a99f6142` of branch
`pkc-to-gadget-v3`). An install built before that: `git -C /storage/$USER/pkdgrav3_io/pkdgrav3 pull --ff-only`,
then rebuild only the tools (`cmake --build /storage/$USER/pkdgrav3_io/build_snapcomp -j 16`; the `pkdgrav3`
binary is not affected).

### 2b. This repository and the halo codes

```bash
cd /user/$USER/Projects            # anywhere
git clone -b rockstar-pkc-reader https://github.com/a-paranjape/sahyadri-sandbox.git
bash sahyadri-sandbox/scripts/pkdgrav3_pipeline/setup_pawna.sh /storage/$USER/sahyadri_codes /storage/$USER/pkdgrav3_io
```

`setup_pawna.sh <CODES> [PKD_IO_ROOT]` builds into `CODES`:

| Directory | What | Source |
|---|---|---|
| `rockstar/` | Rockstar, `make with_hdf5_pkc` (`FILE_FORMAT = "PKC"`, also AREPO/Gadget HDF5) | [github.com/shadaba/rockstar](https://github.com/shadaba/rockstar), branch `sahyadri`, at the commit pinned in `setup_pawna.sh` (`ROCKSTAR_REF`): upstream `36ce9eea36ee` plus one commit per Sahyadri change with its author (Aseem's curvature and Gadget-4 HDF5 changes, Saee's w0/wa in `gen_merger_cfg.pl`, the `.pkc` reader; see `SAHYADRI.md` there). Its source is byte-identical to the older overlay `code/Rockstar/gfcstanford-rockstar-36ce9eea36ee/` on upstream, which `setup_pawna.sh` uses only if it cannot fetch the repository. New Rockstar changes go to that repository. |
| `ctrees/` | consistent-trees, `make` | `git clone https://bitbucket.org/pbehroozi/consistent-trees.git` at `7c20add`, then `code/ConsistentTrees/consistent-trees-7c20add/` copied over (w0-wa and curvature background, `README_sahyadri.txt`) |
| `venv/` | self-contained Python 3 venv with numpy, scipy, pandas, h5py, fitsio, zstandard (pip; the system scipy on pawna needs numpy < 1.25), and Corrfunc from the `adematti/Corrfunc` fork, branch `desi` (section 14) | PyPI, GitHub |
| `lib/` | `cic_update.so`, `ngp_update.so`, `vpot_sums.so` (`.vpot`), `halo_pair_sums.so` (section 12) compiled from `scripts/post-process/*.c` | this repository |

It needs the pkdgrav3 install of 2a (for zstd), `git`, `cc`, `make`, `perl`, `python3` and network access
to bitbucket.org and PyPI (all available on the pawna login node). Rerunning it rebuilds everything.

The post-processing scripts are used from the clone itself (`scripts/post-process/`); `SAHYADRI_SCRATCH` and
`SAHYADRI_LIBDIR` (set by the jobs) tell `utilities.Paths` where the data and the compiled helpers are, so
nothing in the code needs editing per user.

## 3. Configure a run

Copy [`example_512.conf`](../scripts/pkdgrav3_pipeline/example_512.conf) and edit it. It is a bash file with:

| Group | Keys | Notes |
|---|---|---|
| paths | `SANDBOX`, `PKD_IO_ROOT`, `CODES`, `SCRATCH`, `LOGS` | from section 2 |
| run | `STEM`, `REAL`, `NPART`, `LBOX`, `SEED` | `STEM` like `sahyadri/default512`; `NPART > 512` needs approval |
| cosmology | `HUBBLE`, `OMEGA_M`, `OMEGA_L`, `OMEGA_DE`, `W0`, `WA`, `SIGMA8`, `NS`, `TF_FILE` | LCDM: `OMEGA_DE=0, W0=-1, WA=0`. w0-wa: `OMEGA_L=0`, `OMEGA_DE=1-OMEGA_M`. `TF_FILE`: z=0 transfer function file read by pkdgrav3 (`achTfFile`) |
| outputs | `Z_START`, `OUT_Z`, `STEPS` | one snapshot per entry of `OUT_Z` (decreasing), `STEPS[i]` steps to reach it. For the Sahyadri 101 outputs write the 101 redshifts and steps |
| snapshot format | `SUBSAMPLES`, `ALSO_HDF5`, `EXTRA_PAR` | `.pkc` subsample percents; `ALSO_HDF5=1` also writes HDF5 (validation only, large); `EXTRA_PAR="key = value; key = value"` adds pkdgrav3 parameters (e.g. `dCompPotStep = 0.0` for no potential) |
| resources | `*_NCPU`, `*_MEM`, `*_WALL` per stage | a pawna node has 192 cores and 755 GB |
| Rockstar | `FORCE_RES`, `RS_NUM_BLOCKS`, `RS_WRITERS`, `RS_WRITERS_FIXED`, `RS_LOCAL_OUT`, `RS_LOCAL_DIR`, `RS_SYNC_S`, `RS_PKC_SUBSAMPLES`, `MIN_HALO_OUTPUT_SIZE`, `MIN_HALO_PARTICLES`, `UNBOUND_THRESHOLD` | `FORCE_RES` empty = `LBOX/NPART/50` (pkdgrav3's default softening); writers = cores of the job unless `RS_WRITERS_FIXED=1`; `RS_LOCAL_OUT=1` (default) writes on the node's disk (section 5.2); other values as in the production `config/halos/rockstar_template.cfg` |
| trees | `CLEAN_ROCKSTAR` | 1: as production (`clean_trees.pl`), 0: keep all Rockstar outputs |
| value added | `PP_GRID`, `PP_DOWN_TO`, `PP_TENS`, `PP_NPROC`, `PP_BINARY`, `PP_SNAPS` | grid 512 (production: `min(N, 1024)`, section 15); `PP_DOWN_TO=0` (all particles, streamed); `PP_BINARY=1` binary catalogues |
| FITS | `FITS_NPROC`, `FITS_CHECK_NPROC`, `FITS_DELETE_ASCII` | delete only after looking at the check log |
| trees | `CT_BOX_DIVISIONS` | 5 above 512^3 (125 tree files, merged in the FITS stage; section 15.4) |
| Parquet, verify | `PQ_*`, `VF_*`, `VF_PKC_NPROC`, `KEPT_SUMS_NPROC` | section 15 |

The pkdgrav3 parameter file is generated by `make_pkdgrav3_par.py` (ordered particles, IDs = lattice index,
2LPT, `classic` accuracy switches, `.pkc` output only); look at `sims/<STEM>/r<REAL>/p.par` after the run, or
generate it beforehand with `set -a; source my.conf; CONF=my.conf python3 make_pkdgrav3_par.py test.par`.

## 4. Submit

```bash
P=/user/$USER/Projects/sahyadri-sandbox/scripts/pkdgrav3_pipeline
$P/submit_pipeline.sh my.conf --dry-run                 # print the qsub commands
$P/submit_pipeline.sh my.conf                           # all five stages, chained with afterok
```

Output (one line per job, also saved in `$LOGS/chain.<run>.<date>.txt`):
```
sim 2536500.pawna (select=1:ncpus=192:mem=700gb, 24:00:00)
rockstar 2536501.pawna (select=1:ncpus=16:mem=200gb, 12:00:00, -W depend=afterok:2536500.pawna)
...
```

Options:

| Option | Use |
|---|---|
| `--stages ctrees,vahc,fits` | run a subset (in pipeline order), e.g. redo the later stages after a change |
| `--stages xcorr` | the optional halo-halo potential cross-correlation (section 12); never part of the default chain |
| `--stages ximult` | the optional correlation multipoles in real/redshift space (section 14); never part of the default chain |
| `--after <jobid>` | the first stage waits for an existing job (afterok), e.g. chain the halo stages after a simulation that is already running |
| `--hold` | submit the first job held; `qrls <id>` starts the chain |
| `--dry-run` | print only |

Examples:

```bash
# simulation now, everything else later by hand
$P/submit_pipeline.sh my.conf --stages sim
# halos + trees + value added + FITS after the simulation job 2536500 finishes successfully
$P/submit_pipeline.sh my.conf --stages rockstar,ctrees,vahc,fits --after 2536500.pawna
# several realisations: one config per realisation (REAL=1..5, different SEED), each its own chain
for r in 1 2 3 4 5; do sed "s/^REAL=.*/REAL=$r/; s/^SEED=.*/SEED=$((1000+r))/" my.conf > my_r$r.conf
  $P/submit_pipeline.sh my_r$r.conf; done
# only the FITS conversion of an existing run, deleting the ASCII after verification
sed -i 's/^FITS_DELETE_ASCII=.*/FITS_DELETE_ASCII=1/' my.conf; $P/submit_pipeline.sh my.conf --stages fits
```

Monitoring: `qstat -u $USER` (dependent jobs wait in state `H`); the log of each job is
`$LOGS/<stage>.<STEM>_r<REAL>.<jobid>.log` (the jobs write it themselves; PBS's `-o` copy is unreliable on
pawna). If a stage fails, PBS deletes the jobs that depend on it with `afterok`; fix the cause and resubmit
from that stage with `--stages`.

Restarting: every stage can be resubmitted on its own. `vahc` and `fits` skip snapshots/files that are
already done; `sim`, `rockstar` and `ctrees` start over (`rockstar` with `RS_RESTART=1` in the config uses
Rockstar's own `restart.cfg`).

Running a stage without PBS (e.g. on an interactive node):
`CONF=$PWD/my.conf PIPE_ENV=$P/pipeline_env.sh STAGE=vahc bash $P/job_vahc.pbs`.

## 5. The stages

### 5.1 sim — `job_sim.pbs`
Writes `p.par` (from the config) in `$SIM_DIR` and runs
`mpirun --bind-to none -np 1 $PKD_IO_ROOT/build/pkdgrav3 -sz <ncpus> p.par` after sourcing pkdgrav3's
`bench/phase0/env_pawna.sh`. Each output is a set of `.pkc` files, one per subsample percent and writer
(`snap.<step>.s<pct>.<writer>.pkc`, about 10 bytes per particle at default precision; format in pkdgrav3's
`docs/snapcomp_format.md`). At the end it writes `snapnames.txt` from the `snap.<step>.done` markers and runs
`pkc_check` on every output; it fails if an output is missing.
For runs above 512^3 (approval needed), follow pkdgrav3's `docs/running_on_pawna.md` (memory, `/dev/shm`,
staging the snapshots on the node's SSD); the template here writes straight to `$SCRATCH`.

### 5.2 rockstar — `job_rockstar.pbs`
Writes `rs_input.cfg` with `FILE_FORMAT = "PKC"`, `FILENAME = "snap.<snap>.<block>"`, `SNAPSHOT_NAMES` =
`snapnames.txt`, all outputs, and the production halo settings, then starts the server and the client
(`auto-rockstar.cfg`) in the same job on one node. Cosmology, box, particle mass and scale factor come from
the `.pkc` headers (also w0/wa). `RS_NUM_BLOCKS` readers each read an equal share of the `.pkc` blocks.
`RS_PKC_SUBSAMPLES="1,3"` would find halos in a 4% subsample (quick looks only).
Reader details and validation: [rockstar_pkc_reader_plan.md](rockstar_pkc_reader_plan.md).

**Output on the node's disk (`RS_LOCAL_OUT=1`, default).** Rockstar writes its halo files (`halos_<snap>.*.bin`,
`.ascii`, `out_<snap>.list`) inside the halo-finding phase, so on `/storage` that phase waits for the disk. The job
sets `OUTBASE` to `/var/tmp/$USER_rockstar_<job>` (`RS_LOCAL_DIR` to change the disk) and copies to the halo
directory every `RS_SYNC_S` seconds (default 600) and at the end: the finished `out_<snap>.list`, `rockstar.cfg`
(with `OUTBASE` pointed back to the halo directory, which `gen_merger_cfg.pl` reads), `restart.cfg` and the `.bin`
files of the newest complete snapshot only (the previous snapshot's halos are all a restart needs). After a
successful run every copied `out_<snap>.list` is compared with the local one before the local directory is deleted;
after a failure it is left on the node. `RS_RESTART=1` copies the saved `.bin` files back to the local disk.
`RS_LOCAL_OUT=0` writes to the halo directory as before.

**Writers.** One writer per core of the job (`PBS_NCPUS`, else `NCPUS`, else `RS_NCPU`); the job writes the
configuration it runs to `rs_job.cfg` (`NUM_WRITERS`, `FORK_PROCESSORS_PER_MACHINE`, `OUTBASE` set, the rest from
`RS_CFG` or the generated `rs_input.cfg`). `RS_WRITERS_FIXED=1` uses `RS_WRITERS` instead.

**Measured** (pawna job 2536781, node3, 64 cores: the last two outputs of a 512^3/200 Mpc/h run, z = 0.1 and 0, the
second with the merger-tree step; wall time of the server and client, phase times from the server log, memory
summed over all Rockstar processes every 1 s):

| Writers | Output on | Wall [s] | Peak memory (PSS) [GB] |
|---|---|---|---|
| 16 | `/storage` | 188.5 | 7.49 |
| 16 | node SSD | 113.8 | 7.49 |
| 32 | node SSD | 77.7 | 7.38 |
| 64 | node SSD | 64.1 | 8.12 |

Halo finding of the z = 0 output took 73 s (16 writers, `/storage`), 42 s (16, SSD), 25 s (32) and 16 s (64);
reading the `.pkc` files 2-4 s. Copying the two `out_<snap>.list` back took 6-14 s. Halo counts at z = 0 were
254,279-254,442 in all runs (0.06%, Rockstar is not reproducible run to run at the 0.1% level).
`OUTPUT_FORMAT = "BINARY"` still wrote the ascii files and saved nothing.

**Memory.** Measured at 1 s intervals the peak is in the friends-of-friends step, not in reading: reading holds
the particle array (4.21 GB = 134M x 32 B), the FoF step 6.9-7.4 GB, halo finding 4.5-6.5 GB. The FoF peak is
the particles plus their copies in the writers' overlap shells (`OVERLAP_LENGTH = 3`; +26% with 16 writers in
this box, +40% with 64) plus an 8-byte group index per particle; at z = 0 the previous output's halos (for the
merger tree) add ~0.5 GB. Peak = 1.75-1.9 x the particle array. The job requests `RS_MEM`; `setup_production.py`
sets it to 1.7 x 32 B x N^3 + 8 GB, rounded up to 16 GB (`rockstar_mem`): 16 GB at 128^3, 16 GB at 512^3, 80 GB at
1024^3, 480 GB at 2048^3 (measured 422-430 GB on pawna, L200N2048 r3-r5 Rockstar batches of snapshots 0-34, Nov 2026). 3200^3
would need ~1.8 TB, more than one pawna node.

### 5.3 ctrees — `job_ctrees.pbs`
`gen_merger_cfg.pl rockstar.cfg` (writes `outputs/merger_tree.cfg` with Om, Ol, h0 and w0, wa from the
`out_<snap>.list` header), then `do_merger_tree.pl` and `halo_trees_to_catalog.pl` of the patched
consistent-trees, then `scripts/assist/clean_trees.pl` as on pegasus: `hlists/hlist_<a>.list` →
`out_<snap>.trees`, `outputs/scales.txt` → `../scales.txt`, configs to `logs/`, Rockstar outputs deleted.
Note: `clean_trees.pl` matches hlists by the first 5 characters of the scale factor, so outputs closer than
0.001 in `a` would collide (true of the pegasus pipeline too). `CT_BOX_DIVISIONS=5` (default above 512^3) writes
125 tree files instead of one, about 2.5 times faster (section 15.4).

### 5.4 vahc — `job_vahc.pbs` → `scripts/post-process/run_addvalue.py`
For each snapshot: matter density (CIC on `PP_GRID^3`) from the particles, then `AddValue` on all halos of
`out_<snap>.trees` in file order: `b1` (halo-by-halo bias, k < 0.1 h/Mpc), tidal-tensor eigenvalues at
2, 4, 6, 8 R200b and 2, 3, 5 Mpc/h, density-Hessian eigenvalues at 3, 5 Mpc/h → `out_<snap>.vahc`; with
`PP_TENS=1` the six tidal-tensor components at the same scales → `out_<snap>.tens`. The same computation as
the `add_value` branch of `postprocess.py` (production settings: kmax 0.1, m200b, seed 42), but one
density field serves both outputs, snapshots run in parallel (`PP_NPROC`), finished snapshots are skipped and
files are written under a temporary name and renamed when complete.

Particles: `readers.SnapshotReader` reads `.pkc` through `pkcreader.py` (copied from pkdgrav3
`tools/pkcreader`) when no Gadget HDF5 snapshot exists. With `PP_DOWN_TO=512` it decodes only the smallest
`.pkc` subsamples that hold 512^3 particles (1% + 3% of a 2048^3 run) and keeps a random 512^3 of them, the
same kind of random subset production took from the full Gadget snapshot, at about 1/25 of the reading.

**No downsampling (`PP_DOWN_TO=0`, the default and the production setting):** with `.pkc` input the CIC density
is built block by block (65,536 particles at a time, decoded by `--threads` threads), so the particles add almost no
memory at any resolution. It gives byte-identical `.vahc`/`.tens` to loading all positions at once (pawna job
2536526, 128^3 and 512^3, all 22 files compared with `cmp`) in the same time (33-35 s per 512^3 snapshot with 4
threads; ~4-5 min expected at 1024^3, ~35 min at 2048^3, against ~30 min of `AddValue` per snapshot at grid 512).
`--no-stream` restores the old path.

**`vahc_fast.py` (`PP_FAST=1`, the default).** `run_addvalue.py --fast` computes `.vahc` and `.tens` in one pass:
one real FFT of the density, a separable Gaussian kernel, one inverse FFT per tensor component and scale on
`--threads` threads, vectorized eigenvalues and interpolation, and b1 as one filtered field. Its output is identical
to `AddValue`'s: in-process comparison on the same density and halos, every column equal in float32 and the same
halo IDs (pawna job 2536678, 128^3 snap 4, grid 128; job 2536679, 512^3 snap 5, grid 512, 234,258 halos). Time for
that 512^3 snapshot: 147 s (8 FFT threads) against 1126 s (`.vahc`) + 722 s (`.tens`) for `AddValue`; the tidal
FFTs are 84% of the 147 s. `PP_FAST=0` restores the original path.

**`.vpot` (`PP_POT=1`, needs `PP_FAST=1`).** For snapshots whose `.pkc` files store the potential
(`dCompPotMaxRedshift`), `out_<snap>.vpot` holds the mean pkdgrav3 particle potential around each halo, in
(km/s)^2, with pkdgrav3's constant softened self term -(105/64) m/eps removed: particle counts and means within
rs, R200b, 0.5, 1 and 1.5 Rvir (`Np_<R>`, `phi_<R>`; R200b from M200b) and Gaussian-weighted means at the seven
`.vahc` scales; nan where undefined. The FITS stage appends these columns (except `haloID`) to
`out_<snap>_vahc.fits.gz` (card `HAS_VPOT`). Checks: the sphere values equal a KD-tree sum over all particles
for 300 random halos per sphere (0 count and 0 mean mismatches, float32 rounding; 128^3 snaps 70 and 100,
512^3 snap 5); the grid-scale values agree with direct Gaussian sums to <= 1e-3 (one 1e-2 at 5 Mpc/h); pkdgrav3's
potential with the self term removed matches an FFT Poisson solution of the same particles with slope 1.002-1.007,
correlation >= 0.998. Cost: 47-68 s per snapshot (grid 512, 4-5 threads) on top of the tidal FFTs.

Memory: about 6 × 16 × `PP_GRID`^3 bytes per process with `PP_FAST=1` (measured: 93 GB per process at grid 1024
on 2048^3 particles, section 15.3; 12-13 GB per process at grid
512, 128^3 and 512^3 runs; job peak 78 GB for 6 processes on the 512^3 run with `.vpot`), about 12 × 16 ×
`PP_GRID`^3 with `PP_FAST=0` (~26 GB at grid 512, ~85 GB at 768). Choose `PP_NPROC` so that `PP_NPROC` times that
fits in `PP_MEM`. Measured with `PP_FAST=0` (512^3 run, `PP_GRID=512`, `PP_NPROC=4`): largest process 25.5 GB,
job peak 91.6 GB.

**When to run it.** `.vahc`/`.tens` rows are the halos of `out_<snap>.trees`, with consistent-trees IDs
(`haloID` = `id` of the hlist), so this stage needs the finished consistent-trees catalogs and cannot start
before the ctrees stage. It must run before the FITS stage: the FITS stage puts `b1` into
`out_<snap>_basic.fits.gz` and writes `_vahc`/`_tens` files only if the ASCII files exist, and
`FITS_DELETE_ASCII=1` removes `out_<snap>.trees`. After Rockstar the particles are needed only here. With
`PP_DOWN_TO=0` (all particles) keep every `.pkc` file until this stage has finished; only with `PP_DOWN_TO=512`
would the 1% and 3% files (4% of the snapshot volume) suffice to (re)compute `.vahc`/`.tens` later.

### 5.5 fits — `job_fits.pbs` → `scripts/post-process/compress_halo_outputs.py`
`compress_halo_outputs.py all` converts `out_<snap>.trees/.vahc/.tens` and `trees/` to `compressed/` with
the cosmology, Rockstar and consistent-trees settings and the `.pkc` simulation header in the FITS headers
(details: [`README_compress_halo_outputs.md`](../scripts/post-process/README_compress_halo_outputs.md)),
using node-local `/var/tmp` for temporary files (or the halo directory when `/var/tmp` lacks room), then
`check_compressed_halos.py` compares every column with the catalogues (`FITS_CHECK_NPROC` snapshots at a time).
Several tree files (`CT_BOX_DIVISIONS`) are merged into one tree FITS (section 15.4). `readers.HaloReader.prep_halos(use_fits=True)` reads the result.

### 5.6 parquet — `job_parquet.pbs` → `scripts/post-process/fits_to_parquet.py` (`PARQUET=1`)
A Parquet copy of every FITS table and tree file in a parallel directory, same names:
`compressed/out_<snap>_<table>.fits.gz → parquet/out_<snap>_<table>.parquet`, `compressed/trees/*.fits.gz →
parquet/trees/*.parquet`. FITS stays the reference and is not touched; `PARQUET=0` in run.conf turns the stage off
(nothing converted or archived; the directory can then be deleted).

- **Low memory at any size:** each `.fits.gz` is decompressed as a stream and the binary table converted in blocks
  of about 256 MB (`--block-mb`), each written as one Parquet row group. Nothing holds a whole file.
- **Threads per file:** decompression, decoding + hashing and encoding overlap (`--threads`); 2.7 times faster on a
  2048^3 tree file, same bytes (section 15.3).
- **Checks per file:** the decompressed data unit against the FITS `DATASUM`; the Parquet file read back row group
  by row group, with per-column BLAKE2b hashes equal to those of the values converted (numbers bit for bit, strings
  by value). The decoder (TFORM/TZERO → types) matches fitsio on every column (`fits_to_parquet.py --selftest`).
- **Metadata** (Parquet footer): the primary and table FITS headers, the PROVENANCE table, and the source name,
  size and mtime. A file whose Parquet records the current FITS size and mtime is skipped, so a restart is a rerun.
- **Encodings:** zstd level 3; dictionary for basic, extended and trees; no dictionary plus BYTE_STREAM_SPLIT on
  the float columns of tens and vahc (the smaller choice per table, `parquet_layout_test.py`).
- **Archive and verify:** the Parquet files are kept files (`retention.py`: `halo_parquet`, `trees_parquet`),
  copied and read back like the FITS. Stage verify fails unless every kept FITS file has its Parquet copy from the
  current FITS (size, mtime), with the same rows and columns, made by a parquet stage that passed, after the last
  fits stage. So nothing is trashed or purged before the Parquet copy is archived and checked.
- **Cost** (pawna, L500N512 r1, 8 processes, 18 Nov 2026): 379 files, 13.7 GB FITS → 11.5 GB Parquet, 81 s wall,
  391 CPU-s, largest process 1.4 GB. The tree file (2.3 GB Parquet, 19.2 M rows) takes 81 s in one process, as gzip
  is sequential. Expected about 100 GB / 1.5 core-h at 1024^3, about 0.8 TB / 12 core-h at 2048^3 (L200).
- **Resources** (`setup_production.py`): 8 cores / 16 GB / 2 h up to 512^3; 8 / 24 GB / 6 h at 1024^3, 12 h
  above (one process per file; the tree file sets the time).
- **Manager:** with `PARQUET=1` the chain is `ctrees → vahc → fits → parquet → verify`; a run whose FITS were made
  before the stage existed gets `parquet → verify` before its archive. Runs already archived are left as they are.

## 6. Other post-processing (`postprocess.py`)

`postprocess.py` (P(k), mass functions, VVF, kNN, 2PCF; flags at the top of the file) also reads `.pkc`
snapshots now, through the same `SnapshotReader`. Run it on pawna with
`SAHYADRI_SCRATCH=$SCRATCH SAHYADRI_LIBDIR=$CODES/lib $CODES/venv/bin/python postprocess.py <STEM> <snap0> <snap1> <real> <grid> <down_to> <Lbox> <nproc>`.
Its `add_value` branch now keeps the `.trees` order and has an `add_tens` flag; `run_addvalue.py` is the
pipeline's version of the same step.

## 7. Fixes made while setting this up (2026-11-11)

- `correlations.density_hessian_field` smoothed the caller's Fourier density in place. In
  `add_value`, the Hessian is computed at 3 then 5 Mpc/h, so **`lamH1/2/3_R5Mpch` in all existing `.vahc` files
  were computed from a field smoothed twice** (3 and 5 Mpc/h Gaussians). Fixed (copy, as in
  `tidal_tensor_field`); on the 128^3 test the corrected values have about 1.8 times the rms of the old ones
  and every other column is unchanged.
- `correlations.halobyhalo_b1` raised an exception since commit e77922b (Pk_grid needs `nbar`), so `.vahc`
  could not be made with the current code. Now `nbar=inf`, i.e. no shot-noise subtraction for `b1`, as in the
  production version; outputs are byte-identical to the production code.
- `postprocess.py` (repo version) sorted halos by mass before `add_value`, so `.vahc` rows did not follow
  `.trees`; production (Saee's copy) used `sorthalos=False`. Fixed; `compress_halo_outputs.py` refuses
  misaligned files.
- `readers.read_block` (Gadget HDF5): `down_to` was skipped by an early return, so the repo version read and
  used every particle. Fixed (and files are closed).
- The production `cic_update.c` (pegasus) passes the particle count as `int`, which overflows above 2^31
  particles; the repo version uses `long`. With `down_to=512` production stayed below the limit.

## 8. Larger runs, resources and pegasus

Measured wall time and peak memory (pawna, 2026-11-11). Every job runs `memmon.sh`, which every 5 s sums
the PSS of the job's process tree (shared pages counted once; RSS sum in brackets where it differs) and writes
`$LOGS/<stage>.<run>.<jobid>.mem.tsv`; the job log ends with a `peak memory:` line. pawna keeps no per-job
memory accounting itself (all PBS jobs share one cgroup), so this is the only record. Peaks shorter than 5 s
can be missed.

| Run | Stage | Cores (procs) | Wall time | Peak memory |
|---|---|---|---|---|
| 512^3 / 200 Mpc/h, 6 outputs (`example_512.conf`, `SIM_NCPU=64`) | sim | 64 | 4780 s | 25.1 GB |
| | rockstar (8 readers, 16 writers) | 16 | 475 s | 8.0 GB (8.2) |
| | ctrees | 16 | 506 s | 0.8 GB |
| | vahc (`PP_GRID=512`, `PP_NPROC=4`) | 16 | 3666 s | 91.6 GB |
| | fits | 16 | 153 s | 5.9 GB |
| 128^3 / 200 Mpc/h, 5 outputs (`example_test128.conf`) | sim / rockstar / ctrees / vahc / fits | 32 / 8 / 8 / 8 / 8 | 110 / 41 / 9 / 49 / 6 s | 5.9 / 0.2 / < 0.1 / 2.3 / 0.1 GB |
| 512^3 / 50 Mpc/h, 6 outputs (validation) | rockstar, any input route (8 readers) | 8 | 800-850 s | 8.2-8.4 GB (8.9) |
| | rockstar, 1 reader, z = 0 only | | 764-798 s | 6.5 GB |
| | rockstar on the 1%+3% subsamples (`RS_PKC_SUBSAMPLES="1,3"`) | 8 | 105 s | 0.8 GB |
| | `pkc_to_gadget` of 6 outputs | | 388 s | 10.0 GB |
| | ctrees | 8 | 450-650 s | 1.4 GB |

Memory of vahc is `PP_NPROC` × 12 × 16 × `PP_GRID`^3 bytes and does not grow with the resolution of the
run: at the same `PP_GRID` a 2048^3 run needs the same ~26 GB per process as the 512^3 run (with no downsampling
the density is built block by block); only a larger grid raises it (768: ~87 GB per process). Not measured
above 512^3: a 2048^3 run has 64 times the particles. From 128^3 to 512^3 the simulation grew by a factor
4.3 (fixed overhead at 128^3) and Rockstar by 40. Rockstar needs 1.75-1.9 times its particle array (32 B per
particle; section 5.2): 422-430 GB measured at 2048^3 of a node's 755 GB (requested: 480 GB; was 624 GB).
Rockstar time at 512^3: 64 s for two z ~ 0 outputs with 64 writers and the output on the node's disk, 188 s with 16
writers writing to `/storage` (section 5.2); give large runs many cores (`RS_NCPU`) and a long `RS_WALL`. Plots and the
full measurement: [report](report/pkdgrav3_pipeline_report.pdf), section 6.

- Above 512^3 nothing may be submitted without approval. The 2048^3 production settings and their measurements
  are in section 15 (Rockstar 422-430 GB; vahc at grid 1024: 93 GB per process).
- pegasus: the same scripts work with a different `CODES`/`SCRATCH` and PBS resource names; not tested there.

## 9. What was tested

Rockstar reader validation: [rockstar_pkc_reader_plan.md](rockstar_pkc_reader_plan.md). Summary of the
512^3 / 50 Mpc/h test on pawna (job 2536453): with one reader the PKC route gives byte-identical Rockstar
output to the same particles read as Gadget-2; with 8 readers the PKC and Gadget catalogs differ only as much
as two PKC runs differ from each other (median |dM/M| ~ 1e-5). consistent-trees ran on both.

**Effect of compression on halos** (raw pkdgrav3 particles vs `.pkc`, same run, z = 0; method and all
redshifts in [rockstar_pkc_reader_plan.md](rockstar_pkc_reader_plan.md#effect-of-pkc-compression-on-halos-raw-pkdgrav3-particles-vs-pkc-5123-test)).
Position step 1.95 kpc/h (= softening), velocity step 2.44 km/s:

| | 123-1230 particles | 1230-12300 | > 12300 |
|---|---|---|---|
| halo counts | same within Poisson (-0.09 %) | +0.05 % | within Poisson |
| mass, Vmax: bias / median scatter | 0 / 0.4-0.5 % | 0 / 0.1 % | < 0.01 % / < 0.05 % |
| centre / bulk velocity offset (median) | 0.9 kpc/h / 1 km/s | 0.6 kpc/h / 1.7 km/s | 0.7 kpc/h / 0.9 km/s |
| Rs scatter (bias) | 5.5 % (+0.4 %) | 1.7 % (+0.1 %) | 0.3 % (< 0.1 %) |
| spin scatter (bias) | 3.8 % (-0.3 %) | 1.1 % (-0.1 %) | 0.3 % (< 0.1 %) |

Counts, masses, Vmax, positions and velocities are unbiased and agree at the per-mille level; the shape
of the cores of halos with fewer than ~1000 particles (Rs, concentration, spin) is the most sensitive.

- **pawna, full chain, 2026-11-11** (`example_test128.conf`: 128^3 / 200 Mpc/h, 5 outputs z = 3, 2, 1, 0.5, 0;
  fresh clone of this branch, `setup_pawna.sh`, then `submit_pipeline.sh`; times from the job logs):

  | Stage | Job resources | Wall time | Result |
  |---|---|---|---|
  | setup (login node) | - | 40 min, of which ~37 min pip (fitsio and others built from source) | Rockstar, consistent-trees, venv, helpers |
  | sim | 32 cores | 93 s (46 steps), 6.1 GB peak | 5 outputs, `pkc_check` OK, 98 MB |
  | rockstar | 8 cores | 40 s | 5 `out_<snap>.list`, 5205 halos at z = 0 |
  | ctrees | 8 cores | 7 s (+ cleaning) | 5 `out_<snap>.trees`, 3655 halos at z = 0, 1 tree file |
  | vahc | 8 cores, 5 processes | 52 s total (23-30 s per snapshot), 0.6 GB peak | 5 `.vahc`, 5 `.tens` |
  | fits | 8 cores | < 1 min | 20 halo FITS files + trees, 6.9 MB; `check_compressed_halos.py`: 38 checks pass |

  The first vahc job failed because the venv was being rebuilt at the time (fixed setup, see section 2b);
  `submit_pipeline.sh ... --stages vahc,fits` then completed the chain.
- Mac, 128^3 / 200 Mpc/h, 3 outputs (z = 3, 1, 0), Rockstar on `.pkc`, patched consistent-trees:
  `.vahc` from `.pkc` byte-identical to `.vahc` from a Gadget-4 HDF5 file holding the same particles;
  `run_addvalue.py` (3 processes) byte-identical to `postprocess.py`; `.tens` equal except 2 of 204,000
  values in the last printed digit; FITS of `.trees/.vahc/.tens` pass `check_compressed_halos.py` (21 checks);
  a `.vahc` with rows sorted by mass is refused by the FITS converter.

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| A stage fails and the later jobs disappear from `qstat` | Normal for `afterok`: PBS deletes the dependants. Fix the cause, then `submit_pipeline.sh my.conf --stages <failed>,<later...>` |
| No log under `$LOGS` | The job did not start the script: look at `$LOGS/pbs.<stage>.<run>.out` and `qstat -xf <id>` (if the server keeps history) |
| `ModuleNotFoundError: numpy` (or scipy) in vahc/fits | `setup_pawna.sh` did not finish its venv step; rerun it (do not rerun it while jobs use the venv) |
| scipy warning `NumPy version >=1.17.3 and <1.25.0 is required` | The system scipy got picked up instead of the venv's; jobs use `$CODES/venv/bin/python`, check `CODES` in the config |
| Rockstar: `zstd not found` at setup | pkdgrav3 (`build_snapcomp`) is not built under `PKD_IO_ROOT`; do section 2a first |
| Rockstar fails reading an output | The simulation did not write all outputs: compare `snapnames.txt` with the `snap.*.pkc` files and the `pkc_check` lines in the sim log |
| ctrees: hlists overwrite each other | Two outputs closer than 0.001 in scale factor (`clean_trees.pl` uses 5 characters of `a`); space the outputs |
| vahc: `Please specify number density nbar` | Old `correlations.py`; use this branch (section 7) |
| FITS converter refuses a `.vahc` | Its rows are not in `.trees` order (made by the old `postprocess.py`); remake with `run_addvalue.py --overwrite` |
| Out of memory in vahc | Lower `PP_NPROC` or raise `PP_MEM` (≈ 12 × 16 × `PP_GRID`^3 bytes per process) |

## 11. Production runs (`setup_production.py`)

`scripts/pkdgrav3_pipeline/production/setup_production.py` creates the directories, all configuration files and a
submit script for one or more realisations, and stops: check the files, edit them if needed, then submit.

```bash
SB=/storage/shadab/pipeline_test/sahyadri-sandbox          # clone of this branch on pawna
$SB/../codes/venv/bin/python $SB/scripts/pkdgrav3_pipeline/production/setup_production.py --L 200 --N 1024 --cosmo default --real 1
#   options: --root (default /storage/$USER/sahyadri_sims), --real 1,2 or 1-5, --codes, --pkd-io-root,
#            --step-da, --check-interval, --pot-zmax, --soft-frac, --force (overwrite the configuration files)
D=/storage/shadab/sahyadri_sims/sims/Sahyadri_L200N1024/default1024/r1
less $D/run.conf $D/p.par /storage/shadab/sahyadri_sims/halos/Sahyadri_L200N1024/default1024/r1/rockstar_input.cfg
$D/submit.sh dry        # the qsub commands
$D/submit.sh            # sim -> rockstar -> ctrees -> vahc -> fits
$D/submit.sh restart    # after the simulation stopped at its wall time: continue from the newest checkpoint
$D/submit.sh halos      # only the halo stages (simulation finished)
```

Layout (`SIM = Sahyadri_L<L>N<N>`, `COSMO = <cosmology><N>`, e.g. `default1024`; realisation `r<R>`):

| Directory | Contents |
|---|---|
| `ROOT/sims/SIM/COSMO/` | `cosmology.txt` (parameters, sigma8), `outputs_a.txt`, `transfer_pkdgrav3.dat`, `class_pk_z49.dat` (shared by the realisations) |
| `ROOT/sims/SIM/COSMO/r<R>/` | `run.conf` (pipeline: paths, resources, halo and post-processing settings), `p.par` (pkdgrav3), `output_steps.txt` (planned output steps, a, z), `submit.sh`, `README.txt`, `snapnames.txt` (after the run), `logs/`, `checkpoints/` |
| `ROOT/sims/SIM/COSMO/r<R>/compressed/` | snapshots `snap.<step>.s<pct>.<writer>.pkc` |
| `ROOT/halos/SIM/COSMO/r<R>/` | `rockstar_input.cfg`; then `out_<snap>.trees`, `.vahc`, `.tens`, `trees/`, `compressed/*.fits.gz`, `logs/` |
| `ROOT/halos/SIM/COSMO/scales.txt` | Rockstar snapshot number → scale factor (ctrees stage) |

What `p.par` contains:

| Item | Setting | Source |
|---|---|---|
| Cosmology | `cosmologies/default.cosmo`: Ωm = 0.3137721, Ωb = 0.0493017, Ωk = 0, h = 0.6736, ns = 0.9649, As = 2.098903e-9 | pegasus production `config/submit/sahyadri/sub_default.conf` |
| ICs | 2LPT at z = 49; seed of `r<R>` from `config/sims/seeds.txt` (r1 = 12345678, as production) | |
| Linear P(k) | the CLASS P(k) at z = 49 that production gave NGenIC (`class_default2048_pk.dat`, not renormalised), as `transfer_pkdgrav3.dat` = sqrt(P/k^ns) with `dSigma8` = 0.83048 | pkdgrav3 scales `dSigma8` by its own growth D(z=49)/D(0), so the IC power at z = 49 is the CLASS table. 0.83048 = sigma8 of the table (0.021102) × D(0)/D(49) for matter + Λ; this is the linear sigma8 that Gadget (production) and pkdgrav3 actually evolve. CLASS itself, with radiation, gives a lower sigma8(z=0) for this As |
| Gravity | `dSoft` = 1/60 of the mean spacing (3.255 kpc/h at 1024^3/200), `dEta = 0.3`, `bEpsAccStep`, `classic_theta_switch()`, `classic_replicas_switch()`, Ewald, `bNewKDK` | the 256^3 reference run, `/storage/swarajit/pkdgrav3_sims/pkdgrav3_n256_gadget4ic_v4_corrected/examples/cosmology_test.par` |
| Steps | top-level steps of Δa ≈ 4.6e-4 (= that run's 10 steps per Δa = 0.0046), i.e. 124 steps to the first output, then ~20 per output: 2126 in total | |
| Outputs | the 101 scale factors of `production/sahyadri_outputs_a.txt` (z = 11.98 … 0, the last exactly z = 0) | |
| Snapshots | `.pkc` with subsamples 1,3,5,9,10,12,15,20,25 %, in `compressed/`; potential (step 100 (km/s)^2) in every snapshot at z ≤ 0.5 (`dCompPotMaxRedshift = 0.5`: the 37 outputs from a = 0.667) | |
| Checkpoints | every 250 steps in `checkpoints/` (newest 2 kept, `SIM_KEEPCHK`); `iWallRunTime` = PBS wall time − 1 h (− 2 h at 1024^3, − 4 h at 2048^3), so pkdgrav3 writes a checkpoint and stops cleanly and the last files can still be copied | |
| I/O | staged (`SIM_STAGING=1` in `run.conf`): pkdgrav3 writes snapshots and checkpoints to the node's local SSD (`/var/tmp/$USER_stage_<run>`), and pkdgrav3's `pkd_drain.py` copies them to `compressed/` and `checkpoints/` during the run (`tools/staging/run_staged.pbs`) | writing straight to `/storage` (~25 MB/s) stalled the whole simulation 39-87 s per 512^3 snapshot (pawna job 2536502); at 1024^3 that would be ~5-10 min per snapshot and ~30-35 min per checkpoint |

Halo finding uses `FORCE_RES` = the softening and the production Rockstar settings (`MIN_HALO_OUTPUT_SIZE = 30`,
`MIN_HALO_PARTICLES = 20`, `UNBOUND_THRESHOLD = 0.5`, temporal halo finding); `.vahc` uses all particles
(`PP_DOWN_TO=0`, density built block by block) on a grid of `min(N, 1024)` (section 15).

Logs of the sim job (in `r<R>/`): `run.<job>.log` (pkdgrav3), `staged.<job>.log` (staging summary),
`drain.<job>.log` (every copy); the job log in `logs/` ends with the time pkdgrav3 spent writing. A snapshot is
counted when all its files are complete (headers, trailers and particle count checked with `pkcreader`); the
last one is also fully checked with `pkc_check`. If the job ends before the drain has copied everything, the
data stay on that node's SSD and `staged.<job>.log` prints the recover command (pkdgrav3
`docs/compressed_output_guide.md` section 3.9); the sim stage then fails.

Restarts: when pkdgrav3 stops at `iWallRunTime` (or on `touch r<R>/STOP`) it also writes a snapshot at that step.
That snapshot is not a planned output: `job_sim.pbs` builds `snapnames.txt` from `output_steps.txt` and ignores it
(it is reported in the log and can be deleted). The sim job then exits with an error, so the rest of the chain is
not started; `submit.sh restart` continues from the newest checkpoint and resubmits the chain.

**Tested (pawna, 2026-11-11, `--N 128 --root /storage/shadab/pipeline_test/prod_smoke`, `PP_GRID` set to 128 in
`run.conf`; jobs 2536527-2536536):** 2126 steps and 101 outputs. A `STOP` file after 22 outputs stopped pkdgrav3 cleanly at step 593
(checkpoint written); `submit.sh restart` continued from `checkpoints/snap.00593.chk` to all 101 planned outputs,
and the extra snapshot at step 593 was left out of `snapnames.txt`. Rockstar found halos in all 101 snapshots
(5329 at z = 0), consistent-trees wrote 85 hlists (the first 16 snapshots have no halos to track), `.vahc`/`.tens`
followed, and the FITS check passed 598 of 598. The potential is stored in exactly the 37 outputs at z ≤ 0.5 (first
at z = 0.4992; none at z = 0.5165). The power spectrum of the first output (z = 11.98; CIC on 256^3, window
deconvolved, no shot-noise subtraction for these lattice-start particles) agrees with the linear CLASS P(k) × D^2
to within the scatter of the few large-scale modes (ratios 1.01, 0.78, 1.11 for k < 0.13 h/Mpc with 17-122 modes)
and is 4-9% high at k = 0.13-0.8 h/Mpc, where early non-linear growth is expected (Δ² ≈ 0.03 at k = 0.65).

Default resources (in `run.conf`; one pawna node has 192 cores and 755 GB, `workq` allows 168 h):
1024^3: sim 192 cores / 700 GB / 168 h; Rockstar 64 cores (32 readers, 64 writers) / 400 GB / 96 h; ctrees 2 /
32 GB / 48 h (serial: one process plus a forked writer, so it shares a node; 2048^3: 2 / 128 GB / 168 h); vahc 32 / 400 GB / 72 h (8 snapshots at a time); FITS 16 / 200 GB / 24 h. These are estimates
from the 512^3 measurements (section 8), not measured at 1024^3. Runs above 512^3 need approval before
submission; 2048^3 needs ~750 GB on one node and 8-10 days (more than the 168 h `workq` limit: plan restarts).
**Superseded for 1024^3 and above by section 15** (post-processing defaults of 20 Nov 2026, per site).

## 12. Halo-halo potential cross-correlation (`halo_potential_xcorr.py`, stage `xcorr`)

For host halos in two mass bins A and B, statistics of the per-halo potentials (the `.vpot` columns `phi_*` of
`out_<snap>_vahc.fits.gz`: mean particle potential within rs, R200b, 0.5/1/1.5 Rvir, ...) over all pairs
(i in A, j in B) at comoving separation r:

| Statistic | Definition |
|---|---|
| `mean_diff` | <phi_i - phi_j> (tends to <phi>_A - <phi>_B at large r; 0 for A = B by symmetry) |
| `sf` | <(phi_i - phi_j)^2> (structure function) |
| `cov`, `corr` | <phi_i phi_j> - <phi_i><phi_j>, and divided by the pair-weighted sd of each side |

each with a jackknife error (`*_err`, jk^3 sub-cubes by the position of i), for several `phi` columns at once.
Files: `scripts/post-process/halo_potential_xcorr.py` (driver), `scripts/post-process/halo_pair_sums.c` (pair
kernel), `scripts/pkdgrav3_pipeline/job_xcorr.pbs` (stage). Example results and figures: section "Halo-halo potential
cross-correlation" of `reports/gadget_pkdgrav3_potential/report.pdf`.

### 12a. Set-up and installation

Requirements: Python 3 with numpy and fitsio, and a C compiler; OpenMP is used if the compiler supports it.
The input is the FITS stage's output with the `.vpot` columns, so the run needs `PP_FAST=1 PP_POT=1` in its
config (`vahc_fast.py` writes `.vpot`), then the `fits` stage (header `HAS_VPOT = T` in `out_<snap>_vahc.fits.gz`).

- **pawna, new set-up:** `setup_pawna.sh` (section 2b) installs everything: numpy and fitsio in `$CODES/venv`, and
  `$CODES/lib/halo_pair_sums.so` (`cc -O3 -fopenmp`, gcc 11 on pawna).
- **pawna, existing set-up** (`$CODES` from before 2026-11-12): build the library once on the login node (a few
  seconds):
  ```bash
  cc -O3 -fopenmp -shared -fPIC -o $CODES/lib/halo_pair_sums.so <repo>/scripts/post-process/halo_pair_sums.c
  ```
- **Anywhere else (e.g. a Mac):** `python3 -m pip install numpy fitsio`. The library is built automatically on first
  use, next to the source, with `$CC` if set, else `cc`, else `gcc-16` ... `gcc-10`. Apple's `cc` has no
  OpenMP, so on a Mac install gcc (`brew install gcc`) to get threads. Without an OpenMP compiler it builds a serial
  library and prints a warning. To force a compiler: `CC=gcc-16 python halo_potential_xcorr.py ...`.

The script uses `$SAHYADRI_LIBDIR/halo_pair_sums.so` (the jobs set `SAHYADRI_LIBDIR=$CODES/lib`) when it is not
older than the source. Otherwise it builds `scripts/post-process/halo_pair_sums.so` itself, so a change to the C
source is picked up automatically.

### 12b. Running

As a pipeline stage (after `fits`; for 2048^3 catalogues this is the way to run it):
```bash
$P/submit_pipeline.sh my.conf --stages xcorr                # z = 0 (the last snapshot)
$P/submit_pipeline.sh my.conf --stages vahc,fits,xcorr      # chained after the value-added and FITS stages
```
Config variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `XC_SNAPS` | last snapshot | snapshots, comma separated, e.g. `100,64,46` |
| `XC_ARGS` | (tool defaults) | options of `halo_potential_xcorr.py` (below), e.g. `"--rbins lin:0,50,0.2 --pairs auto"` |
| `XC_TAG` | `log` or `lin` (from `--rbins`) | output name tag |
| `XC_NCPU`, `XC_MEM`, `XC_WALL` | 16, 16gb, 02:00:00 | PBS resources |

Output: `$HALO_DIR/xcorr/out_<snap>_xcorr_<tag>.npz`; log `$LOGS/xcorr.<run>.<jobid>.log`. Snapshots whose
output exists are skipped, so to rerun with other options either delete the output or set another `XC_TAG`.
Without PBS: `CONF=$PWD/my.conf PIPE_ENV=$P/pipeline_env.sh STAGE=xcorr bash $P/job_xcorr.pbs`.

Directly (any machine, any FITS directory):
```bash
$CODES/venv/bin/python scripts/post-process/halo_potential_xcorr.py <HALO_DIR>/compressed 100 out.npz \
    --rbins lin:0,50,0.2 --phi phi_Rvir,phi_rs --mbins 1e12,1e13,1e14,1e16 --pairs 0-0,0-2 --threads 16
```

| Option | Default | Meaning |
|---|---|---|
| `--phi` | `phi_Rvir,phi_rs,phi_R1p5vir,phi_R0p5vir,phi_R200b` | columns of the vahc FITS file; any `phi_*` (also `phi_R2R200b` ... `phi_R5Mpch`) |
| `--rbins` | `log:0.2,50,30` | `log:rmin,rmax,nbins` or `lin:rmin,rmax,width` (comoving Mpc/h; rmax <= L/2) |
| `--mbins`, `--mass-col` | `2e12,1e13,5e13,2e14,1e16`, `Mvir` | mass bin edges (Msun/h) and column of the basic FITS file (the stacked-profile bins) |
| `--pairs` | `all` | `all` (every A <= B, including A = A), `auto` (A = A only), or a list `0-0,0-3,1-2` |
| `--all-halos` | hosts only | include subhalos |
| `--jk` | 3 | jackknife sub-cubes per side |
| `--target`, `--min-a` | 2e7, 2000 | dilution: about this many pairs per shell, but at least this many A halos (`--target 0`: none) |
| `--threads` | `PBS_NCPUS`, `NCPUS`, else all cores | OpenMP threads |
| `--seed` | 0 | seed of the dilution hash |

### 12c. Output (`.npz`, compressed)

Index order `[pair, column, shell]`; `pairs[p] = (A, B)` are mass-bin indices, `phi_cols[k]` the columns.

| Key | Shape | Content |
|---|---|---|
| `r_edges`, `r_mid`, `binning` | nbin+1, nbin | shell edges and centres (geometric for log), `log`/`lin` |
| `r_mean` | P x nbin | mean pair separation in the shell |
| `m_edges`, `mass_col`, `n_halos` | | mass bins and halos per bin |
| `pairs`, `phi_cols`, `phi_shift` | P x 2, K, K | bin pairs; columns; the per-column mean subtracted before summing (cancels in every statistic) |
| `mean_diff`, `sf`, `cov`, `corr` (+ `_err`) | P x K x nbin | the statistics and jackknife errors ((km/s)^2, (km/s)^4; corr dimensionless) |
| `npair`, `frac`, `level`, `expected_pairs` | P x nbin | pairs used, dilution fraction 2^-level of A, expected undiluted pairs |
| `S` | P x K x nbin x 6 | sums N, sum phi_i, sum phi_j, sum phi_i phi_j, sum phi_i^2, sum phi_j^2 (shifted phi) |
| `S_jk`, `npair_jk` | P x nreg x K x nbin x 6, P x nreg x nbin | the same per jackknife region of i (to recombine) |
| `a`, `z`, `box`, `target`, `jk`, `hosts_only`, `timing_s`, `source`, `snap` | | metadata |

```python
import numpy as np
X = np.load('out_100_xcorr_log.npz')
p = [tuple(q) for q in X['pairs']].index((0, 3)); k = list(X['phi_cols']).index('phi_Rvir')
r, md, err = X['r_mid'], X['mean_diff'][p, k], X['mean_diff_err'][p, k]
```

### 12d. Method, cost and resources

- One pass over the pairs per bin pair and distance range collects the six sums of every column at once (C,
  `halo_pair_sums.c`): B is sorted into a cell grid of about a third of the range's outer radius, each A halo visits
  the cells within that radius, and OpenMP threads split the A halos (dynamic schedule; per-thread accumulators, no
  locks). Nothing is stored per pair.
- Dilution: shell i expects n_A n_B V_i / L^3 pairs. It uses the A halos with u(id) < 2^-l, where u is the
  splitmix64 hash of the halo ID (nested, independent of phi) and l is the smallest level that keeps the shell at
  about `--target` pairs (and at least `--min-a` A halos). Shells with the same l share one pass. Each shell uses one
  subsample, so its statistics are unbiased; dilution only adds noise. Pairs with the same ID are skipped (A = B).
- Memory: the halo arrays ((4 + K) doubles per halo, read column by column) plus, per thread,
  27 x nbins x (2 + 6K) doubles. 2.5e6 halos: 1.2 GB in total. For 2048^3, 16 GB is ample.
- Time: about (number of shells) x target x 2.5 pair tests per bin pair. 2.5e6 synthetic (unclustered) halos in
  1 Gpc/h on 8 Mac threads: 9 s (30 log bins), 21 s (250 linear bins), all 10 bin pairs. 256^3 (35k hosts):
  0.3 / 0.6 s. Real clustering adds undiluted small-r pairs, so allow more; the 2 h default is generous.
- Checks (2026-11-12): sums equal a numpy brute force to 1e-13 (bin pairs 0-1, 2-2, 3-3, all columns and jackknife
  regions, NaN phi_rs included); diluted vs undiluted within 0.4 sigma; the stage ran on the Mac with a test config,
  including the restart skip. Gadget-4 vs pkdgrav3 at 256^3 agree within the jackknife errors (report).


## 13. Run manager, scratch on home, retention and archive (`tools/sahyadri_manager`, branch `run-manager`)

New production runs work in home and keep only part of their data. `setup_production.py` now defaults to:

- **scratch:** `--root /user/$USER/sahyadri_scratch`. Writes to home measured 219 MB/s for one stream and about
  540 MB/s for four; `/storage` measured 23–31 MB/s (job 2536812).
- **archive:** `--archive /storage/$USER/sahyadri_sims`.

What is kept is set in `run.conf` (`KEEP_FULL_Z`, `KEEP_SUBSAMPLE_PCT`, `KEEP_HALO_TABLES`, ...).
`scripts/pkdgrav3_pipeline/retention.py run.conf` prints the resolved plan.

Two new stages, both submitted by the manager:

- **`verify`** (`verify_run.py`): checks that everything was computed and is consistent (what and how: section 15.4).
- **`archive`** (`archive_run.py`): copies the kept files with SHA-256 checksums, reads them back from disk with
  `O_DIRECT`, writes `MANIFEST.json` and `SHA256SUMS`, then makes the archive read-only.

The manager runs on the head node, because compute nodes cannot `qsub`, `qrls` or `qalter`. It:

- reacts to the jobs' event files through `inotifywait`, and runs `qstat` only every 30 min;
- submits Rockstar in batches while the simulation runs, then `ctrees → vahc → fits → parquet → verify` (parquet
  with `PARQUET=1`, section 5.6), then `archive`;
- moves the scratch data to a trash directory, then purges it, only after you approve each step in its web page.

The page is opened on the Mac with `tools/sahyadri_manager/open_gui.sh`. See `tools/sahyadri_manager/README.md`.

### 13.1 Queueing many runs

```bash
cd scripts/pkdgrav3_pipeline/production
python setup_production.py --cosmo default --N 512 --L 200 --real 1-4 --schedule      # priority 0
python setup_production.py --cosmo 'var_*' --real 1 --schedule 5                            # lower priority
tools/sahyadri_manager/manager.py limits max_sim_running=3 max_sim_queued=1 max_rockstar=1
```

- `--cosmo` takes a name, a comma list, shell patterns or `all` (`cosmologies/<name>.cosmo`); see 13.4 for the
  cosmologies.
- `--schedule [P]` registers each run with the manager and queues its simulation. The manager submits queued
  simulations by priority, then age, while running + waiting simulations < `max_sim_running`, waiting <
  `max_sim_queued`, and the run fits in scratch next to the running ones plus `scratch_reserve_gb`.
- A simulation that stops cleanly at its wall time is queued again at priority −1 and restarts from its newest
  checkpoint, at most `max_sim_restarts` (5) times.
- With `max_rockstar` Rockstar jobs running, the run whose next ready snapshot has waited longest goes next.

### 13.2 Power spectrum checks

With `PP_PK=1` (default) the vahc stage also measures, from the same density grid, the matter P(k) and the
halo–matter cross spectra and bias in mass bins `PK_MBINS` (particles; host haloes) for every output
(`out_<snap>.pk`, `pk_measure.py`). Outputs with no halo catalogue get the matter P(k) only.
`class_pk.py` computes CLASS linear and halofit spectra at every output redshift (`classy` 3.4.2 in the venv).
`verify` then checks, on k ≤ `PK_KLIN` (0.1 h/Mpc):

| check | what | fails when |
|---|---|---|
| absolute | R = ⟨P_mm/P_ref⟩, P_ref = P_CLASS(z_start)·[D(z)/D(z_start)]²·halofit/linear | \|R − 1\| > 4σ + 0.02, σ = √(2/N_modes) |
| growth | G = ⟨[P(z)/P(z₀)]/[P_ref(z)/P_ref(z₀)]⟩ on bins with Δ²_lin ≤ 0.01 | \|G − 1\| > 0.03 |
| IC table | CLASS at z_start vs the table the ICs were made from | > 2 % |

D is the linear growth without radiation, as in pkdgrav3. CLASS at z itself includes radiation and is only
reported: at z = 0, σ8 is 0.8227 from CLASS against 0.8305 from the no-radiation growth.

Two lessons from the tests are built into these checks:

- **No shot-noise subtraction.** The particles start on a lattice, so their discreteness noise on these scales is far
  below Poisson 1/n̄. Subtracting 1/n̄ removed about 7 % of the z = 12 power at k = 0.063 h/Mpc (64³,
  100 Mpc/h) and faked 7 % excess growth.
- **Only nearly linear bins in the growth check.** The lowest bins hold 6–12 modes each. Mode coupling moves a
  single bin by about (2–3)·Δ²_lin, with either sign. Measured at 128³, 200 Mpc/h, z = 1.24, Δ²_lin = 0.02–0.04:
  bins at +6 %, −6 % and +10 %. A growth error in the code would show at high z, where many bins are linear.

The plot `<halo dir>/pk/pk_summary.png` has three panels: the matter P(k) at a few redshifts with CLASS linear
and halofit and the 1/n̄ level; the ratio to the reference against k; and the halo bias b_hm(k) at the last output. It is shown on the manager page and kept in the archive.

### 13.3 Tests (12–13 Nov 2026)

The full write-up, with stage times, the 14 problems found and fixed, and the open items, is in
`docs/reports/sim_manager.pdf`. Stage times below come from the `start`/`end` event lines each job writes on its
own node.

- **64³, L100, r1–r9:** the full chain on every run, through verify and archive (read-back: 0 mismatches). Also
  tested: Rockstar batches during the simulation, three simulations with `max_rockstar=1`, the space-check refusal,
  the queue limits, restarts at the wall time (r9: five jobs), and deletion (refusal after a changed file, trash,
  undo, purge).
- **128³, L200, r1–r2, `PP_GRID=512`:** all stages passed. vahc took 73–89 min (the 512³ grid sets the cost,
  about 200 s per output, so `PP_GRID=256` would do for small runs). The archives are 0.37 GB, 668–674 files.
- **512³, L200, r1, 96 cores:** simulation 4.4 h (2126 steps). Rockstar ran in 11 batches of up to 10 snapshots,
  3.4–6.3 min each, and kept pace with the simulation. ctrees 53 min, vahc 52 min, fits 29 min, verify 6 min. The
  archive is 23.3 GB, 778 files, written in 18 min (21 MB/s including read-back). Scratch use was 178.2 GB against
  an estimate of ≤ 180.9 GB. Start to archived took 7.2 h. P(k): G = 0.987–0.999. The IC table agrees with CLASS to 0.11 %.
- Clocks: the compute nodes run 1.5–3 min and `/storage` about 15 min behind the head node. All ordering uses PBS
  job IDs or file times from the same server.

### 13.4 Cosmologies

`production/cosmologies/` holds the default and 14 one-parameter variations, each a `.cosmo` file with its CLASS
P(k) at z = 49 (the initial conditions are made from that table). The run directories are named `<name><N>`, e.g.
`var_Om_p2048`, as on pegasus. σ8 is the linear σ8 at z = 0 that pkdgrav3 evolves (no radiation; `dSigma8`).

| name | change from default | σ8 | source of the table |
|---|---|---|---|
| default | — | 0.83048 | pegasus `class_default2048_pk.dat` |
| var_As_m / p | A_s = 1.8890127e-9 / 2.3087933e-9 (±10 %) | 0.78786 / 0.87102 | pegasus, run there |
| var_h_m / p | h = 0.63992 / 0.70728 (±5 %) | 0.78483 / 0.87598 | pegasus, run there |
| var_ns_m / p | n_s = 0.916655 / 1.013145 (±5 %) | 0.81569 / 0.84595 | pegasus, run there |
| var_Om_m / p | Ω_m = 0.2980835 / 0.3294607 (±5 %) | 0.79935 / 0.86017 | pegasus, run there |
| var_w_m / p | (w0, wa) = (−1.03278258, 0.12063991) / (−0.96721742, −0.12063991) | 0.82948 / 0.83128 | pegasus, run there |
| var_Ok_m / p | Ω_k = −0.05 / +0.05 (Ω_Λ = 1 − Ω_m − Ω_k) | 0.87260 / 0.79223 | pegasus inis, not run there |
| var_Ob_m / p | Ω_b = 0.04437153 / 0.05423187 (±10 %) | 0.85550 / 0.80598 | made here with CLASS 3.4.2 |

The pegasus tables are `class_var_*2048_pk.dat` from
`/mnt/home/student/csaee/chpc2501005/sahyadri-sandbox/config/transfer/sahyadri/`. Their `_pk.txt` versions, which
NGenIC read, are log10 (k, Δ²) of the same tables to 1e-8. The Ω_m, h, w0 and wa of the pegasus runs agree with the
headers of their snapshots on pawna (`/storage/shadab/sahyadri_sims/sims/sahyadri/var_*2048`). CLASS 3.4.2 (pawna
venv) reproduces every pegasus table from the `.cosmo` parameters to 1.4e-5 for k = 1e-3–10 h/Mpc. The Ω_b tables
were made with it, using the same settings.

For (w0, wa) cosmologies `setup_production.py` writes `dLambda = 0`, `dOmegaDE = 1 − Ω_m − Ω_k`, `w0` and `wa` to
`p.par`, and `OMEGA_DE`, `W0`, `WA` to `cosmology.txt` (read by `class_pk.py`). `dSigma8` comes from the
growth ODE that pkdgrav3 integrates (matter, curvature and CPL dark energy, no radiation), which gives the
default 0.83048239 unchanged. Rockstar's PKC reader takes Ω_DE, w0 and wa from the snapshot header. Ω_k ≠ 0 enters
pkdgrav3 as 1 − Ω_m − dLambda.

### 13.5 Runs on pegasus, archived on pawna

A run can be computed on pegasus (IUCAA) and still be driven by the manager on pawna and archived to pawna's
`/storage`. Every stage up to and including verify runs on pegasus. Only the kept files cross to pawna, pulled from
pegasus's data node `data.iucaa.in`; the login node is used for qsub and qstat only.

| | pawna run | pegasus run |
|---|---|---|
| set up with | `setup_production.py` on pawna | `setup_production.py --site pegasus [--queue debug]` on pegasus |
| scratch | `/user/$USER/sahyadri_scratch`, staged I/O | `/data/project/hpc2502016/shadab/sahyadri_scratch` (Lustre), no staging |
| simulation | 1 node, threads | `SIM_NODES` × 32 cores, 2 MPI ranks per node (3 nodes from 1024³), 96 h wall (debug: 4 h) |
| codes | `/storage/shadab/{pkdgrav3_io,pipeline_test/codes}` | `~/sahyadri_pipeline/{pkdgrav3_io,codes,sahyadri-sandbox}` in the pegasus home |
| PBS | qsub here | qsub/qstat over ssh on `pegasus.iucaa.in` |
| archive | PBS job copies SCRATCH → ARCHIVE | after verify, the verify job writes `KEPT_FILES.json` (SHA-256 of every kept file); a process on the pawna head node pulls those files with rsync from `data.iucaa.in` into ARCHIVE and compares each copy, read back with O_DIRECT, with the pegasus checksum |
| deletion | after approval, on pawna | after approval, on pegasus (`scratch_ops.py` over ssh) |

The pull runs on the head node because pawna's compute nodes have no route out: from node1 the name `data.iucaa.in`
does not resolve, and its address times out (tested 13 Nov 2026, jobs 2536961/2536962). At most `max_pulls` (1)
pulls run at a time. Earlier pegasus→pawna transfers ran at 4–24 MB/s.

Install on pegasus (once; the home quota is 150 GB, about 7 GB of it is used by these):
```bash
# pkdgrav3 a99f6142 (same as pawna) with its conda env; package cache on /data
R=~/sahyadri_pipeline/pkdgrav3_io; mkdir -p $R; ln -s /data/project/hpc2502016/shadab/cache/conda_pkgs $R/conda_pkgs
git clone -b main https://github.com/shadaba/pkdgrav3.git $R/pkdgrav3 && bash -l $R/pkdgrav3/tools/pegasus/setup_pegasus.sh $R
git clone -b run-manager https://github.com/a-paranjape/sahyadri-sandbox.git ~/sahyadri_pipeline/sahyadri-sandbox
bash ~/sahyadri_pipeline/sahyadri-sandbox/scripts/pkdgrav3_pipeline/setup_pegasus.sh ~/sahyadri_pipeline/codes $R
```

Set up a run on pegasus and register it with the manager on pawna:
```bash
# on pegasus
cd ~/sahyadri_pipeline/sahyadri-sandbox/scripts/pkdgrav3_pipeline/production
~/sahyadri_pipeline/codes/venv/bin/python setup_production.py --site pegasus --cosmo var_Om_p --N 2048 --real 1
# on pawna
tools/sahyadri_manager/manager.py add pegasus:/data/project/hpc2502016/shadab/sahyadri_scratch/sims/Sahyadri_L200N2048/var_Om_p2048/r1/run.conf
tools/sahyadri_manager/manager.py schedule pegasus:Sahyadri_L200N2048/var_Om_p2048/r1   # runs of pegasus: pegasus:<stem>/r<N>
tools/sahyadri_manager/manager.py limits pegasus.max_sim_running=2 pegasus.max_sim_queued=1
```
The pegasus queues allow per user: `debug` 2 running jobs, 128 cores, 4 h; `project` 4 running jobs, 2048 cores;
`regular` 4 jobs, 1280 cores, 48 h (`qstat -Qf`, 13 Nov 2026). Rockstar at 2048³ needs about 430 GB (480 GB requested), more
than one pegasus node: run it over several nodes with `RS_NODES` (below).

**Rockstar over several nodes (`RS_NODES`).** Rockstar has no MPI: its server and clients talk over TCP sockets, so the
single-node build runs across nodes unchanged. With `RS_NODES=N` in run.conf, `submit_pipeline.sh` asks for
`select=N:ncpus=RS_NCPU:mem=RS_MEM` with `place=scatter:excl`; `job_rockstar.pbs` starts the server on the first node
(listening on its name instead of `localhost`) and one client per node (the others over ssh), each forking `RS_NCPU`
writers, so the particles and the memory are spread over the nodes. Output goes straight to `$HALO_DIR` on the shared
file system (`RS_LOCAL_OUT=0`, as on pegasus anyway); each node's client writes `client.<node>.log/.time`.
Tested 17 Nov 2026 on the pegasus debug queue with the 128³ test run (101 outputs, jobs 37895–37897): 2 nodes × 32
writers and 1 node × 64 writers both finished in 24 min. The catalogues differ between the two by as much as two
repeated 1-node runs differ from each other (Rockstar is not deterministic run to run: 95–97 % of halos matched within
50 kpc/h, 89–92 % with identical Mvir, counts above 1e11–1e14 within 0.5 %), so the multi-node mode adds nothing
beyond that. `setup_production.py --site pegasus` sets `RS_NODES` itself: the fewest nodes that hold 1.1 × Rockstar's memory
estimate (which already has a margin over the measured peak) at most 360 GB each (`--rs-nodes N` or plan option
`rs_nodes=N` to choose): 2048³ (480 GB) → 2 nodes × 264 GB, 64 writers; up to 1024³ one node. The manager's core-hour estimate counts every node.

**Simulations on the pegasus GPU nodes.** pegasus has four GPU nodes (`gpu001`–`gpu004`): one Tesla P100 (16 GB,
compute capability 6.0, driver 525 = CUDA 12.0) each, with the same 32 Skylake cores and ~370 GB as the CPU nodes.
They are reached through `regular` or `project` with `ngpus=1` per node; the `debug` queue has none.
- Build (once, on the login node; a separate build next to the CPU one, which is not touched):
  `bash scripts/pkdgrav3_pipeline/build_pkdgrav3_gpu_pegasus.sh` → `~/sahyadri_pipeline/pkdgrav3_io/build_gpu/pkdgrav3`.
  It creates the conda env `pkdgrav3_io/env_cuda` (CUDA 12.0 nvcc, gcc 11) and builds the whole code with gcc 11
  against the CPU build's MPI/HDF5/FFTW env. Mixing compilers (gcc 15 C++ with gcc 11 CUDA host code) crashes at the
  first GPU work package; gcc 12 headers do not compile under nvcc 12.0.
- Use: `setup_production.py --site pegasus --sim-gpu` (plan option `gpu=1`) writes `SIM_GPU=1` in run.conf;
  `submit_pipeline.sh` then asks for `ngpus=1` per node and `job_sim.pbs` runs `build_gpu/pkdgrav3` (`SIM_BUILD`
  overrides the directory). Nodes: 3 by default from 2048³ (~785 GB needed; 2 GPU nodes hold 740 GB), 1 below;
  memory per node at most 360 GB. The job log reports the share of gravity tiles done on the GPUs.
- Measured (16 Nov 2026, 512³ L200 restart at z = 0.06, 2 ranks × 16 threads per node, wall clock per step from
  `snap.timing`): CPU node 36.1 s, GPU node 10.3 s (3.5×), 2 GPU nodes 5.7 s (1.8× of one). The step-2005 snapshot
  of the GPU run differs from the CPU run's less than two CPU runs with 16 and 32 threads per rank differ from each
  other (positions rms 0.010 kpc/h, velocities 0.10 km/s, P(k) < 2e-7). `amgr` charges a GPU like one more core
  (a GPU node-hour = 33 units, from the reservations of the test jobs).

### 13.6 Production install, planned runs, budgets

**Install.** One script installs everything a site needs, in a new directory; production runs use only these:
```bash
# pawna (head node): /user/$USER/sahyadri_prod/{pkdgrav3_io,codes,sahyadri-sandbox}
bash install_production.sh pawna
# pegasus (login node): ~/sahyadri_pipeline/{pkdgrav3_io,codes,sahyadri-sandbox}
bash install_production.sh pegasus
```
It clones or updates the repository (`run-manager`) and pkdgrav3 (GitHub main) and builds them. It also builds
Rockstar with the `.pkc` reader, consistent-trees, the Python venv with the pinned versions of
`scripts/pkdgrav3_pipeline/requirements.txt` (the same on both sites), and the post-processing C helpers. It ends
by importing every pipeline and post-processing module. `setup_production.py --site pawna|pegasus` uses these
paths by default. The manager runs from `/user/shadab/sahyadri_prod/sahyadri-sandbox` (`start_manager.sh`).

**Planned runs.** `scripts/pkdgrav3_pipeline/production/planned_runs.txt` lists the runs to make, one line per
cosmology and set of realisations (`site queue cosmology N L realisations priority [options]`; format in the
file). Every `plan_s` (300 s), the manager sets up what is not set up yet (`setup_production.py` locally, or over
ssh on pegasus), registers it and puts its simulation in the queue with that priority. Option `hold` registers the
run paused. An existing `run.conf` is registered, never overwritten. A failed set-up is shown in the Upcoming tab and
is not retried until its line changes.

**Pegasus queues and core hours.** `--queue project` submits with `-q project -P hpc2502016` and is charged to the
project's half-year budget (96 h jobs). `--queue regular` is charged to the user's budget (48 h jobs; more
restarts). The user budget is **monthly**: 50,000 core-h credited on the 1st of each month (period `2026.mNN`),
nothing carried over (`amgr report user -n calam.shadab -p 2026.m11`: opening balance 0, credit 50,000; October
used 4). The project budget is per half-year and shared with the project's other users. PBS holds cores × requested
wall when a job starts and refunds the unused part at the end. On 13 Nov 2026: project hpc2502016, 2026.h2:
172,248 core-h; user calam.shadab: 49,110. On 25 Nov 19:15: user 8,688 left (32,000 charged, 9,312 held for the
two running jobs), project 79,492. A 96 h job on 3 nodes reserves 9,216 core-h, a 48 h job on 4 nodes 6,144.

The manager reads the balances every `budget_check_min` minutes (default 30, set on the page; 0 = only when a job is
submitted or on "read now"). `amgr checkbalance` needs a login token made with `ssh -t calam.shadab@pegasus.iucaa.in
amgr login` (password, expires after about 3 h; `amgr sshlogin` is refused for this account), so most reads fail. The
last good reading is kept in `~/.sahyadri_mgr/budgets.json` and turned into an estimate: reading − what the manager
submitted since (cores × requested wall) − our jobs waiting in PBS on that budget (from `qstat`). Refunds are not
counted, so the estimate errs low. In a new month without a reading, the user budget is taken as
`budget.monthly_coreh` (50,000) minus what was submitted that month. A project reading older than
`budget.project_stale_h` (6 h) is not used (others spend it too). A job is submitted only if its core hours fit in
the estimate minus `budget.reserve_coreh` (2000); otherwise the run waits (not a failure), and its card says why.
The page (Summary → Core-hour budgets) shows each budget's estimate, its state (ok, estimate, assumed, unknown), the
last amgr reading and the login command when the token has expired. `manager.py queue-once <stem/rN> project hpc2502016` puts only the
run's next simulation job on the project queue (charged to the project); later jobs go back to the run.conf's queue.
Used 25 Nov for one restart each of L200N2048 r2 and L200N1600 r2 (user budget short until 1 Dec).

**Page.** Tabs: Summary (runs per site and state, budgets, disk, plan file, the simulation queue and its limits,
the log), Active, Upcoming (planned and scheduled runs not started yet), Finished (archived runs with the trash and
purge buttons; purged runs in a table).

**Post-processing beyond the pipeline.** Every run directory has `postprocess.sh` (not run by the manager):
`vpot` (halo potential profiles `.vpot`, then the vahc FITS tables again with the phi columns), `xcorr` (halo-halo
potential cross-correlation), `all`. The older driver `scripts/post-process/postprocess.py` (two-point correlations,
VVF, kNN, Voronoi web) reads Gadget snapshots, not `.pkc`, and has no submission script yet.

## 14. Correlation multipoles in real and redshift space (`halo_xi_multipoles.py`, stage `ximult`)

Halo cross-correlation multipoles xi_l(s), l = 0..4 (the dipole included), between halo mass bins A and B, in a
periodic box with a fixed line of sight (plane parallel), in four spaces:

| Space | Line-of-sight displacement of each halo (comoving Mpc/h; wraps periodically) |
|---|---|
| `real` | none |
| `rsd` | v_los / (a H) (Rockstar bulk velocity, peculiar km/s) |
| `grav` | v_g / (a H), v_g = -phi / c: gravitational redshift of a source at potential phi (observer at phi = 0) |
| `rsd+grav` | both |

H = 100 E(a) km/s/(Mpc/h), with E from Omega_m and Omega_L in the FITS header. phi is a `.vpot` column (default
`phi_rs`, the mean potential within rs, the closest to a central galaxy; where it is undefined `phi_R0p5vir`). A
constant zero point of phi moves every halo alike and changes nothing. Convention: s = x_B - x_A and
mu = s_los / s in (-1, 1), so a B population shifted by +delta along the line of sight gives
xi_1 = -delta dxi_0/ds. For A = B the dipole is identically 0.
Files: `scripts/post-process/halo_xi_multipoles.py`, `scripts/pkdgrav3_pipeline/job_ximult.pbs`. Results at 256^3:
section "Correlation multipoles in real and redshift space" of `reports/gadget_pkdgrav3_potential/report.pdf`.

### 14a. Set-up and installation

Requirements: Python 3 with numpy and fitsio, and **Corrfunc from Arnaud de Mattia's fork**:
github.com/adematti/Corrfunc, branch `desi` (also mirrored as cosmodesi/Corrfunc). It counts pairs in *signed*
mu in (-mu_max, mu_max). Stock Corrfunc counts |mu| only, which loses the odd multipoles. Corrfunc builds its C
extensions with its own `common.mk` and **ignores the CC environment variable**.
The input needs the `.vpot` columns for the `grav` spaces (`PP_FAST=1 PP_POT=1`, then the `fits` stage). The
`real` and `rsd` spaces need only `out_<snap>_basic.fits.gz`.

Install it with `scripts/pkdgrav3_pipeline/install_corrfunc.sh <python> [workdir]`. The script clones the fork,
applies the two patches below only where they are needed, builds against that Python's numpy
(`--no-build-isolation`), and ends with a smoke test. The test checks that mu is signed (bins from -1 to 1) and
reports whether the extension is linked to OpenMP.

- **pawna, new set-up:** `setup_pawna.sh` (section 2b) runs it for `$CODES/venv`.
- **pawna, existing set-up** (on the login node; the build takes several minutes on the busy node):
  ```bash
  bash <repo>/scripts/pkdgrav3_pipeline/install_corrfunc.sh $CODES/venv/bin/python
  ```
  To keep it out of a shared `$CODES`, use your own venv and run the script directly with that Python (below):
  ```bash
  python3 -m venv my_venv && my_venv/bin/pip install numpy fitsio
  bash <repo>/scripts/pkdgrav3_pipeline/install_corrfunc.sh $PWD/my_venv/bin/python
  ```
  Tested 2026-11-12 on pawna in such a venv (Python 3.10, numpy 2.2.6, gcc 11, libgomp).
- **Mac:** `brew install libomp`, then the same command with your Python (a venv with numpy and fitsio). Tested with
  Python 3.13 / numpy 2.1 (Anaconda) on Apple Silicon.

The patches of Corrfunc's `common.mk`:
- **Linux:** numpy's include directory goes first. Corrfunc passes `-isystem <python include>` before numpy's, and on
  Debian/Ubuntu with the `python3-numpy` package, `<python include>/numpy` holds the *system* numpy's (1.x) headers.
  A venv with numpy 2 then gets an extension that fails at import with `numpy.core.multiarray failed to import`
  (`A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6`). Plain `pip install
  git+https://github.com/adematti/Corrfunc@desi` fails this way on pawna.
- **Mac:** Apple clang has no OpenMP, and Homebrew gcc rejects Corrfunc's `-march=apple-m1`. The script adds
  Homebrew's libomp to the clang branch. Without it the build is serial, about 7x slower on 8 cores.

### 14b. Running

As a pipeline stage (after `fits`):
```bash
$P/submit_pipeline.sh my.conf --stages ximult              # z = 0 (the last snapshot)
$P/submit_pipeline.sh my.conf --stages vahc,fits,ximult
```
Config variables (all optional): `XM_SNAPS` (snapshots, default the last), `XM_ARGS` (options below, e.g.
`"--los x,y,z"`), `XM_TAG` (output tag, default `ximult`), `XM_NCPU`/`XM_MEM`/`XM_WALL` (default 32 cores, 32gb,
6 h). Output: `$HALO_DIR/xcorr/out_<snap>_<tag>.npz`. Existing outputs are skipped. The job stops at once with a
message if Corrfunc is missing from `$CODES/venv`. Without PBS:
`CONF=$PWD/my.conf PIPE_ENV=$P/pipeline_env.sh STAGE=ximult bash $P/job_ximult.pbs`.

Directly:
```bash
python scripts/post-process/halo_xi_multipoles.py <HALO_DIR>/compressed 100 out.npz --los x,y,z --threads 32
```

| Option | Default | Meaning |
|---|---|---|
| `--spaces` | `real,rsd,grav,rsd+grav` | any subset |
| `--phi`, `--phi-fallback` | `phi_rs`, `phi_R0p5vir` | potential column(s) for grav (several: `grav[<col>]` spaces each); fallback where undefined (`''`: drop those halos) |
| `--pot-amp` | 1 | multiply the potential shift (response tests; 1 = physical) |
| `--los` | `z` | line of sight `x`, `y` or `z`; a list (`x,y,z`) measures each and stores their mean (3x the pairs: less noise) |
| `--sbins` | `lin:0,100,2` | `lin:smin,smax,width` or `log:smin,smax,n`, comoving Mpc/h, smax <= L/2 |
| `--nmu`, `--ells` | 200, `0,1,2,3,4` | mu bins in (-1, 1); multipoles |
| `--mbins`, `--mass-col`, `--pairs` | profile bins, `Mvir`, `all` | as section 12 (`all`, `auto`, `cross` or `0-3,1-2`) |
| `--jk` | 3 | jackknife sub-cubes per side (by the real-space position of the A halo) |
| `--subsample-a` | 1 | use A halos with hash(id) < f (fewer pairs for very large catalogues) |
| `--threads`, `--all-halos` | `PBS_NCPUS`, `NCPUS`, all cores; hosts | |

### 14c. Output (`.npz`, compressed)

| Key | Shape | Content |
|---|---|---|
| `s_edges`, `s_mid`, `mu_edges`, `ells` | | bins |
| `spaces`, `los`, `pairs`, `phi_cols` | S, NL, P x 2, | names; mass-bin pairs (A, B) |
| `xi_ell`, `xi_ell_err` | S x NL x P x nell x ns | multipoles per line of sight, jackknife errors |
| `xi_ell_mean`, `xi_ell_mean_err` | S x P x nell x ns | mean over the lines of sight and its jackknife error |
| `xi_ell_jk` | S x P x nreg x nell x ns | leave-one-region-out multipoles of the mean (to rebin or form differences) |
| `diff_names`, `dxi_ell`, `dxi_ell_err` | D, D x P x nell x ns | potential effect: `grav - real`, `rsd+grav - rsd` (same halos; most noise cancels) |
| `DD` | S x NL x P x ns x nmu | pair counts |
| `n_halos`, `n_a`, `n_b`, `vg_mean` | | halos per bin, halos used per pair, mean v_g (km/s) per phi column and bin |
| `a`, `z`, `box`, `aH`, `omega_m`, `omega_l`, `pot_amp`, `jk`, `source`, `snap` | | metadata |

```python
import numpy as np
X = np.load('out_100_ximult.npz')
sp, pairs = list(X['spaces']), [tuple(p) for p in X['pairs']]
s, ip, k = X['s_mid'], pairs.index((0, 3)), sp.index('rsd')
xi0, xi2 = X['xi_ell_mean'][k, ip, 0], X['xi_ell_mean'][k, ip, 2]
d = list(X['diff_names']).index('rsd+grav - rsd')
dip, dip_err = X['dxi_ell'][d, ip, 1], X['dxi_ell_err'][d, ip, 1]        # dipole from the potential
```
A wider s bin is the volume-weighted mean of the narrow ones (exact for DD/RR - 1 in a periodic box). Apply the same
to `xi_ell_jk` for its error (`reports/gadget_pkdgrav3_potential/make_ximult_figures.py`, `rebin`).

### 14d. Method, cost and checks

- xi(s, mu) = DD / RR - 1 with RR analytic, N_A N_B V_s (dmu/2) / L^3 (N_B - 1 for A = B). xi_l uses bin-integrated
  Legendre polynomials. DD is counted with Corrfunc `DDsmu`, A halos one jackknife region at a time against all B.
- Time: Corrfunc, OpenMP. 2.5e6 synthetic (unclustered) halos in 1 Gpc/h to 100 Mpc/h (2.2e10 pairs, all 10 bin
  pairs) take 56 s per space and line of sight on 8 Mac threads, 0.8 GB. So 4 spaces x 3 lines of sight take about
  11 min there, and less on 32 pawna cores. 256^3: 15 s for everything.
- Checks (2026-11-12): DD(s, mu) equals a numpy brute force in every bin (real and rsd, cross and auto). Single-pair
  tests fix the mu sign and the periodic wrap. A constant 1 Mpc/h shift of one bin gives xi_1 ~ -delta dxi_0/ds.
  Same-bin dipoles are 0 and the real-space quadrupole is 0 within errors. The OpenMP and serial builds give
  identical counts. A potential shift x50 gives a dipole linear in the shift, 0.5-1.2 times the mean-shift
  prediction.


## 15. Production defaults for 1024^3 and above (2048^3 runs; set 19–21 Nov 2026)

What `setup_production.py` writes into `run.conf` for `--N 1024` and larger, and what the run manager then does
by default. The seven 2048^3 runs in production (pawna r3–r5, swarm r6–r7, pegasus r2 and r10) were changed to
these values in place (backups `run.conf.bak_20261120pp`, `.bak_20261120vf`, `.bak_20261120pq` next to each
`run.conf`) and reloaded by the manager. Up to 512^3 the earlier defaults are unchanged.

### 15.1 What runs by default

The manager runs `sim → rockstar → ctrees → vahc → fits → parquet → verify → archive` for every run, each stage
after the previous one succeeded.

| Stage | What it does | When |
|---|---|---|
| sim | pkdgrav3, 101 outputs as `.pkc` (subsamples 1,3,5,9,10,12,15,20,25 %), potential at z ≤ 0.5, checkpoints every `SIM_CHECK_INTERVAL` steps | restarts from the newest checkpoint after each wall time (manager, up to `max_sim_restarts`) |
| rockstar | halo finding on all outputs | in batches of ≥ `RS_BATCH_MIN` (10) snapshots while the simulation runs |
| ctrees | consistent-trees with `CT_BOX_DIVISIONS=5` (125 tree files `tree_<i>_<j>_<k>.dat`), `out_<snap>.trees`; Rockstar intermediate files deleted (`CLEAN_ROCKSTAR=1`) | after the simulation and the last Rockstar batch |
| vahc | `.vahc` (environment, tidal and Hessian eigenvalues, b1), `.tens`, `.vpot` (outputs with the potential), `.pk` (matter and halo P(k)), on a `PP_GRID`=1024 density grid from all particles, `PP_NPROC`=3 snapshots at a time, written binary (`PP_BINARY=1`) | after ctrees |
| fits | `compressed/out_<snap>_{basic,extended,vahc,tens}.fits.gz`; the 125 tree files merged into one `trees/tree_0_0_0.fits.gz` (+ `locations`, `forests`); then `check_compressed_halos.py` compares every FITS column with the catalogues, row by row, `FITS_CHECK_NPROC`=4 snapshots at a time | after vahc |
| parquet | a Parquet copy of every FITS file (`parquet/`), with DATASUM and read-back hash checks | after fits |
| verify | everything complete and consistent (15.4); on swarm/pegasus also `KEPT_FILES.json` (SHA-256 of every kept file) | after parquet |
| archive | kept files copied to `ARCHIVE` (`/storage/shadab/sahyadri_sims`), read back with O_DIRECT, SHA-256 compared, `MANIFEST.json`, read-only. Kept particle files are archived in batches while the simulation runs; swarm/pegasus runs are pulled to pawna from their data nodes | after verify |

**Kept** (`KEEP_*`): full particles at the 13 outputs nearest z = 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7137, 0.8,
1.0, 1.9821, 2.9563, 4.9089; 9 % of the particles (subsamples 1+3+5) of every other output; FITS halo tables
basic, extended, vahc, tens at every snapshot; the merged tree FITS; their Parquet copies; no checkpoint.

**Not automatic:** moving the scratch copy to the trash and purging it (only after approval on the manager's
page); the optional stages `vpot` (only needed with `PP_POT=0`), `xcorr`, `vprof`, `ximult` (`postprocess.sh` in
the run directory).

### 15.2 Resources per site (2048^3)

Nodes: pawna 192 cores / 755 GB (`/var/tmp` 1.5 TB SSD, shared with the simulation's staging); swarm 64 cores /
504 GB (`/var/tmp` ~610 GB; per-user limit 310 running cores, all queues); pegasus 32 cores / 385 or 514 GB
(`/var/tmp` 138 GB; scratch on Lustre). pegasus jobs are charged to a core-hour budget (section 13.6).

| Stage | pawna (`workq`) | swarm (`workq`) | pegasus (`regular`, 48 h max) |
|---|---|---|---|
| sim | 1 node, 192 cores, 700 GB, 168 h; staged on the SSD; checkpoint every 250 steps | 2 nodes × 64 cores, 480 GB each, 72 h; straight to Lustre; checkpoint every 100 steps | 3 nodes × 32 cores, 480 GB each, 48 h; checkpoint every 100 steps |
| rockstar | 1 node, 64 cores, 480 GB, 96 h | 2 nodes × 32 cores, 264 GB, 72 h | 2 × 32 cores, 264 GB, 48 h |
| ctrees | 20 cores, 128 GB, 168 h | 20, 128 GB, 72 h | 20, 128 GB, 48 h |
| vahc | 96 cores, 370 GB, 72 h | 32, 400 GB, 72 h | 32, 360 GB, 48 h (one automatic retry at the wall) |
| fits | 16 cores, 160 GB, 24 h | same | same |
| parquet | 8 cores, 24 GB, 12 h | same | same |
| verify | 32 cores, 16 GB, 12 h | 32, 16 GB, 6 h | 32, 16 GB, 6 h |
| archive | 2 cores, 8 GB, 72 h (pawna) | pulled by pawna | pulled by pawna |

`SIM_CHECK_INTERVAL` is also applied on every restart of an unstaged run (`job_sim.pbs` passes
`iCheckInterval` to `PKDGRAV.restore`), so an existing run can change it in `run.conf`.

### 15.3 Why these values: measurements

| Item | How measured | Result |
|---|---|---|
| Rockstar memory | pawna, L200N2048 r3–r5, batches of snapshots 0–34 | 422–430 GB peak → `RS_MEM` 480 GB (1.7 × 32 B × N^3 + 8 GB) |
| ctrees divisions | swarm, 128^3, `BOX_DIVISIONS` 1 vs 5 | 32 s → 16 s; 4739 trees, 309,526 rows in both; 4738/4739 trees identical (one phantom halo differs by 1e-5 in position); 2048^3 estimate ~33 h → ~13 h |
| vahc grid 1024 | swarm, 2048^3 particles, 3 processes × ~10 threads, snapshots 20, 64, 100 | 93 GB per process, 240 GB per job (after the `overwrite_x` FFT change, 286bc3c; outputs byte-identical to before) |
| binary `.vahc/.tens/.vpot` | swarm, 128^3, text vs binary run to FITS and Parquet | 368 FITS tables and 368 Parquet files identical |
| tree merge and FITS check | swarm, 128^3, 125 tiles | merged tree = single-file tree (309,525 of 309,526 rows; the phantom halo), 902 checks passed |
| pkc_check (verify) | pegasus debug node, 32 cores, r2 2048^3 snapshots of 54–60 GB on Lustre | 1 × 8 threads 1127 s; 1 × 32 threads 350 s; 4 × 8 threads 795 s for 4; 2 × 16 threads 297 s for 2 (fastest: 13 kept snapshots ~35 min); 1.1–1.2 GB each |
| SHA-256 (verify, swarm/pegasus) | same node, ~60 GB snapshot | 1.1 GB/s one file at a time; 4.8 GB/s with 8 threads (`archive_run.py`); ~3.6 TB kept → ~13 min |
| Parquet | swarm compute-1-8, 8 cores, Gadget 2048^3 tree file (1.61 GB gz) | 59.6 s → 21.9 s with the threaded converter (same SHA-256), 1.66 GB peak per process; 128^3 run: 371 files identical |

Estimated, not yet measured at 2048^3: ctrees ~13 h; vahc ~40 h on a pegasus node (less on pawna with 96
cores); fits ~4–5 h (halo tables, tree merge and check each ~1–2 h, ~400 GB of temporary disk for the merged
tree: `/var/tmp` when `df` shows room, else `$HALO_DIR/.fits_tmp_<job>`); parquet ~40 min for the merged tree
(~225 GB); verify ~1 h. The first real check is pawna r3 reaching ctrees (about 24–25 Nov 2026).

### 15.4 Stage details changed for 2048^3

- **ctrees** (`CT_BOX_DIVISIONS`): consistent-trees splits the box into 5^3 regions and writes one tree file per
  region; the FITS stage merges them into one FITS file (`MERGED=T`, `NTILES=125` in its header) and
  `locations.fits` points into the merged file.
- **vahc**: grid `PP_GRID = min(N, 1024)`. Binary catalogues (`scripts/post-process/halo_table_io.py`): the same
  `#` header lines as the text files, then `#SAHYADRI-BINARY v1 rows=N dtype=<json>` and packed little-endian
  records, written via `.part` and renamed; every reader (`compress_halo_outputs.py`, `check_compressed_halos.py`,
  `verify_run.py`, `readers.py`, `ConvertToFITS.py`) reads both. `PP_BINARY=0` writes text. Eigenvalues are
  computed in chunks of 200,000 halos. `PP_SNAPS` limits the job to some snapshots.
- **fits**: the check runs `FITS_CHECK_NPROC` snapshots at a time (and the tree check); temporary FITS go to
  `/var/tmp` only when it has room for the tree files plus `FITS_NPROC` × the largest `.trees`, × 1.2. gzip of the
  merged tree uses `pigz` when installed (pawna: `/user/shadab/sahyadri_prod/codes/bin`).
- **parquet**: per file, decompression, decoding + hashing and Parquet encoding run in three overlapping threads
  (`fits_to_parquet.py --threads`, default 4 hash threads); output byte-identical to the serial code; an error in
  any thread fails the file and leaves no `.part`.
- **verify** (`verify_run.py`):
  - snapshots: header and trailer of every `.pkc` file of all outputs (particle count, redshift, subsample
    percents, potential); `pkc_check` (every particle ID exactly once) on the kept full snapshots, `VF_PKC_NPROC`
    at a time sharing the job's threads; a pass is cached in `<state>/pkc_check_cache.json` by file name, size and
    mtime, so a rerun does not read them again;
  - subsample fraction within 5 σ; Rockstar batches cover every snapshot; ctrees outputs present;
  - vahc: halo counts from the basic FITS table headers (the `.trees` text is parsed only if that table is
    missing); `.vahc`, `.tens`, `.vpot` present with that many rows and the same halo IDs in the same order (their
    equality with `.trees`, row by row, is the fits stage's check, required to have passed after the last vahc);
  - fits and parquet as in 5.6; P(k) against CLASS (13.2);
  - on swarm/pegasus, `archive_run.py --source-sums` writes `KEPT_FILES.json` with `KEPT_SUMS_NPROC` files at a
    time (default: the job's cores).
  - Measured old vs new on the 128^3 test run (swarm): 107 s → 10 s, all checks and all 92 halo counts the same;
    the checksum files identical (1221 files).

### 15.5 Deploying code changes to the sites

The manager and the jobs run the code of each site's install (`/user/shadab/sahyadri_prod/sahyadri-sandbox`,
`/user1/shadab/sahyadri_pipeline/sahyadri-sandbox`, `~/sahyadri_pipeline/sahyadri-sandbox` on pegasus,
`/user1/shadab/sahyadri_pride/sahyadri-sandbox` on pride, `/user1/shadab/sahyadri_c7/sahyadri-sandbox` on leap). Every
install stays on branch `run-manager` (`git pull --ff-only`): switching an install to another branch removes the files
only that branch has, and the next job of that site fails (leap r100 945009, 24 Nov 2026: `leap/env_leap.sh` missing
after the leap install was switched from pride-probe to run-manager before the merge).
Pull there only while no vahc, fits or parquet job of that site is running: `run_addvalue.py` starts a new Python
process per snapshot, which loads the new module with the old job's arguments (this broke a running vahc job once,
19 Nov). Jobs already queued keep the job script they were submitted with (PBS copies it at `qsub`). After editing
a `run.conf`, reload it in the manager (`reload_conf`); for a swarm/pegasus run refresh the mirror config first
(`manager.mirror_conf`).

## 16. TIFR sites pride and leap (merged into run-manager 25 Nov 2026)

Branch `pride-probe` (probe of the TIFR theory clusters, 20–23 Nov; notes in
[handover_pride_probe_20261123.md](handover_pride_probe_20261123.md)) was merged into `run-manager` on 25 Nov 2026.
Only site-specific code came with it; the production settings of pawna, swarm and pegasus are unchanged, and the
run-manager production defaults (integer positions `bMemIntegerPosition`, `SIM_CHK_BPP`, memory model, watchdog,
swap alert) now apply to pride and leap as well.

### 16.1 The sites

| | pride | leap |
|---|---|---|
| Login | `shadab@pride.theory.tifr.res.in` | `shadab@leap.theory.tifr.res.in` |
| Nodes | 16 × 20 Haswell (E5-2660 v3, AVX2), ~252 GB (one 189, one 220); 1 GbE | 12 × 32 Skylake (Gold 6142), 187 GB; 100 Gb/s hfi1 (Omni-Path) |
| OS | Ubuntu, gcc 11 | CentOS 7: glibc 2.17, bash 4.2, kernel 3.10 |
| PBS | `workq`, 72 h; compute nodes cannot qsub | `workq`, 72 h; compute nodes cannot qsub |
| Install | `install_production.sh pride` → `/user1/shadab/sahyadri_pride/{codes,pkdgrav3_io,sahyadri-sandbox}` (pkdgrav3 for Haswell) | `leap/build_c7.sh` → `/user1/shadab/sahyadri_c7/{pkdgrav3_io,sahyadri-sandbox}` (glibc-2.17 conda env, skylake-avx512); Python steps use pride's venv |
| Scratch | `/user1/shadab/sahyadri_scratch_pride` | `/user1/shadab/sahyadri_scratch_leap` |
| Run-time env | `pride/env_pride.sh` | `leap/env_leap.sh`: `--mca pml ucx`, `UCX_TLS=rc_verbs,ud_verbs,self,sm`, `UCX_NET_DEVICES=hfi1_0:1`; `leap/ssh_noconda` |
| Use | halo stages; simulations up to 1024^3 | 2048^3 simulation on 6 nodes; no halo stages (Rockstar etc. not built for glibc 2.17) |

Both share swarm's Lustre `/user1` (read-only for anything of swarm's). The pawna manager has `sites.pride` and
`sites.leap` in its code defaults (`~/.sahyadri_mgr/config.json` overrides them; leap: `max_sim_restarts 15`), and
`queues.txt` has `leap workq open`.

### 16.2 Resources (`setup_production.py --site pride|leap`)

| Stage (2048^3) | pride (`resources_pride`) | leap (`resources_leap`) |
|---|---|---|
| sim | 5 × 20 cores, 230 GB, 72 h (untested over 1 GbE; not recommended) | 6 × 32 cores, 175 GB each, 72 h, 2 ranks per node, `place=scatter:excl`, nodes named by `SIM_SELECT` |
| rockstar | 20 writers per node, nodes from the Rockstar memory estimate (≤ 240 GB each) | (not installed) |
| ctrees | 20 cores, 128 GB | – |
| vahc | 20 cores, 245 GB, 2 snapshots at a time (measured peak 238.9 GB PSS, grid 1024) | – |
| fits | 16 cores, 160 GB, 24 h | – |

**leap `SIM_SELECT`.** PBS cannot leave a node out, so a multi-node simulation names its nodes: the first `SIM_NODES` of
`LEAP_FABRIC_NODES` (compute-4-2, 4-4, 4-5, 4-6, 4-8, 4-9, 4-11, 4-12: an active hfi1 port on 22 Nov; 4-3 has its port
down, 4-7 is down). Restarts reuse it; if a named node goes down, edit `SIM_SELECT` in `run.conf` (or update
`LEAP_FABRIC_NODES` and rerun `setup_production.py ... --force`), then refresh the manager's mirror config and reload.

Measured simulation speed (512^3 L200 restart, 2 ranks/node, s/step): leap 34.3 (1 node), 20.0 (2), 10.5 (4),
7.8 (6); pride 63.3, 40.7, 25.5 (1, 2, 4); swarm 22–24.5 (1 × 32 cores). Pride 2048^3 post-processing test: Rockstar
~57 min per snapshot on 3 nodes, vahc ~75 min per snapshot (2 at a time), consistent-trees 43 GB.

### 16.3 leap r100 (2048^3, seed 471913)

- **Set up:** 22 Nov. **Approved:** 23 Nov. **First submission:** 945009, started 24 Nov 19:10. It failed after 20 s
  (`env_leap.sh: No such file or directory`, pkdgrav3 exit 255, no output written) because the leap install had been
  switched to run-manager before the merge (section 15.5).
- **25 Nov:**
  - Leap install updated to the merged run-manager (88065e3).
  - r100's configuration rewritten with the current production settings:
    `setup_production.py --site leap --L 200 --N 2048 --cosmo default --real 100 --force`.
  - The only changes: `bMemIntegerPosition = True` in `p.par` and `SIM_CHK_BPP=36` in `run.conf`. Steps, outputs,
    seed and resources are the same. The old files are in `r100/config_before_20261125/`.
  - Manager: mirror config refreshed (`manager.mirror_conf`), `reload_conf`, `retry` (acknowledges 945009),
    `submit_sim` → **945011.leap**, running since 25 Nov 00:08 on compute-4-2, 4-4, 4-5, 4-6, 4-8, 4-9. Log:
    `/user/shadab/r5_rescue/restart_r100_leap.log` on pawna.
- **Expected:** about 1.1–1.4 × swarm's s/step and about 10 72-hour jobs. The manager restarts it from the newest
  checkpoint (at most 15 times).
- **Halo stages:** not decided yet. Options: run them on pride (same `/user1`) with a second run.conf, or add a
  separate post-processing site per run to the manager.

### 16.4 Code fixes, 24–25 Nov 2026 (post-processing)

Found by pyflakes and smoke tests while documenting the code (API reference: [docs/api](api/README.md)). Each name
below was undefined or never set, so the line raised an error whenever it ran:
- `readers.SnapshotReader.hdf5_to_json`
- `compression12bit.compress_integer_array` (the automatic bit depth)
- `Nbody_Compression.print_this`
- `correlations.PowerSpectrum.wtd_ngp_field`, used by `voronoi_web` for the Voronoi volume functions: three missing
  names, and `ndata` is now passed as `c_long`
- `postprocess.py`: `int_`; and the switched-off VVF writer that used an undefined `stats`, now removed
- `submit_CiPod.sh`: `SCRATH_DIR`

`correlations.density_field` and `wtd_ngp_field` also checked positions with `np.any(pos) < 0.0`, which is never
true. They now raise ValueError for a negative position.

The fixed functions above are not used by the pkdgrav3 production stages. The exception is `density_field`:
`run_addvalue.py` and `pk_measure.py` call it, but only with positions already wrapped into [0, L). The halo positions
go through `% Lbox` in `readers.HaloReader`, and the particle positions are decoded into [0, L). So the new check does
not change their results.

## 17. Maintenance windows (manager option, 25 Nov 2026)

The manager plans around announced shutdowns of a site. Windows are kept in `~/.sahyadri_mgr/maintenance.txt`
(`manager.py maint <site> <start> <end|-> [note]`, `manager.py maint <site> clear`, `manager.py maint` lists; local
time `YYYY-MM-DDTHH:MM`). The full behaviour is in `tools/sahyadri_manager/README.md` (Maintenance windows). In short:

| When | What the manager does |
|---|---|
| any time before the start | submits a job only if it is expected to end 30 min (`maint_guard_min`) before the start: 1.25 x the longest earlier job of that stage (same run, else same site and N; Rockstar per snapshot), else the requested wall time. A wall time that would run past the start is cut (`submit_pipeline.sh --wall HH:MM:SS`) |
| simulations before the start | started or restarted only with >= 4 h (`maint_min_sim_h`) of computing before the STOP time |
| `maint_stop_h` before the start (pawna 6 h, other sites 2 h) | `touch <run dir>/STOP` for every running simulation of the site: pkdgrav3 finishes the step, writes a checkpoint, exits cleanly and deletes the file (`MSR::CheckForStop` in `master.cxx`); waiting simulation jobs are deleted (qdel) and scheduled again |
| during the window (until 30 min after the end; with end `-` until the line is changed) | nothing submitted to the site |
| after it | simulations restart from their checkpoint (not counted in `max_sim_restarts`); jobs killed by the shutdown (started before the window, ended at most 1 h before its start or vanished) are resubmitted instead of stopping the automation |

pawna is the manager's host: when pawna is down the manager is down too. After the reboot start it again
(`tools/sahyadri_manager/start_manager.sh start`) once `/user` and `/storage` answer. It then works through the jobs
killed by the shutdown.

### 17.1 Shutdown of pawna, swarm, leap and pride, Fri 27 Nov 2026 08:00-20:00 IST

Announced: all four TIFR clusters down at 08:00 with an expected downtime of about 12 h. They were entered as
`<site> 2026-11-27T08:00 2026-11-27T20:00`. The manager submits again from 20:30 (`maint_settle_min`). pawna's
manager has to be started again after the reboot. If the work runs late, move the end
(`manager.py maint <site> clear`, then add the window again with the new end). Before the queues reopen, check node1's
NFS client (section 16, admin message of 25 Nov). workq stays in `drain` until then.

State on Wed 25 Nov, 12:00 IST (qstat on pawna and swarm; steps from the run logs):

| Run | Site / job | State | Plan |
|---|---|---|---|
| L200N2048 r3 | pawna node4, 2539100 | step ~2085 of 2126 | finishes Wed afternoon; Rockstar and post-processing go in only when they fit before Fri 07:30 |
| L200N2048 r5 | pawna node2, 2539063 (168 h wall) | step ~928 | STOP Fri 02:00 IST, checkpoint drained to NFS before 08:00; restarts after the window |
| L1000N2048 r1 | pawna, 2539156 (queued, node4, 168 h wall) | waits for r3 | starts on node4 when r3 ends, ~34 h of computing, STOP Fri 02:00; deleted and rescheduled if still waiting at 02:00 |
| L200N2048 r4 | pawna | paused (node1 incident) | unchanged |
| r6, r7 (swarm) | swarm 1786786/1786787 (72 h wall) | stop at iWallRunTime Thu ~08:45 | restarts submitted Thu with the wall time cut to Fri 07:30; STOP Fri 06:00; restart after the window |
| r7 Rockstar | swarm 1787341 (queued, 72 h) | waits for cores | submitted before the window was set: if killed by the shutdown it is resubmitted afterwards |
| leap r100 | leap, 945011 (6 nodes, 72 h wall) | step 22 at 12:50 Wed (~35 min/step) | STOP Fri 06:00 (`maint_stop_h` 2 h; checkpoint straight to Lustre); restarts after 20:30 |
| pride | | no jobs | nothing submitted 08:00-20:30 |
| pegasus runs | | not affected | jobs keep running; while pawna is down nothing is submitted for them |

## 18. Node health checks, leftovers and checkpoint checks (25 Nov 2026)

These follow the node1 incident (sections 16 and 17). The full description is in `tools/sahyadri_manager/README.md` (Node health):

1. **The job checks its node** (`scripts/pkdgrav3_pipeline/node_check.sh`, called by `pipeline_env.sh` in every
   PBS job on pawna). A bad node means the job exits 75 before doing anything. The manager then avoids the node and
   resubmits the job elsewhere.
2. **The manager handles bad nodes:** it avoids the node for 6 h or until a probe passes, and pins new jobs to a
   healthy node with room (`submit_pipeline.sh --host`).
3. **Periodic probe** of all nodes by the manager, every `node_probe_min` minutes, set on the page (Summary, Nodes).
4. **Leftovers** in the nodes' `/var/tmp` are listed on the page and deleted with a button and a typed
   confirmation. A restart also takes the newest *complete* checkpoint (`chk_skipped` otherwise).

First probe of all four nodes (25 Nov, 16:25 IST, `node_check.sh` over ssh from pawna):

| Node | Result | Details |
|---|---|---|
| node1 | ok | rebooted 15:36; r4 restart 2539173 loading (MemAvailable 31 GB); leftovers `shadab_rockstar_2539124` 450 GB, `shadab_rockstar_2539163` 327 GB |
| node2 | warn | Writeback 824 kB unchanged, `flush-0:56` (`/user`) in D state for 5.3 d, 30 hung `sync`; r5's drain still copies at 34 MB/s |
| node3 | ok | |
| node4 | warn | Writeback 3240 kB unchanged, `flush-0:58` (`/user`) and a `python3` in `wait_on_commit` for 5.2 d, 30 hung `sync` |

A stuck Writeback alone is a warning, not a reason to avoid the node. On nodes 2 and 4 the rest of the I/O still
works. On node1 the same symptom came with the memory pressure and I/O hang that stopped every job. Nodes 2 and 4
need the reboot of 27 Nov.

## 19. Halo stages on another TIFR cluster (POST_SITE; 25 Nov 2026)

swarm, leap and pride share the Lustre file system `/user1`. A run computed on swarm or leap can have its halo
stages (rockstar, ctrees, vahc, vpot, fits, parquet, verify, xcorr, vprof) on pride instead:

- **Why:** on swarm the per-user limit is 310 running cores (`max_run_res.ncpus`, and 384 running plus queued).
  Two 2048^3 simulations take 256, so a 64-core Rockstar job cannot start next to them (r7's `rs0_7` waited for
  this reason). leap has no Rockstar build. pride has no per-user limits and 10 of its 16 nodes were up and idle on
  25 Nov.
- **Speed on pride** (23 Nov test with two r3 snapshots, section 16): Rockstar ~57 min per 2048^3 snapshot on 3 nodes
  × 20 cores (pawna ~23 min), head process 141 GB; consistent-trees 32 min, 43 GB; vahc ~75 min per snapshot.
  A whole run: Rockstar ~95 h in batches; three runs can use pride at once (3 nodes each).

**Set-up** (on pride, with pride's install):

    /user1/shadab/sahyadri_pride/codes/venv/bin/python \
        /user1/shadab/sahyadri_pride/sahyadri-sandbox/scripts/pkdgrav3_pipeline/production/post_site_conf.py \
        <run dir>/run.conf --site pride

It writes `<run dir>/run.post_pride.conf` (the run.conf with pride's SANDBOX/CODES/PKD_IO_ROOT, `SITE=pride`, pride's
halo-stage resources from `setup_production.py resources_pride`, `RS_LOCAL_OUT=0`) and appends a marked block with
`POST_SITE=pride` and `POST_CONF=...` to the run.conf. SCRATCH, LOGS and the state directory stay those of the run, so
the pride jobs read the snapshots where the simulation writes them and write the halo products, logs and events into
the run's own directories. Then on pawna: `manager.py reload-conf swarm:<stem>/r<R>` (fetches the run.conf again)
or, for a new run, `manager.py add swarm:<run.conf>`.

**What the manager does with it:** `Run.stage_site` puts those stages on POST_SITE (the simulation and the archive pulls
stay with the run's site). It submits them with pride's `submit_pipeline.sh` and the post configuration, reads pride's
qstat with the run's own, and counts Rockstar jobs (`max_rockstar`), queued jobs (`max_pbs_queued`), closed queues and
maintenance windows on the site where each stage runs. The events of the pride jobs arrive through the run's normal
mirror (they are in its state directory), so the snapshot deletion and archive checks see Rockstar's results as if
it had run on swarm. The run card shows a "halos: pride" badge. Undo: delete the marked block from the run.conf and
`reload-conf`.

**Test (25 Nov 2026):** `swarm:Sahyadri_L200N128/default128/r3` in `/user1/shadab/sahyadri_scratch_posttest` (no
archive): simulation on swarm (job 1787903, 16 cores), the halo stages on pride. Results below.

Results (25 Nov 2026, from the manager's job list and the job logs):

| Stage | Site | Jobs | Notes |
|---|---|---|---|
| sim | swarm | 1787903 | 2126 steps; `qalter` to 12 cores (only compute-1-1/1-2 had free workq cores; compute-1-8 belongs to queue `test`) |
| rockstar | pride | 764489–764497 | 9 batches while the simulation ran (first: 0–11, 80 s, 12/12 lists) |
| ctrees, vahc, fits, parquet | pride | 764498–764501 | one at a time (pride `max_pbs_queued`) |
| verify | pride | 764502, 764503, 764504 | 764504 passed: snapshots, subsample, rockstar, ctrees, vahc, halos, fits, parquet, pk |

Two fixes came out of it. (1) Lustre file times are the writing node's clock, and pride's nodes disagree by up to ~8 min
(offsets against the login node, measured with parallel `ssh date`: −383 s on compute-2-6 to +85 s on 2-14). verify's
"catalogues newer than their FITS tables" and archive_run.py's "kept files changed after verify" now allow 900 s
(`VF_CLOCK_SLACK_S`, `AR_CLOCK_SLACK_S`); the manager's Rockstar redo margin is 15 min. (2) job_verify.pbs no longer
writes the KEPT_FILES.json checksums when the run has no ARCHIVE. Run phase: "verified (no archive)".
