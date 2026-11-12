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
| `venv/` | self-contained Python 3 venv with numpy, scipy, pandas, h5py, fitsio, zstandard (pip; the system scipy on pawna needs numpy < 1.25) | PyPI |
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
| value added | `PP_GRID`, `PP_DOWN_TO`, `PP_TENS`, `PP_NPROC` | grid 512; `PP_DOWN_TO=0` (all particles, streamed; pegasus production used 512 when `NPART > 512`) |
| FITS | `FITS_NPROC`, `FITS_DELETE_ASCII` | delete only after looking at the check log |

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
sets it to 2.2 x 32 B x N^3 + 8 GB, rounded up to 16 GB (`rockstar_mem`): 16 GB at 128^3, 32 GB at 512^3, 96 GB at
1024^3, 624 GB at 2048^3 (estimate: one node, 755 GB, should suffice; to be checked with the 1024^3 run). 3200^3
would need ~2.3 TB, more than one pawna node.

### 5.3 ctrees — `job_ctrees.pbs`
`gen_merger_cfg.pl rockstar.cfg` (writes `outputs/merger_tree.cfg` with Om, Ol, h0 and w0, wa from the
`out_<snap>.list` header), then `do_merger_tree.pl` and `halo_trees_to_catalog.pl` of the patched
consistent-trees, then `scripts/assist/clean_trees.pl` as on pegasus: `hlists/hlist_<a>.list` →
`out_<snap>.trees`, `outputs/scales.txt` → `../scales.txt`, configs to `logs/`, Rockstar outputs deleted.
Note: `clean_trees.pl` matches hlists by the first 5 characters of the scale factor, so outputs closer than
0.001 in `a` would collide (true of the pegasus pipeline too).

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

Memory: about 6 × 16 × `PP_GRID`^3 bytes per process with `PP_FAST=1` (measured: 12-13 GB per process at grid
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
using node-local `/var/tmp` for temporary files, then `check_compressed_halos.py` compares every column with
the ASCII. `readers.HaloReader.prep_halos(use_fits=True)` reads the result.

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
particle; section 5.2): ~0.5 TB at 2048^3 of a node's 755 GB (requested: 624 GB), to be checked with the 1024^3 run.
Rockstar time at 512^3: 64 s for two z ~ 0 outputs with 64 writers and the output on the node's disk, 188 s with 16
writers writing to `/storage` (section 5.2); give large runs many cores (`RS_NCPU`) and a long `RS_WALL`. Plots and the
full measurement: [report](report/pkdgrav3_pipeline_report.pdf), section 6.

- Above 512^3 nothing may be submitted without approval. For 2048^3: simulation as in pkdgrav3's pawna doc;
  Rockstar memory on one node has not been measured for 2048^3; at `PP_GRID=512` vahc needs ~26 GB per process
  (`PP_GRID=768`: ~87 GB, `PP_NPROC` ≤ 6 on one node), with or without downsampling.
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
(`PP_DOWN_TO=0`, density built block by block) on the production grid of 512.

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

- **`verify`** (`verify_run.py`): checks that everything was computed and is consistent.
- **`archive`** (`archive_run.py`): copies the kept files with SHA-256 checksums, reads them back from disk with
  `O_DIRECT`, writes `MANIFEST.json` and `SHA256SUMS`, then makes the archive read-only.

The manager runs on the head node, because compute nodes cannot `qsub`, `qrls` or `qalter`. It:

- reacts to the jobs' event files through `inotifywait`, and runs `qstat` only every 30 min;
- submits Rockstar in batches while the simulation runs, then `ctrees → vahc → fits → verify`, then `archive`;
- moves the scratch data to a trash directory, then purges it, only after you approve each step in its web page.

The page is opened on the Mac with `tools/sahyadri_manager/open_gui.sh`. See `tools/sahyadri_manager/README.md`.

### 13.1 Queueing many runs

```bash
cd scripts/pkdgrav3_pipeline/production
python setup_production.py --cosmo default --N 512 --L 200 --real 1-4 --schedule      # priority 0
python setup_production.py --cosmo 'var_*' --real 1 --schedule 5                            # lower priority
tools/sahyadri_manager/manager.py limits max_sim_running=3 max_sim_queued=1 max_rockstar=1
```

- `--cosmo` takes a name, a comma list, shell patterns or `all` (`cosmologies/<name>.cosmo`). Only
  `default.cosmo` exists so far; the `var_*` cosmologies need their `.cosmo` files made first from
  `config/submit/sinhagad/sub_var_*.conf` with CLASS.
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

Tested on two 64^3 runs, `Sahyadri_L100N64/default64` r1 and r2, 12 Nov 2026:

- Full chain on both, through verify and archive.
- r2: Rockstar batches 0–9, 10–30 and 31–100 submitted while the simulation was running.
- Archive read-back: 0 mismatches.
- Deletion: refused when a kept file changed after the archive. Move to trash, undo, move again and purge all
  worked on r1.
- Clocks: the compute nodes run about 3 min and `/storage` about 15 min behind the head node. All ordering uses PBS
  job IDs or file times from the same server.
