# Sahyadri pipeline with pkdgrav3 on pawna

How to run the full chain on pawna: **simulation (pkdgrav3, compressed `.pkc` snapshots) → halos (Rockstar,
reading `.pkc` directly) → merger trees (consistent-trees) → value-added catalogs (`.vahc`, `.tens`) → FITS**.
The scripts are in [`scripts/pkdgrav3_pipeline/`](../scripts/pkdgrav3_pipeline/). One config file describes a
run; `submit_pipeline.sh` submits the stages as a chain of PBS jobs.

Branch: `rockstar-pkc-reader` (not merged into `master`). Status of the tests is in
[section 9](#9-what-was-tested).

```
 job_sim.pbs        job_rockstar.pbs       job_ctrees.pbs            job_vahc.pbs            job_fits.pbs
 pkdgrav3   ──────► Rockstar (PKC) ──────► consistent-trees ───────► run_addvalue.py ──────► compress_halo_outputs.py
 snap.<step>.s*.pkc out_<snap>.list        out_<snap>.trees          out_<snap>.vahc/.tens   compressed/*.fits.gz
 snapnames.txt      halos_<snap>.*         trees/, ../scales.txt                             (+ check_compressed_halos.py)
        └──────────── particles (.pkc) are read again by job_vahc ──────────┘
```

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

### 2b. This repository and the halo codes

```bash
cd /user/$USER/Projects            # anywhere
git clone -b rockstar-pkc-reader https://github.com/a-paranjape/sahyadri-sandbox.git
bash sahyadri-sandbox/scripts/pkdgrav3_pipeline/setup_pawna.sh /storage/$USER/sahyadri_codes /storage/$USER/pkdgrav3_io
```

`setup_pawna.sh <CODES> [PKD_IO_ROOT]` builds into `CODES`:

| Directory | What | Source |
|---|---|---|
| `rockstar/` | Rockstar, `make with_hdf5_pkc` (`FILE_FORMAT = "PKC"`, also AREPO/Gadget HDF5) | `git clone https://bitbucket.org/gfcstanford/rockstar.git` at `36ce9eea36ee`, then the changed files of `code/Rockstar/gfcstanford-rockstar-36ce9eea36ee/` copied over (`.pkc` reader, Aseem's curvature and Gadget-4 HDF5 changes, w0/wa in `gen_merger_cfg.pl`; see `scripts/rockstar/readme_*.txt`) |
| `ctrees/` | consistent-trees, `make` | `git clone https://bitbucket.org/pbehroozi/consistent-trees.git` at `7c20add`, then `code/ConsistentTrees/consistent-trees-7c20add/` copied over (w0-wa and curvature background, `README_sahyadri.txt`) |
| `venv/` | self-contained Python 3 venv with numpy, scipy, pandas, h5py, fitsio, zstandard (pip; the system scipy on pawna needs numpy < 1.25) | PyPI |
| `lib/` | `cic_update.so`, `ngp_update.so` compiled from `scripts/post-process/*.c` | this repository |

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
| Rockstar | `FORCE_RES`, `RS_NUM_BLOCKS`, `RS_WRITERS`, `RS_PKC_SUBSAMPLES`, `MIN_HALO_OUTPUT_SIZE`, `MIN_HALO_PARTICLES`, `UNBOUND_THRESHOLD` | `FORCE_RES` empty = `LBOX/NPART/50` (pkdgrav3's default softening); other values as in the production `config/halos/rockstar_template.cfg` |
| trees | `CLEAN_ROCKSTAR` | 1: as production (`clean_trees.pl`), 0: keep all Rockstar outputs |
| value added | `PP_GRID`, `PP_DOWN_TO`, `PP_TENS`, `PP_NPROC` | production: grid 512 (768 for 2048^3), `PP_DOWN_TO=512` when `NPART > 512` |
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

Memory: about 12 × 16 × `PP_GRID`^3 bytes per process (complex grids of the tensor calculation) plus the
particles: ~26 GB at grid 512, ~85 GB at 768. Choose `PP_NPROC` so that `PP_NPROC` times that fits in
`PP_MEM`.

**When to run it.** `.vahc`/`.tens` rows are the halos of `out_<snap>.trees`, with consistent-trees IDs
(`haloID` = `id` of the hlist), so this stage needs the finished consistent-trees catalogs and cannot start
before the ctrees stage. It must run before the FITS stage: the FITS stage puts `b1` into
`out_<snap>_basic.fits.gz` and writes `_vahc`/`_tens` files only if the ASCII files exist, and
`FITS_DELETE_ASCII=1` removes `out_<snap>.trees`. The particles are needed only here after Rockstar, and only
the subsample files that `PP_DOWN_TO` selects: keeping `snap.*.s01.*.pkc` and `snap.*.s03.*.pkc` (4% of the
snapshot volume) is enough to (re)compute `.vahc`/`.tens` later with `PP_DOWN_TO=512`.

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

## 8. Larger runs and pegasus

- Above 512^3 nothing may be submitted without approval. For 2048^3: simulation as in pkdgrav3's pawna doc;
  Rockstar memory on one node has not been measured for 2048^3; `PP_GRID=768`, `PP_DOWN_TO=512`, `PP_NPROC`
  ≤ 6 on one node.
- pegasus: the same scripts work with a different `CODES`/`SCRATCH` and PBS resource names; not tested there.

## 9. What was tested

See the end of this section for the results of the pawna test run (filled in after it ran) and
[rockstar_pkc_reader_plan.md](rockstar_pkc_reader_plan.md) for the Rockstar reader validation.
- Mac, 128^3 / 200 Mpc/h, 3 outputs (z = 3, 1, 0), Rockstar on `.pkc`, patched consistent-trees:
  `.vahc` from `.pkc` byte-identical to `.vahc` from a Gadget-4 HDF5 file holding the same particles;
  `run_addvalue.py` (3 processes) byte-identical to `postprocess.py`; `.tens` equal except 2 of 204,000
  values in the last printed digit; FITS of `.trees/.vahc/.tens` pass `check_compressed_halos.py` (21 checks);
  a `.vahc` with rows sorted by mass is refused by the FITS converter.
