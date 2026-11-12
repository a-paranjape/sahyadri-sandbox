# Gadget-4 vs pkdgrav3: gravitational potential from the same initial conditions

Both codes start from one IC file (Gadget-4 NGenIC, 2LPT, CLASS P(k), as in Sahyadri production) and run with
the Sahyadri default cosmology and settings on pawna. The potential is compared particle by particle (same
IDs) and halo by halo (Rockstar + consistent-trees on each run, through `scripts/pkdgrav3_pipeline/`).

```
job_gadget.pbs                                       submit_pipeline.sh compare32.conf
 CLASS P(k), z=49 -> NGenIC ICs (RestartFlag 6)  ───►  sim (pkdgrav3, achInFile = same ICs, outputs at
 -> ics_gadget2.dat (pkdgrav3) -> Gadget-4 run        Gadget's snapshot times) -> rockstar (PKC) -> ctrees ─┐
 -> snapnames.txt, gadget_out_z.txt              ───►  submit_pipeline.sh compare32_gadget.conf             ├─► job_compare.pbs
                                                       rockstar (Gadget-4 HDF5) -> ctrees ──────────────────┘   compare_potential.py
```

## Files

| File | What |
|---|---|
| `compare32.conf`, `compare32_gadget.conf` | the 32^3 run (one config, two views: `SIM_CODE=gadget4` switches `STEM` to the Gadget run) |
| `compare256.conf`, `compare256_gadget.conf` | the 256^3 run, with VAHC for both codes (`CMP_PKD_STAGES`, `CMP_G4_STAGES`) |
| `Config.sh.template`, `setup_gadget4_pawna.sh` | Gadget-4 builds `mesh<2N>-NGenIC` and `mesh<2N>` with the production compile options |
| `make_gadget4_param.py` | Gadget-4 parameter files from `config/sims/run.param.template` (edits of `submit_CiPod.sh`) |
| `gadget4_ic_to_gadget2.py` | Gadget-4 HDF5 ICs -> Gadget-2 binary that pkdgrav3 reads (sorted, IDs 0..N-1, mass for pkdgrav3's G) |
| `job_gadget.pbs` | CLASS, ICs, conversion, Gadget-4 run, snapshot list and redshifts |
| `job_compare.pbs`, `compare_potential.py` | the comparison (numbers only: CSV, JSON, NPZ): potential, positions, velocities |
| `halo_potential_profile.py` | stacked potential around halos in mass bins (stacks only), r in comoving Mpc/h |
| `compare_halos.py` | mass and Vmax functions, matched halos, VAHC quantities |
| `compare_vpot.py` | per-halo pipeline potential (`.vpot` columns of `out_*_vahc.fits.gz`) of matched halos |
| `../post-process/halo_potential_xcorr.py`, `halo_pair_sums.c` | halo-halo potential cross-correlation between mass bins (below) |
| `plot_compare.py` | figures from the comparison output (needs matplotlib; run anywhere) |
| `submit_compare.sh` | submits everything as one PBS chain |

Changes in `scripts/pkdgrav3_pipeline/` used here (backwards compatible):
- `make_pkdgrav3_par.py`: `IC_FILE` in the config reads the ICs from a file. It then leaves out every
  IC-generation parameter (pkdgrav3 generates ICs whenever `nGrid` is given) and writes `dKpcUnit`, `dMsolUnit`,
  because pkdgrav3 derives its physical units only when `nGrid` is set (`units.cxx`); without them the HDF5 and
  `.pkc` headers say `BoxSize = h` and the `.pkc` potential is not in (km/s)^2 (found on the Mac test below).
- `job_rockstar.pbs`: `SIM_CODE=gadget4` reads Gadget-4 HDF5 snapshots (`FILE_FORMAT = "AREPO"`, production
  `config/halos/rockstar_template.cfg` settings).

## Settings (Sahyadri default, 32^3)

| | Gadget-4 | pkdgrav3 |
|---|---|---|
| ICs | NGenIC 2LPT, grid 32, seed 12345678, CLASS P(k) at z = 49 (A_s = 2.098903e-9, n_s = 0.9649) | the same file |
| cosmology | Om 0.3137721, OL 0.6862279, Ob 0.0493017 (DM only), h 0.6736, L = 200 Mpc/h | same (no baryons) |
| compile / accuracy | production Config (PMGRID = 2N, DOUBLEPRECISION, LEAN, ...), ErrTolForceAcc 0.002, ErrTolIntAccuracy 0.01, MaxSizeTimestep 0.004 | `classic` theta/replicas switch, `bEpsAccStep`, dEta 0.3, `bNewKDK`, Ewald (as `sahyadri2048.par`) |
| softening | 0.2083 Mpc/h comoving = L/N/30 (Plummer-equivalent, spline Newtonian at 2.8 eps) | dSoft = 1/(60 N) = 0.1042 Mpc/h (production rule; kernel Newtonian at 2 dSoft) |
| outputs | `config/sims/outputs100.txt` + a = 1 (101) | at the redshifts Gadget actually wrote (`gadget_out_z.txt`), 20 steps between outputs |
| halos | Rockstar (AREPO reader), FORCE_RES 0.2083, MIN_HALO_PARTICLES 20, consistent-trees | Rockstar (PKC reader), the same settings |

Note the softening: the production rule makes pkdgrav3's kernel Newtonian beyond 1.0 eps_Gadget, Gadget's beyond
2.8 eps_Gadget. This is part of what is being compared (it is what the production runs do).

## Run

```bash
ROOT=/storage/$USER/pkdgrav3_io/potcmp
S=$ROOT/sahyadri-sandbox/scripts/gadget_pkdgrav3_compare
# login node, a few minutes; run as a child of a login shell (Intel MPI env; `bash -l script args` loses the args)
bash -lc "bash $S/setup_gadget4_pawna.sh /user/$USER/Projects/code/Gadget-4/gadget4 $ROOT/g4home 32"
$S/submit_compare.sh $S/compare32.conf --dry-run
$S/submit_compare.sh $S/compare32.conf
# results: $ROOT/data/compare/pkd32_vs_gadget32_r1/{particles.csv,halos.csv,summary.json,snap_*.npz}
python3 $S/plot_compare.py $ROOT/data/compare/pkd32_vs_gadget32_r1        # wherever matplotlib is
```

## Conventions in the comparison

- Potentials are peculiar, physical, in (km/s)^2. Gadget-4: `Potential / a` (HDF5 unit attribute a^-1).
  pkdgrav3: `Potential * U^2 / a` with U = 100 L / sqrt(8 pi / 3) km/s (the `.pkc` potential is the same; checked
  per snapshot, `pkc_max_err` against `pkc_step`). pkdgrav3's HDF5 `Header/Time` is code time: a comes from `Redshift`.
- pkdgrav3 keeps the softened self-interaction of every particle, -(105/64) m/dSoft (`gravity/pp.h` at r = 0),
  Gadget-4 removes its own (`gravity.cc`); it is subtracted from pkdgrav3 (`pkd_selfpot` in `particles.csv`).
- A remaining constant per snapshot (zero point of the periodic potential) is reported as `offset_median` and
  removed before comparing halos. On the synthetic lattice test it equals the simple-cubic Madelung term
  2.837 m/d, i.e. the two codes count a particle's own periodic images differently. A constant changes no force.
- Particles: ID k+1 in Gadget-4 = iOrder k in pkdgrav3 (= row of its ordered HDF5 output).
- Reference: an FFT Poisson solution (CIC, deconvolved, grid 4N) of each snapshot's own particles, so a
  difference between the codes can be split into "different positions" and "different potential solver".
- Halos: host halos (pid = -1) of `out_<snap>.trees`, members = particles within Rvir of the centre in the same
  snapshot; phi_min, <phi>, and <phi> of the 10 % most bound. Gadget and pkdgrav3 halos are matched by mutual
  nearest centre (< 0.5 Rvir, |log10 M ratio| < 0.3). `phi*_pkd_of_gadget_members` is pkdgrav3's potential of
  the Gadget halo's member particles (same IDs).

## Stacked potential profiles around halos (`halo_potential_profile.py`)

Mean potential and its scatter in shells of comoving distance (Mpc/h) from host-halo centres, per halo-mass bin:

```bash
P=$S/halo_potential_profile.py; O=(--mbins 2e12,1e13,5e13,2e14,1e16 --rbins 0.02,50,36)
$P gadget <snapshot_100.hdf5> <halos/.../out_100.trees> gadget_out100.npz "${O[@]}"
$P pkd    <snap.02020>        <halos/.../out_100.trees> pkd_out100.npz "${O[@]}" --par p.par --offset <offset_median>
$P pkc    <snap.02020>        <halos/.../out_100.trees> pkc_out100.npz "${O[@]}" --par p.par --offset <offset_median>
```

- Only stacks are kept (no per-halo profiles): per mass bin and shell `stack_mean`, `stack_std` (scatter over all
  (halo, particle) pairs), `stack_err` (jackknife over 3^3 sub-cubes of halo centres), `delta_hm` (1 + xi_hm - 1),
  `npair`, `r_mean`, and the per-sub-cube sums `jk_n`, `jk_s`, `jk_s2`, `jk_nh` (paired code-to-code errors).
- Scaling: particles are read in chunks by parallel processes and paired with KD-trees of halo centres; outer shells
  use nested subsamples u < 8^-l, u = hash(particle ID), so the work per halo is ~ levels x `--target` (16384) pairs
  and the same particles are used in both codes. 256^3 with ~17 000 halos: 3-13 s on 8 cores.
- Tests: brute force (identical without dilution, within 1.4 jackknife errors with it); HDF5 = `.pkc`.
- Binning: `--rbins rmin,rmax,n` (log) or `--rbins lin:rmin,rmax,dr` (e.g. `lin:0,50,0.2`, 250 shells).
- Weighting: `--weight particle` (default; every (halo, particle) pair, i.e. mass-weighted) or `--weight volume`
  (grid points). Volume weighting first puts the potential on an `--vgrid N`^3 grid of cell centres (default
  2 N_part^(1/3)) by a weighted local linear fit to the `--vknn` (16) nearest particles; inverse-distance weighting
  was biased by up to 8% on a clustered test, the linear fit by <= 1.8%. `--vcache grid.npy` keeps the grid (512^3:
  0.5 GB, 6 min on 8 cores) so the log and linear stacks reuse it. The grid step reads all particles into one
  KD-tree, so beyond 512^3 particles it needs a slab-by-slab version (not written). Use `--target 8192` for volume
  stacks. Shells inside one grid cell only see the interpolation and are masked in the notebook.
- All report stacks (both binnings, both weightings): `reports/gadget_pkdgrav3_potential/run_stacks_local.sh`.
- Figures: `reports/gadget_pkdgrav3_potential/stack_potential.ipynb`.

## Gadget-4: `EVALPOTENTIAL` is needed

The first run (2026-11-11, kept in `data_run1_zeropot/`) used exactly the production Config and every Gadget
snapshot had `Potential = 0`: `OUTPUT_POTENTIAL` only allocates and writes the field, `EVALPOTENTIAL` computes it
(src/gravity/gravity.cc). The production Sahyadri Config (snapshot_ics_000 header) has no `EVALPOTENTIAL`, so the
production Gadget snapshots most likely carry a zero potential too. `Config.sh.template` now adds it; it does not
change the forces.

## consistent-trees in small boxes

`gen_merger_cfg.pl` skips catalogues smaller than 20000 bytes ("too few halos"); at 32^3 every catalogue is smaller
(<= 15 halos) and consistent-trees then fails (`find_parents_and_cleanup`). `CT_MIN_LIST_BYTES` in the config makes
`job_ctrees.pbs` use a patched copy (here 1000 B: skip only catalogues without halos).

consistent-trees still fails at 32^3/200 Mpc/h (<= 15 halos per output): `gravitational_consistency` builds its
link tolerances from the halo-to-halo scatter per mass bin and stops with "too few halos at scale factor ... to
calculate consistency metric". This is a limit of consistent-trees, not of the setup (the 128^3/200 pipeline test with
5205 halos ran). `compare_potential.py` and `job_compare.pbs` then use Rockstar's `out_<snap>.list` (all halos, no
host/subhalo flag).

## Results: 32^3, 200 Mpc/h, z = 49 -> 0 (pawna jobs 2536518-2536524, 2026-11-11)

Gadget-4 run 74 s on 16 ranks; pkdgrav3 101 outputs (2020 steps) on 16 threads; Rockstar 13 (Gadget-4) and 8
(pkdgrav3) halos at z = 0. Comparison (`compare_potential.py`, all 101 outputs; run on the Mac on copies of the
pawna outputs, results in `data/compare/pkd32_vs_gadget32_r1/` on pawna):

| z | corr | slope | rms(pkd - Gadget) / rms(phi) | FFT(pkd pos) - FFT(Gadget pos), same units | median / 90% \|dx\| (Mpc/h) | zero-point offset (km/s)^2 |
|---|---|---|---|---|---|---|
| 12.0 | 0.9997 | 1.001 | 0.026 | 0.082 | 0.016 / 0.029 | +1.4e3 |
| 2.8 | 0.9997 | 0.998 | 0.023 | 0.103 | 0.10 / 0.17 | +4.7e2 |
| 0.86 | 0.993 | 0.995 | 0.117 | 0.124 | 0.22 / 0.40 | +1.1e2 |
| 0 | 0.969 | 0.988 | 0.253 | 0.220 | 0.38 / 0.89 | -2.0e3 |

(rms phi of Gadget-4: 7.5e5 at z = 12, 8.6e5 (km/s)^2 at z = 0; pkdgrav3 self-term removed: -1.9e7 at z = 12,
-1.4e6 at z = 0.)
- The zero points agree once pkdgrav3's self-term is removed: the remaining offset is <= 2e3 (km/s)^2, 0.2% of
  rms phi. The Madelung-like constant seen on the synthetic lattice is not visible here.
- The two potentials agree to 2-3% rms while the particles are close (z >= 2); later the difference grows with the
  particle separation (the trajectories diverge: different integrators and softening), and the FFT potential of each
  code's own positions differs by almost the same amount (22% vs 25% at z = 0). The solvers agree much better than
  the positions do.
- Halo profiles at z = 0 (9 Gadget-4 halos with 4e14-1e15 Msun/h): pkdgrav3's potential around the same (Gadget)
  centres agrees within ~5% from 0.2 to 15 Rvir and is ~30% deeper inside 0.2 Rvir, as expected from its smaller
  effective softening (Newtonian beyond 1.0 instead of 2.8 eps_Gadget). pkdgrav3's own halos are a different
  (smaller) set; with 2-9 halos per bin the halo statistics are only indicative at 32^3.

## Tests so far

- Mac, synthetic 16^3 Gadget-4-style IC (shuffled IDs 1..N): converter + `make_pkdgrav3_par.py` (`IC_FILE`) +
  pkdgrav3 3.5.1 (main): reads 4096 particles, total mass = Omega0 exactly (Gadget's mass x 0.999813, pkdgrav3
  uses G = 4.30172e-9), HDF5 rows are the IC particles in ID order, `.pkc` positions = HDF5 within half a step,
  `.pkc` potential = HDF5 x U^2/a within float32 precision, header box 200 Mpc/h (with the unit fix).
- FFT Poisson routine: plane-wave Zel'dovich test on 64^3, slope 1.0003, correlation 1 - 5e-8.

## Results: 256^3 and the report (2026-11-12)

Full write-up with figures, data and script paths: `reports/gadget_pkdgrav3_potential/report.pdf` (LaTeX source,
`stack_potential.ipynb`, `make_figures.py`, small data copies in `data/`). Data on pawna:
`/storage/shadab/pkdgrav3_io/potcmp/data/{sims,halos}/potcmp/{gadget256,pkd256}/r1`,
`.../data/compare/pkd256_vs_gadget256_r1`. In short (256^3, z = 0): potential rms difference 22 % (2 % at z = 3),
mostly from particle positions (median separation 0.12 Mpc/h); mass functions within 2-3 % above 1e12 Msun/h;
pkdgrav3 halos 20-25 % more concentrated at 1e12-1e13 (softening); stacked potential within 1-2 % from 0.2 to
50 Mpc/h, 3-6 % deeper in pkdgrav3 inside 0.1 Mpc/h.

## VAHC rerun with `vahc_fast` and `.vpot` (2026-11-12)

After merging `rockstar-pkc-reader`, VAHC and FITS were rerun at 256^3 for both codes with `PP_FAST=1 PP_POT=1
PP_OVERWRITE=1` (`compare256.conf`; pawna jobs 2536704/2536705 pkdgrav3, 2536706/2536707 Gadget-4). The old
`.vahc/.tens/.vpot` are kept in `$HALO_DIR/vahc_prev/`. 32^3 cannot be rerun: it has no `out_*.trees`.
- Gadget-4 `.vpot`: `scripts/post-process/vpot_hdf5.py` (used by `run_addvalue.py --fast --pot` when the HDF5
  snapshot has `PartType1/Potential`): phi = Potential / a, self term 0, the same `vpot_sums.c` routines as the
  `.pkc` path, density from `ps.density_field` so `.vahc` is unchanged. Header `code = Gadget-4`.
- `job_fits.pbs` passes `--pkc` only when the `.pkc` files exist; `job_vahc.pbs` `PP_OVERWRITE=1` backs up and
  overwrites.
- Mac check at z = 0: new `.vahc` bit-identical to the old for both codes; Gadget sphere sums = KD-tree brute
  force; matched hosts: pkdgrav3 deeper by a median 5.6 % (rs), 2.2 % (Rvir), 1.3 % (2-5 Mpc/h).
- `compare256_vpotfill.conf`, `compare256_gadget_vpotfill.conf`: fill-in without `--overwrite` (12 h) in case a
  run stops at walltime; not needed this time. Walltime cannot be changed with `qalter` on pawna.
- `vpot_sums.so` must be rebuilt (`cc -O3 -fopenmp -shared -fPIC`) where the source is newer than the library.
- FITS: the first run (2536705/2536707) failed on `trees/tree_0_0_0.fits.gz` (read-back float sums over 2e6-row
  chunks vs the whole column; fixed 27a09dc) and froze `locations.fits.gz` with FitsRow = -1 (restart-safe since
  3155b80). After reruns 2536787/2536788 and 2536792/2536793: 661 PASS / 0 FAIL per code, all 94 `_vahc.fits.gz`
  with `HAS_VPOT = True` and `phi_rs` ... `phi_R5Mpch`.
- Report section "Per-halo potential from the pipeline" (`compare_vpot.py`, `make_vpot_figures.py`): at z = 0
  pkdgrav3 deeper by a median 3-6 % within rs, 1.3-2 % within Rvir, ~1 % at 5 Mpc/h (above 1e12); the outer ~1 %
  follows the particle-mean (not median) potential difference, i.e. the deeper halo cores.

## Halo-halo potential cross-correlation (`scripts/post-process/halo_potential_xcorr.py`)

Set-up, running (also as the optional pipeline stage `xcorr`), options and output format:
`docs/pkdgrav3_pipeline_pawna.md` section 12. Results at 256^3: report section "Halo-halo potential
cross-correlation" (`make_xcorr_figures.py`). z = 0 outputs of both codes on pawna:
`/storage/shadab/pkdgrav3_io/potcmp/data/compare/pkd256_vs_gadget256_r1/xcorr/`.

## Correlation multipoles in real and redshift space (`scripts/post-process/halo_xi_multipoles.py`)

Multipoles l = 0..4 (dipole included) of the halo cross-correlation between mass bins, 0-100 Mpc/h, in real space,
redshift space, with the gravitational-redshift shift from the halo potential, and with both (Corrfunc fork with
signed mu: `adematti/Corrfunc@desi`). Set-up, installation (pawna and Mac), running (stage `ximult`), options and
output: `docs/pkdgrav3_pipeline_pawna.md` section 13. Results at 256^3: report section "Correlation multipoles in
real and redshift space" (`make_ximult_figures.py`).

