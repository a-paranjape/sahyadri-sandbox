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
| `compare32.conf`, `compare32_gadget.conf` | the run (one config, two views: `SIM_CODE=gadget4` switches `STEM` to the Gadget run) |
| `Config.sh.template`, `setup_gadget4_pawna.sh` | Gadget-4 builds `mesh<2N>-NGenIC` and `mesh<2N>` with the production compile options |
| `make_gadget4_param.py` | Gadget-4 parameter files from `config/sims/run.param.template` (edits of `submit_CiPod.sh`) |
| `gadget4_ic_to_gadget2.py` | Gadget-4 HDF5 ICs -> Gadget-2 binary that pkdgrav3 reads (sorted, IDs 0..N-1, mass for pkdgrav3's G) |
| `job_gadget.pbs` | CLASS, ICs, conversion, Gadget-4 run, snapshot list and redshifts |
| `job_compare.pbs`, `compare_potential.py` | the comparison (numbers only: CSV, JSON, NPZ) |
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

Mean potential against distance from the halo centre, in halo-mass bins, with its scatter:

```bash
P=$S/halo_potential_profile.py
$P gadget <snapshot_100.hdf5> <halos/.../out_100.trees> g.npz --mbins 1e13,1e14,4e14,1e15 --rbins 0.1,15,20
$P pkd    <snap.02020>        <halos/.../out_100.trees> p.npz --mbins ... --par p.par --offset <offset_median>
$P pkc    <snap.02020>        <halos/.../out_100.trees> c.npz --mbins ... --par p.par --pkc-subsamples 1,3
python3 $S/plot_potential_profile.py prof.png g.npz:Gadget-4 p.npz:pkdgrav3
```

- Per mass bin and shell: `halo_mean`, `halo_std` (mean and standard deviation over halos of each halo's shell
  mean), `halo_sem`, `stack_mean`, `stack_std` (all particles of the bin's halos together), counts, and every
  halo's own profile (`halo_shell_mean_<b>`). Radii in r/Rvir (`--r-units rvir`, default) or comoving Mpc/h.
- Hosts only by default (`--all-halos` for subhaloes); `--max-halos` (default 2000 per bin, random) bounds the work.
- Scaling: particles are read in chunks (HDF5 slices or `.pkc` blocks) by parallel processes; each chunk is paired
  with a KD-tree of the bin's halo centres (`cKDTree.sparse_distance_matrix`, periodic), and only per-halo,
  per-shell sums (count, sum phi, sum phi^2) are kept, so memory does not grow with the particle number. For
  2048^3, `--pkc-subsamples 1,3` (4 % of the particles) is unbiased for the mean and reads 1/25 of the data.
- `job_compare.pbs` runs it at the outputs closest to `PROF_Z` for Gadget-4 and pkdgrav3 (HDF5 and `.pkc`),
  each code with its own consistent-trees halos, pkdgrav3 without self-term and aligned by `offset_median`.
- Tests (Mac, 32^3 z = 0, Gadget halos): one halo's profile equal to a brute-force calculation (difference 0);
  HDF5 and `.pkc` give the same shell counts up to particles on shell edges (positions differ by <= half the
  1.95 kpc/h step) and potentials within one step; each run < 0.5 s.

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

## Tests so far

- Mac, synthetic 16^3 Gadget-4-style IC (shuffled IDs 1..N): converter + `make_pkdgrav3_par.py` (`IC_FILE`) +
  pkdgrav3 3.5.1 (main): reads 4096 particles, total mass = Omega0 exactly (Gadget's mass x 0.999813, pkdgrav3
  uses G = 4.30172e-9), HDF5 rows are the IC particles in ID order, `.pkc` positions = HDF5 within half a step,
  `.pkc` potential = HDF5 x U^2/a within float32 precision, header box 200 Mpc/h (with the unit fix).
- FFT Poisson routine: plane-wave Zel'dovich test on 64^3, slope 1.0003, correlation 1 - 5e-8.
