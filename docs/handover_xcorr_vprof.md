# Handover: halo potential tools (xcorr, ximult, vprof) and the Sahyadri data on pawna

Written 24 Nov 2026 for a new Claude session that builds on these tools. Repository
`a-paranjape/sahyadri-sandbox`, branch **`run-manager`** (everything below is committed and pushed there).
Local clone on the user's Mac: `/Users/shadab/Documents/Projects/claude_code/sahyadri-sandbox`.

## 1. Rules the user has set (always in force)

- Work on a branch (or a fork); never push to or merge into `master` without asking. Push to the branch as you go.
- No simulation larger than 512^3 without approval.
- Other users' files on pawna are read-only. All jobs go through PBS (`qsub`); never run heavy work on the login
  node and never bypass the scheduler.
- Building third-party source from the internet needs explicit approval (pre-approved: Rockstar, consistent-trees,
  pkdgrav3, pawna jobs up to 512^3 - report them anyway).
- State how every measurement was made (job id, cores, wall time, memory, command).
- Do not delete data (dead runs, archives) without the user's approval; avoid `rm -rf` in scripts.
- The archive under `/storage/shadab/sahyadri_sims` is read-only and SHA-256 manifested: do not write into it
  except through the archive procedure (section 6), and only when asked.
- Another session (the "sahyadri-sandbox" session) runs the production simulations and the run manager
  (`tools/sahyadri_manager`). Do not touch running jobs, the manager, `~/.sahyadri_mgr/`, or the queue states.

## 2. Logging in from the Mac

```
ssh -Y shadab@pawna.theory.tifr.res.in        # zsh alias: pawna
```
(The aliases are in `~/.zshrc`, not `~/.ssh/config`, so inside non-interactive commands use the full host.)
Run commands non-interactively, e.g. `ssh shadab@pawna.theory.tifr.res.in 'bash -lc "qstat -u shadab"'`.

- PBS is in `/opt/pbs/bin` (`bash -lc` puts it on the PATH). Submit only from the login node: compute nodes can
  `qstat` but not `qsub`/`qrls`.
- Nodes node1-4: 192 cores, 755 GiB each, queue `workq`. Other users (e.g. hridya, bhaskar) share them.
- Pegasus (IUCAA), only if needed: `ssh calam.shadab@pegasus.iucaa.in` (alias `pegasus_shadab`).

## 3. Software on pawna

| What | Path |
|---|---|
| Production repo clone (this repo) | `/user/shadab/sahyadri_prod/sahyadri-sandbox` |
| Python venv (numpy, scipy, fitsio, h5py, Corrfunc fork with signed mu) | `/user/shadab/sahyadri_prod/codes/venv/bin/python` |
| Compiled helpers (`halo_pair_sums.so`, `vpot_sums.so`, ...) | `/user/shadab/sahyadri_prod/codes/lib` (set `SAHYADRI_LIBDIR` to it) |
| Rockstar, consistent-trees, pigz | `/user/shadab/sahyadri_prod/codes/{rockstar,ctrees,bin}` |
| pkdgrav3 | `/user/shadab/sahyadri_prod/pkdgrav3_io` |

Use this venv; the system Python lacks fitsio/Corrfunc. Corrfunc install: `scripts/pkdgrav3_pipeline/install_corrfunc.sh`.

## 4. Where the simulations are (pawna)

Archive root `S=/storage/shadab/sahyadri_sims` (NFS; read-only for new work).

| Data | Path |
|---|---|
| **Main Sahyadri suite** (Gadget, 2048^3, 9 cosmologies) halos | `$S/halos/sahyadri/<cosmo>/r1/{compressed,parquet}` with `<cosmo>` = `default2048`, `var_{As,h,ns,Om}_{m,p}2048` |
| Main suite snapshots | `$S/sims/sahyadri/<cosmo>` (also `var_w_{m,p}2048`) |
| pkdgrav3 512^3 boxes (L = 200, 500, 1000 Mpc/h) halos | `$S/halos/Sahyadri_L{200,500,1000}N512/default512/r1/` |
| ... their snapshots | `$S/sims/Sahyadri_L{200,500,1000}N512/default512/r1/` |
| pkdgrav3 L200N1024, L100N128 (tests) | `$S/{halos,sims}/Sahyadri_L200N1024`, `.../Sahyadri_L100N128` |
| pkdgrav3 2048^3 production (L200, in progress) | `$S/sims/Sahyadri_L200N2048/default2048/r{2..7}` (halos appear when finished) |
| Run configs (run.conf, p.par) of the pkdgrav3 runs | `/user/shadab/sahyadri_scratch/sims/<Box>/<default>/r<k>/` |

Inside a pkdgrav3 halo directory (`.../r1/`):
- `out_<snap>.*` Rockstar / consistent-trees products, `out_<snap>.pk` P(k), `scales.txt` snapshot scale factors.
- `compressed/out_<snap>_{basic,extended,tens,vahc}.fits.gz` the halo tables (FITS). The `vahc` table has the
  potential columns `phi_*` (header `HAS_VPOT = T`) for snapshots that store the potential (z <= 0.5).
- `parquet/out_<snap>_*.parquet` the same tables as Parquet.
- `xcorr/out_<snap>_xcorr_log.npz`, `out_100_ximult.npz`, `out_100_ximult_phiall{,_log}.npz` (section 5).
- `vprof/out_<snap>_vprof_volume.npz`.
- `README.txt`. The archive manifest of each run (`MANIFEST.json`, `SHA256SUMS`) is in `$S/state/<Box>/<default>/r<k>/`.

The 12 potential columns `phi_*` (mean particle potential, (km/s)^2): `phi_rs, phi_R0p5vir, phi_Rvir,
phi_R1p5vir, phi_R200b` (spheres) and `phi_R2R200b, phi_R4R200b, phi_R6R200b, phi_R8R200b, phi_R2Mpch,
phi_R3Mpch, phi_R5Mpch` (smoothed grid). Snapshot 100 is z = 0.

## 5. The tools

Each script has a full docstring (usage, options, algorithm, output format): read it first.

| Tool | Script | PBS stage | Guide section (`docs/pkdgrav3_pipeline_pawna.md`) |
|---|---|---|---|
| Halo-halo potential cross-correlation (mean diff, structure function, covariance of phi between mass bins vs r) | `scripts/post-process/halo_potential_xcorr.py` (+ `halo_pair_sums.c`) | `scripts/pkdgrav3_pipeline/job_xcorr.pbs` | 12 |
| Correlation multipoles xi_l (l = 0..4) in real, RSD, grav-redshift and RSD+grav space; dipole from the potential | `scripts/post-process/halo_xi_multipoles.py` (Corrfunc DDsmu) | `job_ximult.pbs` | 14 |
| Stacked potential profile around halos (particle or volume weighted) | `scripts/gadget_pkdgrav3_compare/halo_potential_profile.py` | `job_vprof.pbs` | 5 (stage list) |
| Per-halo potentials `.vpot` (input of the above) | `scripts/post-process/vpot_sums.c`, `vahc_fast.py`, `vpot_hdf5.py` | `job_vpot.pbs` | 5 (`.vpot`) |

Stages run through `submit_pipeline.sh <run.conf> --stages xcorr|ximult|vprof` (options via `XC_ARGS`, `XM_ARGS`,
`VPROF_*` in run.conf; see the headers of the PBS files). For one-off studies write your own PBS script in a new
work directory under `/user/shadab/` (example: `/user/shadab/ximult_phi/`, which is not in git), call the venv
python on the script, and write outputs there, not into the archive.

Example used for the log-bin dipole (pawna job 2539125, 1 core, 47 min for the three boxes, 4.9 GB peak):
```
$PY $REPO/scripts/post-process/halo_xi_multipoles.py $S/halos/Sahyadri_L1000N512/default512/r1/compressed 100 out.npz \
    --spaces real,rsd,grav,rsd+grav --phi <all 12 columns> --los x,y,z --pairs all --sbins log:0.01,100,100 --threads 1
```

Output layouts (ximult phiall files): `spaces` = real, rsd, grav[col] x 12, rsd+grav[col] x 12 (26);
`DD[space, los(3), pair(10), s, mu(200)]`; `xi_ell_mean[space, pair, ell(5), s]` (+ `_err`);
`dxi_ell[diff(24), pair, ell, s]` (grav - real, rsd+grav - rsd per column, + `_err`); `vg_mean[col(12), mass bin(4)]`.
Mass bins `m_edges` = 2e12, 1e13, 5e13, 2e14, 1e16 Msun/h (hosts, Mvir); `pairs` (10, 2) bin indices A <= B.
xcorr: statistics `[pair, phi column, shell]`. vprof: `stack_mean[b, i]` etc. (docstrings).

## 6. Results so far and where they are written up

- Report: `reports/gadget_pkdgrav3_potential/report.tex` (PDF `report.pdf`, 36 pages): Gadget vs pkdgrav3 potential
  (32^3, 256^3), box-size effects at 512^3 (`boxsize.tex`), multipoles with gravitational redshift (`ximult512.tex`,
  sections 7, 7.1 incl. all 12 columns and the log-bin study), Rockstar multi-node.
- Figure scripts: `make_xcorr_figures.py`, `make_boxsize_figures.py`, `make_ximult_box_figures.py`,
  `make_ximult_phi_figures.py`, `make_vpot_figures.py`; they read the reduced copies in
  `reports/gadget_pkdgrav3_potential/data/box512_L200_L500_L1000/*.npz` (xcorr_log, vprof_volume, ximult,
  ximult_phiall, ximult_phiall_log for L200/L500/L1000, snapshots 64/82/100).
- Quick plot snippet for the phiall files: `/user/shadab/ximult_phi/plot_ximult_phi.py` on pawna.
- Archive additions: `/user/shadab/ximult_phi/archive_add.py` copies a new output into
  `.../r1/xcorr/`, makes it read-only, verifies the SHA-256 and updates `$S/state/<run>/{MANIFEST.json,SHA256SUMS}`
  (backups first). Ask the user before adding anything to the archive.

## 7. Not done / open

- No xcorr/ximult/vprof yet on the 2048^3 runs (main Gadget suite or pkdgrav3 L200N2048). For 2048^3 use
  `--subsample-a` (ximult) / `--target` (xcorr) and check memory; vprof volume weighting is limited to 512^3
  particles (slab version not implemented). Ask before large jobs; pawna nodes are shared with the production runs.
