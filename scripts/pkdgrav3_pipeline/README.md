# pkdgrav3 pipeline scripts (pawna)

Guide: [docs/pkdgrav3_pipeline_pawna.md](../../docs/pkdgrav3_pipeline_pawna.md) (quick start in section 0).

| File | Role |
|---|---|
| `setup_pawna.sh <CODES> [PKD_IO_ROOT]` | one-time build of Rockstar (PKC reader), consistent-trees, Python venv, C helpers |
| `example_test128.conf`, `example_512.conf` | run configs (bash `KEY=VALUE`); copy one per run/realisation |
| `submit_pipeline.sh <conf> [--stages ...] [--after ID] [--hold] [--dry-run]` | submits the chained PBS jobs |
| `pipeline_env.sh` | sourced by every job: reads the config, sets paths and the per-job log |
| `make_pkdgrav3_par.py` | writes the pkdgrav3 parameter file from the config |
| `job_sim.pbs` | pkdgrav3 → `.pkc` snapshots, `snapnames.txt`, `pkc_check` |
| `job_rockstar.pbs` | Rockstar on `.pkc` (server + clients in one job) |
| `job_ctrees.pbs` | consistent-trees + `clean_trees.pl` → `out_<snap>.trees` |
| `job_vahc.pbs` | `scripts/post-process/run_addvalue.py` → `.vahc`, `.tens` |
| `job_fits.pbs` | `compress_halo_outputs.py all` + `check_compressed_halos.py` |
| `production/setup_production.py` | production runs: Sahyadri directory layout, `run.conf`, `p.par`, Rockstar config, `submit.sh` (guide section 11) |
| `production/cosmologies/*.cosmo` | cosmologies (`default`: Sahyadri default with the production CLASS P(k) at z = 49) |
| `production/sahyadri_outputs_a.txt` | the 101 Sahyadri output scale factors |
