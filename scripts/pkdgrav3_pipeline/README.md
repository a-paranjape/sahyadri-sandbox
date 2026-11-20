# pkdgrav3 pipeline scripts (pawna)

Guide: [docs/pkdgrav3_pipeline_pawna.md](../../docs/pkdgrav3_pipeline_pawna.md) (quick start in section 0;
production defaults for 1024^3 and above, per site, in section 15).

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
| `job_vahc.pbs` | `scripts/post-process/run_addvalue.py` → `.vahc`, `.tens`, `.vpot`, `.pk` (binary with `PP_BINARY=1`: `halo_table_io.py`) |
| `job_fits.pbs` | `compress_halo_outputs.py all` (merges `CT_BOX_DIVISIONS` tree files) + `check_compressed_halos.py --nproc` |
| `job_parquet.pbs` | `fits_to_parquet.py` → `parquet/` (threaded per file) |
| `job_verify.pbs` | `verify_run.py` (+ `archive_run.py --source-sums` → `KEPT_FILES.json` on swarm/pegasus) |
| `job_archive.pbs` | `archive_run.py`: kept files → `ARCHIVE`, read back and compared |
| `production/setup_production.py` | production runs: Sahyadri directory layout, `run.conf`, `p.par`, Rockstar config, `submit.sh` (guide section 11) |
| `production/cosmologies/*.cosmo` | cosmologies with their CLASS P(k) at z = 49: `default` (Sahyadri default, as production) and the variations `var_{As,h,ns,Om,w,Ok,Ob}_{m,p}` (guide section 13.4) |
| `production/sahyadri_outputs_a.txt` | the 101 Sahyadri output scale factors |
