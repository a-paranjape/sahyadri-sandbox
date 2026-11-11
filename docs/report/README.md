# pkdgrav3 pipeline report

`pkdgrav3_pipeline_report.pdf` covers the pipeline, the bugs found, the input-route tests of the halo
catalogs (with and without compression) and the measured wall time and memory. To reproduce it:

1. On pawna, reduce the catalogs and job logs to `data/`:
   `qsub -v RSVAL=/storage/shadab/pkc_rockstar/rsval2,RSVAL_LOG=/storage/shadab/pkc_rockstar/logs/rsval.2536496.log extract_job.pbs`
   (runs `extract_data.py`; `data/vahc_bug_128.npz` comes from the 128^3 buggy/fixed `.vahc` comparison).
2. Make every figure and table from `data/` only: `jupyter nbconvert --to notebook --execute --inplace make_figures.ipynb`
   (`build_notebook.py` writes the notebook from its source).
3. `latexmk -pdf pkdgrav3_pipeline_report.tex`
