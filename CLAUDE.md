# sahyadri-sandbox

Sahyadri simulation suite: pkdgrav3 / Gadget-4 runs, Rockstar + consistent-trees halo pipeline (pawna, pegasus),
post-processing and reports.

- Pipeline guide: `docs/pkdgrav3_pipeline_pawna.md`.
- Halo potential tools (xcorr, ximult, vprof), data paths on pawna, login, rules: `docs/handover_xcorr_vprof.md`.
- Run manager: `tools/sahyadri_manager/README.md` (operated by a dedicated session; do not change running jobs).

Rules: work on a branch (`run-manager` is current), never push/merge to `master` without asking; no simulation
larger than 512^3 without approval; jobs only through PBS; other users' files on pawna are read-only; ask before
building third-party source; state how every measurement was made; no data deletion without approval.
