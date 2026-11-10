# compress_halo_outputs.py

Rockstar / consistent-trees ASCII outputs of a run -> `.fits.gz`, with metadata, lossless, in parallel.
Replaces `ConvertToFITS.convert_fits` (halo catalogs, via `readers.py`/`compres_halos.py`) and
`consistent_trees_compress.py` (trees). Same file names and HDU 1 layout, so `readers.py` (`prep_halos_fits`)
reads the new files unchanged.

```
python compress_halo_outputs.py all --indir <run>/r1 --nproc 16 --tmpdir /var/tmp/me \
    --rockstar-cfg <OUTBASE>/rockstar.cfg --ctrees-cfg <OUTBASE>/outputs/merger_tree.cfg \
    --pkc <snapshots>/snap.02000 --meta SIMNAME=default2048
python check_compressed_halos.py <run>/r1 --trees      # independent check against the ASCII
```

Outputs: `compressed/out_<snap>_{basic,extended,vahc,tens}.fits.gz`, `compressed/trees/tree_X_Y_Z.fits.gz`,
`locations.fits.gz`, `forests.fits.gz`. Each file: HDU 0/1 headers with cosmology (Omega_M, Omega_L, h0, w0, wa,
box, particle mass), scale factor / redshift / snapshot, Rockstar settings (`RS_*`), consistent-trees settings
(`CT_*`), simulation header from a `.pkc` snapshot (`SIM_*`), converter provenance; HDU 2 `PROVENANCE` with the
full texts (ASCII header, rockstar.cfg, merger_tree.cfg, .pkc info JSON). Already existing outputs are skipped
(`--overwrite` to redo); `--delete-ascii` removes inputs only after a verified write.

Differences from the old converters (all fix data loss):
- Integer columns (IDs, counts, flags) are int64. The old halo files stored everything except id/pid/upid/haloID
  as float32, so `desc_id`, `Orig_halo_ID`, `Depth_first_ID`, `Tree_root_ID`, ... above 2^24 were rounded.
- Positions and scale factors are float64 (exact as printed); other columns float32 (Rockstar/consistent-trees
  compute them in single precision). Files are 8-12% larger than before.
- Duplicate hlist names: the old converter kept only the last of `b_to_a`, `c_to_a`, `A[x..z]` (the 500c values,
  under the plain name). Now the first keeps the plain name and the 500c ones are `b_to_a_500c`, `A[x]_500c`, ...
  (as `readers.py` names them for ASCII).
- `locations.dat`: all six columns (old: FileID stored as filename; Filename, Num_Halos, Num_Bytes lost), plus
  `FitsRow`/`FitsRows` = rows of each tree in its `tree_X_Y_Z.fits.gz`.
- `forests.list`: (TreeRootID, ForestID) pairs as in the file (old: misread as root + members).
- Trees are parsed in chunks (the old script held each multi-GB file as one string).

Tests (2026-11-11, Mac): `check_compressed_halos.py` on consistent-trees output of a 128^3 run: every column of
3 hlists, 1 tree file, locations and forests equal to the ASCII; readers.py access works. Speed on a 0.9 GB
catalog (1.18M rows), one process: parse 8 s, total 21 s at gzip level 6 (15 s at level 1, 106 s at level 9 for
no gain); the old ConvertToFITS took 14 s for the same file. pigz (if installed) or python-isal speed up gzip.
