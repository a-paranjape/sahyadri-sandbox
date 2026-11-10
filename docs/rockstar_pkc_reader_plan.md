# Plan: Rockstar reads compressed pkdgrav3 snapshots natively (`FILE_FORMAT = "PKC"`)

Status: plan reviewed in parts; **step 1 of section 8 in progress** (2026-11-10, branch `rockstar-pkc-reader`). Based on pkdgrav3 main fcd37310 (format version 3 only). Sections 3c (consistent-trees cosmology) and 3d (FITS step) added after reading the pegasus code.

Goal: Rockstar reads `<name>.<step>.s<pct>.<writer>.pkc` snapshots directly, with no conversion step
and no extra disk, and produces the same halo catalogs as the existing `pkc_to_gadget` route.

## 0. What this plan is based on

Read for this plan (no code built or run, nothing measured in this session):

| Source | What was taken from it |
|---|---|
| `code/Rockstar/gfcstanford-rockstar-36ce9eea36ee/` (this repo) | the patched files we deploy over upstream: `io/meta_io.c`, `io/io_gadget.c`, `io/io_tipsy.c`, `io/io_arepo.c`, `hubble.c` (w0/wa, curvature), `Makefile` |
| upstream Rockstar 36ce9eea36ee (unpacked copy in the pkdgrav3 checkout, `tmp.nosync/rockstar_src/`) | `main.c`, `client.c`, `server.c`, `config.c`, `config.template.h`, `particle.h`, `universal_constants.h` |
| pkdgrav3 `main` (local checkout) | `docs/snapcomp_format.md` (versions 1 to 3), `io/snapcomp/snapcomp.{h,cxx}` (reference decoder), `io/snapcomp/tools/pkc_to_gadget.cxx`, `docs/running_on_pawna.md`, `docs/compressed_output_guide.md` |

Facts from upstream Rockstar that drive the design:

- `read_particles(char *filename)` in `io/meta_io.c` dispatches on `FILE_FORMAT` and appends to the
  global `p`/`num_p`. It only receives a **file name**, not the block number.
- With `PARALLEL_IO = 1` the server hands blocks `0 .. NUM_BLOCKS-1` to `NUM_READERS` reader processes
  (`server.c:read_blocks`, contiguous ranges via `particle_range`); each reader calls
  `get_input_filename(snap, block)` then `read_particles()` once per block (`client.c`, command `rdbk`).
  `NUM_READERS` defaults to `NUM_BLOCKS` and must be `<= NUM_BLOCKS` (`config.c`).
- Without `PARALLEL_IO`, `config.c` forces `PERIODIC = 0` and `TEMPORAL_HALO_FINDING = 0`, so production
  must use `PARALLEL_IO = 1`.
- After reading, each reader sends `PARTICLE_MASS, AVG_PARTICLE_SPACING, SCALE_NOW, BOX_SIZE, Ol, Om, h0`
  to the server (`client.c:send_config`). So **the reader** must set these globals from the file header,
  exactly as `gadget2_extract_header_info()` does.
- `struct particle { int64_t id; float pos[6]; }` = 32 B; `pos[3..5]` are peculiar velocities in km/s.
- `W0`, `WA` are Rockstar config parameters (upstream `config.template.h`), used by our `hubble.c`; no
  input format sets them, and the .pkc header does not carry them (see Risks).

## 1. Where the code goes, and which decoder

### Decision: (a) a small C99 decoder written from the spec, using the zstd C API

New files in our Rockstar tree:

| File | Content | Size (est.) |
|---|---|---|
| `io/io_pkc.h` | `void load_particles_pkc(char *filename, struct particle **p, int64_t *num_p);` | 10 lines |
| `io/io_pkc.c` | header/trailer/index parsing, file discovery, block assignment, block decoder, Rockstar globals | ~500 lines |

Why (a) and not (b) (`extern "C"` wrapper around `io/snapcomp/` linked with g++):

- Rockstar is C99 built with one `$(CC)` line; (b) brings in a C++17 toolchain, `libstdc++`, exceptions
  crossing a C boundary, and a second build system (CMake) whose FetchContent may download zstd.
- (a) makes `snapcomp_format.md` the single source of truth: a second, independent implementation
  written from the spec is a test of the spec. Every place where the spec and `snapcomp.cxx` disagree
  becomes a spec fix (one already found, see 7.3).
- The decoder is small: parse 16-byte block header and 12-byte stream table, `ZSTD_decompressDCtx` each
  stream, undo the byte-plane shuffle, zigzag, prefix-sum the cell keys, expand cells, three ID coders.
  `snapcomp.cxx:decode_block` is ~80 lines.
- (b) remains the **test oracle**: `pkc_to_gadget` (built on snapcomp) is the comparison route in
  section 6, and a tiny standalone C test (`util/pkc_dump.c`, not part of Rockstar) can print decoded
  particles for diffing against `pkc_check`/the Python reader.

Decoder rules (from spec + reference decoder):

- Accept **version 3 only** (decision 2026-11-10: all snapshots this reader is for are written by
  current pkdgrav3). Versions 1/2 -> clear error ("format version N: rewrite with current pkdgrav3 or
  use pkc_to_gadget"). When `pot_step > 0` blocks have one extra stream (kind 8, potential): skip it
  (Rockstar does not use the potential), but check its presence matches `pot_step > 0`.
- Check `n_streams == 8 + (id_coder == 2 ? 3 : 1) + has_potential`, each stream's `count`
  (`n_cells` for streams 0-1, `n` otherwise), the stream `kind` bytes, decompressed size
  `== count*width`, offset `< sub`, cell counts summing to `n`, and block `n` == index entry `n`.
  zstd frames carry content checksums, so `ZSTD_decompressDCtx` detects corruption.
- Lagrangian IDs (coder 2): `l = (X*N)/Q` in integer arithmetic (the reference uses `__int128`; in C99 use
  `uint64_t` after asserting `Q*N < 2^63`, true for any realistic run: Q ~ 1e5, N = 2048), then
  `q = floor_mod(unzigzag(r) + l, N)`, `id = (qz*N + qy)*N + qx`.
- All integers little-endian; read with `memcpy` into fixed-width types (no struct overlay). Add a
  compile-time check that the host is little-endian (pawna and macOS are).

### Hook-in: minimal edits

1. `io/meta_io.c` (already a patched file here): 3 additions mirroring the AREPO pattern:
   ```c
   #ifdef ENABLE_PKC
   #include "io_pkc.h"
   #endif
   ...
   else if (!strncasecmp(FILE_FORMAT, "PKC", 3)) {
   #ifdef ENABLE_PKC
     load_particles_pkc(filename, &p, &num_p);
   #else
     fprintf(stderr, "[Error] PKC needs zstd.  Recompile Rockstar using \"make with_pkc\".\n"); exit(1);
   #endif
   }
   ```
2. `config.template.h`: copy the upstream file into our tree (it becomes a patched file, `.bak` kept as
   per the existing convention) and add two lines:
   ```c
   string(PKC_SUBSAMPLES, "");   // e.g. "1,3"; empty = all subsamples
   ```
   `config_vars.{h,c}` and the config parser are generated from this X-macro file, so nothing else changes.
3. `Makefile`: `io/io_pkc.c` added to `CFILES` (the file compiles to nothing without `ENABLE_PKC`, like
   `io_arepo.c`), plus
   ```make
   ZSTD_DIR ?= /usr
   PKC_FLAGS = -DENABLE_PKC -I$(ZSTD_DIR)/include -L$(ZSTD_DIR)/lib -lzstd
   with_pkc:           @make reg EXTRA_FLAGS="$(OFLAGS) $(PKC_FLAGS)"
   with_hdf5_pkc:      @make reg EXTRA_FLAGS="$(OFLAGS) $(HDF5_FLAGS) $(PKC_FLAGS)"
   ```
4. `scripts/rockstar/readme_shadab.txt` (or an addition to the existing readme) listing the changes,
   matching how Aseem's and Saee's patches are documented.

5. `client.c` (becomes a patched file, `.bak` kept): add `W0` and `WA` to `send_config()` and
   `recv_config()` (2 lines each, both sides use the same `snd`/`rcv` macros, so the protocol stays
   symmetric). Needed because only readers parse the file; writers compute H(z) for virial
   overdensities (`groupies.c:57,64`), the unbinding Hubble term (`potential.c:174`) and the time table
   (`init_time_table()` after `cnfg`), and today they would keep the config's `W0`/`WA`. Om, Ol, h0
   already travel this way. Side effect: all formats now propagate W0/WA from reader to writers, which
   changes nothing for formats that take them from the config (same value on both sides).

No changes to `server.c`, `main.c`, `get_input_filename()`.

zstd on pawna: needs checking (`ls /usr/include/zstd.h`, `module avail zstd`, or the copy pkdgrav3's
CMake build fetched). If none is installed system-wide, building zstd from source is third-party code
and needs your approval first.

## 2. Mapping Rockstar blocks onto .pkc files

### Passing the block number

`read_particles()` only sees a file name. Use the existing `<block>` token:

```
FILE_FORMAT    = "PKC"
FILENAME       = "snap.<snap>.<block>"     # -> INBASE/snap.00100.7
SNAPSHOT_NAMES = "snapnames.txt"           # lines "00100", "00200", ... (pkdgrav3 step numbers)
NUM_BLOCKS     = 64
```

`load_particles_pkc` strips a trailing `.<digits>` as the block index `k` and treats the rest as the
snapshot prefix. If there is no numeric suffix and `NUM_BLOCKS == 1`, `k = 0` (so
`FILENAME = "snap.<snap>"` works for a single block); otherwise it is an error. `BLOCK_NAMES` must be
empty (checked). `SNAPSHOT_NAMES` is the existing mechanism for non-contiguous snapshot numbers, and is
needed anyway because Rockstar's `<snap>` is the 0-based snapshot index, not the pkdgrav3 step.

### Which particles block k reads

The .pkc files are very unequal (1% to 25% of particles each), so files cannot map to blocks. Instead,
every reader builds the same global list of `(file, pkc-block)` pairs and takes a share of it.

1. Find files with `glob("<prefix>.s*.pkc")`; open each, read header (4096 B), trailer (32 B) and index
   (32 B per block). Cost: at 2048^3 about 131k pkc blocks in total -> 4 MB of index, read by every reader.
2. Keep files of the selected subsamples. Order canonically by `(subsample, writer)` from the **header**
   (not the file name), blocks in index order. Compute prefix sums `S_i` of the particle counts; `T` = total.
3. Rockstar block `k` reads pair `i` iff `floor(S_i * NUM_BLOCKS / T) == k` (the pair's first particle
   falls in the k-th equal share).

Properties:

- Every pair is read exactly once, for any `NUM_BLOCKS >= 1`; the particle count per Rockstar block is
  `T/NUM_BLOCKS` within one pkc block (<= 65536 particles), so readers are balanced.
- Contiguous ranges keep each reader's particles spatially compact (pkdgrav3 threads write blocks of
  spatially coherent particles), so each reader's particle bounding box is small. A `k mod NUM_BLOCKS`
  round-robin, as first suggested, balances equally well but scatters every reader over the whole box;
  Rockstar reports each reader's bounds to the server (`cnf?`), and with compact readers each one
  sends its particles to fewer writers. Both rules are correct; contiguous is chosen for locality and
  for the exact-match test below.
- Exact match with the Gadget route: add `--blocks K` to `pkc_to_gadget` (pkdgrav3 repo, small change)
  using the **same** rule, so Gadget file k holds exactly the particles of PKC block k, in the same
  order. Then (ii) and (iii) in section 6 read identical particle arrays in every reader.
- Reader count: `NUM_READERS = NUM_BLOCKS` (default) or fewer; any value works since blocks are equal.

## 3. Header -> Rockstar globals

Set in `load_particles_pkc` on every call (all calls on one snapshot give identical values):

| Global | Value | Note |
|---|---|---|
| `SCALE_NOW` | `1.0/(1.0 + redshift)` | same expression as `pkc_to_gadget` (Gadget `time`) |
| `BOX_SIZE` | `box_size` | Mpc/h |
| `Om`, `Ol`, `h0`, `W0`, `WA` | from the version-3 binary header, see 3a | **not** the binary `omega_l`, which is 0 for w0-wa runs |
| `TOTAL_PARTICLES` | `T` = particles in the **selected** subsamples | `n_total` when all are selected |
| `PARTICLE_MASS` | `27.7536627 * Om * L*L*L / T * 1e10` Msun/h | same constant and operation order as `pkc_to_gadget` (`mpart*1e10` via `GADGET_MASS_CONVERSION`), so masses are bit-identical. If `RESCALE_PARTICLE_MASS`, use Rockstar's `CRITICAL_DENSITY` instead, as `io_gadget.c` does |
| `AVG_PARTICLE_SPACING` | `cbrt(PARTICLE_MASS / (Om*CRITICAL_DENSITY))` | as `io_gadget.c` |

Particles:

- Position: `pos[d] = (float)(((double)X + 0.5) / Q * L)`, `Q = ncell*sub`; in `[0, L)`. This is the
  pkdgrav3 box `[-0.5, 0.5)` shifted by `+L/2`, the same convention as `pkc_to_gadget`, the Python
  reader and Rockstar's TIPSY reader (`(pos+0.5)*L`).
- Velocity: peculiar km/s, **no** `sqrt(a)` factor (the header's `vel_step` already includes
  `vel_to_kms`). To be bit-identical with the Gadget route, compute it with the same float operations:
  `vf = (float)V * (float)(vel_step/sqrt(a)); pos[3+d] = (float)((double)vf * sqrt(SCALE_NOW));`.
  The extra rounding is ~1e-7 relative, far below the quantization step (2.44 km/s).
- ID: `id = (int64_t)uint64_id`; lattice IDs are `< N^3` (8.6e9 at 2048^3), far below 2^63. IDs are
  persistent, so `TEMPORAL_HALO_FINDING` and consistent-trees work; keep `IGNORE_PARTICLE_IDS = 0`.
- `LIGHTCONE`: not supported for PKC (error out); `read_particles()` applies the `sqrt(a)` only for
  Gadget anyway.

Validation of the file set (on every call; it only reads headers/trailers):

- Each file: magic, version 3, `header_size == 4096`, trailer magic present, `index_offset +
  32*n_blocks + 32 == file size`, trailer and header agree on `n_blocks`/`n_particles`, sum of index `n`
  equals `n_particles`, every index entry inside `[4096, index_offset)`. **No trailer -> error**
  (no block-scanning recovery inside Rockstar).
- Across files: identical `n_total, lattice_n, ncell, sub, vel_step, box_size, redshift, step,
  omega_m, omega_l, h, n_subsamples, percents, subsample_seed, n_writers`; unique `(subsample, writer)`.
- Completeness: for each selected subsample, all `n_writers` files present. If all subsamples are
  selected: `n_subsamples * n_writers` files and `sum n_particles == n_total`, else error.
  If only some subsamples are present on disk (e.g. only the 1% copied), selecting just those works.
- `<prefix>.done` marker: warn if missing (the trailer check is the real test).
- Per block: decoded `n` == index `n`; total decoded for block k == expected share (else error).

## 3a. Cosmology: .pkc version-3 header -> Rockstar Om, Ol, W0, WA (curvature and w0-wa patches)

State of pkdgrav3 `main` (fcd37310, read 2026-11-10; change in 5697d3c3): pkdgrav3 now writes **format
version 3**: 10 doubles at offset 4000 (`omega_b, omega_rad, omega_de, w0, wa, sigma8, n_s, running,
normalization, pivot`); the info text ends before 4000; `pot_step`/`pot_to_kms2` stay at 4080 and are
optional in version 3 (`pot_step = 0` -> no potential stream). `omega_l` (offset 104) is still pkdgrav3
`dLambda` (0 for w0-wa runs); the dynamical component is `omega_de`. The spec now states the background:
E(a)^2 = omega_m a^-3 + omega_rad a^-4 + omega_l + omega_de a^(-3(1+w0+wa)) exp(-3 wa (1-a)) + omega_k a^-2,
omega_k = 1 - omega_m - omega_l - omega_de - omega_rad (matches `cosmo.h:210-215`).

Checked in this session (how): read the diff of 5697d3c3 (`snapcomp.{h,cxx}`, `writecompressed.{h,cxx}`,
`msr_writecompressed.cxx`, `snapcomp_format.md`); offsets do not overlap (info 288-3999, cosmology
4000-4079, potential 4080-4095; `achInfo[3712]` = 3711 characters + NUL); the decoder's stream count
uses `pot_step > 0` in every version. Built `io/snapcomp` with CMake on the Mac (zstd from the local
anaconda install, nothing downloaded) and ran `snapcomp_test`: all tests pass, including
"cosmology header (version 3): ok"; `pkc_check` on the old version-1 set `tmp.nosync/p4mac2/snap.00012`
(64^3): CHECK OK, so old files still read. No version-3 snapshot is on this machine, so a real
version-3 file was not read here (the commit reports a 64^3 w0-wa Mac run checked with `pkc_check`).

For Rockstar this means **no JSON parsing**: everything comes from fixed binary fields. The info JSON
(`cosmology.E_z`) is kept only as an optional cross-check.

### 3a.1 The two models must match

| | pkdgrav3 (`cosmo.h`) | our Rockstar (`hubble.c`, Aseem + Saee patches) |
|---|---|---|
| E^2(a) | Om a^-3 + Orad a^-4 + OK a^-2 + OLambda + ODE f(a) | Om a^-3 + (1-Om-Ol) a^-2 + Ol a^(-3(1+w_eff(a))) |
| DE density f(a) | a^(-3(1+w0+wa)) exp(-3 wa (1-a)), i.e. w(a) = w0 + wa(1-a) (CPL) | w_eff = W0 + WA - WA (a-1)/ln a |
| curvature | OK = 1 - Om - OLambda - ODE - Orad | 1 - Om - Ol (Aseem's term) |

The DE parts are the same function: ln f = -3 ln a [1 + w0 + wa - wa (a-1)/ln a] = -3(1+w_eff) ln a.
So with **Ol = OLambda + ODE** the two agree exactly when one DE component is present and Orad = 0.
Rockstar has only one dark-energy component, so:

| pkdgrav3 run | Rockstar gets | exact? |
|---|---|---|
| LCDM: ODE = 0 | Om, Ol = Omega_lambda, W0 = -1, WA = 0 | yes |
| w0-wa: OLambda = 0, ODE > 0 | Om, Ol = Omega_DE, W0 = w0, WA = wa | yes |
| both OLambda > 0 and ODE > 0 with (w0, wa) != (-1, 0) | cannot be represented | error |
| curved (OK != 0) | 1-Om-Ol = OK (+ Orad, below) | yes (Aseem's term) |
| Orad > 0 | Rockstar treats Orad as curvature (a^-2 instead of a^-4) | no: relative error in E^2 ~ Orad (1+z)^2 / (Om (1+z)) ~ 1e-3 at z = 3 for Orad = 8.5e-5; warn |
| `class = true` | background from a CLASS table; w0/wa may only approximate it | check E_z (3a.2) |

The upstream Rockstar error on `Om + Ol != 1` is already commented out in our `io_gadget.c`; the PKC
reader never had it.

### 3a.2 Rules in `load_particles_pkc`

1. Version 3 (every pkdgrav3 snapshot from 2026-11-11 on):
   `Om = omega_m`, `h0 = h`, `Ol = omega_l + omega_de`, `W0 = w0`, `WA = wa` (or -1, 0 if
   `omega_de == 0`). Error if `omega_l > 0` and `omega_de > 0` and (w0, wa) != (-1, 0) (two dark-energy
   components; Rockstar has one). Warn if `omega_rad > 0` with the size of the error at this z (3a.1).
   These override the config, as the Gadget reader overrides Om/Ol/h0. Rockstar cannot tell an explicit
   config value from its default, so log `config W0=.. WA=.. Om=.. Ol=.. h0=.. -> header ...` whenever
   they differ.
2. Cross-check (optional, cheap): if the info JSON has `"E_z":`, compare it with Rockstar's
   `hubble_scaling(z)` after setting the globals: > 1e-4 relative -> warning (radiation, CLASS
   backgrounds), > 1e-2 -> error. A plain `strstr` + `strtod`; the key occurs once in the JSON.
3. Versions 1 and 2: rejected (see section 1); no cosmology fallback code.

### 3a.3 Propagation inside Rockstar

- Reader -> server -> writers: Om, Ol, h0 already go through `send_config`/`recv_config`; W0/WA need
  the `client.c` edit in section 1 (item 5).
- `out_<snap>.list` header line `#Om = ..; Ol = ..; h = ..` is printed from the writers' globals, so it
  carries the header values.
- `OUTBASE/rockstar.cfg` is written by the reader of block 0 after its read (`client.c`,
  `output_config`), so it contains the header's Om, Ol, h0, W0, WA.

## 3b. Rockstar + consistent-trees as one pipeline, cosmology carried through

### What is read how often (checked in the code)

- Particles are read **once per snapshot**, by Rockstar's readers. Nothing else touches the snapshot.
- Rockstar's temporal halo finding and descendant links use Rockstar's own `halos_<snap>.*.bin` of the
  previous snapshot (`client.c:908`, `load_previous_halos`): halo member particles only, not the snapshot.
- consistent-trees (`do_merger_tree.pl`, `halo_trees_to_catalog.pl`) reads only the `out_<snap>.list`
  halo catalogs (text, MB to GB), never particles. It cannot run inside Rockstar: it needs the
  catalogs of **all** snapshots (it walks from the last snapshot back), so it always runs after the
  last Rockstar snapshot. Running them "jointly" therefore means one job chain, not one process; the
  extra I/O of running them separately is only the halo catalogs.
- I/O that can be cut (optional): `OUTPUT_FORMAT = "BINARY"` drops the `halos_*.ascii` copies if no one
  reads them (check `scripts/post-process/` first; it uses the trees/hlists); `out_*.list` is always
  written and is what consistent-trees needs. `DELETE_BINARY_OUTPUT_AFTER_FINISHED = 1` removes the
  `.bin` files at the end, if halo particle lists are not wanted.

Recommended run: **one Rockstar invocation for all snapshots** (`NUM_SNAPS`, `SNAPSHOT_NAMES`), as
`scripts/submit_CiPod.sh` does (server job + worker job from `auto-rockstar.cfg`), then the
consistent-trees jobs with `afterok` dependencies. One Rockstar call per snapshot (today's pawna batch
script) also works with temporal halo finding but starts and stops the server per snapshot; no
particle I/O is saved or lost either way. For pawna this becomes one PBS script that runs: Rockstar
server + workers -> `gen_merger_cfg.pl` -> `do_merger_tree.pl` -> `halo_trees_to_catalog.pl`.

### Where consistent-trees gets its cosmology today (the gap)

`gen_merger_cfg.pl` writes `outputs/merger_tree.cfg` with:
- `Om`, `Ol`, `h0` parsed from the `#Om = ..; Ol = ..; h = ..` line of `out_<STARTING_SNAP>.list`.
  These are Rockstar's globals, so with the PKC reader they are the header values (Ol = omega_l +
  omega_de). Correct.
- `w0`, `wa` from `$config->{W0}`, `$config->{WA}` of the config file given on the command line. In
  `submit_CiPod.sh` that is `$ROCKSTAR_CONFIG_FILE`, the **hand-written input config**. With the PKC
  reader, W0/WA come from the snapshot header, so consistent-trees would get the input config's values
  (default -1, 0) and silently disagree with the halo catalogs.

### Fix (two small edits, header values reach consistent-trees for every format)

1. Rockstar `meta_io.c`, `print_ascii_header_info()`: after the `#Om = ..` line print
   `#w0 = %.8g; wa = %.8g` **only if (W0, WA) != (-1, 0)**. LCDM outputs stay byte-identical (so Gadget
   and TIPSY LCDM runs are unchanged); w0-wa runs get the line in every `out_*.list`, `halos_*.ascii`
   and particle file header. Writers have the header values thanks to the `client.c` edit (section 1).
2. `gen_merger_cfg.pl`: read `w0`, `wa` from that line of `out_<STARTING_SNAP>.list`, like Om/Ol/h0;
   if the line is absent, use -1 and 0 (LCDM, or older catalogs) - not `$config->{W0}`. Then it no
   longer matters which config file is passed. Also write the lines with `#` comments
   (`w0 = -0.9 #added by Saee`): the Rockstar-style parser strips after `#` (`read_config.c:151`);
   the present `/* added by Saee/*` only works because `strtod` stops at the space.

Alternative without edit 1: pass `OUTBASE/rockstar.cfg` (written by the reader of block 0 after reading,
`client.c:1308`, so it holds the header W0/WA) to `gen_merger_cfg.pl` instead of the input config.
Simpler, but depends on the job script using the right file; edit 1+2 makes the catalogs themselves
carry the cosmology. Recommended: 1+2.

## 3c. consistent-trees cosmology patch (checked on pegasus 2026-11-10)

Read-only check of the copy the pipeline uses, `/mnt/home/faculty/caseem/code/ConsistentTrees/consistent-trees`
(from the upstream `ctrees-1.0` tarball; `src/` dated 2025-12-21), with `grep`/`cat` over ssh:

- `grep -rn "w0|wa|weff"` over `src/*.c *.h *.pl`: no match. `w0`/`wa` in `merger_tree.cfg` are ignored.
- `src/universe_time.c`: `init_time_table(Om, h0)` takes only Om; scale -> time uses the flat-LCDM
  closed form `a(t) ∝ sinh(1.5 t sqrt(1-Om))^(2/3)`. Ol, curvature and w0/wa are not used.
- `src/masses.c:delta_vir(a)`: Bryan & Norman with `x = 1/(1 + a^3 Ol/Om) - 1`: uses Ol, assumes a
  cosmological constant, no curvature.
- `scale_to_years()` is used by `find_parents_and_cleanup`, `gravitational_consistency` (time between
  snapshots in the consistency test), `halo_evolve` (phantom halos), `recalc_tidal_forces`/`tidal_lib`
  (tidal time), `gen_sussing_forests`; the time-based accretion rates in the hlists (`accrate_100Myr`,
  `_1tdyn`, ...) follow from the same table.

Consequence: every consistent-trees product so far (including the `var_w_m2048`, `var_w_p2048` runs on
pegasus, and any curved run) used flat-LCDM times. Size of the effect: not estimated yet; it grows
with |w0 + 1|, |wa| and |Omega_k|.

Patch (small, kept in this repo like the Rockstar files: `code/ConsistentTrees/` with only the changed
files and `.bak` originals, same convention as `readme_aseem.txt`):

1. `config.template.h`: add `real(W0, -1); real(WA, 0);` (the keys `gen_merger_cfg.pl` already writes).
2. `universe_time.c`: build the time table by integrating dt = da / (a H(a)) with the same E(a) as our
   Rockstar `hubble.c` (Om a^-3 + (1-Om-Ol) a^-2 + Ol a^(-3(1+w_eff))), instead of the closed form;
   keep the table interface (`init_time_table`, `scale_to_time`, `scale_to_years`). For flat LCDM the
   new table must reproduce the old one to < 1e-6 (test).
3. `masses.c:delta_vir`: x = Omega_m(a) - 1 with Omega_m(a) = Om a^-3 / E(a)^2 (Bryan & Norman's x),
   which reduces to the present formula for flat LCDM.
4. Test: for flat LCDM, trees and hlists identical (byte for byte) before and after on a small run;
   for a w0-wa 128^3 run, report the change in accretion rates and in the fraction of halos the
   consistency step modifies.

## 3d. Halo outputs to FITS: last step of the same job chain

What exists (read on pegasus, `scripts/post-process/` in `/mnt/home/project/calam.shadab/sahyadri_sandbox`):

- `compres_halos.py <sim> <snap|-1>` -> `HaloReader.convert_halos_fits` (`readers.py`) ->
  `ConvertToFITS.convert_fits`: `out_<snap>.trees` (the consistent-trees `hlist_<a>.list`, copied by
  `scripts/assist/clean_trees.pl`) -> `out_<snap>_basic.fits.gz` + `_extended.fits.gz`;
  `out_<snap>.vahc` (from our post-processing) -> `_vahc.fits.gz`. Skips a snapshot if the basic file
  exists.
- `consistent_trees_compress.py <in_dir> <out_dir>`: `tree_X_Y_Z.dat`, `locations.dat`,
  `forests.list` -> `.fits.gz` (dtype policy int64/float64/float32 per column, FITS-safe names; writes
  a plain FITS file, then gzip level 6). Job `job_compress_trees.pbs` asks for 32 cores, 95 h.
- Sizes (`ls -l`, `default2048/r1`): `out_99.trees` 7.94 GB -> 0.62 GB basic + 1.57 GB extended;
  `out_99.vahc` 1.99 GB -> 0.38 GB; `trees/tree_0_0_0.dat` 6.14 GB -> 2.14 GB; 125 tree files
  (if all similar, ~750 GB ASCII per run; extrapolated from one file).

Decision: do **not** write FITS from inside consistent-trees. Its stages pass ASCII to each other
(`find_parents_and_cleanup` -> `resort_outputs` -> `assemble_halo_trees` writes `tree_*.dat` ->
`halo_trees_to_catalog` reads them to make the hlists), so native FITS means changing two writers and
one reader in C, adding cfitsio, and duplicating the Python column/split logic in C. Instead:

1. New job after `halo_trees_to_catalog` (`afterok`), before `clean_trees.pl`: converts hlists and tree
   files with the existing Python code, **in parallel, one process per file** (the 32 cores of the
   trees job are otherwise idle; `consistent_trees_compress.py` converts files one after another).
2. Each output is read back (row count, ID column equal to the ASCII, a few columns compared) before
   the ASCII is deleted; a failed check keeps the ASCII and fails the job.
3. FITS headers get the cosmology and run constants: Om, Ol, h0, w0, wa (from `outputs/merger_tree.cfg`,
   which `clean_trees.pl` deletes, so the conversion runs first or `clean_trees.pl` keeps a copy in
   `logs/`), box size, particle mass, scale factor.
4. `compres_halos.py` and `consistent_trees_compress.py` move into this repo
   (`scripts/post-process/`) so the pipeline and the plan refer to versioned code. They are on
   pegasus only today.
5. Optional later: a seekable format (per-column or tiled compression) instead of whole-file gzip,
   which forces reading a whole `.fits.gz` to get one column. Not part of this work.

## 4. Options

| Option | Default | Meaning |
|---|---|---|
| `FILE_FORMAT = "PKC"` | | this reader |
| `FILENAME = "snap.<snap>.<block>"` | | prefix of the .pkc set + block index (section 2) |
| `PKC_SUBSAMPLES = "1,3"` | `""` = all | subsamples by percent, matched to `percents[subsample]` within 1e-9 (as `pkc_to_gadget --percents`) |

Subsamples for quick tests: with `PKC_SUBSAMPLES`, `PARTICLE_MASS` rises as `1/T` (mean density kept,
as `pkc_to_gadget` does) and `AVG_PARTICLE_SPACING` (hence the FOF linking length) rises as `T^(-1/3)`.
`FORCE_RES` and `MIN_HALO_PARTICLES` are not scaled; halos below ~`MIN_HALO_PARTICLES * PARTICLE_MASS`
disappear. Log a line stating the selected percents, `T`, and the resulting particle mass.

## 5. Memory and speed

Decoding: one pkc block at a time, straight into Rockstar's particle array.

- Before decoding, each call knows its particle count from the index, so it does **one** `realloc` of
  `*p` to `num_p + count` (no growth per block).
- Scratch per decoding thread: compressed block (<= ~0.6 MB at 65536 particles, 5-8 B/particle +
  stream table), byte-plane buffer (<= 8 B x 65536 = 0.5 MB), decoded streams (11 x 65536 x 8 B = 5.8 MB
  if kept as `uint64`; can decode in place into narrower arrays), one `ZSTD_DCtx` (~100-200 kB).
  Under 10 MB per thread; negligible next to the particles.

Threads: Rockstar's parallelism is already one process per reader. With `NUM_READERS = 64-96` on a
192-core node, readers already decode in parallel, so `PKC_THREADS = 1` is the default and the first
version is single-threaded (no pthreads in `io_pkc.c`). Add threads (one block per thread, pre-assigned
output offsets, so no locking) only if a measurement shows the reader stage dominates.

Estimates for 2048^3 on one pawna node (inputs are measurements in pkdgrav3 docs, not this session;
the decode rate is **not measured yet**):

| Quantity | Value | Source |
|---|---|---|
| Particles | 8.59e9 | 2048^3 |
| .pkc snapshot size | 45.6-54.1 GB (5.3-6.3 B/particle at z = 45/25/12) | `compressed_output_guide.md`, 1-node 2048^3 run |
| Rockstar particle array | 8.59e9 x 32 B = 275 GB | `sizeof(struct particle)`; same for every input format |
| Read from `/storage` NFS | ~50 GB / 25-30 MB/s = **28-35 min** | `running_on_pawna.md` NFS table |
| Read from node-local SSD | ~50 GB / 0.8-1.4 GB/s = **0.6-1 min** | same table, 4-16 streams |
| TIPSY today, from NFS | 8.59e9 x 36 B = 309 GB -> **~2.9-3.4 h** | TIPSY dark particle record (mass, pos, vel, eps, phi) |
| `pkc_to_gadget` route | +275 GB written to SSD and read back by Rockstar | `running_on_pawna.md` |
| Decode, 64 readers | 8.59e9 / (64 x 45e6/s) = **~3 s** CPU per reader at the Mac rate; allow ~10x for pawna cores, the C99 decoder and writing into Rockstar's array: well under a minute, small next to the read | `snapcomp_test` on the Mac (built 2026-11-10 in the session scratchpad, Release): "decode 45.1 M particles/s" for one thread, block 65536, zstd-3, synthetic lattice data. Not measured on pawna or on real snapshots |

So the native reader cuts the input stage from hours (TIPSY on NFS) to minutes, removes 275 GB of
temporary disk, and the dominant input cost becomes the NFS read; staging .pkc files on the SSD makes it
negligible.

Measurement to do first (before any 2048^3 run): decode throughput per core on the existing 512^3 .pkc
(`time` of a single-reader run with `NUM_BLOCKS = 1` vs. `NUM_BLOCKS = 64`, wall clock from Rockstar's
`timed_output` lines), and peak RSS (`/usr/bin/time -v`) of Rockstar at 512^3 and 1024^3 to extrapolate.

## 6. Validation

Implementation tests (laptop or pawna, small):

1. Decoder unit test: decode the 128^3 test snapshots (pkdgrav3 `tmp.nosync/p1/n128_*`, or regenerate)
   with a standalone C driver and compare every particle (X, V, id) against `pkc_check` / Python
   `pkcreader` output; all three ID coders and version-3 files with and without the potential must appear in the test set
   (versions 1/2 are only tested for the clean rejection).
2. Corruption tests: truncated file (no trailer), file missing from the set, flipped byte in a block
   (zstd checksum), mismatched headers -> clean errors, not silent wrong counts.
3. Block assignment: for NUM_BLOCKS in {1, 7, 64, 1000}, the union of decoded IDs is every ID once,
   counts per block differ by at most one pkc block.

Science validation on a 512^3 run with both outputs (`bCompAlsoStandard = True` with `bHDF5 = True`,
so the same run writes .pkc and pkdgrav3 HDF5 at the same steps; at most 512^3 per the rules; submitted
through PBS).

Rockstar cannot read the pkdgrav3 HDF5 directly: our AREPO reader (`io_arepo.c`) expects Gadget-4
layout and units (cosmology in the `Parameters` group as `Omega0/OmegaLambda/HubbleParam`, positions in
`[0, L)` times `AREPO_LENGTH_CONVERSION`, Gadget velocities), while pkdgrav3 HDF5 holds code units
(positions in box units `[-0.5, 0.5)`, velocities in pkdgrav3 units; see the conversion in
`pkc_convert.cxx`) and writes `OmegaLambda = dLambda` (0 for w0-wa). So route (i) also goes through
Gadget-2, written by the converter from the HDF5 file:

| Run | Input | Rockstar config |
|---|---|---|
| (i) | pkdgrav3 HDF5 (float32, unquantized) -> `pkc_to_gadget --from-hdf5 snap.00100.h5 --cosmo snap.00100` (new mode, below) | `FILE_FORMAT = "GADGET2"`, `GADGET_ID_BYTES = 8`, `NUM_BLOCKS = 64` |
| (ii) | .pkc -> `pkc_to_gadget --blocks 64` | same as (i) |
| (iii) | .pkc, native reader | `FILE_FORMAT = "PKC"`, `NUM_BLOCKS = 64` |

Why this design: (i) and (ii) differ **only** in the quantization (same reader, same Gadget
conventions, same cosmology and particle mass), and (ii) and (iii) differ **only** in the reader. So
(i) vs (ii) measures the effect of compression and (ii) vs (iii) tests the native reader. The TIPSY
route of today's pipeline can be added as (i') if wanted (cosmology from the config there).

`pkc_to_gadget` changes needed (pkdgrav3 repo, `io/snapcomp/tools/pkc_to_gadget.cxx`):

1. `--blocks K`: split the (file, pkc-block) list with the same rule as the native reader (section 2),
   so Gadget file k = PKC block k, same particles, same order.
2. Cosmology from the version-3 header: `OmegaLambda = omega_l + omega_de` (today it copies
   `omega_l`, which is 0 for w0-wa runs); stop if both are > 0 and (w0, wa) != (-1, 0); require
   version-3 input. A Gadget header cannot carry w0/wa, so print the lines
   to put in the Rockstar config (`W0 = .. WA = ..`, also `Om/Ol/h0` for reference) and write them to
   `<out_prefix>.rockstar.cfg` so the job script can append them instead of copying by hand.
3. `--from-hdf5 file.h5 --cosmo <pkc prefix>`: read `/PartType1/Coordinates`, `Velocities`,
   `ParticleIDs` (as `pkc_convert` does, reading the stored type, float or double), convert to the
   Gadget conventions (`(x + 0.5) * L` wrapped into `[0, L)`, peculiar km/s via `pkc_convert`'s
   `vel_to_kms`, then `/ sqrt(a)`), take box, redshift and cosmology from the .pkc header of the same
   step (check that `n_total`, box and redshift agree with the HDF5 `Header`), particle mass with the
   same 2.77536627e11 formula. Taking the cosmology from the .pkc makes it identical in (i)-(iii) by
   construction. Needs HDF5 at build time (already optional in the snapcomp CMake for `pkc_convert`).
   Alternative: a separate `h5_to_gadget` tool sharing the writer; one tool keeps the conventions in
   one place.
4. Conversion check before using it: on the Mac 64^3 run and the pawna 512^3 run, compare (i)'s Gadget
   positions/velocities with (ii)'s per ID: differences must be within half a quantization step
   (0.98 kpc/h, 1.22 km/s), which also verifies the HDF5 unit conversion. Same check as `pkc_convert
   --verify`.

Identical `NUM_WRITERS`, `NUM_READERS`, `FORK_PROCESSORS_PER_MACHINE`, `W0`/`WA` (from item 2's file
for (i), (ii); from the header for (iii)) and everything else; same snapshots (at least z = 0, 0.5, 1,
2). Check that `#Om = ..; Ol = ..; h = ..` and `rockstar.cfg` W0/WA agree across the three runs.

- **(ii) vs (iii): must agree exactly.** Same particles, same order per reader, same float arithmetic
  and globals -> compare `out_*.list` and `halos_*.ascii` byte for byte (after dropping
  version/timestamp lines). Any difference is a bug. If Rockstar turns out not to be deterministic for a
  fixed input (thread/socket timing), first run (ii) twice to establish the noise floor, then require
  (iii) to be within it.
- **(i) vs (ii) (and so vs (iii)): effect of compression** (1.95 kpc/h positions, 2.44 km/s velocities):
  - halo counts above 20/50/100/1000 particles; mass function ratio per mass bin with Poisson errors;
  - match halos by maximum shared particle IDs (IDs are the same in both), then distributions of
    `dlog M_vir`, position offset / R_vir, velocity offset / V_rms, `dVmax/Vmax`, concentration and spin;
    expect scatter dominated by halos near the 20-particle limit;
  - subhalo counts per host (most sensitive to velocity quantization);
  - consistent-trees on 3-4 consecutive snapshots from (i) and (iii): fraction of main progenitors
    matched, merger rates.
  Report numbers with the exact commands and config files used, stored with the outputs.
- Also (iii) with `PKC_SUBSAMPLES = "1"` and `"1,3,5,9,10"` once, to document what the quick-test
  subsets give (mass function at the scaled resolution), not as a correctness test.

## 7. Risks and open questions

1. **Rockstar memory at 2048^3 on one node** (independent of the input format): the particle array alone
   is 275 GB of 755 GB, and readers and writers hold particles concurrently during the transfer, plus
   overlap regions and halo structures. Whether 2048^3 fits on one pawna node at all is unknown; measure
   RSS at 512^3 and 1024^3 first. The PKC reader adds < 10 MB per reader.
2. **Exactness of (ii) vs (iii)** relies on identical particle order per reader and identical float
   operations; designed for (section 2-3), but Rockstar's own determinism is untested here.
3. **Spec vs reference decoder:** the spec says positions in box units are `(X+0.5)/Q - 0.5`
   (`[-0.5, 0.5)`), while the prompt for this work and `pkc_to_gadget` use `(X+0.5)/Q * L` in `[0, L)`.
   Both are right (a shift by L/2, periodic), but the spec should state the `[0, L)` convention used by
   all readers. Also: the spec says Lagrangian coder computes `floor(X*N/Q)`; the reference does it in
   128-bit integers; the spec should say "integer division" to rule out a float implementation.
   The optional potential stream (versions 2 and 3) must be handled by every reader.
4. **Dark energy and curvature**: resolved for Rockstar by section 3a (format version 3 only).
   Remaining: radiation treated as curvature by Rockstar; consistent-trees ignores Ol (in its times),
   curvature and w0/wa (section 3c, patch planned).
5. **Mass constant**: `pkc_to_gadget` (and this plan) use rho_crit = 2.77536627e11, Rockstar's
   `CRITICAL_DENSITY` is 2.77519737e11 (6e-5 relative). Kept as in `pkc_to_gadget` for the exact
   comparison; decide which one Sahyadri catalogs should use.
6. **Index order is write-completion order**, so the particle order (and thus per-reader sets) differs
   between two runs of the same simulation; only matters for bitwise reproducibility across reruns,
   not between readers of the same files.
7. **zstd availability** on pawna and the version (needs >= 1.4 for `ZSTD_decompressDCtx` with
   checksums; any modern one). Building it from source needs approval.
8. **Multi-writer snapshots** (pegasus, 4 writers): the design handles any `n_writers`, but only
   single-writer (pawna) files have been checked so far; include a pegasus 512^3 set in test 1.
9. **`SNAPSHOT_NAMES`** must list zero-padded pkdgrav3 steps; easy to get wrong. Provide a helper in
   `scripts/rockstar/` that writes it from the `*.done` markers.
10. Rockstar's `PARALLEL_IO_*` server/port handling, `FORK_READERS_FROM_WRITERS` and PBS setup are
    unchanged; the existing `run_rockstar_batch.sh` flow should work with only config changes. To check
    on pawna with a 128^3 run before 512^3.

## 7b. Progress (2026-11-10)

Step 1 implemented on this branch (uncommitted), files listed in `scripts/rockstar/readme_shadab.txt`.
Decisions taken while implementing:

- Decoder split: `io/pkc_core.{c,h}` (format only, no Rockstar headers) and `io/io_pkc.{c,h}` (globals).
- Version 3 only; versions 1/2 are rejected with a message.
- The potential stream is skipped without decompressing it (Rockstar does not use it), so damage in that
  stream is not detected; every other stream is checked by its zstd checksum.
- `PKC_THREADS` is not added: one decoding thread per reader (section 5); add only if a measurement
  asks for it.
- A snapshot with fewer pkc blocks (in the selected subsamples) than `NUM_BLOCKS` is an error ("lower
  NUM_BLOCKS"), so no reader gets zero particles.
- `gen_merger_cfg.pl` takes w0/wa from the `#w0` catalog line, else from the config (as before), else -1/0.

Tests (`tests_pkc/run_tests.sh`, Mac, snapcomp built from pkdgrav3 fcd37310 with the anaconda zstd,
reader with Homebrew zstd 1.5; about 3 s): 28 checks, 0 failures.

| Check | Result |
|---|---|
| decode: 4 version-3 sets (32^3 lattice IDs, 2 writers, potential, block 1000; 24^3 random IDs; 24^3 index IDs, 3 writers, block 500; 48^3 lattice IDs, block 65536) | every block identical to `snapcomp::FileReader` (X, V, IDs); all three ID coders exercised (raw/delta/Lagrangian) |
| shares, NB = 1, 7, 64, 1000 | every ID exactly once; shares differ from T/NB by less than one pkc block |
| `PKC_SUBSAMPLES` 1 and 1,3,5,9,10 | only those subsamples' particles (checked with the ID hash); 0.998% and 28.4% of 32^3 |
| cosmology, w0-wa set (Omega_lambda 0, Omega_DE 0.69, w0 -0.9, wa 0.1, z 0.5) | Ol 0.69, W0 -0.9, WA 0.1; Rockstar's `hubble.c` E(z) equals pkdgrav3's E_z to 2e-10 |
| floats vs Rockstar reading `pkc_to_gadget` output (32^3 and 48^3) | positions and velocities bit-identical (0 ulp), particle mass identical |
| rejection | missing trailer, missing writer file, flipped byte in velocity or ID stream (zstd checksum), header of another snapshot, version 2, duplicate file: all refused with a clear message; flipped byte in the potential stream: ignored, as designed |

The float check repeats the expressions of `io_pkc.c` in the test (the glue itself is not linked into
a test). All changed Rockstar files pass `cc -fsyntax-only -std=c99 -Wall` against the upstream
headers, with and without `ENABLE_PKC` (no warnings). Not done yet: building the Rockstar binary
(needs the upstream source: ask first), running it, `pkc_to_gadget --blocks`, consistent-trees patch.

### End-to-end on the Mac (2026-11-10)

Rockstar built from upstream 36ce9eea36ee + our files (`make with_pkc ZSTD_DIR=/opt/homebrew`, scratch
copy; only upstream warnings). `pkc_to_gadget` extended on pkdgrav3 branch `pkc-to-gadget-v3`
(`--blocks K`, version-3 cosmology with `<out>.rockstar.cfg`, `--from-hdf5`).

Input: `tmp.nosync/p1/n128_L200/snap.00090` of the pkdgrav3 checkout (128^3, 200 Mpc/h, z = 0, HDF5,
LCDM). .pkc made with `pkc_convert` (dx 1.95 kpc/h, dv 2.44 km/s; writes version 1) and turned into
version 3 with `tests_pkc/pkc_v1_to_v3.py` (test helper). Rockstar: PARALLEL_IO on one machine,
FORCE_RES 0.03, MIN_HALO_OUTPUT_SIZE 20 (`tests_pkc/run_rockstar_e2e.sh`); about 2 s per run.

1. `pkc_compare blocks`: Gadget file k of `--blocks 8` holds exactly the IDs of native share k, same
   order; `pkc_compare gadget`: all 2,097,152 particles bit-identical (0 ulp), same particle mass.
2. **Rockstar is not deterministic**: the same input run twice gives different catalogs (8 blocks:
   7805/7808/7811 halos native, 7811/7823/7799 Gadget). Causes: `srand(getpid())` in `client.c` (the
   random constant of `inthash.c` changes iteration order), and with several readers the order in which
   particles reach the writers. Even 1 block differs between runs (7809-7811) with the stock binary.
3. **Exact test passes** with a test-only build (`srand(12345)` instead of `srand(getpid())`, scratch
   copy, not in the repo) and NUM_BLOCKS = 1: native PKC and `pkc_to_gadget --blocks 1` give
   byte-identical `out_0.list`, `halos_0.0.ascii`, `halos_0.0.bin` (7808 halos), reproducible over two
   runs each; effective `rockstar.cfg` identical except the format lines. With 8 blocks the fixed seed
   is not enough (7792-7801 halos).
4. First look at the compression (HDF5 route vs native, both 1 block, fixed seed; halos matched by
   nearest centre within 0.1 Rvir, `tests_pkc/cmp_halos.py`), next to Rockstar's own scatter (same
   PKC input, two 8-block runs):

| | exact (HDF5) vs PKC | PKC run 1 vs run 2 |
|---|---|---|
| halos Np >= 20 / 50 / 100 / 1000 | 7799/3484/1531/33 vs 7808/3480/1522/33 | 7801/3485/1523/33 vs 7798/3484/1529/33 |
| matched (Np >= 100) | 95.8% | 96.6% |
| rms dlog10 Mvir (99% bound) | 0.026 (0.040) | 0.015 (0.021) |
| rms dVmax/Vmax | 0.016 | 0.013 |
| position offset / Rvir: median (99%) | 6e-4 (0.089) | 0 (0.082) |
| bulk velocity diff.: median (99%) | 0.16 km/s (62 km/s) | 0 (40 km/s) |

   One 128^3 snapshot, few halos and a crude matching: indicative only. The compression effect is
   of the order of Rockstar's run-to-run scatter, somewhat larger in mass. The 512^3 runs (section 6)
   with ID-based matching are the real test. Note the .pkc here was made from the same HDF5 with the
   same velocity conversion as `--from-hdf5`, so the difference is quantization only.

Consequence for section 6: the exact (ii) vs (iii) comparison uses NUM_BLOCKS = 1 and the fixed-seed
test build; production-like 8+ block runs are compared statistically against the scatter between
repeated runs of the same input.

## 8. Work order

1. **Rockstar reader (this branch, started 2026-11-10)**: decoder split into `io/pkc_core.{c,h}` (no
   Rockstar headers, testable alone) and `io/io_pkc.{c,h}` (Rockstar globals); `meta_io.c` (reader
   branch + `#w0/wa` header line), `client.c` (W0/WA), `config.template.h`, `Makefile`;
   `scripts/gen_merger_cfg.pl` (w0/wa from `out_*.list`, `#` comments). Tests against version-3 files
   made with the snapcomp library (section 6, tests 1-3). Building the full Rockstar binary needs the
   upstream source (third-party): ask first.
2. `pkc_to_gadget`: `--blocks K`, version-3 cosmology (`Ol = omega_l + omega_de`, W0/WA config file),
   `--from-hdf5` (section 6), in the pkdgrav3 repo (separate branch/PR there).
3. consistent-trees cosmology patch (section 3c), with the flat-LCDM identity test.
4. Pipeline: one job chain Rockstar -> `gen_merger_cfg.pl` -> consistent-trees -> FITS conversion
   with checks -> `clean_trees.pl` (sections 3b, 3d); bring `compres_halos.py` and
   `consistent_trees_compress.py` into this repo.
5. Build on pawna (`make with_pkc`, zstd question first), 128^3 end-to-end, then the 512^3 validation
   runs (i)-(iii) through PBS, then the comparison report.
6. Spec clarifications (7.3) as a PR to pkdgrav3 `docs/snapcomp_format.md`.
