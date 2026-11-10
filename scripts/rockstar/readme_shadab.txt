Shadab (with Claude) made the following changes, 2026-11-10 (plan: docs/rockstar_pkc_reader_plan.md):
---- io/pkc_core.c, io/pkc_core.h (new)
	C99 reader for pkdgrav3 compressed snapshots (snapcomp format version 3), written from the format
	spec; zstd + POSIX, no Rockstar headers. Compiled only with -DENABLE_PKC.
---- io/io_pkc.c, io/io_pkc.h (new)
	FILE_FORMAT = "PKC": load_particles_pkc(); sets SCALE_NOW, BOX_SIZE, Om, Ol (= Omega_lambda + Omega_DE),
	h0, W0, WA, TOTAL_PARTICLES, PARTICLE_MASS, AVG_PARTICLE_SPACING from the snapshot header.
	FILENAME = "<prefix>.<snap>.<block>"; Rockstar block k reads the k-th equal share of the pkc blocks.
---- io/meta_io.c
	PKC branch in read_particles() (#ifdef ENABLE_PKC, as for AREPO);
	print_ascii_header_info() prints "#w0 = ..; wa = .." when (W0, WA) != (-1, 0) (LCDM output unchanged)
---- client.c
	send_config()/recv_config() also pass W0 and WA from readers to writers
---- config.template.h
	PKC_SUBSAMPLES (only with ENABLE_PKC): subsamples to read by percent, e.g. "1,3"; "" = all
---- Makefile
	io/io_pkc.c io/pkc_core.c in CFILES; targets with_pkc, with_hdf5_pkc; ZSTD_DIR (default /usr)
---- scripts/gen_merger_cfg.pl
	w0/wa taken from the "#w0" line of out_<STARTING_SNAP>.list, else from the config, else -1/0;
	comments written with '#'
---- tests_pkc/ (new, not part of the build)
	make_testset.cxx, pkc_compare.cxx, run_tests.sh: tests of the reader against pkdgrav3's snapcomp library

Original files should be stored in:
	client.c.bak, config.template.h.bak, io/meta_io.c.bak, Makefile.bak (upstream 36ce9eea36ee)
