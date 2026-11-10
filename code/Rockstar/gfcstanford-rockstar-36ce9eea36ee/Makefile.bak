CFLAGS=-m64 -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 -D_BSD_SOURCE -D_POSIX_SOURCE -D_POSIX_C_SOURCE=200809L -D_SVID_SOURCE -D_DARWIN_C_SOURCE -Wall -fno-math-errno -fPIC
LDFLAGS=-shared
OFLAGS = -lm -O3 -std=c99 -rdynamic -g
DEBUGFLAGS = -lm -g -O0 -std=c99 -rdynamic
PROFFLAGS = -lm -g -pg -O2 -std=c99
#CC = gcc
CFILES = rockstar.c check_syscalls.c fof.c groupies.c subhalo_metric.c potential.c nfw.c jacobi.c fun_times.c interleaving.c universe_time.c hubble.c integrate.c distance.c config_vars.c config.c bounds.c inthash.c io/read_config.c client.c server.c merger.c inet/socket.c inet/rsocket.c inet/address.c io/meta_io.c io/io_internal.c io/io_ascii.c io/stringparse.c io/io_gadget.c io/io_generic.c io/io_art.c io/io_tipsy.c io/io_bgc2.c io/io_util.c io/io_arepo.c io/io_hdf5.c
DIST_FLAGS =
HDF5_FLAGS = -DH5_USE_16_API -lhdf5 -DENABLE_HDF5 -I/opt/local/include -L/opt/local/lib

all:
	@make reg EXTRA_FLAGS="$(OFLAGS)"

with_hdf5:
	@make reg EXTRA_FLAGS="$(OFLAGS) $(HDF5_FLAGS)"

debug:
	@make reg EXTRA_FLAGS="$(DEBUGFLAGS)"

with_hdf5_debug:
	@make reg EXTRA_FLAGS="$(DEBUGFLAGS) $(HDF5_FLAGS)"

prof:
	@make reg EXTRA_FLAGS="$(PROFFLAGS)"

.REMAKE:

dist: .REMAKE
	cd ../ ; perl -ne 'print "$$1\n" if (/VERSION\s*\"([^\"]+)/)' Rockstar/version.h > Rockstar/VERSION; tar -czvf rockstar.tar.gz Rockstar/Makefile Rockstar/*.[ch] Rockstar/examples/Makefile Rockstar/[^sto]*/*.[ch] Rockstar/quickstart.cfg Rockstar/parallel.cfg Rockstar/scripts/*.pbs Rockstar/scripts/*.cfg Rockstar/scripts/*.pl Rockstar/SOURCE_LAYOUT Rockstar/README.pdf Rockstar/README Rockstar/LICENSE Rockstar/VERSION Rockstar/ACKNOWLEDGMENTS Rockstar/CHANGELOG; mv rockstar.tar.gz Rockstar

versiondist:
	$(MAKE) dist DIST_FLAGS="$(DIST_FLAGS)"
	rm -rf dist
	mkdir dist
	cd dist; tar xzf ../rockstar.tar.gz ; perl -ne '/\#define.*VERSION\D*([\d\.rcRC-]+)/ && print $$1' Rockstar/version.h > NUMBER ; mv Rockstar Rockstar-`cat NUMBER`; tar czf rockstar-`cat NUMBER`.tar.gz Rockstar-`cat NUMBER`

reg:
	$(CC) $(CFLAGS) main.c $(CFILES) -o rockstar  $(EXTRA_FLAGS)

lib:
	$(CC) $(CFLAGS) $(LDFLAGS) $(CFILES) -o librockstar.so $(OFLAGS)

bgc2:
	$(CC) $(CFLAGS) io/extra_bgc2.c util/redo_bgc2.c $(CFILES) -o util/finish_bgc2  $(OFLAGS)
	$(CC) $(CFLAGS) io/extra_bgc2.c util/bgc2_to_ascii.c $(CFILES) -o util/bgc2_to_ascii  $(OFLAGS)

parents:
	$(CC) $(CFLAGS) util/find_parents.c io/stringparse.c check_syscalls.c  -o util/find_parents $(OFLAGS)

substats:
	$(CC) $(CFLAGS) util/subhalo_stats.c $(CFILES) -o util/subhalo_stats  $(OFLAGS)


clean:
	rm -f *~ io/*~ inet/*~ util/*~ rockstar util/redo_bgc2 util/subhalo_stats

