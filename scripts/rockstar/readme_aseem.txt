Aseem made following changes:
---- io/io_arepo.c
	modified arepo_readheader_float() and arepo_readheader_array() to take char *gid as argument
	modified load_particles_arepo() to separately read Header and Parameters from Gadget-4 output
---- io/meta_io.c
	wrapped io_tipsy.h include in #ifdef HAVE_TIRPC #endif as already done for AREPO
	wrapped load_particles_tipsy in #ifdef HAVE_TIRPC #else <error msg> #endif as already done for AREPO
---- io/io_tipsy.c, io/io_tipsy.h
	wrapped #ifdef HAVE_TIRPC #endif like Oliver does in MUSIC 
----  hubble.c
	add term (1-Om-Ol)*(z1*z1) in hubble_scaling()
----  io/io_gadget.c
	comment out exit 1 upon detection of curvature
----  Makefile
	for compatibility, included commented-out definition of HAVE_TIRPC

Original files should be stored in:
	hubble.c.bak, io/io_gadget.c.bak, io/io_arepo.c.bak
