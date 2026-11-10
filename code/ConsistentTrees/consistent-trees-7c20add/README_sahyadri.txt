consistent-trees as used by the Sahyadri pipeline: upstream https://bitbucket.org/pbehroozi/consistent-trees
commit 7c20add (2021-04-20, version 1.01). Only changed files are kept here; the originals are the .bak files.
Copy these files over a clone of 7c20add, then build as usual (cd src; make).

---- do_merger_tree.pl, halo_trees_to_catalog.pl (Aseem, 08 Nov 2023)
	take the code directory as second argument ($code), so the scripts run from anywhere
	(copied from the pegasus install /mnt/home/faculty/caseem/code/ConsistentTrees/consistent-trees)
---- cosmology patch (Shadab with Claude, 2026-11-10; plan: docs/rockstar_pkc_reader_plan.md, section 3c)
	Upstream uses flat LCDM everywhere: scale -> time from the closed form with Om only (Ol and curvature
	ignored), delta_vir and Dyn_time with a cosmological constant, and no w0/wa. Now, with the keys Ol, w0, wa
	of merger_tree.cfg (written by our gen_merger_cfg.pl), the background is
	    E(a)^2 = Om a^-3 + (1 - Om - Ol) a^-2 + Ol a^(-3(1 + w_eff(a))),  w(a) = w0 + wa (1 - a),
	the same as our Rockstar hubble.c (Aseem's curvature term, Saee's w0/wa).
	src/config.template.h       w0, wa (defaults -1, 0)
	src/grav_config.c           set_dark_energy(Ol, w0, wa) after reading the config (all programs that
	                            read merger_tree.cfg; halo_evolve, which takes Om on its command line, stays LCDM)
	src/universe_time.[ch]      set_dark_energy(), hubble_E(); numerical time table (Simpson in sqrt(a)) when the
	                            background is not flat LCDM; otherwise the original closed-form code
	src/masses.c                delta_vir(): x = Om a^-3 / E^2 - 1 when not flat LCDM; otherwise unchanged
	halo_trees_to_catalog.pl    passes Ol, w0, wa to Universe::Time::init
	src/Universe/Time.pm        same as universe_time.c, in Perl (accretion rates, times in the hlists)
	src/MassConversions.pm      Delta_vir (and so Dyn_time) with Omega_m(z) from the same background
	Flat LCDM (|Om + Ol - 1| < 1e-6, w0 = -1, wa = 0) runs the original code paths, so LCDM outputs are
	byte-identical to upstream.
---- tests_sahyadri/run_tests.sh <clone of 7c20add> <scratch> [Rockstar OUTBASE]
	2026-11-10, Mac (Apple clang, built without -fopenmp), Rockstar PKC outputs of three 128^3 snapshots
	(z = 3, 1, 0): flat LCDM time table / delta_vir identical to the original in C and Perl; numerical path
	vs closed form 1.4e-7 (the original's Newton tolerance); C vs Perl 3e-9; full LCDM run original vs
	patched: 38 output files (trees, hlists, cleaned catalogs) byte-identical. With w0 = -0.9, wa = 0.1 on
	the same catalogs (a code exercise, not a w0-wa simulation), accretion rates change by 1.6% (a = 0.5)
	to 2.2% (a = 1) and about 1-2% of halos are handled differently by the consistency step.
