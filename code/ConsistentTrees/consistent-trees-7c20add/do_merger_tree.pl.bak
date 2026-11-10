#!/usr/bin/perl -w

my $cfg = $ARGV[0];
check_system("./gravitational_consistency", $cfg);
check_system("./find_parents_and_cleanup", $cfg);
check_system("./resort_outputs", $cfg);
check_system("./assemble_halo_trees", $cfg);
 
sub check_system {
    system(@_) == 0 or
	die "Tree creation failed while executing \"@_\".\n(See errors above).\n";
}
