#!/usr/bin/perl -w

my $cfg = $ARGV[0];
# $code references added by Aseem 08 Nov 2023
my $code = $ARGV[1];
check_system("$code/gravitational_consistency", $cfg);
check_system("$code/find_parents_and_cleanup", $cfg);
check_system("$code/resort_outputs", $cfg);
check_system("$code/assemble_halo_trees", $cfg);
 
sub check_system {
    system(@_) == 0 or
	die "Tree creation failed while executing \"@_\".\n(See errors above).\n";
}
