#!/usr/bin/perl -w

use strict;
use warnings;

my $halos = $ARGV[0];
my $trees = "$halos/hlists";
my $scales = "$halos/outputs/scales.txt";

open (my $handle, '<', $scales) or die "Could not open file '$scales'";
chomp(my @scale_values = <$handle>);
close $handle;

foreach ( @scale_values )
{
	my ( $snap, $scale ) = split;
	$scale = sprintf("%.5f", $scale);
	my $out_tree = "out_" . $snap . ".trees";
	my $hlist = "hlist_" . $scale . ".list";
	system("cp $trees/$hlist $halos/$out_tree");
}

system("cp $scales $halos/../scales.txt");
system("rm -rf $halos/outputs $halos/out_*.list $halos/hlists");

#print "@scale_values\n";
#print "$#scale_values\n";