#!/usr/bin/perl -w

use strict;
use warnings;

my @scale_values_wrong = (0.137165, 0.234165, 0.390705, 0.583525, 0.783095, 0.816705, 0.861575);
my @scale_values_right = (0.178535, 0.266645, 0.371775, 0.832455, 0.929985, 0.940705, 0.984835);

system("echo these values were wrong");
foreach my $scale ( @scale_values_wrong )
{
	my $scale_then = sprintf("%.5f", $scale);
	my $scale_now = sprintf("%.5f", $scale+1e-10);
	system("echo $scale --: then $scale_then, now $scale_now");
}

system("echo these values were right");
foreach my $scale ( @scale_values_right )
{
	my $scale_then = sprintf("%.5f", $scale);
	my $scale_now = sprintf("%.5f", $scale+1e-10);
	system("echo $scale --: then $scale_then, now $scale_now");
}

# above will always be a problem due to buggy sprintf rounding

system("echo instead, do this:");
foreach my $scale ( @scale_values_wrong )
{
    my $scale = substr("$scale",0,5);
    my $hlist = "hlist_" . $scale . "*.list";
    system("echo $hlist");
}
foreach my $scale ( @scale_values_right )
{
    my $scale = substr("$scale",0,5);
    my $hlist = "hlist_" . $scale . "*.list";
    system("echo $hlist");
}
