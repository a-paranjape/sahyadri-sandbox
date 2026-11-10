#!/usr/bin/perl -w
package Universe::Time;
use Math::Trig;
use base qw(Exporter);

our @EXPORT = qw(scale_to_time scale_to_years exact_time_to_scale);

our $omega_0 = 0.27;
our $a_0 = 1;
our $t_0 = 0;
our $steps = 1024;
our @scales;
our @times;
our $to_years = 1.39687707e10;
our $exact_t0_conv = 0;


sub init {
    my ($om, $h)  = @_;
    $to_years = 1.39687707e10*0.7/$h;
    $omega_0 = $om;
    $exact_t0_conv = 0;
    $exact_t0_conv = exact_scale_to_time(1.0);
    for (1..($steps+2)) {
	my $a = $_/$steps;
	$times[$_] = exact_scale_to_time($a);
    }
    $times[0] = -$exact_t0_conv;
}

sub exact_time_to_scale {
    my $t = shift;
    return (($t*3/2)**(2/3)) if ($omega_0 == 1);
    my $m = (sinh(1.5*$t*sqrt(1-$omega_0)))**2;
    return ((($omega_0*$m)/(1-$omega_0))**(1/3));
}

sub exact_scale_to_time {
    my $scale = shift;
    my $t = $scale;
    my $a = exact_time_to_scale($t);
    my $dt = $scale/10.0;
    my $count = 0;
    while (abs($a-$scale)>1e-7 && ($count < 10)) {
	$count++;
	my $a2 = exact_time_to_scale($t+$dt);
	my $move = ($scale-$a)*($dt)/($a2-$a);
	$t += $move;
	$a = exact_time_to_scale($t);
	if ($move/10.0 < 0.5*$dt) { $dt = $move/10.0; }
	else { $dt /= 2.0; }
    }
    return ($t-$exact_t0_conv);
}


sub scale_to_time {
    my $s = shift;
    my $l = int($s*$steps);
    my $f = $s*$steps - $l;
    if ($s > 1) { return exact_scale_to_time($s); }
    elsif ($s < 0) { return $times[0]; } 
    return ($times[$l]*(1-$f) + $times[$l+1]*$f);
}

sub scale_to_years {
    return scale_to_time($_[0])*$to_years;
}

init(0.27, 0.7);

1;
