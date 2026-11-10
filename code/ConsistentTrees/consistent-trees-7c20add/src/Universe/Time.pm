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

# Sahyadri (2026-11-10): dark energy w(a) = w0 + wa (1 - a) and curvature 1 - Om - Ol, the same
# background as our Rockstar (hubble.c) and the patched C code (universe_time.c). init($om, $h) alone,
# or flat LCDM, keeps the original closed-form code.
our ($de_set, $de_Ol, $de_w0, $de_wa) = (0, 0, -1, 0);

sub is_flat_lcdm {
    my $om = shift;
    return (!$de_set || (abs($om + $de_Ol - 1) < 1e-6 && $de_w0 == -1 && $de_wa == 0));
}

sub hubble_E {
    my ($a, $om) = @_;
    my $ol = $de_set ? $de_Ol : 1 - $om;
    my $weff = $de_set ? ($a != 1 ? $de_w0 + $de_wa - $de_wa*($a - 1)/log($a) : $de_w0) : -1;
    return sqrt($om/($a**3) + (1 - $om - $ol)/($a**2) + $ol*($a**(-3*(1 + $weff))));
}

# int_{a1}^{a2} da/(a E) in units of 1/H0, with a = s^2 (Simpson, n even)
sub _time_integral {
    my ($a1, $a2, $n) = @_;
    my ($s1, $s2) = (sqrt($a1), sqrt($a2));
    my $ds = ($s2 - $s1)/$n;
    my $sum = 0;
    for my $i (0..$n) {
        my $s = $s1 + $i*$ds;
        next if ($s <= 0);
        my $f = 2/($s*hubble_E($s*$s, $omega_0));
        $sum += ($i == 0 || $i == $n) ? $f : (($i % 2) ? 4*$f : 2*$f);
    }
    return $sum*$ds/3;
}


sub init {
    my ($om, $h, $ol, $w0, $wa)  = @_;
    $to_years = 1.39687707e10*0.7/$h;
    $omega_0 = $om;
    if (defined $ol) { ($de_set, $de_Ol, $de_w0, $de_wa) = (1, $ol, defined $w0 ? $w0 : -1, defined $wa ? $wa : 0); }
    else { $de_set = 0; }
    $exact_t0_conv = 0;
    if (!is_flat_lcdm($om)) { # Sahyadri: numerical table (cumulative, 64 Simpson steps per table step)
        my $t = 0;
        my @t_of;
        $t_of[0] = 0;
        for (1..($steps+2)) {
            $t += _time_integral(($_-1)/$steps, $_/$steps, 64);
            $t_of[$_] = $t;
        }
        $exact_t0_conv = $t_of[$steps];
        $times[$_] = $t_of[$_] - $exact_t0_conv for (0..($steps+2));
        return;
    }
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
    return _time_integral(0, $scale, 4096) - $exact_t0_conv if (!is_flat_lcdm($omega_0)); # Sahyadri
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
