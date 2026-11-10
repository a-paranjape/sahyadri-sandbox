#!/usr/bin/perl -w
package MassConversions;
use Math::Trig;

gen_lookups();

# From http://arxiv.org/abs/0710.5520v2
sub ConvertMvirToMstatic {
    my ($m, $z) = @_; # M is in log units
    my $a = 1/($z+1);
    my $m0 = log(1e-3*2.8*1e15*($a**8.9)/0.7)/log(10);
    my $x = $m-$m0;
    return ($m + 0.24 + 0.054*$x) if ($x<0);
    return ($m + 0.24 - 0.075*$x);
}

sub ConvertDeltaPcToDeltaPm {
    my ($delta, $z) = @_;
    $z ||= 0;
    my $om = 0.27;
    return ($delta * ($om*((1+$z)**3) + (1-$om))/($om*((1+$z)**3)));
}

sub RvirFromMvir{
    my ($m, $z) = @_;
    my $delta_vir = Delta_vir($z);
    my $bdens = $delta_vir * 7.49234837e10;
    return ((10**($m)/($bdens*4/3*pi))**(1/3));
}

sub ConvertToMvir {
    my ($m, $delta, $z, $c) = @_;
    $z ||= 0;
    if ($delta =~ s/c$//) { $delta = ConvertDeltaPcToDeltaPm($delta, $z); }
    my $delta_vir = Delta_vir($z);
    $c ||= c($m, $z);
     my $r_frac = inv_f($delta/$delta_vir * f(1/$c));
    #print $c*$r_frac, "\n";
    my $conv_factor = $delta/$delta_vir * (($r_frac*$c)**(-3));
    #print "$delta\n$delta_vir\n";
    return $m - log($conv_factor)/log(10);
}

sub ConvertFromMvir {
    my ($m, $delta, $z, $c) = @_;
    my $delta_vir = Delta_vir($z);
    if ($delta =~ s/c$//) { $delta = ConvertDeltaPcToDeltaPm($delta, $z); }
    $c ||= c($m, $z);
    
    my $r_frac = inv_f($delta/$delta_vir * f(1/$c));
    #print $c*$r_frac, "\n";
    my $conv_factor = $delta/$delta_vir * (($r_frac*$c)**(-3));
    #print "$delta\n$delta_vir\n";
    return $m + log($conv_factor)/log(10);
}

sub Delta_vir {
    #From http://arxiv.org/pdf/astro-ph/9710107v1
    #WRT to background density
    my ($z, $om) = @_;
    $om ||= 0.27;
    $z ||= 0;
    my $omega = $om*((1+$z)**3)/($om*((1+$z)**3) + 1-$om);
    my $x = $omega - 1;
    return 1/(1+$x) * (18*pi*pi + 82*$x - 39*$x*$x);
}

sub Dyn_time {
    my ($z, $om, $h) = @_;
    $om ||= 0.27;
    $z ||= 0;
    $h ||= 0.7;
    my $dvir = Delta_vir($z, $om); 
    my $g = 4.30117902e-9; #Actually, Gc * (Msun / Mpc) in (km/s)^2
    my $cd = 2.77519737e11; #3H^2/8piG in (Msun / h) / (Mpc / h)^3
    my $bd = $cd*$om*((1+$z)**3); #background density
    my $dyn_time = 1.0/(sqrt((4/3)*pi*$g*$dvir*$bd)*$h); # in Mpc / (km/s)
    $dyn_time *= 9.77813106e11; # Mpc/(km/s) to yrs
    return $dyn_time;
}

sub c {
    my ($m, $z) = @_;
    $z ||= 0;
    return ((10**(2.2358 - 0.10*$m))/(1+$z));
}

sub f {
    my $x = shift;
    return ($x*$x*$x *(log(1+1/$x) - 1/(1+$x)));
}

sub _inv_f {
    my $f = shift;
    my $inv_f = 1;
    my $dif = 0.1;
    my $ft = f($inv_f);
    while (abs($ft-$f)/$f > 0.01) {
	my $ft2 = f($inv_f + $dif);
	my $m = ($f - $ft)*($dif)/($ft2-$ft);
	$dif = $m/5;
	$inv_f += $m;
	$ft = f($inv_f);
    }
    return $inv_f;
}

sub inv_f {
    my $f = shift;
    my $of = $f;
    my $index = 800*($f - $f1)/($f2-$f1);
    $f = $index - int($index);
    $index = int($index);
    return _inv_f($of) unless (defined($inv_f[$index+1]));
    return $inv_f[$index] + $f*($inv_f[$index+1]-$inv_f[$index]);
}

sub gen_lookups {
    our @f = map { f($_/1600) } (1..800);
    our @inv_f;
    our ($f1, $f2) = ($f[0], $f[-1]);
    my $i = 1;
    my $step = ($f2 - $f1)/799;
    $inv_f[0] = 1/1600;
    $inv_f[399] = 800/1600;
    for (0..($#f-1)) {
	while ($f[$_+1]> $f1+$i*$step and $i<799) {
	    my $f = ($f1+$i*$step - $f[$_])/($f[$_+1]-$f[$_]);
	    $inv_f[$i] = ($_+$f)/1600;
	    $i++;
	}
    }
}

1;
