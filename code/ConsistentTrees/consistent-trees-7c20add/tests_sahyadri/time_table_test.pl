use strict; use warnings;
use Universe::Time; use MassConversions;
my ($mode) = @ARGV;
my ($om, $h, $ol) = (0.3137721, 0.6736, 0.6862279);
if ($mode == 0) { Universe::Time::init($om, $h); }
elsif ($mode == 1) { Universe::Time::init($om, $h, $ol, -1 + 1e-12, 0); }
elsif ($mode == 2) { Universe::Time::init($om, $h, $ol, -0.9, 0.1); }
else { Universe::Time::init($om, $h, $ol, -1, 0); }
for (my $a = 0.1; $a <= 1.0001; $a += 0.15) {
  printf "%.2f %.12e %.12e %.12e\n", $a, scale_to_years($a), MassConversions::Delta_vir(1/$a - 1, $om), MassConversions::Dyn_time(1/$a - 1, $om, $h);
}
