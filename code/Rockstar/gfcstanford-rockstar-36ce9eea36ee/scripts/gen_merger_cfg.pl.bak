#!/usr/bin/perl -w
use Cwd 'abs_path';

$ENV{PWD} = abs_path(".") unless (abs_path($ENV{PWD}) eq abs_path("."));

for (@ARGV) {
    next if (/-c/);
    our $config = new ReadConfig($_);
    last;
}
die "Usage: perl $0 -c rockstar.cfg\n" unless ($config);
$config->set_defaults(OUTBASE => ".", STARTING_SNAP => 0, PERIODIC => 1,
		      MASS_DEFINITION => "vir",
		      MASS_DEFINITION2 => "200b", MASS_DEFINITION3 => "200c",
		      MASS_DEFINITION4 => "500c", MASS_DEFINITION5 => "2500c");

my $outbase = $config->{OUTBASE};
if ($outbase !~ m!^/!) {
    $outbase = $ENV{PWD}."/$outbase";
}
my $first_fn = "$outbase/out_$config->{STARTING_SNAP}.list";
open OUT, "<", $first_fn or
    die "Couldn't open first halo output catalog $first_fn!\n";
while (<OUT>) {
    last unless (/^\#/);
    if (/Om/) {
	our $Om = (/Om\s*=\s*([\d.e+-]+)/)[0];
	our $Ol = (/Ol\s*=\s*([\d.e+-]+)/)[0];
	our $h0 = (/h\s*=\s*([\d.e+-]+)/)[0];
	next;
    }
    if (/Box size/) {
	our $box_size = (/(\d[\d.e+-]+)/)[0];
    }
    if (/Particle/) {
	our $part_mass = (/(\d[\d.e+-]+)/)[0];
    }
}
close OUT;

our $num_snaps = 0;
if ($config->{SNAPSHOT_NAMES}) {
    open INPUT, "<", $config->{SNAPSHOT_NAMES} or
	die "Couldn't open snapshot names file $config->{SNAPSHOT_NAMES}!\n";
    while (<INPUT>) {
	chomp;
	$num_snaps++ if (length());
    }
}
elsif ($config->{NUM_SNAPS}) {
    $num_snaps = $config->{NUM_SNAPS};
}

my @scales;
my $size = 0;
for my $num ($config->{STARTING_SNAP}..($num_snaps-1)) {
    open INPUT, "<", "$outbase/out_$num.list" or
	die ("Couldn't open merger tree file $outbase/out_$num.list");
    my $scale;
    while (<INPUT>) {
	if (/a = (\d+\.\d+)/) {
            $scale = $1;
        }
	last unless (/^#/);
    }
    die "Couldn't extract scale factor from file $outbase/out_$num.list!" unless defined $scale;
    if ((-s "$outbase/out_$num.list") < 20000) {
	warn "Skipping $outbase/out_$num.list (too few halos).\n";
    }
    else {
	push @scales, "$num $scale\n";
	$size += -s "$outbase/out_$num.list";
    }
    close INPUT;
}

our $box_divisions = int(($size / 2e9)**(1/3));
$box_divisions = 1 if ($box_divisions < 1);
our $timesteps = int(@scales / 34);
our $sub_timesteps = 2*$timesteps;
$timesteps = 1 if ($timesteps < 1);
$sub_timesteps = 1 if ($sub_timesteps < 1);
our $max_phantoms = $timesteps;
our $max_phantoms_small = int($timesteps/2);
$max_phantoms_small = 1 if ($max_phantoms_small < 1);
our $mass_res_ok = sprintf("%g", $part_mass*1000);

my (@m) = map { $config->{$_} } qw(MASS_DEFINITION MASS_DEFINITION2 MASS_DEFINITION3 MASS_DEFINITION4 MASS_DEFINITION5);

for ("$outbase/outputs", "$outbase/trees", "$outbase/hlists") {
    unless (-d $_) {
	mkdir $_ or die "Couldn't make directory $_!\n";
    }
}



open SCALES, ">$outbase/outputs/scales.txt" or
    die "Couldn't output to scale file $outbase/outputs/scales.txt!\n";
print SCALES @scales or die "Couldn't write scales to $outbase/outputs/scales.txt!\n";
close SCALES;

open CONFIG, ">$outbase/outputs/merger_tree.cfg" or
    die "Couldn't output to merger tree config file $outbase/outputs/merger_tree.cfg!\n";
print CONFIG << "EOL"
Om=$Om #Omega_Matter
Ol=$Ol #Omega_Lambda
h0=$h0 #h0

SCALEFILE = "$outbase/outputs/scales.txt"
INBASE = "$outbase"
OUTBASE = "$outbase/outputs"
TREE_OUTBASE = "$outbase/trees"
HLIST_OUTBASE = "$outbase/hlists"

BOX_DIVISIONS=$box_divisions
BOX_WIDTH=$box_size

MASS_DEFINITION = $m[0]

MIN_TIMESTEPS_TRACKED = $timesteps
MIN_TIMESTEPS_SUB_TRACKED = $sub_timesteps
MIN_TIMESTEPS_SUB_MMP_TRACKED = $sub_timesteps
MAX_PHANTOM_FRACTION = 0.25

MAJOR_MERGER=0.3 #The merger ratio which constitutes a major merger
MIN_MMP_MASS_RATIO=0.3 #The minimum mass for a progenitor to be considered an MMP
MIN_MMP_VMAX_RATIO=0.7 #The minimum vmax for a progenitor to be considered an MMP

PADDING_TIMESTEPS=0 #Don't kill halos if they have link problems during the last 1 timestep.

LAST_DITCH_SEARCH_LIMIT=1.0 #For connecting halos which have "moved" up 
			    #to this amount times their virial radius
LAST_DITCH_VMAX_RATIO_1=1.4  # For connecting halos which have "moved"
			     # unphysical amounts
LAST_DITCH_VMAX_RATIO_2=2.5 
MAX_PHANTOM=$max_phantoms       # max timesteps to keep phantom halo
MAX_PHANTOM_SMALL=$max_phantoms_small # max timesteps to keep small phantom halo
SMALL_PARTICLE_LIMIT=49 # Halos smaller than this size get
			# kept around for less time.
TIDAL_FORCE_LIMIT=0.4
RECURSION_LIMIT=5
METRIC_LIMIT=7
METRIC_BREAK_LIMIT=3.2 #Below which we break a link.
MASS_RES_OK=$mass_res_ok #Halo mass above which there are probably
		 #not resolution issues.

EXTRA_PARAMS = 23
EXTRA_PARAM_LABELS = "Rs_Klypin M$m[0]_all M$m[1] M$m[2] M$m[3] M$m[4] Xoff Voff Spin_Bullock b_to_a c_to_a A[x] A[y] A[z] b_to_a($m[3]) c_to_a($m[3]) A[x]($m[3]) A[y]($m[3]) A[z]($m[3]) T/|U| M_pe_Behroozi M_pe_Diemer Halfmass_Radius"
EXTRA_PARAM_DESCRIPTIONS = "#Rs_Klypin: Scale radius determined using Vmax and Mvir (see Rockstar paper)\\n#M$m[0]_all: Mass enclosed within the specified overdensity, including unbound particles (Msun/h)\\n#M$m[1]--M$m[4]: Mass enclosed within specified overdensities (Msun/h)\\n#Xoff: Offset of density peak from average particle position (kpc/h comoving)\\n#Voff: Offset of density peak from average particle velocity (km/s physical)\\n#Spin_Bullock: Bullock spin parameter (J/(sqrt(2)*MVR))\\n#b_to_a, c_to_a: Ratio of second and third largest shape ellipsoid axes (B and C) to largest shape ellipsoid axis (A) (dimensionless).\\n#  Shapes are determined by the method in Allgood et al. (2006).\\n#  ($m[3]) indicates that only particles within R$m[3] are considered.\\n#A[x],A[y],A[z]: Largest shape ellipsoid axis (kpc/h comoving)\\n#T/|U|: ratio of kinetic to potential energies\\n#M_pe_*: Pseudo-evolution corrected masses (very experimental)\\nHalfmass_Radius: Radius within which 1/2 of M$m[0] is enclosed."
EXTRA_PARAM_INTERPOLATIONS = "cllllllllllllllllllllc"

EOL
    ;
close CONFIG;

print "Merger tree config file generated in $outbase/outputs/merger_tree.cfg\n";
print "\nTo generate a merger tree, change to the consistent_trees directory and run\n";
print "    make\n";
if ($config->{PERIODIC}==0) {
    print "    perl do_merger_tree_np.pl $outbase/outputs/merger_tree.cfg\n";
} else {
    print "    perl do_merger_tree.pl $outbase/outputs/merger_tree.cfg\n";
}
print "\nTrees will be generated in\n";
print "   $outbase/trees\n";
print "\nNote that if $outbase is not accessible from the machine you intend to run the merger tree code on, you will have to change the directories in $outbase/outputs/merger_tree.cfg appropriately.\n";

package ReadConfig;

sub new {
    my ($class, $file) = @_;
    return bless {}, $class unless defined $file;
    open FILE, "<", $file or 
	die "Couldn't open file $file for reading!\n";
    local $_;
    my %config;
    while (<FILE>) {
	s/\#.*//;
	my ($key, $value) = /([^=]+)=(.*)/;
	next unless defined $value;
	#print "$key = $value\n";
	($key, $value) = map { trim($_) } ($key, $value);
	next unless length($key) and length($value);
	$config{$key} = $value;
    }
    close FILE;
    bless \%config, $class;
}

sub trim {
    my $val = shift;
    $val =~ s/^\s*['"]?//;
    $val =~ s/['"]?\s*$//;
    return $val;
}

sub set_defaults {
    my $self = shift;
    my %opts = @_;
    %$self = (%opts, %$self);
}

sub print_config {
    my ($self, $fh) = @_;
    $fh = \*STDOUT unless $fh;
    for (sort keys %$self) {
	print $fh "\"$_\" = \"$self->{$_}\"\n";
    }
}
