#/usr/bin/perl
#
#	calculation of proper motion
#
#	2012-11-08	yamamoto
#

use strict;
use warnings;

use Math::Trig qw(pi);

my $filename;
my @calc_dates;
my $epsilon = 23.42929111;
my $Omega_and_omega = 282.9400;	# degree
my ($star, $RA, $DEC, $pRA, $pDEC, $RA_h, $RA_m, $RA_s, $DEC_d, $DEC_m, $DEC_s, $dist, $switch);
my ($maxJD,$minJD, $temp, $alpha, $delta, $std_alpha, $std_delta, $std_JD);
my $deg2rad = pi()/180;
my ($Year, $Month, $Day, $y, $m, $A, $B, $C, $T, $M, $L_sun_temp, $n, $L, $g);
my @L_sun;
my @JD;
my @i = (1 ... 360);
my %L_sun = (
		"degree"	=> 0,
		"minutes"	=> 0,
		"second"	=> 0,
		);
my $count=0;

########### read params #################################

print "\n###====================================================\n";
print "\n###============read params($ARGV[0])===================\n";
print "\n###====================================================\n";

$switch=0;
while(defined($_ = <>)){
	chomp($_);
	if(/^[a-zA-Z]/){
		/^star:/ and (undef, $star) = split /:\s*/
			and print "#star::".$star."\n" and $switch=1;
		/^RA:/ and (undef, $temp) = split /:\s*/
			and ($RA_h, $RA_m, $RA_s) = split /\s+/, $temp
			and print "#RA=$RA_h:$RA_m:$RA_s\n";
		/^DEC:/ and (undef, $temp) = split /:\s*/
			and ($DEC_d, $DEC_m, $DEC_s) = split /\s+/, $temp
			and print "#DEC=$DEC_d:$DEC_m:$DEC_s\n";
		/^pRA:/ and (undef, $pRA) = split /:\s*/
			and print "#pRA=$pRA\n";
		/^pDEC:/ and (undef, $pDEC) = split /:\s*/
			and print "#pDEC=$pDEC\n";
		/^Dist:/ and (undef, $dist) = split /:\s*/
			and print "#Distance=$dist\n";
	}elsif(/^\d/){
		push @calc_dates, $_;
	}
}

$filename = ${star}."_orbit.dat";
open my $outfile, "> $filename" or die "Can't open file($!)";

########### calc JD #################################
foreach(@calc_dates){
	print "\n###============calc JD($_)===================\n";

	/^\d{4}-\d{2}-\d{2}/ and ($Year, $Month, $Day) = split /-/, $_;

#	printf "Year=%d, Month=%d, Day=%d\n",$Year,$Month,$Day;

	if($Month > 2){
		$y = $Year, $m = $Month;
	}elsif($Month <= 2){
		$y = $Year-1, $m = $Month+12;
	}
	if($Year > 1582 || $Year == 1582 && ($Month > 10) || ($Month == 10 && $Day >= 15)){
		$A = int($y/100);
		$B = 2 - $A + int($A/4);
	}
	if($Year > 0){
		$C = int(365.25 * $y);
	}else{
		$C = int(365.25 * $y - 0.75);
	}
	$JD[$count] = 1720994.5 + int(30.6001 * ($m + 1)) + $B + $C + $Day;
	printf "Year=%d, Month=%d, Day=%d\n => JD=%lf\n",$Year,$Month,$Day,$JD[$count];
	$count++;
}

########### max / min ################################
{
	$minJD = 100000000000000000000000;
	$maxJD = 0;
	foreach(@JD){
		$maxJD < $_ and $maxJD = $_;
		$minJD > $_ and $minJD = $_;
	}
#print $maxJD-$minJD;
	my $i=0;
	while($i < ($maxJD - $minJD)){
		push @JD, ($minJD + $i);
		$i++;
	}
#$i = @JD and print "$i\n";
}

########### calc orbit elem ############################
my $countemp=0;
foreach my $JD  (@JD){

	$T = ($JD - 2451545.0) / 36525.0;
	$M = (357.5256 + 35999.049 * $T) ;
	$L_sun_temp = $Omega_and_omega + $M + 6892/3600 * sin($M*$deg2rad) + 72/3600 * sin(2*$M*$deg2rad);
	push @L_sun , $L_sun_temp % 360 + ($L_sun_temp-int($L_sun_temp));

	$L_sun{"degree"} = int($L_sun[$countemp]);
	$L_sun{"minutes"} = int(($L_sun[$countemp]-$L_sun{"degree"}) * 60);
	$L_sun{"second"} = (($L_sun[$countemp]-$L_sun{"degree"})*60-$L_sun{"minutes"})*60;
	$countemp < $count and print "\n\n###============calc Longitude of Sun at $JD===================\n"
		and printf "T=%lf, M=%lf, L_sum=%lf\n", $T, $M, $L_sun[$countemp]
		and printf "L_sun= %d : %d : %lf\n",$L_sun{"degree"},$L_sun{"minutes"},$L_sun{"second"};

	$countemp++;
}



print "\n\n###============calc proper motion===================\n";

########### calc proper motion  #################################
if($switch==1){
	$RA = ($RA_h + $RA_m / 60 + $RA_s / 3600) * 360 / 24;
	$DEC = ($DEC_d + $DEC_m /60 + $DEC_s / 3600);
	print "Distance:$dist, RA:$RA, DEC:$DEC, obliquity:$epsilon\n";
	print "pRA:$pRA, pDEC:$pDEC, obliquity:$epsilon\n";
	print $outfile "#Distance:$dist, RA:$RA, DEC:$DEC, obliquity:$epsilon\n";
	print $outfile "#pRA:$pRA, pDEC:$pDEC, obliquity:$epsilon\n";
	my ($countemp, $now_alpha, $now_delta, $now_JD)=(0,0,0, 0);
	foreach(@L_sun){
		$alpha = 1/$dist * ( -sin($RA*$deg2rad) * cos($_*$deg2rad)
							+cos($epsilon*$deg2rad) * cos($DEC*$deg2rad) * sin($_*$deg2rad));
		$delta = 1/$dist * ( -sin($DEC*$deg2rad) * cos($RA*$deg2rad) * cos($_*$deg2rad)
							-cos($epsilon*$deg2rad) * sin($DEC*$deg2rad) * sin($RA*$deg2rad) * sin($_*$deg2rad)
							+sin($epsilon*$deg2rad) * cos($DEC*$deg2rad) * sin($_*$deg2rad));
		if($countemp<$count){
			printf "JD= $JD[$countemp], dRA= $alpha, dDEC= $delta\n";
			printf $outfile "#Date=$calc_dates[$countemp], JD= $JD[$countemp], dRA= $alpha, dDEC= $delta\n";
		}else{
			if($countemp==$count){
				$std_alpha=$alpha;
				$std_delta=$delta;
				$std_JD=$JD[$countemp];
			}
			$now_JD = ($JD[$countemp]-$std_JD)/365.25;
			$now_alpha = $alpha-$std_alpha;
			$now_delta = $delta-$std_delta;
			printf $outfile "$JD[$countemp] $alpha $delta " . ($now_alpha+$pRA/1000 * $now_JD) . " " . ($now_delta+$pDEC/1000 * $now_JD) . " " . ($RA+($now_alpha+$pRA/1000 * $now_JD)/3600) . " ". ($DEC+($now_delta+$pDEC/1000 * $now_JD)/3600) . "\n";
		}
		$countemp++;
	}
}

#print @calc_dates;

