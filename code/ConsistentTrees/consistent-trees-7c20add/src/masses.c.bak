#include <stdio.h>
#include <math.h>
#include <math.h>
#include "masses.h"
#include "universal_constants.h"

#define FF_CACHE_SIZE 10001
#define HALO_PROFILE_CUTOFF 100 /* maximum value of rh/rs cached */
#define FF_CACHE_STEP (((double)HALO_PROFILE_CUTOFF)/ \
		       ((double)FF_CACHE_SIZE-1.0))
#define FF_INV_CACHE_STEP (((double)FF_CACHE_SIZE-1.0)/ \
			   ((double)HALO_PROFILE_CUTOFF))

float ff_cache[FF_CACHE_SIZE] = {0};

//In terms of rs / rvir
inline double filling_fraction(double x) {
  return (log1p(1.0/x) - 1.0/(1.0+x));
}

inline double kravtsov_f(double x) {
  return (x*x*x*filling_fraction(x));
}

double inv_ff(double f) {
  double c = 1.0/10.0;
  double tc = kravtsov_f(c);
  double new_c;
  while (fabs((f-tc) / f) > 1e-7) {
    double tc2 = kravtsov_f(c+0.1);
    double slope = (tc2 - tc) / 0.1;
    new_c = c + (f-tc)/slope;
    if (new_c < 0) c/=2;
    else c = new_c;
    tc = kravtsov_f(c);
  }
  return (c);
}

double delta_vir(double a) {
  double x = 1.0/(1.0+a*a*a*Ol/Om)-1.0;
  return ((18*M_PI*M_PI + 82.0*x - 39*x*x)/(1.0+x));
}

double concentration(double mvir, double scale) {
  if (!(mvir>0)) return 0;
  double logm = log10(mvir);
  return (pow(10.0, 2.2358 - 0.10*logm)*scale);
}


/*See Appendix C of http://arxiv.org/abs/astro-ph/0203169 for mass conversions*/
double calculate_mass_factor(double mvir, double rvir, double rs)
{
  /* mvir = 4pi rho_s rvir^3 * (rs/rvir)^3 * filling_fraction(rs/rvir) */
  /* mass_factor = 4pi rho_s rs^3 = mvir / filling_fraction(rs/rvir) */
  /* Hence, m(r) = mass_factor * filling_fraction(rs/r) */
  double invc = rs / rvir;
  return (mvir / filling_fraction(invc));
}

void gen_ff_cache(void) {
  int i;
  double f;
  for (i=0; i<FF_CACHE_SIZE; i++) {
    f = FF_CACHE_STEP*i;
    ff_cache[i] = i ? filling_fraction(1.0/f) : 0;
  }
}

/* Returns the filling fraction as a function of r_h / r_s */
float ff_cached(float x) {
  int i;
  x *= FF_INV_CACHE_STEP;
  if (!(x>=0)) return 0;
  if (x>=FF_CACHE_SIZE-1) return ff_cache[FF_CACHE_SIZE-1];
  i = x;
  x -= i;
  return (ff_cache[i] + x*(ff_cache[i+1]-ff_cache[i]));
}





//Scale radius calculation stuff
inline double c_to_f(double c) {
  double cp1 = 1.0+c;
  return (c*cp1 / (log1p(c)*cp1 - c));
}

double f_to_c(double f) {
  double c = f;
  double tc = c_to_f(c);
  double new_c;
  while (fabs((f-tc) / f) > 1e-7) {
    double tc2 = c_to_f(c+0.1);
    double slope = (tc2 - tc) / 0.1;
    new_c = c + (f-tc)/slope;
    if (new_c < 0) c/=2;
    else c = new_c;
    tc = c_to_f(c);
  }
  return c;
}

float calc_scale_radius(float mvir, float rvir, float vmax, float rvmax, float scale)
{
  float f, c, vm2;
  if (!mvir || !rvir || !vmax) return (rvmax / RMAX_TO_RS);
  vm2 = vmax/VMAX_CONST;
  vm2 = vm2*vm2 * scale;
  f = (rvir/1.0e3)*vm2 / (mvir*RS_CONSTANT);
  if (f < 4.625) return (rvmax / RMAX_TO_RS);
  c = f_to_c(f);
  if (c <= 0) return (rvmax / RMAX_TO_RS);
  return (rvir/c);
}


double ConvertDeltaPcToDeltaPm(float delta, float z) {
  return ((delta * (Om*pow(1+z,3) + (1.0-Om)))/(Om*pow(1+z, 3)));
}

void convert_mvir_to_delta(float mvir, float rvir, float rs, float scale,
			   float delta, char type,
			   float *mdelta, float *rdelta) {
  float dvir = delta_vir(scale);
  float c = (rs > 0) ? (rvir / rs) : concentration(mvir, scale);
  if (type == 'v') { *mdelta = mvir; *rdelta = rvir; return; }
  if (type == 'c') { delta = ConvertDeltaPcToDeltaPm(delta, 1.0/scale - 1.0); }
  float r_frac = inv_ff(delta/dvir * kravtsov_f(1.0/c));
  *rdelta = rvir / (r_frac * c);
  *mdelta = mvir * (delta / dvir) * pow(r_frac*c, -3);
}
