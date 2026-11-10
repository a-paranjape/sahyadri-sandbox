#include <math.h>
#include <inttypes.h>
#include "universe_time.h"

#define STEPS 1024

double omega_0 = 0.3; // Matter density
double t_0 = 0;       // Time now (in Hubble time units).
double times[STEPS+1]={0};
double H_CONV = 9.77813952e9/0.7;
static double exact_t0_conv = 0;

/* Sahyadri (2026-11-10): general background, see universe_time.h */
static int de_set = 0;
static double de_Ol = 0, de_w0 = -1, de_wa = 0;
static double de_Om = 0.3; /* set by init_time_table */

void set_dark_energy(double Ol, double w0, double wa) {
  de_set = 1; de_Ol = Ol; de_w0 = w0; de_wa = wa;
}

int dark_energy_is_flat_lcdm(double Om) {
  return (!de_set || (fabs(Om + de_Ol - 1.0) < 1e-6 && de_w0 == -1.0 && de_wa == 0.0));
}

double hubble_E(double a, double Om) {
  double Ol = de_set ? de_Ol : 1.0 - Om;
  double weff = (a != 1.0 && de_set) ? de_w0 + de_wa - de_wa*(a - 1.0)/log(a) : (de_set ? de_w0 : -1.0);
  return sqrt(Om/(a*a*a) + (1.0 - Om - Ol)/(a*a) + Ol*pow(a, -3.0*(1.0 + weff)));
}

/* t(a) = int_0^a da'/(a' E(a')) in units of 1/H0; with a' = s^2 the integrand 2/(s E(s^2)) is smooth */
static double integrated_time(double a) {
  const int n = 4096; /* even */
  double s1 = sqrt(a), ds = s1/n, sum = 0;
  int i;
  for (i = 1; i <= n; i++) {
    double s = i*ds, f = 2.0/(s*hubble_E(s*s, de_Om));
    sum += (i == n) ? f : ((i % 2) ? 4*f : 2*f);
  }
  return sum*ds/3.0; /* the s = 0 end contributes 0 */
}


void init_time_table(double Om, double h0) {
  double a = 1;
  int64_t i;
  omega_0 = Om;
  de_Om = Om;
  H_CONV = 9.77813952e9/h0; // 1/(1 km/s/Mpc) in years
  if (!dark_energy_is_flat_lcdm(Om)) { /* Sahyadri: numerical table for w0-wa / curvature */
    exact_t0_conv = integrated_time(1.0);
    times[STEPS] = 0;
    for (i=1; i<=STEPS; i++) {
      a = 1.0-(double)i/((double)STEPS);
      times[STEPS-i] = (a > 0) ? integrated_time(a) - exact_t0_conv : -exact_t0_conv;
    }
    return;
  }

  times[STEPS] = t_0;
  exact_t0_conv = exact_scale_to_time(1.0);
  for (i=1; i<=STEPS; i++) {
    a = 1.0-(double)i/((double)STEPS);
    times[STEPS-i] = exact_scale_to_time(a);
  }
}

double _exact_time_to_scale(double t) {
  double m = sinh(1.5*t*sqrt(1.0-omega_0));
  return pow(omega_0*m*m/(1.0-omega_0), 1.0/3.0);
}

double exact_scale_to_time(double scale) {
  if (!dark_energy_is_flat_lcdm(omega_0)) return integrated_time(scale) - exact_t0_conv; /* Sahyadri */
  double t = scale;
  double a = _exact_time_to_scale(t);
  double dt = scale/10.0;
  int64_t count = 0;
  while (fabs(a-scale)>1e-7 && (count < 10)) {
    count++;
    double a2 = _exact_time_to_scale(t+dt);
    double move = (scale-a)*(dt)/(a2-a);
    t += move;
    a = _exact_time_to_scale(t);
    if (move/10.0 < 0.5*dt) dt = move/10.0;
    else dt /= 2.0;
  }
  return t-exact_t0_conv;
}


// Linearly interpolate between calculated values.
double scale_to_time(double scale) {
  double s = scale;
  int64_t l = (int)(s*STEPS);
  double f = s*STEPS - l;
  if (scale > 1) return exact_scale_to_time(scale);
  if (scale < 0) return times[0];
  return (times[l]+f*(times[l+1]-times[l]));
}

double scale_to_years(double scale) {
  return (scale_to_time(scale)*H_CONV);
}
