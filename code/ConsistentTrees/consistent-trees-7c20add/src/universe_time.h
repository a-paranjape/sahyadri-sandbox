#ifndef __UNIVERSE_TIME_H__
#define __UNIVERSE_TIME_H__

void init_time_table(double Om, double h0);
double scale_to_time(double scale);
double scale_to_years(double scale);
double _exact_time_to_scale(double t);
double exact_scale_to_time(double scale);

/* Added for Sahyadri (2026-11-10): dark energy with w(a) = w0 + wa (1 - a) and curvature
 * Ok = 1 - Om - Ol, the same background as our Rockstar (hubble.c). Call before init_time_table().
 * Without a call, or for flat LCDM (|Om + Ol - 1| < 1e-6, w0 = -1, wa = 0), the original closed-form
 * flat-LCDM code is used unchanged. */
void set_dark_energy(double Ol, double w0, double wa);
int dark_energy_is_flat_lcdm(double Om);
double hubble_E(double a, double Om); /* H(a)/H0 */

#endif
