#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "universe_time.h"
#include "masses.h"
double Om = 0.3137721, Ol = 0.6862279;
int main(int argc, char **argv) {
  int mode = argc > 1 ? atoi(argv[1]) : 0;
  double a;
#ifdef NEWCODE
  if (mode == 1) set_dark_energy(Ol, -1.0 + 1e-12, 0); /* forces the numerical path, same background */
  if (mode == 2) set_dark_energy(Ol, -0.9, 0.1);
  if (mode == 3) set_dark_energy(Ol, -1.0, 0);         /* flat LCDM explicitly: closed form */
#endif
  init_time_table(Om, 0.6736);
  for (a = 0.1; a <= 1.0001; a += 0.15)
    printf("%.2f %.12e %.12e\n", a, scale_to_years(a), delta_vir(a));
  return 0;
}
