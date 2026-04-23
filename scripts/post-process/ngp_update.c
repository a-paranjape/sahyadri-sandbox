#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void update_density(long int ndata, int grid, double *density,
                    double *x, double *y, double *z,
                    double *weights, double cell_size)
{
  long int i;
  int ix, iy, iz;
  int c0, c1, c2;
  int grid2 = grid * grid;

  for(i = 0; i < ndata; i++) {

    ix = (int) floor(x[i] / cell_size);
    iy = (int) floor(y[i] / cell_size);
    iz = (int) floor(z[i] / cell_size);

    c0 = ix % grid; if (c0 < 0) c0 += grid;
    c1 = iy % grid; if (c1 < 0) c1 += grid;
    c2 = iz % grid; if (c2 < 0) c2 += grid;

    density[c0 + grid*c1 + grid2*c2] += weights[i];
  }

  return;
}
