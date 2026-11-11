/* vpot_sums.c: helpers for vahc_fast.py (.vpot, mean particle potential around halos).
 *
 * deposit_weighted: CIC deposit of a weight per particle, with exactly the cell and weight arithmetic of
 *   cic_update.c's update_density (so the potential grid and the density grid are built the same way).
 * sphere_sums: for one chunk of particles, add for every halo the potential sum and the particle count
 *   inside two spheres around its centre (radii r1, r2; periodic box). Particles are put in a cell grid
 *   (counting sort); each halo visits only the cells its larger sphere overlaps. Halos are split over
 *   threads (each halo's sums are written by one thread only, so no locks are needed).
 * Build: cc -O3 -fopenmp -shared -fPIC -o vpot_sums.so vpot_sums.c
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void deposit_weighted(long int ndata, int grid, double *field, const double *x, const double *y, const double *z,
                      const double *w, double cell_size_x, double cell_size_y, double cell_size_z)
{
  long int i;
  int grid2 = grid * grid;
  for (i = 0; i < ndata; i++) {
    double cx = x[i] / cell_size_x, cy = y[i] / cell_size_y, cz = z[i] / cell_size_z;
    int ix = (int)cx, iy = (int)cy, iz = (int)cz;
    int c0 = ix % grid, c1 = iy % grid, c2 = iz % grid;
    int cp10 = (c0 + 1) % grid, cp11 = (c1 + 1) % grid, cp12 = (c2 + 1) % grid;
    double dx = fabs(cx - (double)ix), dy = fabs(cy - (double)iy), dz = fabs(cz - (double)iz);
    double tx = 1.0 - dx, ty = 1.0 - dy, tz = 1.0 - dz, wi = w[i];
    field[c0 + grid * c1 + grid2 * c2] += tx * ty * tz * wi;
    field[cp10 + grid * c1 + grid2 * c2] += dx * ty * tz * wi;
    field[c0 + grid * cp11 + grid2 * c2] += tx * dy * tz * wi;
    field[c0 + grid * c1 + grid2 * cp12] += tx * ty * dz * wi;
    field[cp10 + grid * cp11 + grid2 * c2] += dx * dy * tz * wi;
    field[cp10 + grid * c1 + grid2 * cp12] += dx * ty * dz * wi;
    field[c0 + grid * cp11 + grid2 * cp12] += tx * dy * dz * wi;
    field[cp10 + grid * cp11 + grid2 * cp12] += dx * dy * dz * wi;
  }
}

static inline int cell_of(double v, double inv, int nc)
{
  int c = (int)floor(v * inv);
  c %= nc;
  if (c < 0) c += nc;
  return c;
}

static inline double wrap(double d, double L)
{
  if (d > 0.5 * L) d -= L;
  else if (d < -0.5 * L) d += L;
  return d;
}

/* returns 0 on success, 1 if memory could not be allocated */
int sphere_sums(long int np, const double *x, const double *y, const double *z, const double *phi,
                long int nh, const double *hx, const double *hy, const double *hz,
                const double *r1, const double *r2, double L, int nc, int nthreads,
                double *s1, long int *n1, double *s2, long int *n2)
{
  long int ncell = (long int)nc * nc * nc, i;
  double inv = nc / L, csize = L / nc;
  long int *start = calloc(ncell + 1, sizeof(long int));
  long int *fill = NULL;
  int *cell = malloc(np * sizeof(int));
  double *px = malloc(np * sizeof(double)), *py = malloc(np * sizeof(double)), *pz = malloc(np * sizeof(double));
  double *pp = malloc(np * sizeof(double));
  if (!start || !cell || !px || !py || !pz || !pp) goto fail;
  for (i = 0; i < np; i++) {
    int c = (cell_of(x[i], inv, nc) * nc + cell_of(y[i], inv, nc)) * nc + cell_of(z[i], inv, nc);
    cell[i] = c;
    start[c + 1]++;
  }
  for (i = 0; i < ncell; i++) start[i + 1] += start[i];
  fill = malloc(ncell * sizeof(long int));
  if (!fill) goto fail;
  memcpy(fill, start, ncell * sizeof(long int));
  for (i = 0; i < np; i++) {
    long int k = fill[cell[i]]++;
    px[k] = x[i]; py[k] = y[i]; pz[k] = z[i]; pp[k] = phi[i];
  }
  free(fill); fill = NULL;
  free(cell); cell = NULL;

#ifdef _OPENMP
  if (nthreads > 0) omp_set_num_threads(nthreads);
#endif
#pragma omp parallel for schedule(dynamic, 256)
  for (long int h = 0; h < nh; h++) {
    double R = r1[h] > r2[h] ? r1[h] : r2[h];
    if (!(R > 0)) continue;
    double R1 = r1[h] * r1[h], R2 = r2[h] * r2[h];
    double cx = hx[h], cy = hy[h], cz = hz[h];
    int lo[3], nn[3];
    double cc[3] = {cx, cy, cz};
    for (int d = 0; d < 3; d++) {
      int a = (int)floor((cc[d] - R) * inv), b = (int)floor((cc[d] + R) * inv);
      nn[d] = b - a + 1;
      if (nn[d] > nc) nn[d] = nc;
      lo[d] = a;
    }
    double a1 = 0, a2 = 0;
    long int m1 = 0, m2 = 0;
    for (int ia = 0; ia < nn[0]; ia++) {
      int gx = ((lo[0] + ia) % nc + nc) % nc;
      for (int ib = 0; ib < nn[1]; ib++) {
        int gy = ((lo[1] + ib) % nc + nc) % nc;
        for (int ic = 0; ic < nn[2]; ic++) {
          int gz = ((lo[2] + ic) % nc + nc) % nc;
          long int c = ((long int)gx * nc + gy) * nc + gz;
          for (long int k = start[c]; k < start[c + 1]; k++) {
            double dx = wrap(px[k] - cx, L), dy = wrap(py[k] - cy, L), dz = wrap(pz[k] - cz, L);
            double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 <= R1) { a1 += pp[k]; m1++; }
            if (d2 <= R2) { a2 += pp[k]; m2++; }
          }
        }
      }
    }
    s1[h] += a1; n1[h] += m1; s2[h] += a2; n2[h] += m2;
  }
  (void)csize;
  free(start); free(px); free(py); free(pz); free(pp);
  return 0;
fail:
  free(start); free(fill); free(cell); free(px); free(py); free(pz); free(pp);
  return 1;
}
