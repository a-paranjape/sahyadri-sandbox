/* halo_pair_sums.c: weighted halo-halo pair sums for halo_potential_xcorr.py.
 *
 * For halos i of set A and j of set B at periodic separation d in shells [edges[b], edges[b+1]), b in [i0, i1),
 * and K weight columns phi[.,k] (nan = undefined for that halo), accumulates per jackknife region of i:
 *   npair[reg][b]           pairs (all columns)
 *   dsum[reg][b]            sum of d
 *   s[reg][b][k][0..5]      Nk = #pairs with phi_i,k and phi_j,k defined, sum phi_i, sum phi_j, sum phi_i phi_j,
 *                           sum phi_i^2, sum phi_j^2 (over those pairs)
 * Per pair the cost is 1 + 3K additions to per-halo accumulators (count of defined j, sum phi_j, sum phi_j^2 per
 * column); they are folded with phi_i once per halo i and shell.
 * B is put in a cell grid (counting sort, nc^3 cells); each halo i visits the cells within edges[i1] of it.
 * Pairs with the same id (A = B) are skipped. Threads split A; each has its own accumulators (no locks).
 * Memory: B copy nb * (4 + K) doubles + nc^3 + 1 longs; per thread nreg * nbin * (2 + 6K) doubles.
 * Build: cc -O3 -fopenmp -shared -fPIC -o halo_pair_sums.so halo_pair_sums.c
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

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

/* returns 0 on success, 1 on allocation failure */
int pair_sums(long na, const double *ax, const double *ay, const double *az, const double *aphi, const int *areg,
              const long *aid,
              long nb, const double *bx, const double *by, const double *bz, const double *bphi, const long *bid,
              int K, double L, const double *edges, int nbin, int i0, int i1, int nc, int nreg, int nthreads,
              double *npair, double *dsum, double *s)
{
  long ncell = (long)nc * nc * nc, i;
  double inv = nc / L;
  double rlo = edges[i0], rhi = edges[i1], rlo2 = rlo * rlo, rhi2 = rhi * rhi;
  int nl = i1 - i0;
  long *start = calloc(ncell + 1, sizeof(long)), *fill = NULL;
  int *cell = malloc((nb > 0 ? nb : 1) * sizeof(int));
  double *px = malloc((nb > 0 ? nb : 1) * sizeof(double)), *py = malloc((nb > 0 ? nb : 1) * sizeof(double));
  double *pz = malloc((nb > 0 ? nb : 1) * sizeof(double)), *pp = malloc((nb > 0 ? nb : 1) * K * sizeof(double));
  long *pid = malloc((nb > 0 ? nb : 1) * sizeof(long));
  if (!start || !cell || !px || !py || !pz || !pp || !pid) goto fail;
  for (i = 0; i < nb; i++) {
    int c = (cell_of(bx[i], inv, nc) * nc + cell_of(by[i], inv, nc)) * nc + cell_of(bz[i], inv, nc);
    cell[i] = c;
    start[c + 1]++;
  }
  for (i = 0; i < ncell; i++) start[i + 1] += start[i];
  fill = malloc((ncell > 0 ? ncell : 1) * sizeof(long));
  if (!fill) goto fail;
  memcpy(fill, start, ncell * sizeof(long));
  for (i = 0; i < nb; i++) {
    long k = fill[cell[i]]++;
    px[k] = bx[i]; py[k] = by[i]; pz[k] = bz[i]; pid[k] = bid[i];
    memcpy(pp + k * K, bphi + i * K, K * sizeof(double));
  }
  free(fill); fill = NULL;
  free(cell); cell = NULL;

  int nt = 1;
#ifdef _OPENMP
  if (nthreads > 0) omp_set_num_threads(nthreads);
  nt = omp_get_max_threads();
#endif
  long sz_n = (long)nreg * nbin, sz_s = sz_n * K * 6;
  double *tn = calloc((size_t)nt * sz_n, sizeof(double)), *td = calloc((size_t)nt * sz_n, sizeof(double));
  double *ts = calloc((size_t)nt * sz_s, sizeof(double));
  int stride = 1 + 3 * K;  /* per-halo accumulator per shell: n, then per column: cnt, sum, sum2 */
  double *loc = malloc((size_t)nt * (nl * (stride + 1)) * sizeof(double));
  if (!tn || !td || !ts || !loc) { free(tn); free(td); free(ts); free(loc); goto fail; }

#pragma omp parallel
  {
    int t = 0;
#ifdef _OPENMP
    t = omp_get_thread_num();
#endif
    double *myn = tn + (size_t)t * sz_n, *myd = td + (size_t)t * sz_n, *mys = ts + (size_t)t * sz_s;
    double *acc = loc + (size_t)t * nl * (stride + 1);   /* [nl][stride], then dsum [nl] */
    double *accd = acc + (size_t)nl * stride;
#pragma omp for schedule(dynamic, 64)
    for (long h = 0; h < na; h++) {
      memset(acc, 0, (size_t)nl * (stride + 1) * sizeof(double));
      double cx = ax[h], cy = ay[h], cz = az[h];
      long idh = aid[h];
      int lo[3], nn[3];
      double cc[3] = {cx, cy, cz};
      for (int d = 0; d < 3; d++) {
        int a = (int)floor((cc[d] - rhi) * inv), b = (int)floor((cc[d] + rhi) * inv);
        nn[d] = b - a + 1;
        if (nn[d] > nc) nn[d] = nc;
        lo[d] = a;
      }
      for (int ia = 0; ia < nn[0]; ia++) {
        int gx = ((lo[0] + ia) % nc + nc) % nc;
        for (int ib = 0; ib < nn[1]; ib++) {
          int gy = ((lo[1] + ib) % nc + nc) % nc;
          for (int ic = 0; ic < nn[2]; ic++) {
            int gz = ((lo[2] + ic) % nc + nc) % nc;
            long c = ((long)gx * nc + gy) * nc + gz;
            for (long k = start[c]; k < start[c + 1]; k++) {
              double dx = wrap(px[k] - cx, L), dy = wrap(py[k] - cy, L), dz = wrap(pz[k] - cz, L);
              double d2 = dx * dx + dy * dy + dz * dz;
              if (d2 < rlo2 || d2 >= rhi2 || pid[k] == idh) continue;
              double d = sqrt(d2);
              /* shell by bisection on the edges (exact for linear and log bins alike) */
              int a = i0, b = i1;
              while (b - a > 1) { int m = (a + b) >> 1; if (d >= edges[m]) a = m; else b = m; }
              double *q = acc + (size_t)(a - i0) * stride;
              const double *w = pp + k * K;
              q[0] += 1.0;
              accd[a - i0] += d;
              for (int j = 0; j < K; j++) {
                double v = w[j];
                if (v == v) { q[1 + 3 * j] += 1.0; q[2 + 3 * j] += v; q[3 + 3 * j] += v * v; }
              }
            }
          }
        }
      }
      int r = areg[h];
      const double *wi = aphi + h * K;
      for (int bb = 0; bb < nl; bb++) {
        double *q = acc + (size_t)bb * stride;
        if (q[0] == 0) continue;
        long o = (long)r * nbin + i0 + bb;
        myn[o] += q[0];
        myd[o] += accd[bb];
        for (int j = 0; j < K; j++) {
          double v = wi[j];
          if (!(v == v) || q[1 + 3 * j] == 0) continue;
          double c = q[1 + 3 * j], s1 = q[2 + 3 * j], s2 = q[3 + 3 * j];
          double *o6 = mys + (o * K + j) * 6;
          o6[0] += c; o6[1] += v * c; o6[2] += s1; o6[3] += v * s1; o6[4] += v * v * c; o6[5] += s2;
        }
      }
    }
  }
  for (int t = 0; t < nt; t++) {
    for (long o = 0; o < sz_n; o++) { npair[o] += tn[(size_t)t * sz_n + o]; dsum[o] += td[(size_t)t * sz_n + o]; }
    for (long o = 0; o < sz_s; o++) s[o] += ts[(size_t)t * sz_s + o];
  }
  free(tn); free(td); free(ts); free(loc);
  free(start); free(px); free(py); free(pz); free(pp); free(pid);
  return 0;
fail:
  free(start); free(fill); free(cell); free(px); free(py); free(pz); free(pp); free(pid);
  return 1;
}
