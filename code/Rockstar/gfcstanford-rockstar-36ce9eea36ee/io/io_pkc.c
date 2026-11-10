/* FILE_FORMAT = "PKC": Rockstar reads pkdgrav3 compressed snapshots (snapcomp format version 3)
 * directly. Decoding is in pkc_core.c; this file sets Rockstar's globals and fills its particle array.
 * Added for Sahyadri (2026-11-10); design: sahyadri-sandbox docs/rockstar_pkc_reader_plan.md.
 *
 * Conventions (identical, bit for bit, to Rockstar reading pkc_to_gadget output):
 *   positions  (X + 0.5) / (ncell*sub) * box_size, Mpc/h in [0, L)
 *   velocities peculiar km/s: float(V) * float(vel_step/sqrt(a)), then * sqrt(a) (as io_gadget.c)
 *   mass       27.7536627 * Om * L^3 / N * 1e10 Msun/h, N = particles in the selected subsamples
 *   cosmology  Om = omega_m, Ol = omega_l + omega_de, h0 = h, W0 = w0, WA = wa (from the header,
 *              overriding the config, as the Gadget reader does for Om, Ol, h0) */
#ifdef ENABLE_PKC

#include <ctype.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../check_syscalls.h"
#include "../config_vars.h"
#include "../config.h"
#include "../hubble.h"
#include "../particle.h"
#include "../universal_constants.h"
#include "io_pkc.h"
#include "pkc_core.h"

static void pkc_die(const char *msg) {
  fprintf(stderr, "[Error] PKC reader: %s\n", msg);
  exit(1);
}

/* INBASE/snap.00100.7 -> prefix INBASE/snap.00100, block 7 */
static int64_t split_block(const char *filename, char *prefix, size_t len) {
  const char *dot = strrchr(filename, '.');
  const char *q;
  int digits = (dot && dot[1]);
  if (strlen(filename) >= len) pkc_die("file name too long");
  if (digits) for (q = dot + 1; *q; q++) if (!isdigit((unsigned char)*q)) { digits = 0; break; }
  if (digits) {
    memcpy(prefix, filename, (size_t)(dot - filename));
    prefix[dot - filename] = 0;
    return strtoll(dot + 1, NULL, 10);
  }
  if (NUM_BLOCKS != 1) {
    char msg[2048];
    snprintf(msg, sizeof(msg), "%s has no block number; use FILENAME = \"<prefix>.<snap>.<block>\" "
             "(e.g. \"snap.<snap>.<block>\") when NUM_BLOCKS > 1", filename);
    pkc_die(msg);
  }
  strcpy(prefix, filename);
  return 0;
}

void load_particles_pkc(char *filename, struct particle **p, int64_t *num_p) {
  static int announced = 0;
  char err[PKC_ERRLEN], prefix[4096], msg[PKC_ERRLEN + 256];
  double want[PKC_MAX_SUBSAMPLES];
  int n_want = 0;
  int64_t block, first, count, i;
  uint64_t np, j;
  struct pkc_set s;
  struct pkc_decoder d;
  struct pkc_block b;
  const struct pkc_header *h;
  double Om_h, Ol_h, h0_h, W0_h, WA_h, Q, L, vel_rescale, Ez;
  float vfac;
  struct particle *out;

  if (LIGHTCONE) pkc_die("LIGHTCONE is not supported for FILE_FORMAT = \"PKC\"");
  if (strlen(BLOCK_NAMES)) pkc_die("BLOCK_NAMES must be empty for FILE_FORMAT = \"PKC\" (blocks are shares of the snapshot)");
  block = split_block(filename, prefix, sizeof(prefix));
  if (block < 0 || block >= NUM_BLOCKS) {
    snprintf(msg, sizeof(msg), "block %" PRId64 " out of range (NUM_BLOCKS = %" PRId64 ")", block, NUM_BLOCKS);
    pkc_die(msg);
  }
  if (strlen(PKC_SUBSAMPLES)) {
    n_want = pkc_parse_percents(PKC_SUBSAMPLES, want, PKC_MAX_SUBSAMPLES);
    if (n_want <= 0) pkc_die("cannot parse PKC_SUBSAMPLES (expected a list like \"1,3\")");
  }

  if (pkc_set_open(prefix, want, n_want, &s, err)) pkc_die(err);
  h = &s.files[0].h;
  if (s.n_pairs < NUM_BLOCKS) {
    snprintf(msg, sizeof(msg), "%s has %" PRId64 " pkc blocks in the selected subsamples, fewer than NUM_BLOCKS = %"
             PRId64 "; lower NUM_BLOCKS", prefix, s.n_pairs, NUM_BLOCKS);
    pkc_die(msg);
  }

  /* globals from the header (every block of a snapshot gives the same values) */
  if (pkc_cosmology(h, &Om_h, &Ol_h, &h0_h, &W0_h, &WA_h, err)) pkc_die(err);
  if (!announced && (Om != Om_h || Ol != Ol_h || h0 != h0_h || W0 != W0_h || WA != WA_h))
    fprintf(stderr, "[Info] PKC reader: cosmology from the snapshot header overrides the config: "
            "Om %g -> %g, Ol %g -> %g, h0 %g -> %g, W0 %g -> %g, WA %g -> %g\n",
            Om, Om_h, Ol, Ol_h, h0, h0_h, W0, W0_h, WA, WA_h);
  Om = Om_h; Ol = Ol_h; h0 = h0_h; W0 = W0_h; WA = WA_h;
  SCALE_NOW = 1.0 / (1.0 + h->redshift);
  BOX_SIZE = h->box_size;
  TOTAL_PARTICLES = (int64_t)s.n_selected;
  L = h->box_size;
  if (!RESCALE_PARTICLE_MASS)
    PARTICLE_MASS = 27.7536627 * Om * L * L * L / (double)s.n_selected * 1e10; /* as pkc_to_gadget + io_gadget.c */
  else
    PARTICLE_MASS = Om * CRITICAL_DENSITY * pow(BOX_SIZE, 3) / TOTAL_PARTICLES;
  AVG_PARTICLE_SPACING = cbrt(PARTICLE_MASS / (Om * CRITICAL_DENSITY));

  /* checks of the background against pkdgrav3's own E(z) */
  if (!announced) {
    Ez = pkc_info_E_z(h);
    if (h->omega_rad > 0)
      fprintf(stderr, "[Warning] PKC reader: Omega_rad = %g is treated as curvature by Rockstar (relative error in "
              "H^2 about %.1e at z = %g)\n", h->omega_rad,
              h->omega_rad * pow(1 + h->redshift, 2) / (Om * (1 + h->redshift)), h->redshift);
    if (Ez > 0) {
      double rel = fabs(hubble_scaling(h->redshift) / Ez - 1.0);
      if (rel > 1e-2) {
        snprintf(msg, sizeof(msg), "Rockstar's H(z)/H0 = %.8g differs from pkdgrav3's E_z = %.8g by %.2e",
                 hubble_scaling(h->redshift), Ez, rel);
        pkc_die(msg);
      }
      if (rel > 1e-4)
        fprintf(stderr, "[Warning] PKC reader: Rockstar's H(z)/H0 = %.8g, pkdgrav3's E_z = %.8g (rel. diff. %.2e)\n",
                hubble_scaling(h->redshift), Ez, rel);
    }
    fprintf(stderr, "[Info] PKC reader: %s: z = %g, L = %g Mpc/h, %" PRIu64 " of %" PRIu64 " particles (%s), "
            "%d files, %" PRId64 " pkc blocks, m_p = %.6e Msun/h, Om %g Ol %g h0 %g W0 %g WA %g\n",
            prefix, h->redshift, L, s.n_selected, h->n_total, strlen(PKC_SUBSAMPLES) ? PKC_SUBSAMPLES : "all subsamples",
            s.n_files, s.n_pairs, PARTICLE_MASS, Om, Ol, h0, W0, WA);
    announced = 1;
  }

  /* this block's share of the (file, pkc block) pairs; one allocation, then decode one pkc block at a time */
  if (pkc_set_range(&s, block, NUM_BLOCKS, &first, &count, &np, err)) pkc_die(err);
  check_realloc_s(*p, (*num_p) + (int64_t)np, sizeof(struct particle));
  out = *p + *num_p;

  Q = (double)((uint64_t)h->ncell * h->sub);
  vel_rescale = sqrt(SCALE_NOW);
  vfac = (float)(h->vel_step / sqrt(SCALE_NOW));
  pkc_decoder_init(&d);
  memset(&b, 0, sizeof(b));
  for (i = first, j = 0; i < first + count; i++) {
    const struct pkc_pair *pr = s.pairs + i;
    uint32_t k;
    int dim;
    if (pkc_read_block(s.files + pr->file, pr->block, &d, &b, err)) pkc_die(err);
    for (k = 0; k < b.n; k++, j++) {
      struct particle *q = out + j;
      q->id = (int64_t)b.id[k];
      for (dim = 0; dim < 3; dim++) {
        float vf = (float)b.V[3 * k + dim] * vfac;
        q->pos[dim] = (float)(((double)b.X[3 * k + dim] + 0.5) / Q * L);
        q->pos[dim + 3] = (float)((double)vf * vel_rescale);
      }
    }
  }
  if (j != np) pkc_die("decoded particle count differs from the index");
  *num_p += (int64_t)np;
  pkc_block_free(&b);
  pkc_decoder_free(&d);
  pkc_set_close(&s);
}

#endif /* ENABLE_PKC */
