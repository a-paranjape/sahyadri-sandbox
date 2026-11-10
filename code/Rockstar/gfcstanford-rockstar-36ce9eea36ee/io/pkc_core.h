/* pkc_core: C99 reader for pkdgrav3 compressed snapshots (snapcomp format version 3).
 *
 * Written from the format spec (pkdgrav3 docs/snapcomp_format.md); no Rockstar or pkdgrav3 headers,
 * so it can be tested alone. Only version 3 files are accepted. Depends on zstd and POSIX.
 *
 * A snapshot is a set of files <prefix>.s<pct>.<writer>.pkc. pkc_set_open() opens the files of the
 * selected subsamples, checks them, and lists every (file, block) pair in a fixed order (subsample,
 * writer, index order). pkc_set_range() gives the pairs of share k out of nshare, by particle count.
 *
 * Added for Sahyadri (2026-11-10); see docs/rockstar_pkc_reader_plan.md in sahyadri-sandbox.
 */
#ifndef PKC_CORE_H
#define PKC_CORE_H
#ifdef ENABLE_PKC

#include <stddef.h>
#include <stdint.h>

#define PKC_MAX_SUBSAMPLES 16
#define PKC_ERRLEN 512

struct pkc_header {
  uint32_t version;
  uint64_t n_particles, n_blocks, lattice_n, n_total;
  uint32_t ncell, sub;
  double vel_step, vel_to_kms, box_size, redshift, time, omega_m, omega_l, h;
  uint32_t subsample, n_subsamples;
  double percents[PKC_MAX_SUBSAMPLES];
  uint64_t subsample_seed;
  uint32_t writer, n_writers, block_size;
  int32_t zstd_level;
  uint64_t step;
  /* version 3 cosmology block */
  double omega_b, omega_rad, omega_de, w0, wa, sigma8, n_s, running, normalization, pivot;
  double pot_step, pot_to_kms2; /* pot_step > 0: blocks carry a potential stream */
  char info[4000 - 288];        /* NUL-terminated */
};

struct pkc_index_entry {
  uint64_t offset, size;
  uint32_t n, thread;
};

struct pkc_file {
  char *path;
  int fd;
  uint64_t file_size;
  struct pkc_header h;
  struct pkc_index_entry *index; /* h.n_blocks entries */
};

/* One (file, block) pair and the number of selected particles before it. */
struct pkc_pair {
  int32_t file;
  uint64_t block;
  uint64_t start;
};

struct pkc_set {
  int n_files;              /* selected files, sorted by (subsample, writer) */
  struct pkc_file *files;
  int64_t n_pairs;
  struct pkc_pair *pairs;
  uint64_t n_selected;      /* particles in the selected subsamples */
  int all_subsamples;       /* 1 if every subsample is selected */
};

/* Decoded block: n particles in block order. X: quantized positions in [0, ncell*sub) per axis;
 * V: velocities in units of vel_step; id: particle IDs. Buffers are reused between calls. */
struct pkc_block {
  uint32_t n;
  int id_coder;
  int64_t *X, *V;
  uint64_t *id;
  size_t cap;
};

/* Per-thread decoder scratch (zstd context and buffers). */
struct pkc_decoder {
  void *dctx;
  uint8_t *raw;     size_t raw_cap;    /* compressed block */
  uint8_t *planes;  size_t planes_cap; /* one decompressed stream */
  uint64_t *st[12]; size_t st_cap;     /* decoded streams */
};

/* All functions return 0 on success and -1 on error, with a message in err (PKC_ERRLEN bytes). */
int pkc_file_open(const char *path, struct pkc_file *f, char *err);
void pkc_file_close(struct pkc_file *f);

/* want_pct: percents of the subsamples to read (matched to percents[subsample] within 1e-9);
 * n_want = 0 selects all. Checks every file (trailer, index, header consistency, completeness). */
int pkc_set_open(const char *prefix, const double *want_pct, int n_want, struct pkc_set *s, char *err);
void pkc_set_close(struct pkc_set *s);

/* Pairs of share k (0 <= k < nshare): those whose first particle falls in
 * [k*T/nshare, (k+1)*T/nshare), T = n_selected. Sets *first, *count (pair indices) and *np. */
int pkc_set_range(const struct pkc_set *s, int64_t k, int64_t nshare, int64_t *first, int64_t *count,
                  uint64_t *np, char *err);

void pkc_decoder_init(struct pkc_decoder *d);
void pkc_decoder_free(struct pkc_decoder *d);
void pkc_block_free(struct pkc_block *b);
int pkc_read_block(const struct pkc_file *f, uint64_t i, struct pkc_decoder *d, struct pkc_block *b,
                   char *err);

/* Rockstar-side cosmology from a version-3 header: Om, Ol = omega_l + omega_de, h0, W0, WA.
 * Error if both omega_l and omega_de are > 0 with (w0, wa) != (-1, 0) (Rockstar has one DE component). */
int pkc_cosmology(const struct pkc_header *h, double *Om, double *Ol, double *h0, double *W0, double *WA,
                  char *err);

/* E(z) = H(z)/H0 stored by pkdgrav3 in the info JSON ("E_z"), or -1 if absent. */
double pkc_info_E_z(const struct pkc_header *h);

/* Parse a list like "1,3,5" into out (at most max values); returns the count or -1. */
int pkc_parse_percents(const char *s, double *out, int max);

#endif /* ENABLE_PKC */
#endif /* PKC_CORE_H */
