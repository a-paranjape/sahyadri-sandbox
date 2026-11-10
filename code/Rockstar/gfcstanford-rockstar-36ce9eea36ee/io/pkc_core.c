/* pkc_core: C99 reader for pkdgrav3 compressed snapshots (snapcomp format version 3).
 * See pkc_core.h. Written from pkdgrav3 docs/snapcomp_format.md. Added for Sahyadri (2026-11-10). */
#ifdef ENABLE_PKC

#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zstd.h>

#include "pkc_core.h"

#define HEADER_SIZE 4096u
#define INFO_OFFSET 288u
#define COSMO_OFFSET 4000u
#define POT_OFFSET 4080u
#define TRAILER 32u
#define INDEX_ENTRY 32u
#define BLOCK_HEADER 16u
#define STREAM_ENTRY 12u

enum { K_CELLKEY = 1, K_COUNT = 2, K_OFFSET = 3, K_VEL = 4, K_ID_RAW = 5, K_ID_DELTA = 6, K_ID_LAGR = 7, K_POT = 8 };
enum { ID_RAW = 0, ID_DELTA_IN_CELL = 1, ID_LAGRANGIAN = 2 };

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "pkc_core assumes a little-endian host (as the snapcomp format)"
#endif

static int fail(char *err, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  if (err) vsnprintf(err, PKC_ERRLEN, fmt, ap);
  va_end(ap);
  return -1;
}

static uint32_t get_u32(const uint8_t *b, size_t o) { uint32_t v; memcpy(&v, b + o, 4); return v; }
static uint64_t get_u64(const uint8_t *b, size_t o) { uint64_t v; memcpy(&v, b + o, 8); return v; }
static int32_t get_i32(const uint8_t *b, size_t o) { int32_t v; memcpy(&v, b + o, 4); return v; }
static double get_f64(const uint8_t *b, size_t o) { double v; memcpy(&v, b + o, 8); return v; }
static uint16_t get_u16(const uint8_t *b, size_t o) { uint16_t v; memcpy(&v, b + o, 2); return v; }

static int64_t unzigzag(uint64_t u) { return (int64_t)(u >> 1) ^ -(int64_t)(u & 1); }
static int64_t floor_mod(int64_t a, int64_t m) { int64_t r = a % m; return r < 0 ? r + m : r; }

static int pread_all(int fd, void *buf, size_t n, uint64_t off) {
  uint8_t *p = (uint8_t *)buf;
  while (n) {
    ssize_t r = pread(fd, p, n, (off_t)off);
    if (r < 0) { if (errno == EINTR) continue; return -1; }
    if (r == 0) return -1;
    p += r; n -= (size_t)r; off += (uint64_t)r;
  }
  return 0;
}

static void *grow(void *p, size_t *cap, size_t need, size_t elem) {
  if (need <= *cap) return p;
  size_t c = *cap ? *cap : 1;
  while (c < need) c *= 2;
  void *q = realloc(p, c * elem);
  if (!q) return NULL;
  *cap = c;
  return q;
}

/* ---------------------------------------------------------------------------------------------- */
/* Header and file                                                                                 */

static int unpack_header(const uint8_t *b, struct pkc_header *h, const char *path, char *err) {
  static const char magic[8] = {'P', 'K', 'C', 'O', 'M', 'P', 0, 1};
  int i;
  if (memcmp(b, magic, 8)) return fail(err, "%s: not a snapcomp file (bad header magic)", path);
  memset(h, 0, sizeof(*h));
  h->version = get_u32(b, 8);
  if (h->version != 3)
    return fail(err, "%s: format version %u; this reader accepts only version 3 (current pkdgrav3)", path,
                (unsigned)h->version);
  if (get_u32(b, 12) != HEADER_SIZE) return fail(err, "%s: unexpected header size", path);
  h->n_particles = get_u64(b, 16);
  h->n_blocks = get_u64(b, 24);
  h->lattice_n = get_u64(b, 32);
  h->n_total = get_u64(b, 40);
  h->ncell = get_u32(b, 48);
  h->sub = get_u32(b, 52);
  h->vel_step = get_f64(b, 56);
  h->vel_to_kms = get_f64(b, 64);
  h->box_size = get_f64(b, 72);
  h->redshift = get_f64(b, 80);
  h->time = get_f64(b, 88);
  h->omega_m = get_f64(b, 96);
  h->omega_l = get_f64(b, 104);
  h->h = get_f64(b, 112);
  h->subsample = get_u32(b, 120);
  h->n_subsamples = get_u32(b, 124);
  for (i = 0; i < PKC_MAX_SUBSAMPLES; i++) h->percents[i] = get_f64(b, 128 + 8 * i);
  h->subsample_seed = get_u64(b, 256);
  h->writer = get_u32(b, 264);
  h->n_writers = get_u32(b, 268);
  h->block_size = get_u32(b, 272);
  h->zstd_level = get_i32(b, 276);
  h->step = get_u64(b, 280);
  memcpy(h->info, b + INFO_OFFSET, COSMO_OFFSET - INFO_OFFSET);
  h->info[sizeof(h->info) - 1] = 0;
  h->omega_b = get_f64(b, COSMO_OFFSET + 0);
  h->omega_rad = get_f64(b, COSMO_OFFSET + 8);
  h->omega_de = get_f64(b, COSMO_OFFSET + 16);
  h->w0 = get_f64(b, COSMO_OFFSET + 24);
  h->wa = get_f64(b, COSMO_OFFSET + 32);
  h->sigma8 = get_f64(b, COSMO_OFFSET + 40);
  h->n_s = get_f64(b, COSMO_OFFSET + 48);
  h->running = get_f64(b, COSMO_OFFSET + 56);
  h->normalization = get_f64(b, COSMO_OFFSET + 64);
  h->pivot = get_f64(b, COSMO_OFFSET + 72);
  h->pot_step = get_f64(b, POT_OFFSET);
  h->pot_to_kms2 = get_f64(b, POT_OFFSET + 8);

  if (!(h->ncell > 0 && h->sub > 0 && h->vel_step > 0 && h->box_size > 0))
    return fail(err, "%s: invalid grid, velocity step or box size in header", path);
  if (!(h->pot_step >= 0)) return fail(err, "%s: negative potential step", path);
  if (h->n_subsamples < 1 || h->n_subsamples > PKC_MAX_SUBSAMPLES || h->subsample >= h->n_subsamples)
    return fail(err, "%s: invalid subsample %u of %u", path, (unsigned)h->subsample, (unsigned)h->n_subsamples);
  if (h->n_writers < 1 || h->writer >= h->n_writers)
    return fail(err, "%s: invalid writer %u of %u", path, (unsigned)h->writer, (unsigned)h->n_writers);
  if (h->lattice_n) {
    /* Lagrangian IDs need X*N < 2^63 in 64-bit integers (X < ncell*sub) */
    uint64_t q = (uint64_t)h->ncell * h->sub;
    if (h->lattice_n >= (1ull << 21) || q >= (1ull << 40))
      return fail(err, "%s: lattice_n or grid too large for 64-bit Lagrangian ID decoding", path);
  }
  return 0;
}

int pkc_file_open(const char *path, struct pkc_file *f, char *err) {
  static const char tmagic[8] = {'P', 'K', 'C', 'I', 'D', 'X', 0, 1};
  struct stat st;
  uint8_t t[TRAILER], *hb = NULL, *ib = NULL;
  uint64_t idx, nb, np, i, sum = 0;
  memset(f, 0, sizeof(*f));
  f->fd = open(path, O_RDONLY);
  if (f->fd < 0) return fail(err, "%s: cannot open (%s)", path, strerror(errno));
  f->path = strdup(path);
  if (fstat(f->fd, &st)) { fail(err, "%s: fstat failed", path); goto bad; }
  f->file_size = (uint64_t)st.st_size;
  if (f->file_size < HEADER_SIZE + TRAILER) { fail(err, "%s: file too small (incomplete?)", path); goto bad; }
  if (pread_all(f->fd, t, TRAILER, f->file_size - TRAILER)) { fail(err, "%s: cannot read trailer", path); goto bad; }
  if (memcmp(t, tmagic, 8)) { fail(err, "%s: no trailer: file incomplete (writer did not close)", path); goto bad; }
  idx = get_u64(t, 8); nb = get_u64(t, 16); np = get_u64(t, 24);
  if (nb > (f->file_size / INDEX_ENTRY) || idx < HEADER_SIZE || idx + nb * INDEX_ENTRY + TRAILER != f->file_size) {
    fail(err, "%s: index does not match file size", path); goto bad;
  }
  hb = (uint8_t *)malloc(HEADER_SIZE);
  if (!hb || pread_all(f->fd, hb, HEADER_SIZE, 0)) { fail(err, "%s: cannot read header", path); goto bad; }
  if (unpack_header(hb, &f->h, path, err)) goto bad;
  if (f->h.n_blocks != nb || f->h.n_particles != np) {
    fail(err, "%s: header and trailer disagree on block or particle count", path); goto bad;
  }
  f->index = (struct pkc_index_entry *)calloc(nb ? nb : 1, sizeof(struct pkc_index_entry));
  ib = (uint8_t *)malloc(nb ? nb * INDEX_ENTRY : 1);
  if (!f->index || !ib) { fail(err, "%s: out of memory for the index", path); goto bad; }
  if (nb && pread_all(f->fd, ib, nb * INDEX_ENTRY, idx)) { fail(err, "%s: cannot read index", path); goto bad; }
  for (i = 0; i < nb; i++) {
    struct pkc_index_entry *e = f->index + i;
    e->offset = get_u64(ib, i * INDEX_ENTRY);
    e->size = get_u64(ib, i * INDEX_ENTRY + 8);
    e->n = get_u32(ib, i * INDEX_ENTRY + 16);
    e->thread = get_u32(ib, i * INDEX_ENTRY + 20);
    if (e->offset < HEADER_SIZE || e->size < BLOCK_HEADER || e->offset + e->size > idx || e->n == 0) {
      fail(err, "%s: index entry %llu out of range", path, (unsigned long long)i); goto bad;
    }
    sum += e->n;
  }
  if (sum != np) { fail(err, "%s: index particle counts do not add up to n_particles", path); goto bad; }
  free(hb); free(ib);
  return 0;
bad:
  free(hb); free(ib);
  pkc_file_close(f);
  return -1;
}

void pkc_file_close(struct pkc_file *f) {
  if (f->path) close(f->fd);
  free(f->path);
  free(f->index);
  memset(f, 0, sizeof(*f));
  f->fd = -1;
}

/* ---------------------------------------------------------------------------------------------- */
/* Snapshot set                                                                                    */

static int cmp_files(const void *a, const void *b) {
  const struct pkc_file *x = (const struct pkc_file *)a, *y = (const struct pkc_file *)b;
  if (x->h.subsample != y->h.subsample) return x->h.subsample < y->h.subsample ? -1 : 1;
  if (x->h.writer != y->h.writer) return x->h.writer < y->h.writer ? -1 : 1;
  return 0;
}

static int same_snapshot(const struct pkc_header *a, const struct pkc_header *b) {
  int i;
  if (a->n_total != b->n_total || a->lattice_n != b->lattice_n || a->ncell != b->ncell || a->sub != b->sub ||
      a->vel_step != b->vel_step || a->box_size != b->box_size || a->redshift != b->redshift ||
      a->step != b->step || a->omega_m != b->omega_m || a->omega_l != b->omega_l || a->h != b->h ||
      a->n_subsamples != b->n_subsamples || a->subsample_seed != b->subsample_seed ||
      a->n_writers != b->n_writers || a->omega_de != b->omega_de || a->w0 != b->w0 || a->wa != b->wa ||
      a->omega_rad != b->omega_rad || (a->pot_step > 0) != (b->pot_step > 0))
    return 0;
  for (i = 0; i < (int)a->n_subsamples; i++)
    if (a->percents[i] != b->percents[i]) return 0;
  return 1;
}

int pkc_set_open(const char *prefix, const double *want_pct, int n_want, struct pkc_set *s, char *err) {
  char pattern[4096];
  glob_t g;
  size_t i;
  int j, nf = 0, rc;
  uint64_t all_particles = 0;
  int64_t p;
  memset(s, 0, sizeof(*s));
  if (snprintf(pattern, sizeof(pattern), "%s.s*.pkc", prefix) >= (int)sizeof(pattern))
    return fail(err, "snapshot prefix too long: %s", prefix);
  memset(&g, 0, sizeof(g));
  rc = glob(pattern, 0, NULL, &g);
  if (rc == GLOB_NOMATCH || (rc == 0 && g.gl_pathc == 0)) { globfree(&g); return fail(err, "no files %s", pattern); }
  if (rc) { globfree(&g); return fail(err, "glob failed for %s", pattern); }

  s->files = (struct pkc_file *)calloc(g.gl_pathc, sizeof(struct pkc_file));
  if (!s->files) { globfree(&g); return fail(err, "out of memory"); }
  for (i = 0; i < g.gl_pathc; i++) {
    if (pkc_file_open(g.gl_pathv[i], s->files + nf, err)) { globfree(&g); s->n_files = nf; pkc_set_close(s); return -1; }
    nf++;
  }
  globfree(&g);
  s->n_files = nf;
  qsort(s->files, (size_t)nf, sizeof(struct pkc_file), cmp_files);

  /* consistency and uniqueness over every file found */
  for (j = 0; j < nf; j++) {
    if (!same_snapshot(&s->files[0].h, &s->files[j].h)) {
      fail(err, "%s and %s: headers belong to different snapshots or runs", s->files[0].path, s->files[j].path);
      pkc_set_close(s); return -1;
    }
    if (j && !cmp_files(s->files + j - 1, s->files + j)) {
      fail(err, "%s and %s: same subsample and writer", s->files[j - 1].path, s->files[j].path);
      pkc_set_close(s); return -1;
    }
    all_particles += s->files[j].h.n_particles;
  }

  {
    const struct pkc_header *h0 = &s->files[0].h;
    int sel[PKC_MAX_SUBSAMPLES] = {0}, k, nsel = 0, w;
    int count[PKC_MAX_SUBSAMPLES] = {0};
    if (n_want == 0) {
      for (k = 0; k < (int)h0->n_subsamples; k++) sel[k] = 1;
    } else {
      for (j = 0; j < n_want; j++) {
        int found = 0;
        for (k = 0; k < (int)h0->n_subsamples; k++)
          if (fabs(h0->percents[k] - want_pct[j]) < 1e-9) { sel[k] = 1; found = 1; }
        if (!found) { fail(err, "%s: no subsample with %g percent", prefix, want_pct[j]); pkc_set_close(s); return -1; }
      }
    }
    for (k = 0; k < (int)h0->n_subsamples; k++) nsel += sel[k];
    s->all_subsamples = (nsel == (int)h0->n_subsamples);
    for (j = 0; j < nf; j++) count[s->files[j].h.subsample]++;
    for (k = 0; k < (int)h0->n_subsamples; k++) {
      if (!sel[k]) continue;
      if (count[k] != (int)h0->n_writers) {
        fail(err, "%s: subsample %g%%: %d of %u writer files present", prefix, h0->percents[k], count[k],
             (unsigned)h0->n_writers);
        pkc_set_close(s); return -1;
      }
    }
    if (s->all_subsamples && all_particles != h0->n_total) {
      fail(err, "%s: files hold %llu particles, header n_total is %llu", prefix,
           (unsigned long long)all_particles, (unsigned long long)h0->n_total);
      pkc_set_close(s); return -1;
    }
    /* keep only selected files (array is sorted, so order is preserved) */
    for (j = 0, w = 0; j < nf; j++) {
      if (sel[s->files[j].h.subsample]) s->files[w++] = s->files[j];
      else pkc_file_close(s->files + j);
    }
    s->n_files = w;
  }

  /* pairs in canonical order with particle prefix sums */
  for (j = 0, p = 0; j < s->n_files; j++) p += (int64_t)s->files[j].h.n_blocks;
  s->n_pairs = p;
  s->pairs = (struct pkc_pair *)calloc(p ? (size_t)p : 1, sizeof(struct pkc_pair));
  if (!s->pairs) { fail(err, "out of memory"); pkc_set_close(s); return -1; }
  {
    uint64_t start = 0, b;
    int64_t q = 0;
    for (j = 0; j < s->n_files; j++)
      for (b = 0; b < s->files[j].h.n_blocks; b++, q++) {
        s->pairs[q].file = j;
        s->pairs[q].block = b;
        s->pairs[q].start = start;
        start += s->files[j].index[b].n;
      }
    s->n_selected = start;
  }
  return 0;
}

void pkc_set_close(struct pkc_set *s) {
  int j;
  for (j = 0; j < s->n_files; j++) pkc_file_close(s->files + j);
  free(s->files);
  free(s->pairs);
  memset(s, 0, sizeof(*s));
}

int pkc_set_range(const struct pkc_set *s, int64_t k, int64_t nshare, int64_t *first, int64_t *count,
                  uint64_t *np, char *err) {
  uint64_t T = s->n_selected, sum = 0;
  int64_t a, b, i;
  if (nshare < 1 || k < 0 || k >= nshare) return fail(err, "share %lld of %lld out of range", (long long)k, (long long)nshare);
  *first = 0; *count = 0; *np = 0;
  if (T == 0) return 0;
  if ((uint64_t)nshare > UINT64_MAX / T) return fail(err, "too many shares for %llu particles", (unsigned long long)T);
  /* pair i belongs to share floor(start_i * nshare / T); starts are increasing, so shares are contiguous */
  for (a = 0; a < s->n_pairs && (s->pairs[a].start * (uint64_t)nshare) / T < (uint64_t)k; a++) {}
  for (b = a; b < s->n_pairs && (s->pairs[b].start * (uint64_t)nshare) / T == (uint64_t)k; b++) {}
  *first = a;
  *count = b - a;
  for (i = a; i < b; i++) sum += s->files[s->pairs[i].file].index[s->pairs[i].block].n;
  *np = sum;
  return 0;
}

/* ---------------------------------------------------------------------------------------------- */
/* Block decoder                                                                                   */

void pkc_decoder_init(struct pkc_decoder *d) { memset(d, 0, sizeof(*d)); }

void pkc_decoder_free(struct pkc_decoder *d) {
  int k;
  if (d->dctx) ZSTD_freeDCtx((ZSTD_DCtx *)d->dctx);
  free(d->raw);
  free(d->planes);
  for (k = 0; k < 12; k++) free(d->st[k]);
  memset(d, 0, sizeof(*d));
}

void pkc_block_free(struct pkc_block *b) {
  free(b->X); free(b->V); free(b->id);
  memset(b, 0, sizeof(*b));
}

static int read_stream(struct pkc_decoder *d, const uint8_t *src, size_t csize, int w, uint32_t count,
                       uint64_t *out, const char *path, char *err) {
  size_t raw = (size_t)count * (size_t)w, r, i;
  int b;
  uint8_t *p = (uint8_t *)grow(d->planes, &d->planes_cap, raw ? raw : 1, 1);
  if (!p) return fail(err, "%s: out of memory", path);
  d->planes = p;
  r = ZSTD_decompressDCtx((ZSTD_DCtx *)d->dctx, d->planes, raw, src, csize);
  if (ZSTD_isError(r) || r != raw)
    return fail(err, "%s: stream decompression failed or size mismatch (%s)", path,
                ZSTD_isError(r) ? ZSTD_getErrorName(r) : "size");
  for (i = 0; i < count; i++) out[i] = 0;
  for (b = 0; b < w; b++) {
    const uint8_t *q = d->planes + (size_t)b * count;
    int sh = 8 * b;
    for (i = 0; i < count; i++) out[i] |= (uint64_t)q[i] << sh;
  }
  return 0;
}

int pkc_read_block(const struct pkc_file *f, uint64_t bi, struct pkc_decoder *d, struct pkc_block *out, char *err) {
  static const char bmagic[4] = {'P', 'K', 'B', '1'};
  const struct pkc_header *h = &f->h;
  const char *path = f->path;
  const struct pkc_index_entry *e;
  const uint8_t *b;
  uint32_t n, n_cells, c, m;
  int ns, coder, n_id, expect, k, has_pot = h->pot_step > 0;
  size_t off, data, size, cap, j, i;
  uint8_t kinds[12];
  uint64_t key = 0;
  const uint64_t nc = h->ncell, sub = h->sub, Q = (uint64_t)h->ncell * h->sub;

  if (bi >= h->n_blocks) return fail(err, "%s: block %llu out of range", path, (unsigned long long)bi);
  e = f->index + bi;
  size = (size_t)e->size;
  if (!d->dctx) {
    d->dctx = ZSTD_createDCtx();
    if (!d->dctx) return fail(err, "ZSTD_createDCtx failed");
  }
  d->raw = (uint8_t *)grow(d->raw, &d->raw_cap, size, 1);
  if (!d->raw) return fail(err, "%s: out of memory", path);
  if (pread_all(f->fd, d->raw, size, e->offset)) return fail(err, "%s: cannot read block %llu", path, (unsigned long long)bi);
  b = d->raw;

  if (memcmp(b, bmagic, 4)) return fail(err, "%s: block %llu: bad block magic", path, (unsigned long long)bi);
  n = get_u32(b, 4);
  n_cells = get_u32(b, 8);
  ns = b[12];
  coder = b[13];
  if (get_u16(b, 14) != 0) return fail(err, "%s: block %llu: reserved field not 0", path, (unsigned long long)bi);
  if (coder > ID_LAGRANGIAN) return fail(err, "%s: block %llu: unknown ID coder %d", path, (unsigned long long)bi, coder);
  n_id = coder == ID_LAGRANGIAN ? 3 : 1;
  expect = 8 + n_id + has_pot;
  if (ns != expect || n != e->n || n == 0 || n_cells == 0 || n_cells > n)
    return fail(err, "%s: block %llu: unexpected stream count, particle or cell count", path, (unsigned long long)bi);
  if (coder == ID_LAGRANGIAN && !h->lattice_n)
    return fail(err, "%s: block %llu: Lagrangian ID coder needs lattice_n", path, (unsigned long long)bi);
  if (size < BLOCK_HEADER + STREAM_ENTRY * (size_t)ns)
    return fail(err, "%s: block %llu: truncated (stream table)", path, (unsigned long long)bi);

  /* expected kinds: keys, counts, 3 offsets, 3 velocities, ID stream(s), potential */
  kinds[0] = K_CELLKEY; kinds[1] = K_COUNT;
  kinds[2] = kinds[3] = kinds[4] = K_OFFSET;
  kinds[5] = kinds[6] = kinds[7] = K_VEL;
  for (k = 0; k < n_id; k++) kinds[8 + k] = coder == ID_RAW ? K_ID_RAW : coder == ID_DELTA_IN_CELL ? K_ID_DELTA : K_ID_LAGR;
  if (has_pot) kinds[8 + n_id] = K_POT;

  if (n > d->st_cap) {
    cap = d->st_cap ? d->st_cap : 1;
    while (cap < n) cap *= 2;
    for (k = 0; k < 12; k++) {
      uint64_t *q = (uint64_t *)realloc(d->st[k], cap * sizeof(uint64_t));
      if (!q) return fail(err, "%s: out of memory", path);
      d->st[k] = q;
    }
    d->st_cap = cap;
  }

  off = BLOCK_HEADER;
  data = BLOCK_HEADER + STREAM_ENTRY * (size_t)ns;
  for (k = 0; k < ns; k++) {
    int w = b[off], kind = b[off + 1];
    uint32_t count = get_u32(b, off + 4), cs = get_u32(b, off + 8);
    if (kind != kinds[k]) return fail(err, "%s: block %llu: stream %d has kind %d, expected %d", path, (unsigned long long)bi, k, kind, kinds[k]);
    if (!(w == 1 || w == 2 || w == 4 || w == 8)) return fail(err, "%s: block %llu: bad stream width", path, (unsigned long long)bi);
    if (count != (k < 2 ? n_cells : n)) return fail(err, "%s: block %llu: stream count mismatch", path, (unsigned long long)bi);
    if (data + cs > size) return fail(err, "%s: block %llu: truncated (stream data)", path, (unsigned long long)bi);
    if (kind != K_POT) /* the potential is not used by Rockstar: not decompressed */
      if (read_stream(d, b + data, cs, w, count, d->st[k], path, err)) return -1;
    off += STREAM_ENTRY;
    data += cs;
  }
  if (data != size) return fail(err, "%s: block %llu: size differs from its streams", path, (unsigned long long)bi);

  /* output buffers */
  if (n > out->cap) {
    size_t c2 = out->cap ? out->cap : 1;
    int64_t *X, *V; uint64_t *id;
    while (c2 < n) c2 *= 2;
    X = (int64_t *)realloc(out->X, 3 * c2 * sizeof(int64_t));
    if (X) out->X = X;
    V = (int64_t *)realloc(out->V, 3 * c2 * sizeof(int64_t));
    if (V) out->V = V;
    id = (uint64_t *)realloc(out->id, c2 * sizeof(uint64_t));
    if (id) out->id = id;
    if (!X || !V || !id) return fail(err, "%s: out of memory", path);
    out->cap = c2;
  }
  out->n = n;
  out->id_coder = coder;

  /* cells -> positions */
  for (c = 0, j = 0; c < n_cells; c++) {
    uint64_t cell[3];
    key += d->st[0][c];
    if (key >= nc * nc * nc) return fail(err, "%s: block %llu: cell key out of range", path, (unsigned long long)bi);
    cell[0] = key % nc; cell[1] = (key / nc) % nc; cell[2] = key / (nc * nc);
    if (j + d->st[1][c] > n) return fail(err, "%s: block %llu: cell counts exceed block size", path, (unsigned long long)bi);
    for (m = 0; m < d->st[1][c]; m++, j++)
      for (k = 0; k < 3; k++) {
        uint64_t o = d->st[2 + k][j];
        if (o >= sub) return fail(err, "%s: block %llu: offset out of range", path, (unsigned long long)bi);
        out->X[3 * j + k] = (int64_t)(cell[k] * sub + o);
      }
  }
  if (j != n) return fail(err, "%s: block %llu: cell counts do not add up to block size", path, (unsigned long long)bi);

  for (i = 0; i < n; i++)
    for (k = 0; k < 3; k++) out->V[3 * i + k] = unzigzag(d->st[5 + k][i]);

  if (coder == ID_RAW) {
    for (i = 0; i < n; i++) out->id[i] = d->st[8][i];
  } else if (coder == ID_DELTA_IN_CELL) {
    uint64_t prev_first = 0;
    for (c = 0, i = 0; c < n_cells; c++)
      for (m = 0; m < d->st[1][c]; m++, i++) {
        uint64_t ref = (m == 0) ? prev_first : out->id[i - 1];
        out->id[i] = ref + (uint64_t)unzigzag(d->st[8][i]);
        if (m == 0) prev_first = out->id[i];
      }
  } else {
    const uint64_t N = h->lattice_n;
    for (i = 0; i < n; i++) {
      int64_t q[3];
      for (k = 0; k < 3; k++) {
        int64_t l = (int64_t)(((uint64_t)out->X[3 * i + k] * N) / Q); /* integer division, as the reference */
        q[k] = floor_mod(unzigzag(d->st[8 + k][i]) + l, (int64_t)N);
      }
      out->id[i] = ((uint64_t)q[2] * N + (uint64_t)q[1]) * N + (uint64_t)q[0];
    }
  }
  return 0;
}

/* ---------------------------------------------------------------------------------------------- */
/* Cosmology and options                                                                           */

int pkc_cosmology(const struct pkc_header *h, double *Om, double *Ol, double *h0, double *W0, double *WA, char *err) {
  if (h->omega_l > 0 && h->omega_de > 0 && (h->w0 != -1.0 || h->wa != 0.0))
    return fail(err, "run has both a cosmological constant (Omega_lambda = %g) and dark energy with w0 = %g, "
                     "wa = %g (Omega_DE = %g); Rockstar has one dark-energy component",
                h->omega_l, h->w0, h->wa, h->omega_de);
  *Om = h->omega_m;
  *h0 = h->h;
  *Ol = h->omega_l + h->omega_de;
  if (h->omega_de > 0) { *W0 = h->w0; *WA = h->wa; }
  else { *W0 = -1.0; *WA = 0.0; }
  return 0;
}

double pkc_info_E_z(const struct pkc_header *h) {
  const char *p = strstr(h->info, "\"E_z\":");
  char *end;
  double v;
  if (!p) return -1;
  p += 6;
  v = strtod(p, &end);
  return (end == p) ? -1 : v;
}

int pkc_parse_percents(const char *s, double *out, int max) {
  int n = 0;
  const char *p = s;
  char *end;
  while (*p) {
    while (*p == ' ' || *p == ',' || *p == '\t') p++;
    if (!*p) break;
    if (n >= max) return -1;
    out[n] = strtod(p, &end);
    if (end == p) return -1;
    n++;
    p = end;
  }
  return n;
}

#endif /* ENABLE_PKC */
