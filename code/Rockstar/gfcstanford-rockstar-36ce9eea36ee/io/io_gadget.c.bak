#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include "io_gadget.h"
#include "io_util.h"
#include "../universal_constants.h"
#include "../check_syscalls.h"
#include "../config_vars.h"
#include "../config.h"
#include "../particle.h"

#define GADGET_BUFFER_SIZE 100000
#define GHPT GADGET_HALO_PARTICLE_TYPE

void gadget2_detect_filetype(FILE *input, char *filename) {
  int32_t first_word;
  SWAP_ENDIANNESS=0;
  check_fread(&first_word, sizeof(int32_t), 1, input);
  check_limited_funread(&first_word, sizeof(int32_t), 1);
  if ((first_word != 4) && (first_word != GADGET_HEADER_SIZE) &&
      (first_word != 8)) {
    SWAP_ENDIANNESS = 1;
    swap_endian_4byte((int8_t *)(&first_word));
  }
  if (first_word == 4) GADGET_VARIANT = 1;
  else if (first_word == 8) GADGET_VARIANT = 2;
  else if (first_word == GADGET_HEADER_SIZE) GADGET_VARIANT = 0;
  else {
    fprintf(stderr, "[Error] Unrecognized GADGET2 file type in %s!\n", filename);
    exit(1);
  }
}

void gadget2_read_stride(FILE *input, int64_t p_start, int64_t nelems, int64_t stride, int64_t width, struct particle *p, int64_t offset, int64_t skip, int64_t skip2, char *filename)
{
  int64_t to_read = nelems, i, n;
  char *buffer = NULL;
  uint32_t readsize;

  if (SWAP_ENDIANNESS) fread_swap(&readsize, sizeof(uint32_t), 1, input);
  else check_fread(&readsize, sizeof(uint32_t), 1, input);
  if ((stride == 1) && (((char *)p)+offset == (char *)&(p[0].id))) { 
    //reading IDs
    if (readsize == (uint32_t)((nelems+skip+skip2)*4))
      GADGET_ID_BYTES = width = 4;
    else if (readsize == (uint32_t)((nelems+skip+skip2)*8))
      GADGET_ID_BYTES = width = 8;
    else {
      fprintf(stderr, "[Error] Invalid particle ID block size in file %s!\n", filename);
      exit(1);
    }
  }

  //support 8-byte positions / velocities
  if (stride == 3) { 
    //reading IDs
    if (readsize == (uint32_t)(stride*(nelems+skip+skip2)*4)) width = 4;
    else if (readsize == (uint32_t)(stride*(nelems+skip+skip2)*8))
      width = 8;
    else {
      fprintf(stderr, "[Error] Invalid position/velocity block size in file %s!\n", filename);
      exit(1);
    }
  }

  check_realloc_s(buffer, GADGET_BUFFER_SIZE,stride*width);
  check_fskip(input, (skip*width*stride), buffer, 
	      GADGET_BUFFER_SIZE*stride*width);

  while (nelems > 0) {
    to_read = nelems;
    if (to_read > GADGET_BUFFER_SIZE) to_read = GADGET_BUFFER_SIZE;
    if (!SWAP_ENDIANNESS) n=check_fread(buffer, width, stride*to_read, input);
    else if (width==4) n=fread_swap(buffer, width, stride*to_read, input);
    else n=fread_swap8(buffer, width, stride*to_read, input);
    n/=stride;
    if (n<to_read) {
      fprintf(stderr, "[Error] Input file %s is truncated!\n", filename);
      exit(1);
    }
    if ((stride != 3) || (width != 8)) {
      for (i=0; i<n; i++) memcpy(((char *)&(p[p_start+i])) + offset,
				 buffer+(i*stride*width), stride*width);
    } else {
      for (i=0; i<n; i++) {
	float *dest = (float *)((char *)&(p[p_start+i]) + offset);
	for (int64_t j=0; j<3; j++)
	  dest[j] = (float)(*((double *)(buffer + (i*stride*width) + j*width)));
      }
    }
    p_start += n;
    nelems -= n;
  }
  check_fskip(input, 4+(skip2*width*stride), buffer, 
	      GADGET_BUFFER_SIZE*stride*width);
  free(buffer);
}

void gadget2_extract_header_info(struct gadget_header *header)
{
  int64_t i;
  if (SWAP_ENDIANNESS) {
#define SWAP8(x) { swap_4byte_to_8byte((int32_t *)(void *)&(x)); }
    for (i=0; i<6; i++) SWAP8(header->particle_masses[i]);
    SWAP8(header->scale_factor);
    SWAP8(header->redshift);
    SWAP8(header->box_size);
    SWAP8(header->omega_0);
    SWAP8(header->omega_lambda);
    SWAP8(header->h_0);
#undef SWAP8
  }

  if (fabs(header->omega_0 + header->omega_lambda - 1.0) > 1e-5) {
    fprintf(stderr, "[Error] Halo Finder Not Currently Configured to Run on Cosmologies with Curvature. (Omega_Matter = %f, Omega_Lambda = %f!)\n", header->omega_0, header->omega_lambda);
    exit(1);
  }

  Ol = header->omega_lambda;
  Om = header->omega_0;
  h0 = header->h_0;

  BOX_SIZE = header->box_size*GADGET_LENGTH_CONVERSION;
  TOTAL_PARTICLES = (((int64_t)header->num_total_particles_hw[GHPT])<<32) +
    (int64_t)header->num_total_particles[GHPT];
  //According to Matt Becker, LGADGET uses the header fields in a different way:
  if (!strncasecmp(FILE_FORMAT, "LGADGET", 7)) {
    TOTAL_PARTICLES = (((int64_t)header->num_total_particles[GHPT+1])<<32) +
      (int64_t)header->num_total_particles[GHPT];
  }

  SCALE_NOW = header->scale_factor;
  if (header->particle_masses[GHPT] || !PARTICLE_MASS || RESCALE_PARTICLE_MASS) {
    if (!RESCALE_PARTICLE_MASS)
      PARTICLE_MASS = header->particle_masses[GHPT]*GADGET_MASS_CONVERSION;
    else
      PARTICLE_MASS = Om*CRITICAL_DENSITY * pow(BOX_SIZE, 3) / TOTAL_PARTICLES;
  }
  AVG_PARTICLE_SPACING = cbrt(PARTICLE_MASS / (Om*CRITICAL_DENSITY));
}

void gadget2_rescale_particles(struct particle *p, int64_t p_start, int64_t nelems) {
  int64_t i, j;
  uint32_t id;
  double vel_rescale = sqrt(SCALE_NOW);
  if (LIGHTCONE) vel_rescale = 1;
  for (i=0; i<nelems; i++) {
    if (GADGET_ID_BYTES == 4) {
      memcpy(&id, &(p[p_start+i].id), sizeof(uint32_t));
      p[p_start+i].id = id;
    }
    for (j=0; j<3; j++) {
      p[p_start+i].pos[j]   *= GADGET_LENGTH_CONVERSION;
      p[p_start+i].pos[j+3] *= vel_rescale;
    }
  }
}

void load_particles_gadget2(char *filename, struct particle **p, int64_t *num_p)
{
  FILE *input;
  char tag[10] = {0};
  struct gadget_header header;
  int64_t total_particles, skip, skip2, halo_particles, i;

  input = check_fopen(filename, "rb");
  gadget2_detect_filetype(input, filename);

#define gadget_variant_block(a)		\
  if (GADGET_VARIANT) { \
    fread_fortran(tag, sizeof(char)*4*GADGET_VARIANT,1,input, SWAP_ENDIANNESS);\
    if (SWAP_ENDIANNESS) swap_endian_4byte((int8_t *)(&tag)); \
    assert(!strncmp(tag, a, strlen(a)));		      \
  }

  gadget_variant_block("HEAD");
  fread_fortran(&header, GADGET_HEADER_SIZE, 1, input, SWAP_ENDIANNESS);
  gadget2_extract_header_info(&header);

  total_particles = 0;
  for (i=0; i<6; i++) total_particles += header.num_particles[i];

  if (GADGET_SKIP_NON_HALO_PARTICLES) {
    for (skip=0,i=0; i<GHPT; i++) skip += header.num_particles[i];
    halo_particles = header.num_particles[GHPT];
    skip2 = total_particles - skip - halo_particles;
  } else {
    skip = skip2 = 0;
    halo_particles = total_particles;
  }

  *p = (struct particle *)check_realloc(*p, ((*num_p)+halo_particles)*sizeof(struct particle), "Allocating particles.");

  gadget_variant_block("POS");
  gadget2_read_stride(input, *num_p, halo_particles, 3, sizeof(float), 
    *p, (char *)&(p[0][0].pos[0])-(char*)(p[0]), skip, skip2, filename);

  gadget_variant_block("VEL");
  gadget2_read_stride(input, *num_p, halo_particles, 3, sizeof(float), 
    *p, (char *)&(p[0][0].pos[3])-(char*)(p[0]), skip, skip2, filename);

  gadget_variant_block("ID");
  gadget2_read_stride(input, *num_p, halo_particles, 1, GADGET_ID_BYTES, 
    *p, (char *)&(p[0][0].id)-(char*)(p[0]), skip, skip2, filename);

  gadget2_rescale_particles(*p, *num_p, halo_particles);

  *num_p += halo_particles;
  fclose(input);
}
