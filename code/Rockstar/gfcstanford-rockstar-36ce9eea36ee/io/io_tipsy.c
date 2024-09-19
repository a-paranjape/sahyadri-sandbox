// added by Aseem 03 Nov 2023
#ifdef HAVE_TIRPC

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <rpc/types.h>
#include <rpc/xdr.h>
#include "../check_syscalls.h"
#include "../universal_constants.h"
#include "../config_vars.h"
#include "../config.h"
#include "../particle.h"
#include "io_tipsy.h"

//#define TIPSY_READ_GAS_STARS


void load_particles_tipsy(char *filename, struct particle **p, int64_t *num_p) {
  FILE *input;
  struct tipsy_dump header;
  struct tipsy_dark_particle dark;
  struct tipsy_gas_particle gas;
#ifdef TIPSY_READ_GAS_STARS
  struct tipsy_star_particle star;
#endif /*TIPSY_READ_GAS_STARS*/

  int i, j;
  int xdrfmt=1,haveiords=0;
  int *iords=NULL, num_iords=0;
  XDR xdrs;

  input = check_fopen(filename, "r");
  xdrstdio_create(&xdrs, input, XDR_DECODE);
  tipsy_xdr_header(&xdrs, &header);
  if (header.ndim != 3) {
      xdr_destroy(&xdrs);
      fseek(input, 0L, SEEK_SET);
      check_fread((char *)&header,sizeof(header),1,input);
      assert(header.ndim == 3);
      xdrfmt = 0;
      }
  SCALE_NOW = header.time;
#ifndef TIPSY_READ_GAS_STARS
  check_realloc_s(*p, (*num_p) + header.ndark, sizeof(struct particle));
#else
  check_realloc_s((*p), ((*num_p) + header.ndark+header.nsph+header.nstar),
		  sizeof(struct particle));
#endif

  haveiords = load_ids_tipsy(filename,header,&iords,&num_iords);
  if (haveiords && (!iords || num_iords < header.nsph+header.ndark+header.nstar)) {
    fprintf(stderr, "[Error] Not enough IDs for total number of particles in Tipsy file %s!\n", filename);
    exit(1);
  }

#ifdef TIPSY_READ_GAS_STARS
  for(i = 0; i < header.nsph; i++) {
      if (xdrfmt) assert(tipsy_xdr_gas(&xdrs, &gas) > 0);
      else fread((char *)&gas,sizeof(struct tipsy_gas_particle), 1, input) ;
      for (j=0; j<3; j++) {
	if (haveiords) (*p)[i+(*num_p)].id = iords[i];
	else (*p)[i+(*num_p)].id = i;
	(*p)[i+(*num_p)].pos[j] = (gas.pos[j] + 0.5) * TIPSY_LENGTH_CONVERSION;
	(*p)[i+(*num_p)].pos[j+3] = gas.vel[j] * TIPSY_VELOCITY_CONVERSION * SCALE_NOW;
	  }
      }
  (*num_p) += header.nsph;  
#else
  if (!xdrfmt) {
    fseek(input, sizeof(struct tipsy_gas_particle)*header.nsph, SEEK_CUR);
  }
  else {
    for (i=0; i<header.nsph; i++) assert(tipsy_xdr_gas(&xdrs, &gas) > 0);
  }
  fprintf(stderr, "[Warning] Rockstar is skipping TIPSY gas and star particles and calculating halos from dark matter only.  If this is not what you want, inquire with the authors about Rockstar-Galaxies.\n");
#endif

  for(i = 0; i < header.ndark; i++) {
      int ip = i+header.nsph;
      if (xdrfmt) assert(tipsy_xdr_dark(&xdrs, &dark) > 0);
      else check_fread((char *)&dark,sizeof(struct tipsy_dark_particle), 1, input) ;
      if (haveiords) (*p)[i+(*num_p)].id = iords[ip];
      else (*p)[i+(*num_p)].id = ip;
      for (j=0; j<3; j++) {
	(*p)[i+(*num_p)].pos[j] = (dark.pos[j] + 0.5) * TIPSY_LENGTH_CONVERSION;
	(*p)[i+(*num_p)].pos[j+3] = dark.vel[j] * TIPSY_VELOCITY_CONVERSION * SCALE_NOW;
	  /*
	  if (haveiords) (*p)[ip].id = iords[ip];
	  else (*p)[ip].id = ip;
	  (*p)[ip].pos[j] = (dark.pos[j] + 0.5) * TIPSY_LENGTH_CONVERSION;
	  (*p)[ip].pos[j+3] = dark.vel[j] * TIPSY_VELOCITY_CONVERSION;
	  */
	  }
      }

  //printf("Read %d dark matter particles.\n",header.ndark);
  //TOTAL_PARTICLES += header.ndark;
  (*num_p) += header.ndark;

#ifdef TIPSY_READ_GAS_STARS
  for(i = 0; i < header.nstar; i++) {
      int ip = i+header.nsph+header.ndark;
      if (xdrfmt) assert(tipsy_xdr_star(&xdrs, &star) > 0);
      else fread((char *)&star,sizeof(struct tipsy_star_particle), 1, input);
      for (j=0; j<3; j++) {
	if (haveiords) (*p)[i+(*num_p)].id = iords[ip];
	else (*p)[i+(*num_p)].id = ip;
	(*p)[i+(*num_p)].pos[j] = (star.pos[j] + 0.5) * TIPSY_LENGTH_CONVERSION;
	(*p)[i+(*num_p)].pos[j+3] = star.vel[j] * TIPSY_VELOCITY_CONVERSION * SCALE_NOW;
      }
  }
  (*num_p) += header.nstar;
#endif /* TIPSY_READ_GAS_STARS */

  if (xdrfmt) xdr_destroy(&xdrs);
  if (iords) free(iords);
  fclose(input);
}

/* open iord file for ids */
int load_ids_tipsy(char *filename, struct tipsy_dump header, int **iords, int *num_iords) {
  FILE *iordf;
  XDR xdrs;  
  char iofilename[256];
  int i, nbodies=0, count=0, bStandard = 0, bASCII=0;
  sprintf(iofilename,"%s.iord",filename);
  iordf = fopen(iofilename, "r");
  if (iordf != NULL) {
      count=fscanf(iordf, "%d%*[, \t]%*d%*[, \t]%*d",&nbodies) ;
      if ( (count == EOF) || (count==0) ){
	  /* try binary instead */
	  rewind(iordf);
	  count=fread(&nbodies, sizeof(int), 1, iordf) ;

	  if ( (count == EOF) || (count==0) ){
	      printf("<%s format is wrong>\n",iofilename);
	      fclose(iordf);
	      return 0;
	      } else if(nbodies <= 0 || nbodies > 10000000){
	      fseek(iordf,0,SEEK_SET);
	      xdrstdio_create(&xdrs,iordf,XDR_DECODE);
	      xdr_int(&xdrs,&nbodies);
	      if (nbodies <= 0 || nbodies > 10000000) {
		  printf("<%s doesn't appear standard or binary or nbodies > 10 mil.>\n",iofilename);
		  xdr_destroy(&xdrs);
		  fclose(iordf);
		  return 0;
		  } else bStandard = 1;
	      }
	  } else bASCII = 1;

      check_realloc_s(*iords, sizeof(**iords), nbodies);
      for(i = 0, count = 0; i < nbodies; i++) {
	int idummy, check=0;
	if (bStandard) check = xdr_int(&xdrs,&idummy);
	else if (bASCII) check = (fscanf(iordf, "%d", &idummy)<1) ? EOF : 0;
	else check = check_fread(&idummy, sizeof(int), 1, iordf) ;
	(*iords)[count] = idummy;

	if (check == EOF) {
	  printf("<%s format is wrong>\n",iofilename);
	  free(*iords) ;
	  *iords = NULL;
	  break;
	}
	count++;
      }
      fclose(iordf);
  }

  if (iordf && (count == nbodies)) {
    printf("Read %d iords.\n",count);
    *num_iords = nbodies;
    return 1;
  }
  else {
    fprintf(stderr, "[Warning] Did not read TIPSY iords file; particle IDs may not be correct.\n");
    if (*iords) free(*iords);
    *iords=NULL;
    *num_iords = 0;
    return 0;
  }
}

int tipsy_xdr_header(XDR *pxdrs,struct tipsy_dump *ph) {
    int pad = 0;
    
    if (!xdr_double(pxdrs,&ph->time)) return 0;
    if (!xdr_int(pxdrs,&ph->nbodies)) return 0;
    if (!xdr_int(pxdrs,&ph->ndim)) return 0;
    if (!xdr_int(pxdrs,&ph->nsph)) return 0;
    if (!xdr_int(pxdrs,&ph->ndark)) return 0;
    if (!xdr_int(pxdrs,&ph->nstar)) return 0;
    if (!xdr_int(pxdrs,&pad)) return 0;
    return 1;
    }

int tipsy_xdr_gas(XDR *pxdrs,struct tipsy_gas_particle *ph) { 
    int i;
    if (!xdr_float(pxdrs,&ph->mass)) return 0;
    for(i=0; i<TIPSY_MAXDIM; i++) if (!xdr_float(pxdrs,&ph->pos[i])) return 0;
    for(i=0; i<TIPSY_MAXDIM; i++) if (!xdr_float(pxdrs,&ph->vel[i])) return 0;
    if (!xdr_float(pxdrs,&ph->rho)) return 0;
    if (!xdr_float(pxdrs,&ph->temp)) return 0;
    if (!xdr_float(pxdrs,&ph->hsmooth)) return 0;
    if (!xdr_float(pxdrs,&ph->metals)) return 0;
    if (!xdr_float(pxdrs,&ph->phi)) return 0;
    return 1;
    }

int tipsy_xdr_dark(XDR *pxdrs,struct tipsy_dark_particle *ph) {
    int i;
    if (!xdr_float(pxdrs,&ph->mass)) return 0;
    for(i=0; i<TIPSY_MAXDIM; i++) if (!xdr_float(pxdrs,&ph->pos[i])) return 0;
    for(i=0; i<TIPSY_MAXDIM; i++) if (!xdr_float(pxdrs,&ph->vel[i])) return 0;
    if (!xdr_float(pxdrs,&ph->eps)) return 0;
    if (!xdr_float(pxdrs,&ph->phi)) return 0;
    return 1;
    }

int tipsy_xdr_star(XDR *pxdrs,struct tipsy_star_particle *ph) {
    int i;
    if (!xdr_float(pxdrs,&ph->mass)) return 0;
    for(i=0; i<TIPSY_MAXDIM; i++) if (!xdr_float(pxdrs,&ph->pos[i])) return 0;
    for(i=0; i<TIPSY_MAXDIM; i++) if (!xdr_float(pxdrs,&ph->vel[i])) return 0;
    if (!xdr_float(pxdrs,&ph->metals)) return 0;
    if (!xdr_float(pxdrs,&ph->tform)) return 0;
    if (!xdr_float(pxdrs,&ph->eps)) return 0;
    if (!xdr_float(pxdrs,&ph->phi)) return 0;
    return 1;
    }

#endif //HAVE_TIRPC
