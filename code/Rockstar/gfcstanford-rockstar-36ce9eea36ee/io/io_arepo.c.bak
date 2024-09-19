/*
 * Arepo I/O for Rockstar
 * Dylan Nelson (dnelson@cfa.harvard.edu)
 */
#ifdef ENABLE_HDF5

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <hdf5.h> /* HDF5 required */
#include "io_hdf5.h"
#include "io_arepo.h"
#include "io_util.h"
#include "../universal_constants.h"
#include "../check_syscalls.h"
#include "../config_vars.h"
#include "../config.h"
#include "../particle.h"

#define AREPO_NTYPES 6

void arepo_read_dataset(hid_t HDF_FileID, char *filename, char *gid, char *dataid, struct particle *p, int64_t to_read, int64_t offset, int64_t stride, hid_t type) {
  int64_t width = (type == H5T_NATIVE_LLONG) ? 8 : 4;
  void *buffer = check_malloc_s(buffer, to_read, width*stride);
  int64_t *ibuffer = buffer;
  float *fbuffer = buffer;

  hid_t HDF_GroupID = check_H5Gopen(HDF_FileID, gid, filename);
  hid_t HDF_DatasetID = check_H5Dopen(HDF_GroupID, dataid, gid, filename);
  hid_t HDF_DataspaceID = check_H5Dget_space(HDF_DatasetID);

  check_H5Sselect_all(HDF_DataspaceID);
  hssize_t npoints = H5Sget_select_npoints(HDF_DataspaceID);

  if (npoints != to_read*stride) {
    fprintf(stderr, "[Error] dataspace %s/%s in HDF5 file %s not expected size!\n  (Actual size = %"PRId64" elements; expected size = %"PRId64" elements\n", 
	    gid, dataid, filename, (int64_t)(npoints), stride*to_read);
    exit(1);
  }

  check_H5Dread(HDF_DatasetID, type, buffer, dataid, gid, filename);
  
  H5Sclose(HDF_DataspaceID);
  H5Dclose(HDF_DatasetID);
  H5Gclose(HDF_GroupID);

  if (width == 8)
    for (int64_t i=0; i<to_read; i++)
      p[i].id = ibuffer[i];
  else
    for (int64_t i=0; i<to_read; i++)
      memcpy(((char *)&(p[i]))+offset, fbuffer+(i*stride), stride*width);

  free(buffer);
}

float arepo_readheader_float(hid_t HDF_GroupID, char *filename, char *objName)
{
  char *gid = "Header";
  hid_t HDF_Type = H5T_NATIVE_FLOAT;
  hid_t HDF_AttrID = check_H5Aopen_name(HDF_GroupID, objName, gid, filename);
  hid_t HDF_DataspaceID = check_H5Aget_space(HDF_AttrID);

  check_H5Sselect_all(HDF_DataspaceID);
  
  float data = 0.0;
  check_H5Aread( HDF_AttrID, HDF_Type, &data, objName, gid, filename);

  H5Sclose(HDF_DataspaceID);
  H5Aclose(HDF_AttrID);
  return data;
}


void arepo_readheader_array(hid_t HDF_GroupID, char *filename, char *objName, hid_t type, void *data)
{
  char *gid = "Header";
  hid_t HDF_AttrID = check_H5Aopen_name(HDF_GroupID, objName, gid, filename);
  hid_t HDF_DataspaceID = check_H5Aget_space(HDF_AttrID);
  check_H5Sselect_all(HDF_DataspaceID);

  int64_t ndims = check_H5Sget_simple_extent_ndims( HDF_DataspaceID );
  assert(ndims == 1);
  hsize_t dimsize = 0;
  check_H5Sget_simple_extent_dims(HDF_DataspaceID, &dimsize);
  assert(dimsize == AREPO_NTYPES);
  
  check_H5Aread(HDF_AttrID, type, data, objName, gid, filename);

  H5Sclose(HDF_DataspaceID);
  H5Aclose(HDF_AttrID);
}

void arepo_rescale_particles(struct particle *p, int64_t p_start, int64_t nelems) {
  double vel_rescale = sqrt(SCALE_NOW);
  if (LIGHTCONE) vel_rescale = 1;
	
  for (int64_t i=0; i<nelems; i++) {
    for (int64_t j=0; j<3; j++) {
      p[p_start+i].pos[j]   *= AREPO_LENGTH_CONVERSION;
      p[p_start+i].pos[j+3] *= vel_rescale;
    }
  }
}

void load_particles_arepo(char *filename, struct particle **p, int64_t *num_p)
{	
  hid_t HDF_FileID = check_H5Fopen(filename, H5F_ACC_RDONLY);
  hid_t HDF_Header = check_H5Gopen(HDF_FileID, "Header", filename);
  
  Ol = arepo_readheader_float(HDF_Header, filename, "OmegaLambda");
  Om = arepo_readheader_float(HDF_Header, filename, "Omega0");
  h0 = arepo_readheader_float(HDF_Header, filename, "HubbleParam");
  SCALE_NOW = arepo_readheader_float(HDF_Header, filename, "Time");
  BOX_SIZE = arepo_readheader_float(HDF_Header, filename, "BoxSize");
  BOX_SIZE *= AREPO_LENGTH_CONVERSION;  
  
  uint32_t npart_low[AREPO_NTYPES], npart_high[AREPO_NTYPES] = {0};
  int64_t npart[AREPO_NTYPES];
  float massTable[AREPO_NTYPES];
  
  arepo_readheader_array(HDF_Header, filename, "NumPart_ThisFile", H5T_NATIVE_UINT64, npart);
  arepo_readheader_array(HDF_Header, filename, "NumPart_Total_HighWord", H5T_NATIVE_UINT32, npart_high);
  arepo_readheader_array(HDF_Header, filename, "NumPart_Total", H5T_NATIVE_UINT32, npart_low);
  arepo_readheader_array(HDF_Header, filename, "MassTable", H5T_NATIVE_FLOAT, massTable);
  
  TOTAL_PARTICLES = ( ((int64_t)npart_high[AREPO_DM_PARTTYPE]) << 32 ) 
    + (int64_t)npart_low[AREPO_DM_PARTTYPE];
  
  H5Gclose(HDF_Header);
    
  PARTICLE_MASS   = massTable[AREPO_DM_PARTTYPE] * AREPO_MASS_CONVERSION;
  AVG_PARTICLE_SPACING = cbrt(PARTICLE_MASS / (Om*CRITICAL_DENSITY));
	
  if(RESCALE_PARTICLE_MASS)
    PARTICLE_MASS = Om*CRITICAL_DENSITY * pow(BOX_SIZE, 3) / TOTAL_PARTICLES;
 
  printf("AREPO: filename:       %s\n", filename);
  printf("AREPO: box size:       %g Mpc/h\n", BOX_SIZE);
  printf("AREPO: h0:             %g\n", h0);
  printf("AREPO: scale factor:   %g\n", SCALE_NOW);
  printf("AREPO: Total DM Part:  %" PRIu64 "\n", TOTAL_PARTICLES);
  printf("AREPO: ThisFile DM Part: %" PRIu64 "\n", npart[AREPO_DM_PARTTYPE]);
  printf("AREPO: DM Part Mass:   %g Msun/h\n", PARTICLE_MASS);
  printf("AREPO: avgPartSpacing: %g Mpc/h\n\n", AVG_PARTICLE_SPACING);
  
  if (!npart[AREPO_DM_PARTTYPE]) {
    H5Fclose(HDF_FileID);
    printf("   SKIPPING FILE, PARTICLE COUNT ZERO.\n");
    return;
  }

  int64_t to_read = npart[AREPO_DM_PARTTYPE];
  check_realloc_s(*p, ((*num_p)+to_read), sizeof(struct particle));

  // read IDs, pos, vel
  char buffer[100];
  snprintf(buffer, 100, "PartType%"PRId64, AREPO_DM_PARTTYPE);
  arepo_read_dataset(HDF_FileID, filename, buffer, "ParticleIDs", *p + (*num_p),
	 to_read, (char *)&(p[0][0].id)-(char*)(p[0]), 1, H5T_NATIVE_LLONG);
  arepo_read_dataset(HDF_FileID, filename, buffer, "Coordinates", *p + (*num_p),
	 to_read, (char *)&(p[0][0].pos[0])-(char*)(p[0]), 3, H5T_NATIVE_FLOAT);
  arepo_read_dataset(HDF_FileID, filename, buffer, "Velocities", *p + (*num_p),
	 to_read, (char *)&(p[0][0].pos[3])-(char*)(p[0]), 3, H5T_NATIVE_FLOAT);

  H5Fclose(HDF_FileID);
  
  arepo_rescale_particles(*p, *num_p, to_read);
  
  *num_p += npart[AREPO_DM_PARTTYPE];
}

#endif /* ENABLE_HDF5 */
