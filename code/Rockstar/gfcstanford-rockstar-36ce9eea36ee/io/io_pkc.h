#ifndef _IO_PKC_H_
#define _IO_PKC_H_
#ifdef ENABLE_PKC
#include <stdint.h>
#include "../particle.h"

/* FILE_FORMAT = "PKC": pkdgrav3 compressed snapshots (format version 3).
 * filename = INBASE/<snapshot prefix>.<block>, e.g. FILENAME = "snap.<snap>.<block>". */
void load_particles_pkc(char *filename, struct particle **p, int64_t *num_p);

#endif /* ENABLE_PKC */
#endif /* _IO_PKC_H_ */
