#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include "gravitational_consistency.h"
#include "read_config.h"
#include "check_syscalls.h"

void output_config(char *filename) {
  FILE *output;
  char buffer[1024];
  if (!filename) filename = "grav_consistency.cfg";
  snprintf(buffer, 1024, "%s/%s", OUTBASE, filename);
  output = check_fopen(buffer, "w");

#define string(a,b) fprintf(output, "%s = \"%s\"\n", #a, a);
#define real(a,b) fprintf(output, "%s = %g\n", #a, a);
#define real3(a,b) fprintf(output, "%s = (%g, %g, %g)\n", #a, a[0], a[1], a[2]);
#define integer(a,b) fprintf(output, "%s = %"PRId64"\n", #a, a);
#include "config.template.h"
#undef string
#undef real
#undef real3
#undef integer

  fclose(output);
}


void grav_config(char *filename, int write_cfile) {
  struct configfile c = {0};
  if (filename)
    load_config(&c, filename);

#define string(a,b) a = config_to_string(&c,#a,b)
#define real(a,b) a = config_to_real(&c,#a,b)
#define real3(a,b) config_to_real3(&c,#a,a,b)
#define integer(a,b) a = config_to_real(&c,#a,b)
#include "config.template.h"
#undef string
#undef real
#undef real3
#undef integer
  UNPHYSICAL = 3*METRIC_LIMIT + 1;
  if (EXTRA_PARAMS > MAX_EXTRA_PARAMS) {
    fprintf(stderr, "Number of extra parameters (%"PRId64") exceeds the "
	    " maximum allowed value (%"PRId64").\nPlease change "
	    "MAX_EXTRA_PARAMS in src/tree_halo.h and recompile.\n",
	    EXTRA_PARAMS, MAX_EXTRA_PARAMS);
    exit(1);
  }
  if (EXTRA_PARAMS) {
    char *interp = malloc(EXTRA_PARAMS+1);
    snprintf(interp, EXTRA_PARAMS+1, "%s", EXTRA_PARAM_INTERPOLATIONS);
    if (strlen(interp) < EXTRA_PARAMS) {
      int64_t i;
      for (i=strlen(interp); i<EXTRA_PARAMS; i++) interp[i] = 'l';
      interp[i] = 0;
    }
    free(EXTRA_PARAM_INTERPOLATIONS);
    EXTRA_PARAM_INTERPOLATIONS = interp;    
  }

  free_config(c);

  if (write_cfile) output_config("grav_consistency.cfg");

  if (EXTRA_PARAMS) {
    int64_t i, j=0;
    char *epd = EXTRA_PARAM_DESCRIPTIONS;
    for (i=0; i<strlen(epd); i++,j++) {
      if ((epd[i] == '\\') && (epd[i+1] == 'n')) { 
	epd[j] = '\n'; i++;
	if (epd[i+1] != '#') { j++; epd[j] = '#'; }
      }
      else epd[j] = epd[i];
    }
    epd[j] = 0;
  }
}
