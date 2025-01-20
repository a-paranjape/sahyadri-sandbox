#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void update_density(int ndata, int grid, double *density, 
		    double *x, double *y, double *z, double cell_size, double interlace);


void update_density(int ndata, int grid, double *density, 
		    double *x, double *y, double *z, double cell_size, double interlace)
{

  int i;
  int c0,c1,c2,cp10,cp11,cp12;
  int grid2 = grid*grid;
  double cx,cy,cz,dx,dy,dz,tx,ty,tz;
  int ix,iy,iz;

  for(i=0;i<ndata;i++){

    cx = x[i]/cell_size;// + 0.5*interlace;
    cy = y[i]/cell_size;// + 0.5*interlace;
    cz = z[i]/cell_size;// + 0.5*interlace;

    ix = (int) cx;
    iy = (int) cy;
    iz = (int) cz;
    
    c0 = ix % grid;
    c1 = iy % grid;
    c2 = iz % grid;
    cp10 = (c0+1) % grid;
    cp11 = (c1+1) % grid;
    cp12 = (c2+1) % grid;

    dx = fabs((cx - (double)ix));// - 0.5); // incorrect subtraction pointed out by sujatha r. changed 15 Oct 2018
    dy = fabs((cy - (double)iy));// - 0.5);
    dz = fabs((cz - (double)iz));// - 0.5);

    tx = 1.0 - dx;
    ty = 1.0 - dy;
    tz = 1.0 - dz;

    density[c0+grid*c1+grid2*c2] += tx*ty*tz;
    density[cp10+grid*c1+grid2*c2] += dx*ty*tz;
    density[c0+grid*cp11+grid2*c2] += tx*dy*tz;
    density[c0+grid*c1+grid2*cp12] += tx*ty*dz;
    density[cp10+grid*cp11+grid2*c2] += dx*dy*tz;
    density[cp10+grid*c1+grid2*cp12] += dx*ty*dz;
    density[c0+grid*cp11+grid2*cp12] += tx*dy*dz;
    density[cp10+grid*cp11+grid2*cp12] += dx*dy*dz;
    
  }
  return;
}
