# sahyadri-sandbox
Sandbox for testing codes and scripts related to Sahyadri simulations at IUCAA/TIFR/IISER-Pune/NCRA

## Introduction ##
As an offshoot of the 2nd edition of the [PMCAP](https://www.tifr.res.in/~shadab.alam/PM_CAP_meeting/) meeting held at IUCAA, Pune, it was decided to build a suite of cosmological N-body simulations that explore various cosmological and dark matter models using high-resolution, intermediate volume configurations that are optimal for exploring statistics related to the cosmic web, beyond-2pt observables and the small-scale phenomenology of dark matter.

The simulation suite is named **Sayhadri** (inspired by the mountain range one must cross to travel between Pune and Mumbai), with a smaller pilot study named **Sinhagad** (after a popular mountain in the range).

This repository collects various pieces of code and general information that is needed for setting up and running individual realisations in **Sahyadri** or **Sinhagad**. The information here should, in principle, suffice to set up an end-to-end local pipeline that starts with calculating a transfer function, performs the simulation, finds halos and finally analyses all the output to produce some basic summary statistics like power spectra and mass functions.

## Code list ##
We will use the following codes:
1. Transfer function: `CLASS` -- [see here](https://lesgourg.github.io/class_public/class.html)
2. Initial conditions and N-body simulation: `GADGET-4` (includes `NGen-IC`) -- [see here](https://wwwmpa.mpa-garching.mpg.de/gadget4/); documentation is also copied in [docs/gadget-4](/docs/gadget-4)
3. Halo finding:
   1. Individual snapshots: `ROCKSTAR` -- [see here](https://bitbucket.org/gfcstanford/rockstar/)
   2. Merger trees: `CONSISTENT-TREES` -- [see here](https://bitbucket.org/pbehroozi/consistent-trees/)
4. Post-processing: Included as a customizable Python script [here](/scripts/post-process/postprocess.py)

## Installation tips ##
Here we provide some installation tips for setting up these codes in a [PBS environment](https://en.wikipedia.org/wiki/Portable_Batch_System). For further details, please see the individual documentation pages linked above.

### List of modules ###
Include the following line in the install user's `.bashrc` file, replacing the module names with local ones (the ones below are for installing on [Pegasus](http://hpc.iucaa.in/))
###
	module add null gcc/11.2.0 anaconda3 hdf5-1.14.2 openmpi-4.1.5-gcc-11.2 fftw-3.3.10 gsl-2.6 gcc-8.2.0

Including `anaconda3` ensures that the command `python` links to Python3.

### Installing `CLASS` ###
1. Download the latest `class_public***.tar.gz` file from the `CLASS` repository and unzip it in the local install folder.
2. The Makefile should not need any edits if the module list above is loaded.
3. Run the following in the `class_public` folder
   ```
   make clean
   make
   ```
   which should compile the code with Python support.


### Installing `GADGET-4` ###
**Note 1:** The tips below will produce an installation compatible with the [run_gadget.sh](/scripts/gadget/run_gadget.sh) script in this repository.  
**Note 2:** It is **highly** recommended to read through the `GADGET-4` [code documentation](/docs/gadget4/).  
**Note 3:** All appearances of `Pegasus` or `pegasus` can be consistently replaced with any local cluster name.

1. Create and navigate to the folder `$CODE_HOME/code/Gadget-4/` where `$CODE_HOME` is the path to the install user's home (e.g., `/mnt/home/faculty/caseem`)
2. Clone into the `GADGET-4` git repository [here](http://gitlab.mpcdf.mpg.de/vrs/gadget4).
3. Create `Makefile.systype` in `gadget4/`:
    1. In the `gadget4/` folder, copy the file `Template-Makefile.systype` to `Makefile.systype`.
    2. Edit `Makefile.systype` and add an uncommented line `SYSTYPE="Pegasus"`.  
4. Create `Makefile.comp` and `Makefile.path` in `gadget4/buildsystem/`:
    1. In the `gadget4/buildsystem/` folder, copy `Makefile.comp.gcc` to `Makefile.comp.pegasus`
    2. In the same folder, create a file `Makefile.path.pegasus` with library and include paths. E.g., with the modules loaded above, the contents of this file would be
       ```
       GSL_INCL   = -I/mnt/csoft/libraries/gsl-2.6/include
       GSL_LIBS   = -L/mnt/csoft/libraries/gsl-2.6/lib
       FFTW_INCL  = -I/mnt/csoft/libraries/fftw-3.3.10/include
       FFTW_LIBS  = -L/mnt/csoft/libraries/fftw-3.3.10/lib
       HDF5_INCL  = -I/mnt/csoft/libraries/hdf5-1.14.2/include
       HDF5_LIBS  = -L/mnt/csoft/libraries/hdf5-1.14.2/lib
       HWLOC_INCL = 
       HWLOC_LIBS = 
       ```
5. Edit `gadget4/Makefile`:
   Look for the code block starting with the comment `#define available Systems`. Just beneath the comment, include the following lines of code
   ```
   ifeq ($(SYSTYPE),"Pegasus")
   include buildsystem/Makefile.comp.pegasus
   include buildsystem/Makefile.path.pegasus
   endif
   ```
   appropriately replacing the `Pegasus` references with the local names used for these files.
6. **Compile executables.**
   We will need a binary executable for each PM configuration. We will organize these into different folders under `code/Gadget-4/' (i.e., one level above `gadget4/') labelled by the mesh size. E.g., the binary for a 1024<sup>3</sup> grid with `NGen-IC` support will sit in the folder `mesh1024-NGenIC`, and for the same mesh but without `NGen-IC` in the folder `mesh1024`.  
   > Below, we will **always assume** that a simulation with *N*<sup>3</sup> particles will use a (2*N*)<sup>3</sup> mesh. The primary [job submission script](/scripts/submit_CiPod.sh) is hard-coded for this. To change this assumption, the submission script must be edited.
   
   Here we focus on a 1024<sup>3</sup> grid (i.e., a simulation with 512<sup>3</sup> particles) with `NGen-IC` support, whose binary will sit in `Gadget-4/mesh1024-NGenIC/`. Repeat the following steps for each such required mesh size / IC combination.  
   To disable `NGen-IC` support, simply skip the last 2 lines of step iii and the last line of step iv.
    1. Create the folder `Gadget-4/mesh1024-NGenIC/`.  
    2. Copy the file `Config.sh' from `Gadget-4/gadget4/` to `Gadget-4/mesh1024-NGenIC/` and open it for editing in the latter folder.
    3. Uncomment the following lines:
       ```
       # PERIODIC
       # LEAN
       # INITIAL_CONDITIONS_CONTAIN_ENTROPY
       # OUTPUT_POTENTIAL # if gravitational potential needed in output
       # NGENIC_2LPT # if 2LPT needed, else leave commented for Zeldovich
       # CREATE_GRID
       ```
   4. Uncomment and set the following variables
      ```
      # PMGRID=1024 # mesh size along each dimension
      # DOUBLEPRECISION=1
      # MAX_NUMBER_OF_RANKS_WITH_SHARED_MEMORY=32 # set to number of cores on each node, default 64. 
      # NGENIC=512 # half of mesh size, can be different but see comment above
      ```
   5. Run the code
      ```
      make clean DIR=$CODE_HOME/code/Gadget-4/mesh1024-NGenIC
      make DIR=$CODE_HOME/code/Gadget-4/mesh1024-NGenIC
      ```
    The script [here](/scripts/gadget/makeallgrids.sh) automates the compilation of binaries for mesh sizes 128<sup>3</sup> to 4096<sup>3</sup>, with and without `NGen-IC` support, assuming the required folders and `Config.sh` files exist. It can be run from the folder `Gadget-4/gadget4/`. Executing `./makeallgrids.sh clean` will pass the `clean` flag to all compilations, while `./makeallgrids.sh` will compile the binaries. 

### Installing `ROCKSTAR` ###
We use `ROCKSTAR` version `36ce9eea36ee` with some custom modifications to enable
* reading `GADGET-4` HDF5 output snapshots
* support for non-zero `Omega_k`
The modifications are described in [this file](/scripts/rockstar/readme_aseem.txt).

`ROCKSTAR` can be installed with `GADGET-4` HDF5 support by simply typing `make' at the command prompt in the install folder after making all changes mentioned in the file above.

Contact Aseem Paranjape for help installing this modified version.

### Installing `CONSISTENT-TREES` ###
1. Clone into the `CONSISTENT-TREES` repository [here](https://bitbucket.org/pbehroozi/consistent-trees).
2. Type `make` at the command prompt in the install folder.

## Contact ##
Aseem Paranjape: aseem_at_iucaa_dot_in

