# sahyadri-sandbox
Sandbox for testing codes and scripts related to Sahyadri simulations at IUCAA/TIFR/IISER-Pune/NCRA

## Introduction ##
As an offshoot of the 2nd edition of the [PMCAP](https://www.tifr.res.in/~shadab.alam/PM_CAP_meeting/) meeting held at IUCAA, Pune, it was decided to build a suite of cosmological N-body simulations that explore various cosmological and dark matter models using high-resolution, intermediate volume configurations that are optimal for exploring statistics related to the cosmic web, beyond-2pt observables and the small-scale phenomenology of dark matter.

The simulation suite is named Sahyadri (inspired by the mountain range one must cross to travel between Pune and Mumbai), with a smaller pilot study named Sinhagad (after a popular mountain in the range).

This repository collects various pieces of code and general information that is needed for setting up and running individual realisations in Sahyadri or Sinhagad. The information here should, in principle, suffice to set up an end-to-end local pipeline that starts with calculating a transfer function, performs the simulation, finds halos and finally analyses all the output to produce some basic summary statistics like power spectra and mass functions.

### Under construction ###

## List of modules on Pegasus ##
Include the following line in .bashrc
##
	module add null gcc/11.2.0 anaconda3 hdf5-1.14.2 openmpi-4.1.5-gcc-11.2 fftw-3.3.10 gsl-2.6 gcc-8.2.0
