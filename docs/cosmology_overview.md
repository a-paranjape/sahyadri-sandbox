
# Cosmology Overview
## Summary of cosmological parameters used in fiducial and variation boxes
Here we summarize the sampling strategies and parameter choices for the 1-parameter and stochastic variations of cosmological parameters. In each case, the parameters being varied have the following fiducial values

* $\Omega_{\rm m}$: 0.3137721
* $h$: 0.6736
* $n_{\rm s}$: 0.9649
* $A_{\rm s}$: 2.098903e-09
* $\Omega_{\rm b}$: 0.0493017
* $\Omega_{\rm k}$: 0.0
* $w_0$ : -1.0

## **Sinhagad** simulations (pilot study)
The **Sinhagad** suite uses linear variations of each parameter $\theta$ with step-size $\Delta\theta$ to facilitate Fisher formalism studies. In addition to the common fiducial set $\theta=\theta_{\rm fid}$, samples are produced at
$\theta_{\rm fid}-2\Delta\theta,\theta_{\rm fid}-\Delta\theta,\theta_{\rm fid}+\Delta\theta,\theta_{\rm fid}+2\Delta\theta$. Each realisation of the suite therefore contains $1 + 4\times 7 = 29$ simulations.

The values of $\Delta\Theta$ for each parameter are as follows

* $\Delta\Omega_{\rm m}$: $0.05 \times \Omega_{\rm m,fid}$
* $\Delta h$: $0.05 \times h_{\rm fid}$
* $\Delta n_{\rm s}$: $0.05 \times n_{\rm s,fid}$
* $\Delta A_{\rm s}$: $0.1 \times A_{\rm s,fid}$
* $\Delta\Omega_{\rm b}$: $0.1 \times \Omega_{\rm b,fid}$
* $\Delta\Omega_{\rm k}$: $0.05$
* $\Delta w_0$ : $0.1$

Particle mass for the default simulation is $4.15153e10\,M_\odot/h$.

## **Sahyadri** simulations
**In progress**
Particle mass for the default simulation is $8.108e7\,M_\odot/h$.

## PMCAP 2024 discussion:
### Priority list for running the +/- variation simulations:
*  $\Omega_{\rm m}$
*  $h$
*   $A_{\rm s}$
*   $w_0$
*   $\Omega_{\rm k}$
*  $n_{\rm s}$
Discussion about whether to use $\sigma_8$ instead of $A_s$, inconclusive. Decided to do some Halofits checks.

 ### Fisher Analysis
 #### Observables to be explored now
 * VVF
 * KNN
 * Redshift space 2pcf
 * Real space 2pcf
 * Halo mass function
 * Density profiles around massive clusters
 * Mass accretion history
 * Assembly bias: concentration + tidal field
The first 4 point to be considered immediately, with halos with 2 or more mass cuts.
2PCFs to be obtained for halos + hod
 #### Observables to be explored later
 * Mock HI+ optical catalogues
 * relativistic effects X dipole
 * some kind of intensity mapping

#### Covariance matrix calculations
* L200N256: 20 realisations, jk averages or 100 realisations, proper errors. To be compared to jk error on single realisation
* L50N512: 20 realisations. Compare this covariance matrix with that of L200N256.
* Behaviour as a function of mass cuts.


