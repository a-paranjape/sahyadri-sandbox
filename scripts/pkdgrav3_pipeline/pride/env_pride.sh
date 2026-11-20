# pkdgrav3 run-time environment on pride (TIFR theory; 16 nodes x 20 Haswell cores (2 x Xeon E5-2660 v3, AVX2),
# 189-252 GB, 1 Gb/s Ethernet, no InfiniBand). Sourced by job_sim.pbs (SIM_ENV in run.conf).
# pride shares swarm's Lustre /user1 (home) and Miniforge; only pkdgrav3 differs: swarm's build is for Ice Lake
# (AVX-512) and does not run here, so pride has its own (install_production.sh pride: ARCH=haswell) in
# /user1/$USER/sahyadri_pride. The rest (conda MPI, ssh without conda's library path) is swarm's environment.
export PKD_IO_ROOT=${PKD_IO_ROOT:-/user1/$USER/sahyadri_pride/pkdgrav3_io}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../swarm" && pwd)/env_swarm.sh"
