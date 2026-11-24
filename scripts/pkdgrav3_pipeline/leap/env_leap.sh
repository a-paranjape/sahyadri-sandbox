# pkdgrav3 run-time environment on leap (TIFR theory; 12 nodes x 32 Skylake cores (2 x Xeon Gold 6142), 187 GB,
# 100 Gb/s hfi1 fabric (Omni-Path-like; UCX: rc_verbs/ud_verbs) + 10/1 GbE; CentOS 7, glibc 2.17). Sourced by
# job_sim.pbs (SIM_ENV in run.conf).
# pkdgrav3 and its conda environment (Open MPI 5 + UCX) are in /user1/$USER/sahyadri_c7/pkdgrav3_io, built by
# build_c7.sh (next to this file): a conda environment solved for glibc 2.17 and TARGET_ARCHITECTURE=skylake-avx512
# (swarm's and pride's builds need glibc 2.27-2.34 and do not start here). The same build runs on flock and raft.
# MPI between nodes (measured 20-21 Nov 2026, 512^3 restart: 2/4/6 nodes 86/82/73 % efficient; 6 us, 3.2 GB/s):
#   - pml ucx over the fabric's verbs: with Open MPI's defaults the ranks choose different PMLs (cm on some nodes,
#     ob1 on others) and MPI_Init fails ("selected pml cm, but peer ... selected pml ob1")
#   - compute-4-3's fabric port is DOWN (ucx finds no transport there): run.conf SIM_SELECT names the nodes
#   - mpirun starts its daemons with ssh_noconda (system ssh without the conda library path, and without host-key
#     checks: the home directory, and with it known_hosts, is shared by swarm, pride, flock, leap and raft, whose
#     compute nodes are not in it, so ssh refused them: "Host key verification failed")
export PKD_IO_ROOT=${PKD_IO_ROOT:-/user1/$USER/sahyadri_c7/pkdgrav3_io}
export CONDA_BASE=${CONDA_BASE:-/user1/$USER/miniforge3}
export UCX_TLS=${UCX_TLS:-rc_verbs,ud_verbs,self,sm} UCX_NET_DEVICES=${UCX_NET_DEVICES:-hfi1_0:1}
source "$PKD_IO_ROOT/pkdgrav3/tools/pegasus/env_pegasus.sh"
LEAP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mpirun_pegasus_args="--prtemca plm_ssh_agent $LEAP_DIR/ssh_noconda --mca pml ucx -x UCX_NET_DEVICES $mpirun_pegasus_args"
