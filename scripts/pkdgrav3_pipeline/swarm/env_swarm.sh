# pkdgrav3 run-time environment on swarm (TIFR; 12 nodes x 64 Ice Lake cores, 504 GB; 10 Gb/s Ethernet with RoCE).
# Sourced by job_sim.pbs (SIM_ENV in run.conf). The conda environment and its Open MPI 5 (UCX) are those of pkdgrav3's
# pegasus set-up (tools/pegasus/env_pegasus.sh); swarm adds:
#   - mpirun starts its daemons on the other nodes with ssh; with the conda library path the system ssh loads conda's
#     OpenSSL and fails ("OpenSSL version mismatch"): ssh_noconda (next to this file) runs it without that path
#   - CONDA_BASE: Miniforge in /user1 (install_production.sh swarm)
export PKD_IO_ROOT=${PKD_IO_ROOT:-/user1/$USER/sahyadri_pipeline/pkdgrav3_io}
export CONDA_BASE=${CONDA_BASE:-/user1/$USER/miniforge3}
source "$PKD_IO_ROOT/pkdgrav3/tools/pegasus/env_pegasus.sh"
SWARM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mpirun_pegasus_args="--prtemca plm_ssh_agent $SWARM_DIR/ssh_noconda $mpirun_pegasus_args"
