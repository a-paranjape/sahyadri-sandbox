#!/bin/bash
# End-to-end test: Rockstar (PARALLEL_IO, one machine) on one snapshot through three routes:
#   pkc   native PKC reader            (INBASE <dir>/pkc, files snap.0.s*.pkc)
#   gad   pkc_to_gadget --blocks NB     (INBASE <dir>/gad, files g.000.<k>, g.000.rockstar.cfg)
#   h5g   pkc_to_gadget --from-hdf5     (INBASE <dir>/h5g, same names)
# usage: run_rockstar_e2e.sh <rockstar binary> <dir> [NB=8] [FORCE_RES=0.03]
set -u
RS=${1:?rockstar binary}; DIR=${2:?dir}; NB=${3:-8}; FR=${4:-0.03}
run() { # route, extra config lines
  local route=$1 out=$DIR/out_$1
  rm -rf "$out"; mkdir -p "$out"
  cat > "$out/rs.cfg" <<EOF
INBASE = "$DIR/$route"
OUTBASE = "$out"
NUM_SNAPS = 1
STARTING_SNAP = 0
NUM_BLOCKS = $NB
NUM_WRITERS = $NB
FORK_READERS_FROM_WRITERS = 1
FORK_PROCESSORS_PER_MACHINE = $NB
PARALLEL_IO = 1
PARALLEL_IO_SERVER_ADDRESS = "localhost"
PERIODIC = 1
FORCE_RES = $FR
MIN_HALO_OUTPUT_SIZE = 20
EOF
  printf '%s\n' "$2" >> "$out/rs.cfg"
  ( cd "$out" && "$RS" -c rs.cfg > server.log 2>&1 ) &
  local spid=$!
  for _ in $(seq 1 100); do [ -f "$out/auto-rockstar.cfg" ] && break; sleep 0.1; done
  ( cd "$out" && "$RS" -c auto-rockstar.cfg > client.log 2>&1 )
  wait $spid
  echo "$route: server exit $?, $(grep -c . "$out/out_0.list") lines in out_0.list, $(du -sh "$out" | cut -f1)"
}
t0=$(date +%s)
run pkc 'FILE_FORMAT = "PKC"
FILENAME = "snap.<snap>.<block>"'
t1=$(date +%s)
run gad "$(grep -v '^#' "$DIR/gad/g.000.rockstar.cfg")
FILENAME = \"g.<snap>.<block>\""
t2=$(date +%s)
run h5g "$(grep -v '^#' "$DIR/h5g/g.000.rockstar.cfg")
FILENAME = \"g.<snap>.<block>\""
t3=$(date +%s)
echo "wall: pkc $((t1-t0)) s, gad $((t2-t1)) s, h5g $((t3-t2)) s"
