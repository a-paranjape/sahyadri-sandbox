#!/bin/bash
# The stacked-potential files of the report (data/<run>/profiles/), made on a workstation from copies of the four
# outputs closest to z = 3, 1, 0.5, 0 (snapshots, pkdgrav3 p.par, halo catalogues):
#   <kind>_out<i>.npz          particle-weighted, 36 log bins 0.02-50 Mpc/h        (gadget, pkd, pkc)
#   <kind>_out<i>_lin.npz      particle-weighted, linear bins 0-50 Mpc/h, 0.2 wide  (gadget, pkd, pkc)
#   <kind>_out<i>_vol.npz      volume-weighted (grid points), log bins             (gadget, pkd)
#   <kind>_out<i>_vol_lin.npz  volume-weighted, linear bins                        (gadget, pkd)
# usage: run_stacks_local.sh <raw dir: g/ p/ hg/ hp/> <particles.csv> <out dir> <halo ext: trees|list> <mbins> [vgrid]
# pkdgrav3 potentials are aligned to Gadget-4's zero point by offset_median of particles.csv (as job_compare.pbs).
set -eu
RAW=$1; CSV=$2; OUT=$3; EXT=$4; MB=$5; VG=${6:-0}
S=$(cd "$(dirname "$0")/../../scripts/gadget_pkdgrav3_compare" && pwd)
PY=${PY:-python3}; W=${WORKERS:-8}
mkdir -p "$OUT" "$RAW/vgrid"
LOG=(--rbins 0.02,50,36); LIN=(--rbins lin:0,50,0.2)
for pair in 19:00400 46:00940 64:01300 100:02020; do
  i=${pair%:*}; s=${pair#*:}; g=$(printf %03d $i)
  off=$("$PY" -c "
import csv
print([r['offset_median'] for r in csv.DictReader(open('$CSV')) if r['out']=='$i'][0])")
  G=(gadget "$RAW/g/snapshot_$g.hdf5" "$RAW/hg/out_$i.$EXT")
  for kind in pkd pkc; do eval "K_$kind=($kind $RAW/p/snap.$s $RAW/hp/out_$i.$EXT)"; done
  PK=(--par "$RAW/p/p.par" --offset "$off")
  C=(--mbins "$MB" --workers "$W")
  V=(--weight volume --target 8192 --vgrid "$VG")
  "$PY" -W ignore "$S/halo_potential_profile.py" "${G[@]}" "$OUT/gadget_out$i.npz" "${C[@]}" "${LOG[@]}" | tail -1
  "$PY" -W ignore "$S/halo_potential_profile.py" "${G[@]}" "$OUT/gadget_out${i}_lin.npz" "${C[@]}" "${LIN[@]}" | tail -1
  "$PY" -W ignore "$S/halo_potential_profile.py" "${G[@]}" "$OUT/gadget_out${i}_vol.npz" "${C[@]}" "${LOG[@]}" "${V[@]}" --vcache "$RAW/vgrid/gadget_out$i.npy" | grep -E "gridded|wrote"
  "$PY" -W ignore "$S/halo_potential_profile.py" "${G[@]}" "$OUT/gadget_out${i}_vol_lin.npz" "${C[@]}" "${LIN[@]}" "${V[@]}" --vcache "$RAW/vgrid/gadget_out$i.npy" | tail -1
  for kind in pkd pkc; do
    eval "K=(\"\${K_$kind[@]}\")"
    "$PY" -W ignore "$S/halo_potential_profile.py" "${K[@]}" "$OUT/${kind}_out$i.npz" "${C[@]}" "${LOG[@]}" "${PK[@]}" | tail -1
    "$PY" -W ignore "$S/halo_potential_profile.py" "${K[@]}" "$OUT/${kind}_out${i}_lin.npz" "${C[@]}" "${LIN[@]}" "${PK[@]}" | tail -1
  done
  "$PY" -W ignore "$S/halo_potential_profile.py" "${K_pkd[@]}" "$OUT/pkd_out${i}_vol.npz" "${C[@]}" "${LOG[@]}" "${PK[@]}" "${V[@]}" --vcache "$RAW/vgrid/pkd_out$i.npy" | grep -E "gridded|wrote"
  "$PY" -W ignore "$S/halo_potential_profile.py" "${K_pkd[@]}" "$OUT/pkd_out${i}_vol_lin.npz" "${C[@]}" "${LIN[@]}" "${PK[@]}" "${V[@]}" --vcache "$RAW/vgrid/pkd_out$i.npy" | tail -1
done
