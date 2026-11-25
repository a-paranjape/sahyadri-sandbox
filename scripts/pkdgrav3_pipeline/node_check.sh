#!/bin/bash
# Health check of the node it runs on, before a pipeline job starts (pipeline_env.sh) and as the run manager's
# periodic probe (tools/sahyadri_manager, `ssh <node> bash -s -- ARGS < node_check.sh`: the script comes over
# stdin, so a hung /user does not stop it from being read). Takes ~15 s; no probe can block it (each runs in the
# background with a deadline). Problems seen on pawna (Nov 2026) that it detects:
#   bad   (the job should not start here):
#     io        a small write + fsync + read on each --paths directory (NFS /user, /storage) not done in --io-s (20 s)
#     pressure  memory pressure "full" avg60 > --psi (20 %): all tasks stalled on memory (node1: ~90 % with idle CPUs)
#     mem       MemAvailable < --mem GB (the job's request; page cache that cannot be dropped counts as used)
#     tmp       free space in --tmpdir < --tmp GB (staging of checkpoints and snapshots, Rockstar local output)
#   warn  (shown, the job still starts):
#     writeback Writeback in /proc/meminfo > 0 kB and unchanged over 3 samples 5 s apart: an NFS commit that never
#               completes (kworker flush-0:5x in D state). On node1 (23-25 Nov) it came with the hang that stopped
#               every job; on nodes 2 and 4 (since 20 Nov) the other I/O still works (drains at 34 MB/s). Reboot due
#     dstate    processes in D state for more than 10 min (hung root cron `sync` on nodes 1, 2, 4 since 20 Nov)
#     tmp_low   free space in --tmpdir < --tmp-warn GB
# Also lists this user's directories in --tmpdir (<user>_stage_*, <user>_rockstar_*) with their size, so that
# leftovers of killed jobs can be found (the manager decides which are not in use).
# usage: node_check.sh [--paths "DIR ..."] [--mem GB] [--tmp GB] [--tmp-warn GB] [--tmpdir /var/tmp] [--psi 20]
#                      [--io-s 20] [--user USER]
# Output: a summary on stderr and, last on stdout, `NODECHECK {json}`. Exit 0 (ok or warn) or 3 (bad).
PATHS=; NEED_MEM=0; NEED_TMP=0; WARN_TMP=0; TMPD=/var/tmp; PSI_MAX=20; IO_S=20; U=${USER:-$(id -un)}
while [ $# -gt 0 ]; do
  case $1 in
    --paths) PATHS=$2; shift 2 ;;
    --mem) NEED_MEM=$2; shift 2 ;;
    --tmp) NEED_TMP=$2; shift 2 ;;
    --tmp-warn) WARN_TMP=$2; shift 2 ;;
    --tmpdir) TMPD=$2; shift 2 ;;
    --psi) PSI_MAX=$2; shift 2 ;;
    --io-s) IO_S=$2; shift 2 ;;
    --user) U=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
H=$(hostname -s)
BAD=(); WARN=()
js() { local s=${1//\\/\\\\}; s=${s//\"/\\\"}; printf '"%s"' "$s"; }
jarr() { local out= x; for x in "$@"; do out="$out${out:+, }$(js "$x")"; done; printf '[%s]' "$out"; }

# --- I/O probe on each path, in the background with a deadline (a hung NFS write cannot be killed)
IOJ=
for d in $PATHS; do
  f="$d/.nodecheck.$H.$$"
  ( dd if=/dev/zero of="$f" bs=4k count=1 conv=fsync status=none 2>/dev/null && cat "$f" > /dev/null && rm -f "$f" ) &
  pid=$!; t0=$(date +%s%N); ok=
  for ((i = 0; i < IO_S * 10; i++)); do
    kill -0 $pid 2>/dev/null || { wait $pid && ok=1; break; }
    sleep 0.1
  done
  ms=$(( ($(date +%s%N) - t0) / 1000000 ))
  if [ -n "$ok" ]; then
    IOJ="$IOJ${IOJ:+, }$(js "$d"): $ms"
  else
    kill -0 $pid 2>/dev/null && BAD+=("io: write+fsync+read in $d not done after ${IO_S} s (file system hung?)") ||
      BAD+=("io: write+fsync+read in $d failed")
    IOJ="$IOJ${IOJ:+, }$(js "$d"): -1"
  fi
done

# --- Writeback: three samples 5 s apart (the I/O probes above already took some time)
wb() { awk '/^Writeback:/ {print $2}' /proc/meminfo; }
W1=$(wb); sleep 5; W2=$(wb); sleep 5; W3=$(wb)
if [ "${W1:-0}" -gt 0 ] && [ "$W1" = "$W2" ] && [ "$W2" = "$W3" ]; then
  WARN+=("writeback: ${W1} kB in Writeback, unchanged for 10 s (an NFS commit hung; reboot due)")
fi

# --- memory pressure (PSI)
PSI=$(awk '/^full/ {for (i = 1; i <= NF; i++) if ($i ~ /^avg60=/) {sub("avg60=", "", $i); print $i}}' /proc/pressure/memory 2>/dev/null)
PSI=${PSI:-0}
if awk -v p="$PSI" -v m="$PSI_MAX" 'BEGIN{exit !(p > m)}'; then
  BAD+=("pressure: memory pressure full avg60 = ${PSI} % (> ${PSI_MAX} %): tasks stalled on memory")
fi

# --- available memory
MA=$(awk '/^MemAvailable:/ {printf "%.0f", $2 / 1048576}' /proc/meminfo)
if [ "${NEED_MEM%.*}" -gt 0 ] && [ "$MA" -lt "${NEED_MEM%.*}" ]; then
  BAD+=("mem: ${MA} GB available, the job asks for ${NEED_MEM%.*} GB (page cache that cannot be freed?)")
fi

# --- processes in D state for more than 10 min (comm only: reading cmdline of such a process can block)
DS=$(ps -eo stat=,etimes=,comm= 2>/dev/null | awk '$1 ~ /^D/ && $2 > 600 {n++; c[$3]++; if ($2 > m) m = $2}
     END{for (k in c) s = s (s ? ", " : "") k " x" c[k]; printf "%d|%.1f|%s\n", n, m / 86400, s}')
DN=${DS%%|*}; DS=${DS#*|}; DD=${DS%%|*}; DL=${DS#*|}
[ "$DN" -gt 0 ] && WARN+=("dstate: $DN processes in D state for > 10 min, oldest ${DD} d ($DL)")

# --- local disk
TF=$(df -Pk "$TMPD" 2>/dev/null | awk 'NR == 2 {printf "%.0f", $4 / 1048576}')
TF=${TF:-0}
if [ "${NEED_TMP%.*}" -gt 0 ] && [ "$TF" -lt "${NEED_TMP%.*}" ]; then
  BAD+=("tmp: ${TF} GB free in $TMPD, the job needs ~${NEED_TMP%.*} GB")
elif [ "${WARN_TMP%.*}" -gt 0 ] && [ "$TF" -lt "${WARN_TMP%.*}" ]; then
  WARN+=("tmp_low: ${TF} GB free in $TMPD (estimate ~${WARN_TMP%.*} GB)")
fi

# --- this user's directories in TMPD (sizes with a deadline)
LJ=
for d in "$TMPD/${U}"_stage_* "$TMPD/${U}"_rockstar_*; do
  [ -d "$d" ] || continue
  gb=$(timeout 60 du -sk "$d" 2>/dev/null | awk '{printf "%.1f", $1 / 1048576}')
  LJ="$LJ${LJ:+, }{\"dir\": $(js "$d"), \"gb\": ${gb:--1}, \"mtime\": $(stat -c %Y "$d" 2>/dev/null || echo 0)}"
done

ST=ok; [ ${#WARN[@]} -gt 0 ] && ST=warn; [ ${#BAD[@]} -gt 0 ] && ST=bad
for x in "${BAD[@]}"; do echo "node_check $H BAD  $x" >&2; done
for x in "${WARN[@]}"; do echo "node_check $H WARN $x" >&2; done
echo "node_check $H: $ST (pressure full60 $PSI %, Writeback $W1/$W2/$W3 kB, MemAvailable $MA GB, $TMPD free $TF GB, D>10min $DN)" >&2
printf 'NODECHECK {"host": %s, "t": %s, "status": %s, "bad": %s, "warn": %s, "psi_full60": %s, "writeback_kb": [%s, %s, %s], "memavail_gb": %s, "tmp_free_gb": %s, "dstate": %s, "io_ms": {%s}, "dirs": [%s]}\n' \
  "$(js "$H")" "$(date +%s)" "$(js $ST)" "$(jarr "${BAD[@]}")" "$(jarr "${WARN[@]}")" "$PSI" "${W1:-0}" "${W2:-0}" "${W3:-0}" \
  "${MA:-0}" "$TF" "$DN" "$IOJ" "$LJ"
[ $ST = bad ] && exit 3
exit 0
