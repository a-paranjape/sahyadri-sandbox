#!/bin/bash
# Memory monitor: samples a process and all its descendants until the process exits (Linux /proc).
# usage: memmon.sh <root pid> <out.tsv> [interval s, default 5] [pid to exclude with its descendants]
# Each sample: sum over the processes of PSS (proportional set size: a shared page counts 1/n in each of
# the n processes that map it, so the sum is the memory the job really occupies) and of RSS (shared pages
# counted in every process: an upper bound, e.g. for Rockstar's forked readers).
# Output lines "t_s nproc pss_kb rss_kb"; peaks shorter than the interval can be missed.
# pawna puts all PBS jobs in one cgroup (/system.slice/pbs.service), so there is no per-job memory.peak.
root=$1; out=$2; dt=${3:-5}; excl=${4:-0}
t0=$(date +%s.%N)
echo "# t_s nproc pss_kb rss_kb   (root pid $root on $(hostname), every $dt s, started $(date -Is))" > "$out"
while kill -0 "$root" 2>/dev/null; do
  pids=$(ps -e -o pid=,ppid= | awk -v r="$root" -v x="$excl" '{p[$1]=$2}
         END{for(i in p){j=i; while(j>1){ if(j==x) break; if(j==r){print i; break}; j=p[j]}}}')
  line=$(for p in $pids; do awk '/^Rss:/{r=$2} /^Pss:/{s=$2} END{if(NR) print s+0, r+0}' /proc/$p/smaps_rollup 2>/dev/null; done |
         awk '{n++; s+=$1; r+=$2} END{print n+0, s+0, r+0}')
  echo "$(awk -v a="$(date +%s.%N)" -v b="$t0" 'BEGIN{printf "%.1f", a-b}') $line" >> "$out"
  sleep "$dt"
done
