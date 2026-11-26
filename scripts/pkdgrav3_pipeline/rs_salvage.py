#!/usr/bin/env python3
"""Rescue the finished part of a failed Rockstar batch from its node's local disk, so a rerun continues instead
of starting over.

usage: rs_salvage.py <run.conf> --job <id> --node <host> --first <a> --last <b> [--delete] [--dry-run]

Run on the head (or login) node of the site where the job ran, never on the compute node. job_rockstar.pbs
(RS_LOCAL_OUT=1) writes to <RS_LOCAL_DIR>/<user>_rockstar_<job> on the node and copies the finished catalogues to
$HALO_DIR every RS_SYNC_S. When the node's NFS client hangs, those copies hang too, and the job is killed with
the finished work still on the node's disk. pawna node1, 25 Nov 2026: r4 batch 59-68 (job 2539124) and r3 batch
85-96 (2539163). Their reruns redid about 8 and 5 finished snapshots, about 4.8 h on node3. Reading the node's
disk over ssh and writing $HALO_DIR from here goes around the node's NFS client.

What a rerun of the batch needs to continue after snapshot m (job_rockstar.pbs): out_<a>..out_<m-1>.list,
the complete halos_<name of m>.*.bin set (NUM_WRITERS files, none empty), rockstar.cfg (NUM_WRITERS) and a
restart.cfg whose RESTART_SNAP is above m. m is the newest snapshot of the batch below the node's RESTART_SNAP
(Rockstar had finished it) for which all of these are on the node, and m < b unless out_<b>.list is there too.
The files are copied as .<name>.part and renamed once their size matches. A file is skipped only when $HALO_DIR
already has this job's own copy of it: same size and same modification time (the job's copies use cp -p, which
keeps the node's time stamp, so this needs no agreement between clocks). Every other file is replaced: all
the files come from this one job, because catalogues of two Rockstar runs have different halo IDs, and mixing
them would break the descendant links between snapshots. No time stamp is compared across machines. Clocks differ:
pawna nodes were 2-10 s behind the head node on 27 Nov 2026, and pride nodes up to ~8 min on 25 Nov.
A later job of the run is recognised without clocks: the RESTART_SNAP in $HALO_DIR's restart.cfg is further than
the node's (then nothing is copied), or the caller says so with --no-copy (the manager passes it when a
Rockstar job of the run with a higher PBS id has started). restart.cfg is written last (OUTBASE = $HALO_DIR, RESTART_SNAP = m+1), so an
interrupted rescue never looks complete.
--delete: afterwards remove the node's directory (what a successful job does itself at its end); only after a
rescue that verified every copied file, or when there was nothing to rescue and the directory holds no catalogue
missing from $HALO_DIR.
Prints progress, then one JSON line: {"ok": true, "m": .., "continue_at": .., "files": .., "bytes": ..} or
{"ok": false, "error": ..}.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retention  # noqa: E402

SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30', '-o', 'ServerAliveInterval=30',
       '-o', 'ServerAliveCountMax=6']
LIST = r'''
import json, os, re, sys
d = sys.argv[1]
out = dict(exists=os.path.isdir(d), files={}, mtimes={}, restart_snap=None, num_writers=None, restart_cfg=None, rockstar_cfg=None)
if out["exists"]:
    for n in os.listdir(d):
        p = os.path.join(d, n)
        if os.path.isfile(p) and (n.endswith(".list") or n.endswith(".bin") or n.endswith(".cfg")):
            out["files"][n] = os.path.getsize(p)
            out["mtimes"][n] = os.path.getmtime(p)
    for n, k in (("restart.cfg", "restart_cfg"), ("rockstar.cfg", "rockstar_cfg")):
        p = os.path.join(d, n)
        if os.path.isfile(p):
            out[k] = open(p, errors="replace").read()
    m = re.search(r"^\s*RESTART_SNAP\s*=\s*(\d+)", out["restart_cfg"] or "", re.M)
    out["restart_snap"] = int(m.group(1)) if m else None
    m = re.search(r"^\s*NUM_WRITERS\s*=\s*(\d+)", out["rockstar_cfg"] or "", re.M)
    out["num_writers"] = int(m.group(1)) if m else None
print(json.dumps(out))
'''


def say(*a):
    print('[%s] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), ' '.join(str(x) for x in a)), flush=True)


def node_run(node, cmd, timeout=600):
    return subprocess.run(SSH + [node, cmd], capture_output=True, text=True, timeout=timeout)


def fetch(node, src, dst, size, mtime):
    """Copy ``src`` on the node to ``dst`` here (ssh cat), through .<name>.part; checks the size and gives the copy
    the node file's modification time (as cp -p does), so a later pass recognises it."""
    part = os.path.join(os.path.dirname(dst), '.%s.part' % os.path.basename(dst))
    t0 = time.time()
    with open(part, 'wb') as f:
        r = subprocess.run(SSH + [node, 'cat %s' % shlex.quote(src)], stdout=f, stderr=subprocess.PIPE,
                           timeout=max(3600, size / 20e6))          # at least 20 MB/s, else give up
    got = os.path.getsize(part)
    if r.returncode != 0 or got != size:
        os.remove(part)
        raise IOError('copy of %s:%s failed (rc %d, %d of %d bytes): %s' % (
            node, src, r.returncode, got, size, r.stderr.decode(errors='replace')[-200:]))
    os.utime(part, (mtime, mtime))
    os.replace(part, dst)
    dt = time.time() - t0
    say('copied %s (%.2f GB, %.0f s, %.0f MB/s)' % (os.path.basename(dst), size / 1e9, dt, size / 1e6 / max(dt, 1e-3)))
    return size


def choose(L, first, last, names):
    """The snapshot m to continue after, or None (see the module docstring)."""
    files, r, nw = L['files'], L['restart_snap'], L['num_writers']
    if r is None or not nw:
        return None, 'no restart.cfg / NUM_WRITERS on the node'
    num = {n: i for i, n in enumerate(names)}
    sets = {}
    for n, s in files.items():
        mm = re.match(r'^halos_(.+)\.(\d+)\.bin$', n)
        if mm and mm.group(1) in num:
            sets.setdefault(num[mm.group(1)], []).append(s)
    for m in sorted((x for x in sets if first <= x <= last and x < r), reverse=True):
        if len(sets[m]) != nw or min(sets[m]) == 0:
            continue
        if any(files.get('out_%d.list' % s, 0) == 0 for s in range(first, m)):
            continue
        if m == last and files.get('out_%d.list' % last, 0) == 0:
            continue
        return m, None
    return None, 'no finished snapshot of the batch with its catalogues and halo set on the node (RESTART_SNAP %s)' % r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('conf')
    ap.add_argument('--job', required=True)
    ap.add_argument('--node', required=True)
    ap.add_argument('--first', type=int, required=True)
    ap.add_argument('--last', type=int, required=True)
    ap.add_argument('--delete', action='store_true')
    ap.add_argument('--no-copy', action='store_true', help='a later job wrote the halo directory: copy nothing (with '
                    '--delete: only remove the node directory if every catalogue in it is in the halo directory)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    E = retention.load_conf(a.conf)
    halo = retention.paths(E)['halo']
    src = os.path.join(E.get('RS_LOCAL_DIR') or '/var/tmp', '%s_rockstar_%s' % (os.environ.get('USER', ''), a.job))
    res = dict(ok=False, node=a.node, dir=src, halo=halo, first=a.first, last=a.last)
    try:
        names_f = os.path.join(halo, 'snapnames_b%d.txt' % (a.last + 1))
        names = [x.strip() for x in open(names_f) if x.strip()]
        r = node_run(a.node, 'python3 -c %s %s' % (shlex.quote(LIST), shlex.quote(src)), timeout=300)
        if r.returncode != 0:
            raise RuntimeError('listing %s on %s failed (rc %d): %s' % (src, a.node, r.returncode, r.stderr[-300:]))
        L = json.loads(r.stdout.strip().splitlines()[-1])
        if not L['exists']:
            raise RuntimeError('%s does not exist on %s' % (src, a.node))
        say('%s:%s: %d files, RESTART_SNAP %s, NUM_WRITERS %s' % (a.node, src, len(L['files']), L['restart_snap'], L['num_writers']))
        m, why = choose(L, a.first, a.last, names)
        # a later Rockstar job of the run wrote the halo directory after this one: its files are newer, never
        # replaced by older ones (a leftover rescued after the batch was redone)
        # (its restart.cfg there names a later RESTART_SNAP; the job's own copies, sync_out, have the same or an
        # earlier one, and keep the files' times: cp -p)
        hr = os.path.join(halo, 'restart.cfg')
        hm = re.search(r'^\s*RESTART_SNAP\s*=\s*(\d+)', open(hr, errors='replace').read(), re.M) if os.path.isfile(hr) else None
        if m is not None and hm and int(hm.group(1)) > L['restart_snap']:
            m, why = None, 'the halo directory is further on (RESTART_SNAP %s there, %s on the node): a later job wrote it' % (
                hm.group(1), L['restart_snap'])
        if m is not None and a.no_copy:
            m, why = None, 'a later Rockstar job of the run wrote the halo directory (--no-copy)'

        res.update(m=m, continue_at=(m + 1) if m is not None else a.first)
        todo = []
        if m is not None:
            mn = names[m]
            todo = ['out_%d.list' % s for s in range(a.first, m + (1 if m == a.last else 0))]
            todo += sorted((n for n in L['files'] if re.match(r'^halos_%s\.\d+\.bin$' % re.escape(mn), n)),
                           key=lambda n: int(n.split('.')[-2]))
            say('snapshots %d-%d finished on the node; to copy: %d catalogues, the halos_%s set (%d files); the rerun '
                'continues at %d' % (a.first, m, sum(n.endswith('.list') for n in todo), mn,
                                     sum(n.endswith('.bin') for n in todo), m + 1))
        else:
            say('nothing to rescue: %s' % why)
            res['why'] = why
        nbytes = nfiles = 0
        if not a.dry_run:
            def own_copy(n):
                """$HALO_DIR has this job's own copy of n (cp -p by sync_out, or an earlier rescue)."""
                d = os.path.join(halo, n)
                return (os.path.isfile(d) and os.path.getsize(d) == L['files'][n]
                        and abs(os.path.getmtime(d) - L['mtimes'][n]) <= 2)
            for n in todo:
                if own_copy(n):
                    continue
                d = os.path.join(halo, n)
                nbytes += fetch(a.node, os.path.join(src, n), d, L['files'][n], L['mtimes'][n])
                nfiles += 1
            bad = [n for n in todo if not own_copy(n)]
            if bad:
                raise RuntimeError('after the copy %d file(s) differ in size: %s' % (len(bad), ' '.join(bad[:5])))
            if m is not None:
                fix = lambda t: re.sub(r'(?m)^\s*OUTBASE\s*=.*$', 'OUTBASE = "%s"' % halo, t)    # noqa: E731
                for n, text in (('rockstar.cfg', fix(L['rockstar_cfg'])),
                                ('restart.cfg', re.sub(r'(?m)^(\s*RESTART_SNAP\s*=\s*)\d+', r'\g<1>%d' % (m + 1),
                                                       fix(L['restart_cfg'])))):
                    p = os.path.join(halo, n)
                    with open(p + '.part', 'w') as f:
                        f.write(text)
                    os.replace(p + '.part', p)       # restart.cfg last: the rescue is complete
                say('wrote rockstar.cfg and restart.cfg (RESTART_SNAP %d) in %s' % (m + 1, halo))
        res.update(ok=True, files=nfiles, bytes=nbytes, todo=len(todo), dry_run=a.dry_run)
        if a.delete and not a.dry_run:
            lost = [n for n in L['files'] if re.match(r'^out_\d+\.list$', n)
                    and not (os.path.isfile(os.path.join(halo, n)) and os.path.getsize(os.path.join(halo, n)) > 0)]
            if lost and m is None:
                say('not deleting %s: %d catalogue(s) there are not in %s' % (src, len(lost), halo))
                res['deleted'] = False
            else:
                code = 'import shutil,sys; shutil.rmtree(sys.argv[1])'
                rr = node_run(a.node, 'python3 -c %s %s' % (shlex.quote(code), shlex.quote(src)), timeout=1800)
                res['deleted'] = rr.returncode == 0
                say('deleted %s on %s' % (src, a.node) if rr.returncode == 0 else
                    'deleting %s on %s failed: %s' % (src, a.node, rr.stderr[-200:]))
    except Exception as e:  # noqa: BLE001
        res['error'] = str(e)
        say('ERROR', e)
    print(json.dumps(res), flush=True)
    return 0 if res['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
