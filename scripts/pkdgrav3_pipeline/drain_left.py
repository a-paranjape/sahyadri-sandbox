"""Is what a staged simulation's drain left on the node's disk safe to leave there?
drain_left.py <stage dir> <snapshot dir on NFS> <checkpoint dir on NFS>

Prints one line and exits 0 when it is safe, 1 when it is not (or the check fails). Safe means a restart from NFS
loses nothing:
  * every file left in <stage>/local (snapshots) is on NFS in <snapshot dir> with the same size, and
  * every file left in <stage>/local_chk belongs to a checkpoint older than the newest complete checkpoint in
    <checkpoint dir> (chk_complete.check), so the restart does not need it.
Any other file (an unrecognised name, a snapshot file missing on NFS or of another size, a piece of the newest
checkpoint or of a later one) makes it unsafe. Nothing is deleted or moved: the files stay on the node.

Seen on 27 Nov 2026: r4's job 2539173 stopped cleanly on the maintenance STOP file (pkdgrav3 rc 0), but the drain
returned 2 because 375 pieces of the split checkpoint 1750 were still on node1, while checkpoint 1811 was complete on
NFS; the job reported a failure and the manager stopped the run's automation.
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chk_complete  # noqa: E402

CHK_RE = re.compile(r'(snap\.(\d+)\.chk)(\.\d+(\.dec)?|\.pkl)?$')


def newest_complete(chk_dir):
    """(step, path) of the newest complete checkpoint in ``chk_dir``, or (None, None)."""
    steps = []
    for p in glob.glob(os.path.join(chk_dir, 'snap.*.chk')):
        m = re.match(r'snap\.(\d+)\.chk$', os.path.basename(p))
        if m:
            steps.append((int(m.group(1)), p))
    for s, p in sorted(steps, reverse=True):
        if chk_complete.check(p)[0]:
            return s, p
    return None, None


def verdict(stage, pkc_dir, chk_dir):
    """(safe, reason) for the files left in ``stage``."""
    bad, nsnap, nchk, size = [], 0, 0, 0
    for p in sorted(glob.glob(os.path.join(stage, 'local', '*'))):
        f = os.path.basename(p)
        d = os.path.join(pkc_dir, f)
        n = os.path.getsize(p)
        size += n
        nsnap += 1
        if not (os.path.isfile(d) and os.path.getsize(d) == n):
            bad.append('%s not on NFS with the same size' % f)
    left = sorted(glob.glob(os.path.join(stage, 'local_chk', '*')))
    step, path = newest_complete(chk_dir) if left else (None, None)
    for p in left:
        f = os.path.basename(p)
        size += os.path.getsize(p)
        nchk += 1
        m = CHK_RE.match(f)
        if not m:
            bad.append('%s: not a checkpoint file' % f)
        elif step is None:
            bad.append('%s: no complete checkpoint on NFS' % f)
        elif int(m.group(2)) >= step:
            bad.append('%s: not older than the newest complete checkpoint on NFS (%d)' % (f, step))
    what = '%d snapshot file(s), %d checkpoint file(s), %.1f GB left in %s' % (nsnap, nchk, size / 1e9, stage)
    if bad:
        return False, '%s; needed: %s%s' % (what, '; '.join(bad[:5]), ' ...' if len(bad) > 5 else '')
    return True, '%s; not needed for a restart (snapshots on NFS%s)' % (
        what, ', checkpoint pieces older than %s' % os.path.basename(path) if path else '')


def main(argv):
    if len(argv) != 3:
        print(__doc__.splitlines()[1])
        return 1
    try:
        ok, why = verdict(*argv)
    except Exception as e:  # noqa: BLE001
        ok, why = False, 'check failed: %s' % e
    print(('safe: ' if ok else 'NOT safe: ') + why)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
