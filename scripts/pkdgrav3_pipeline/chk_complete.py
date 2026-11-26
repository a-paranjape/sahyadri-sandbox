"""Is a pkdgrav3 checkpoint complete?  chk_complete.py <dir>/<name>.<step>.chk [...]

Prints one line per checkpoint and exits 0 only if all are complete. Complete means:
  * the restart file (<base>) is non-empty, and its .pkl exists;
  * the data pieces <base>.0 .. <base>.<n-1> are all there and non-empty (no gap in the numbering);
  * there are as many .dec files as data pieces, when the checkpoint has .dec files at all;
  * the data pieces hold whole particle records: their total size is an exact multiple of the particle count,
    which pkdgrav3 writes as the first element of the second object in the .pkl (msr.checkpoint).
The last test is what catches a checkpoint with missing pieces. Counting data files against .dec files does not
catch it. A pawna 2048^3 checkpoint is 191 pieces of about 2.16 GB, 48 bytes per particle (36 with integer
positions). On 26 Nov 2026 r4's step 1000 had 166 data files and 166 .dec, which comes to 41.72 bytes per particle.
Standard library only (the .pkl's first objects are plain ints and lists).
"""
import glob
import os
import pickle
import re
import sys


def particles(pkl):
    """Particle count of the checkpoint: the first element of the second object pickled in its .pkl."""
    with open(pkl, 'rb') as f:
        pickle.load(f)                      # format version
        return int(pickle.load(f)[0])


def check(base):
    """(complete, reason) for checkpoint ``base`` (the restart file's path)."""
    if not (os.path.isfile(base) and os.path.getsize(base) > 0):
        return False, 'no restart file'
    if not os.path.isfile(base + '.pkl'):
        return False, 'no .pkl'
    idx, dec = [], 0
    for p in glob.glob(glob.escape(base) + '.*'):
        tail = p[len(base):]
        if re.fullmatch(r'\.\d+', tail):
            idx.append(int(tail[1:]))
        elif re.fullmatch(r'\.\d+\.dec', tail):
            dec += 1
    if not idx:
        return False, 'no data pieces'
    idx.sort()
    if idx != list(range(len(idx))):
        return False, '%d data pieces, numbering has gaps (0..%d)' % (len(idx), idx[-1])
    if dec and dec != len(idx):
        return False, '%d data pieces but %d .dec' % (len(idx), dec)
    sizes = [os.path.getsize('%s.%d' % (base, i)) for i in idx]
    if min(sizes) == 0:
        return False, '%d empty data piece(s)' % sizes.count(0)
    try:
        n = particles(base + '.pkl')
    except Exception as e:  # noqa: BLE001
        return False, 'particle count not readable from .pkl (%s)' % e
    tot = sum(sizes)
    if n <= 0 or tot % n:
        return False, '%d data pieces, %d bytes = %.4f bytes per particle (N=%d): pieces missing' % (len(idx), tot, tot / max(n, 1), n)
    return True, '%d data pieces, %d bytes per particle (N=%d)' % (len(idx), tot // n, n)


def main(paths):
    ok = True
    for b in paths:
        good, why = check(b)
        ok &= good
        print('%s %s: %s' % ('complete' if good else 'INCOMPLETE', b, why))
    return 0 if ok and paths else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
