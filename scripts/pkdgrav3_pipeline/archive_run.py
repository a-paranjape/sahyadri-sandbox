#!/usr/bin/env python3
"""Copy the kept part of a run (retention.py) from SCRATCH to ARCHIVE, check the copy, then make it read-only
(stage `archive`, job_archive.pbs). Nothing in SCRATCH is changed or deleted.

Steps (restartable: files already copied and checked are skipped):
  1. refuses unless <state>/verify.json passed and is newer than every kept file (no stage ran after it)
  2. each file: read once, SHA-256 while copying to <dest>.part, fsync, rename
  3. each copied file read back from ARCHIVE with O_DIRECT (dd iflag=direct: from the disk, not the page cache)
     and its SHA-256 compared with the source's
  4. MANIFEST.json and SHA256SUMS in ARCHIVE/state/<stem>/r<real>/ (cd ARCHIVE; sha256sum -c that file), a copy in
     the scratch state dir; event archive_done
  5. read-only: files ARCHIVE_FILE_MODE (0444), the run's directories ARCHIVE_DIR_MODE (0555), manifest last
     (shared directories, e.g. sims/<stem>/ with cosmology.txt, keep their mode; their files become read-only)

usage: archive_run.py <run.conf> [--dry-run] [--no-lock]
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retention  # noqa: E402

CHUNK = 16 << 20


def sha_copy(src, dst):
    h = hashlib.sha256()
    tmp = dst + '.part'
    with open(src, 'rb') as fi, open(tmp, 'wb') as fo:
        while True:
            b = fi.read(CHUNK)
            if not b:
                break
            h.update(b)
            fo.write(b)
        fo.flush()
        os.fsync(fo.fileno())
    shutil.copystat(src, tmp)
    os.replace(tmp, dst)
    return h.hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha_direct(path):
    """SHA-256 of the file as stored (O_DIRECT read, bypassing this node's page cache)."""
    r = subprocess.run(['bash', '-c', 'set -o pipefail; dd if="$1" iflag=direct bs=16M status=none | sha256sum', '_', path],
                       capture_output=True, text=True)
    if r.returncode != 0:   # O_DIRECT not supported (some file systems): normal read
        return sha_file(path), False
    return r.stdout.split()[0], True


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dry = '--dry-run' in sys.argv
    E = retention.load_conf(sys.argv[1])
    if not E.get('ARCHIVE'):
        sys.exit('ARCHIVE not set in %s' % E['CONF'])
    scratch, archive = os.path.realpath(E['SCRATCH']), os.path.realpath(E['ARCHIVE'])
    if scratch == archive or archive.startswith(scratch + '/') or scratch.startswith(archive + '/'):
        sys.exit('SCRATCH (%s) and ARCHIVE (%s) must be separate trees' % (scratch, archive))
    pl = retention.plan(E)
    P = pl['paths']
    vf = os.path.join(P['state'], 'verify.json')
    if not os.path.isfile(vf):
        sys.exit('no %s: run stage verify first' % vf)
    V = json.load(open(vf))
    if not V.get('ok'):
        sys.exit('%s: verification failed; not archiving' % vf)
    files = retention.kept_files(E, pl)
    # verify.json's mtime: set by the same file server as the kept files' (job clocks on the nodes differ)
    vt = os.path.getmtime(vf)
    late = [rel for cat, _s, src, rel in files
            if cat not in ('log', 'events', 'meta') and os.path.getmtime(src) > vt]
    if late:
        sys.exit('%d kept files changed after verify (%s): run verify again' % (len(late), ', '.join(late[:5])))
    adir = os.path.join(E['ARCHIVE'], P['rel']['state'])
    man_path = os.path.join(adir, 'MANIFEST.json')
    if os.path.isfile(man_path) and not os.access(adir, os.W_OK):
        M = json.load(open(man_path))
        if M.get('complete'):
            print('already archived and locked: %s (%d files)' % (man_path, len(M['files'])))
            return 0
    partial = os.path.join(P['state'], 'archive_partial.json')
    done = json.load(open(partial)) if os.path.isfile(partial) else {}
    total = sum(os.path.getsize(src) for _c, _s, src, _r in files)
    print('%s r%s: %d files, %s -> %s' % (E['STEM'], E['REAL'], len(files), retention.human(total), E['ARCHIVE']))
    if dry:
        for cat, s, _src, rel in files:
            print('%-11s %4s %s' % (cat, '' if s is None else s, rel))
        return 0
    t0 = time.time()
    copied = 0
    rows = []
    for cat, snap, src, rel in files:
        dst = os.path.join(E['ARCHIVE'], rel)
        st = os.stat(src)
        prev = done.get(rel)
        if (prev and prev['size'] == st.st_size and prev['mtime'] == st.st_mtime and os.path.isfile(dst)
                and os.path.getsize(dst) == st.st_size and prev.get('sha256_dst') == prev['sha256']):
            rows.append(prev)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and not os.access(dst, os.W_OK):
            # already archived by another run (shared cosmology files) or an earlier archive: must be identical
            h = sha_file(src)
            hd, direct = sha_direct(dst)
            if h != hd:
                sys.exit('%s exists read-only in the archive and differs from %s' % (dst, src))
            row = dict(path=rel, cat=cat, snap=snap, size=st.st_size, mtime=st.st_mtime, sha256=h, sha256_dst=hd,
                       direct=direct, existing=True)
        else:
            h = sha_copy(src, dst)
            copied += st.st_size
            row = dict(path=rel, cat=cat, snap=snap, size=st.st_size, mtime=st.st_mtime, sha256=h)
        rows.append(row)
        done[rel] = row
        if len(done) % 50 == 0:
            json.dump(done, open(partial, 'w'))
    json.dump(done, open(partial, 'w'))
    t1 = time.time()
    print('copied %s in %.0f s (%.0f MB/s)' % (retention.human(copied), t1 - t0, copied / 1e6 / max(t1 - t0, 1e-3)))
    # ---------------------------------------------------------------- read back from the archive
    bad = 0
    nback = 0
    for row in rows:
        if row.get('sha256_dst') == row['sha256']:
            continue
        hd, direct = sha_direct(os.path.join(E['ARCHIVE'], row['path']))
        row['sha256_dst'], row['direct'] = hd, direct
        nback += row['size']
        if hd != row['sha256']:
            bad += 1
            print('MISMATCH %s: source %s, archive %s' % (row['path'], row['sha256'], hd))
            row.pop('sha256_dst')
        done[row['path']] = row
    json.dump(done, open(partial, 'w'))
    t2 = time.time()
    print('read back %s in %.0f s (%.0f MB/s), %d mismatches' % (retention.human(nback), t2 - t1,
                                                                  nback / 1e6 / max(t2 - t1, 1e-3), bad))
    if bad:
        sys.exit(1)
    # ---------------------------------------------------------------- manifest
    by = {}
    for r in rows:
        n, b = by.get(r['cat'], (0, 0))
        by[r['cat']] = (n + 1, b + r['size'])
    M = dict(run='%s/r%s' % (E['STEM'], E['REAL']), archive=E['ARCHIVE'], scratch=E['SCRATCH'],
             time=time.strftime('%Y-%m-%dT%H:%M:%S'), t=time.time(), job=os.environ.get('PBS_JOBID', ''),
             verify=dict(time=V['time'], job=V.get('job'), checks={k: c['ok'] for k, c in V['checks'].items()}),
             retention=V['retention'], totals=by, bytes=sum(r['size'] for r in rows), complete=True,
             sandbox=git_head(), files=rows)
    os.makedirs(adir, exist_ok=True)
    with open(os.path.join(adir, 'SHA256SUMS.part'), 'w') as f:
        for r in rows:
            f.write('%s  %s\n' % (r['sha256'], r['path']))
    os.replace(os.path.join(adir, 'SHA256SUMS.part'), os.path.join(adir, 'SHA256SUMS'))
    with open(man_path + '.part', 'w') as f:
        json.dump(M, f, indent=1)
    os.replace(man_path + '.part', man_path)
    shutil.copyfile(man_path, os.path.join(P['state'], 'MANIFEST.json'))
    print('manifest %s: %d files, %s' % (man_path, len(rows), retention.human(M['bytes'])))
    # ---------------------------------------------------------------- read-only
    if '--no-lock' not in sys.argv:
        fm, dm = int(E.get('ARCHIVE_FILE_MODE', '0444'), 8), int(E.get('ARCHIVE_DIR_MODE', '0555'), 8)
        for r in rows:
            os.chmod(os.path.join(E['ARCHIVE'], r['path']), fm)
        own = [os.path.join(E['ARCHIVE'], P['rel'][k]) for k in ('sim', 'halo', 'state')]
        for top in own:
            for d, _subs, _fs in sorted(os.walk(top), key=lambda x: -x[0].count('/')):
                for n in _fs:
                    os.chmod(os.path.join(d, n), fm)
                os.chmod(d, dm)
        # the directories above the run (created with the user's umask, often 0700) must let others through:
        # add read and search (r-x) for the classes the run's own directories grant, keep the owner's write
        rx = dm & 0o555
        for top in own:
            d = os.path.dirname(top)
            while True:
                st = os.stat(d)
                if st.st_uid == os.getuid() and (st.st_mode & rx) != rx:
                    os.chmod(d, (st.st_mode & 0o7777) | rx)
                if os.path.realpath(d) == os.path.realpath(E['ARCHIVE']) or d in ('/', ''):
                    break
                d = os.path.dirname(d)
        print('read-only: files %o, directories %o (%s); parent directories readable' % (fm, dm, ', '.join(own)))
    os.remove(partial)
    return 0


def git_head():
    try:
        return subprocess.run(['git', '-C', HERE, 'log', '-1', '--format=%h %s'], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ''


if __name__ == '__main__':
    sys.exit(main())
