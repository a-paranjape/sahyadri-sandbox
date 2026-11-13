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

Runs computed on another site (REMOTE_ARCHIVE=1 in run.conf, e.g. pegasus), archived on pawna:
  archive_run.py <run.conf> --source-sums   on the computing site, at the end of the verify job: step 1, then size,
                                            mtime and SHA-256 of every kept file -> <state>/KEPT_FILES.json
  archive_run.py <mirror run.conf>          on pawna, when the config sets PULL_HOST (user@data node) and
                                            REMOTE_SCRATCH (the manager writes such a config): the files of
                                            KEPT_FILES.json pulled with rsync over ssh from PULL_HOST into ARCHIVE,
                                            each read back (O_DIRECT) and compared with the source's SHA-256; logs and
                                            event files (still growing after the sums) are compared by rsync only;
                                            then steps 4-5 as above
usage: archive_run.py <run.conf> [--dry-run] [--no-lock] [--source-sums]
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
    if '--source-sums' in sys.argv:
        return source_sums(E)
    if E.get('PULL_HOST'):
        return pull(E, dry)
    pl = retention.plan(E)
    P = pl['paths']
    V, files = checked_files(E, pl)
    return copy_local(E, pl, P, V, files, dry)


def checked_files(E, pl):
    """verify.json passed and no kept file changed after it -> (verify report, kept files)."""
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
    return V, files


def copy_local(E, pl, P, V, files, dry):
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
    return finish(E, P, V, rows, adir, man_path, partial)


def finish(E, P, V, rows, adir, man_path, partial, source=None):
    """Manifest, SHA256SUMS, read-only modes (steps 4-5)."""
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
    if source:
        M.update(source)
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
    if os.path.exists(partial):
        os.remove(partial)
    return 0


LIVE = ('log', 'events', 'meta')     # still written after verify (its own log and events): no source checksum check


def source_sums(E):
    """On the computing site, after verify: KEPT_FILES.json with size, mtime, SHA-256 of every kept file."""
    pl = retention.plan(E)
    P = pl['paths']
    V, files = checked_files(E, pl)
    t0 = time.time()
    rows = []
    for cat, snap, src, rel in files:
        st = os.stat(src)
        rows.append(dict(path=rel, src=os.path.relpath(src, E['SCRATCH']), cat=cat, snap=snap, size=st.st_size,
                         mtime=st.st_mtime, sha256=sha_file(src)))
    out = dict(run='%s/r%s' % (E['STEM'], E['REAL']), scratch=E['SCRATCH'], host=os.uname()[1], time=time.strftime('%Y-%m-%dT%H:%M:%S'),
               verify_time=V['time'], bytes=sum(r['size'] for r in rows), files=rows)
    f = os.path.join(P['state'], 'KEPT_FILES.json')
    with open(f + '.part', 'w') as fh:
        json.dump(out, fh, indent=1)
    os.replace(f + '.part', f)
    print('%s: %d kept files, %s, checksummed in %.0f s' % (f, len(rows), retention.human(out['bytes']), time.time() - t0))
    return 0


def pull(E, dry):
    """On pawna: the kept files of a run computed elsewhere, from PULL_HOST:REMOTE_SCRATCH into ARCHIVE."""
    P = retention.paths(E)       # the manager's mirror of the run's state (SCRATCH = mirror)
    host, rscr = E['PULL_HOST'], E['REMOTE_SCRATCH'].rstrip('/')
    vf, kf = os.path.join(P['state'], 'verify.json'), os.path.join(P['state'], 'KEPT_FILES.json')
    if not os.path.isfile(vf) or not json.load(open(vf)).get('ok'):
        sys.exit('no passed verify.json in %s' % P['state'])
    V = json.load(open(vf))
    if not os.path.isfile(kf):
        sys.exit('no %s (written by the verify job on the computing site)' % kf)
    K = json.load(open(kf))
    if K.get('verify_time') != V.get('time'):
        sys.exit('KEPT_FILES.json (verify %s) is not from the newest verify (%s)' % (K.get('verify_time'), V.get('time')))
    adir = os.path.join(E['ARCHIVE'], P['rel']['state'])
    man_path = os.path.join(adir, 'MANIFEST.json')
    if os.path.isfile(man_path) and not os.access(adir, os.W_OK):
        M = json.load(open(man_path))
        if M.get('complete'):
            print('already archived and locked: %s (%d files)' % (man_path, len(M['files'])))
            return 0
    partial = os.path.join(P['state'], 'archive_partial.json')
    done = json.load(open(partial)) if os.path.isfile(partial) else {}
    print('%s r%s: %d files, %s from %s:%s -> %s' % (E['STEM'], E['REAL'], len(K['files']), retention.human(K['bytes']),
                                                      host, rscr, E['ARCHIVE']))
    if dry:
        return 0
    rows, todo, existing = [], [], []
    for r in K['files']:
        dst = os.path.join(E['ARCHIVE'], r['path'])
        prev = done.get(r['path'])
        if prev and prev.get('src_mtime') == r['mtime'] and prev.get('sha256_dst') == prev['sha256'] and os.path.isfile(dst):
            rows.append(prev)
        elif os.path.exists(dst) and not os.access(dst, os.W_OK):
            existing.append(r)        # already archived by another run (shared cosmology files): must be identical
        else:
            todo.append(r)
    def src_of(r):
        # where the file is in the remote SCRATCH; KEPT_FILES.json before 'src' was recorded: the cosmology copies
        # are archived as sims/<stem>/r<N>/cosmology/<f> from sims/<stem>/<f> (retention.kept_files)
        if r.get('src'):
            return r['src']
        if r['cat'] == 'cosmo':
            return r['path'].replace('/r%s/cosmology/' % E['REAL'], '/')
        return r['path']

    def rsync(args):
        cmd = ['rsync', '-tpW', '--partial', '-e', 'ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10'] + args
        print(' '.join(cmd), flush=True)
        for attempt in range(1, 6):
            rc = subprocess.run(cmd).returncode
            if rc == 0:
                return
            print('rsync exit %d (attempt %d); retrying in 60 s' % (rc, attempt), flush=True)
            time.sleep(60)
        sys.exit('rsync failed %d times' % attempt)
    t0 = time.time()
    same = [r for r in todo if src_of(r) == r['path']]
    moved = [r for r in todo if src_of(r) != r['path']]
    os.makedirs(E['ARCHIVE'], exist_ok=True)
    if same:
        lst = os.path.join(P['state'], 'pull_files.txt')
        with open(lst, 'w') as f:
            f.write(''.join(r['path'] + '\n' for r in same))
        # -t times, -p permissions of the source (made read-only below), whole files (-W: no delta against old copies)
        rsync(['--files-from=' + lst, '%s:%s/' % (host, rscr), E['ARCHIVE'].rstrip('/') + '/'])
    for r in moved:              # stored under another name in the archive: one file at a time
        dst = os.path.join(E['ARCHIVE'], r['path'])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        rsync(['%s:%s/%s' % (host, rscr, src_of(r)), dst])
    nb = sum(r['size'] for r in todo)
    t1 = time.time()
    print('pulled %s in %.0f s (%.2f MB/s)' % (retention.human(nb), t1 - t0, nb / 1e6 / max(t1 - t0, 1e-3)), flush=True)
    bad = 0
    for r in todo + existing:
        dst = os.path.join(E['ARCHIVE'], r['path'])
        hd, direct = sha_direct(dst)
        row = dict(path=r['path'], cat=r['cat'], snap=r['snap'], size=os.path.getsize(dst), mtime=r['mtime'],
                   src_mtime=r['mtime'], sha256=r['sha256'], sha256_dst=hd, direct=direct)
        if r in existing:
            row['existing'] = True
        if r['cat'] in LIVE and r not in existing:
            row.update(sha256=hd, live=True, mtime=os.path.getmtime(dst))   # grew after the sums; rsync checked it
        elif hd != r['sha256'] or row['size'] != r['size']:
            bad += 1
            print('MISMATCH %s: source %s (%d B), archive %s (%d B)' % (r['path'], r['sha256'], r['size'], hd, row['size']))
            continue
        rows.append(row)
        done[r['path']] = row
    json.dump(done, open(partial, 'w'))
    t2 = time.time()
    print('read back %s in %.0f s, %d mismatches' % (retention.human(nb), t2 - t1, bad))
    if bad:
        sys.exit(1)
    return finish(E, P, V, rows, adir, man_path, partial,
                  source=dict(scratch=rscr, source_host=host, kept_files_time=K['time'], pulled_seconds=round(t1 - t0, 1)))


def git_head():
    try:
        return subprocess.run(['git', '-C', HERE, 'log', '-1', '--format=%h %s'], capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ''


if __name__ == '__main__':
    sys.exit(main())
