#!/usr/bin/env python3
"""Scratch operations on the site where a run is computed, for the run manager on another host (it calls this over
ssh, e.g. pawna -> pegasus). Prints one JSON object.

  scratch_ops.py stat  <run.conf>            size and mtime of every kept file (retention.py; not logs/events),
                                             content hash of the cosmology files
  scratch_ops.py usage <run.conf>            bytes in the run's sim and halo directories; free/total of SCRATCH
  scratch_ops.py trash <run.conf> <stamp>    move the sim and halo directories to SCRATCH/.trash/<run>_<stamp>/
  scratch_ops.py undo  <run.conf> <dir>      move them back
  scratch_ops.py purge <run.conf> <dir>      delete the trash directory <dir> (must be inside SCRATCH/.trash)
  scratch_ops.py delete_plan <run.conf> <json>   what `delete` removes, with sizes (json: SCRATCH, STEM, REAL, used
                                                 when run.conf itself is gone)
  scratch_ops.py delete <run.conf> <json>        delete the run from SCRATCH: its sim, halo and state directories, its
                                                 trash directories, and the set-up's shared directories when no other
                                                 realisation is left in them. Never the archive.
The manager checks the archive (on its own host) before it asks for trash or purge; the path guards are here.
"""
import glob
import json
import os
import re
import shutil
import stat
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retention  # noqa: E402


def guard(E, path, under):
    rp, ru = os.path.realpath(path), os.path.realpath(under)
    if not rp.startswith(ru.rstrip('/') + '/') or rp == ru:
        raise SystemExit(json.dumps(dict(ok=False, error='refusing: %s is not inside %s' % (rp, ru))))
    if E['STEM'].split('/')[0] not in rp:
        raise SystemExit(json.dumps(dict(ok=False, error='refusing: %s does not contain the run name' % rp)))
    return rp


def du(top):
    tot = 0
    for dp, _ds, fs in os.walk(top):
        for f in fs:
            try:
                tot += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return tot


def delete_targets(E):
    """(run directories, shared set-up directories) that `delete` removes, all inside SCRATCH: the run's sim, halo and
    state directories and its trash directories; the set-up directories <kind>/<stem> (shared cosmology files,
    scales.txt) only when no other realisation r<N> is left in them."""
    S, stem, real = E['SCRATCH'], E['STEM'], str(E['REAL'])
    P = retention.paths(E)
    own = [P[k] for k in ('sim', 'halo', 'state') if os.path.lexists(P[k])]
    own += sorted(glob.glob(os.path.join(glob.escape(S), '.trash', '%s_r%s_*' % (glob.escape(stem.replace('/', '_')), real))))
    shared = []
    for kind in ('sims', 'halos', 'state'):
        cd = os.path.join(S, kind, stem)
        if os.path.isdir(cd) and not [n for n in os.listdir(cd) if re.match(r'^r\d+$', n) and n != 'r' + real]:
            shared.append(cd)
    return own, shared


def rmtree_w(path):
    """shutil.rmtree, making read-only directories writable first (archive-style 0555 copies)."""
    def onerr(func, p, _exc):
        os.chmod(os.path.dirname(p), os.stat(os.path.dirname(p)).st_mode | stat.S_IWUSR)
        if os.path.isdir(p) and not os.path.islink(p):
            os.chmod(p, os.stat(p).st_mode | stat.S_IWUSR | stat.S_IXUSR)
        func(p)
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, onerror=onerr)
    else:
        os.unlink(path)


def delete_run(E, plan_only):
    """delete_plan / delete: [[path, bytes], ...] of the run's directories and the shared ones."""
    own, shared = delete_targets(E)
    for p in own:
        guard(E, p, E['SCRATCH'])
    for p in shared:
        guard(E, p, E['SCRATCH'])
    sizes = {p: du(p) for p in own}
    inner = lambda d: sum(b for p, b in sizes.items() if p.startswith(d.rstrip('/') + '/'))   # noqa: E731
    # a shared directory's size without the run's own directories inside it (counted once, above)
    out = dict(own=[[p, sizes[p]] for p in own], shared=[[p, du(p) - inner(p)] for p in shared])
    if plan_only:
        return out
    for p in own + shared:
        if os.path.lexists(p):
            rmtree_w(p)
    for p in shared:             # sims/<SIM> etc., when the set-up was its last
        parent = os.path.dirname(p)
        try:
            os.rmdir(parent)
        except OSError:
            pass
    out['deleted'] = True
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, conf = sys.argv[1], sys.argv[2]
    if cmd in ('delete_plan', 'delete'):
        # run.conf may be gone (run directory purged): the manager passes SCRATCH, STEM and REAL as well
        E = retention.load_conf(conf) if os.path.isfile(conf) else {}
        E.update({k: v for k, v in json.loads(sys.argv[3]).items() if k not in E})
        out = dict(ok=True)
        out.update(delete_run(E, cmd == 'delete_plan'))
        print(json.dumps(out))
        return 0
    if cmd in ('undo', 'purge') and not os.path.isfile(conf):
        # the run's directory (with its run.conf) is in the trash: the copy there, <trash dir>/sim/run.conf
        conf = os.path.join(sys.argv[3], 'sim', os.path.basename(conf))
    E = retention.load_conf(conf)
    P = retention.paths(E)
    out = dict(ok=True)
    if cmd == 'stat':
        files, content = {}, {}
        for cat, _s, src, rel in retention.kept_files(E):
            if cat in ('log', 'events') or rel.endswith('verify.json'):
                continue
            st = os.stat(src)
            files[rel] = [st.st_size, st.st_mtime]
            if cat == 'cosmo':      # shared by the realisations, rewritten by each set-up: compared by content
                content[rel] = retention.content_sha(src)
        out['files'], out['content'] = files, content
    elif cmd == 'usage':
        out['bytes'] = sum(du(P[k]) for k in ('sim', 'halo') if os.path.isdir(P[k]))
        u = shutil.disk_usage(E['SCRATCH'])
        out.update(total=u.total, used=u.used, free=u.free)
    elif cmd == 'trash':
        tdir = os.path.join(E['SCRATCH'], '.trash', '%s_r%s_%s' % (E['STEM'].replace('/', '_'), E['REAL'], sys.argv[3]))
        os.makedirs(tdir)
        moved = []
        for k in ('sim', 'halo'):
            src = guard(E, P[k], E['SCRATCH'])
            if os.path.isdir(src):
                dst = os.path.join(tdir, k)
                os.rename(src, dst)
                moved.append([src, dst])
        out.update(dir=tdir, moved=moved)
    elif cmd == 'undo':
        tdir = guard(E, sys.argv[3], os.path.join(E['SCRATCH'], '.trash'))
        for k in ('sim', 'halo'):
            src, dst = P[k], os.path.join(tdir, k)
            if os.path.isdir(dst):
                if os.path.exists(src):
                    raise SystemExit(json.dumps(dict(ok=False, error='%s exists again; not moving back' % src)))
                os.rename(dst, src)
        try:
            os.rmdir(tdir)
        except OSError:
            pass
    elif cmd == 'purge':
        tdir = guard(E, sys.argv[3], os.path.join(E['SCRATCH'], '.trash'))
        shutil.rmtree(tdir)
        out['dir'] = tdir
    else:
        sys.exit(__doc__)
    print(json.dumps(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
