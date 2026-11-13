#!/usr/bin/env python3
"""Scratch operations on the site where a run is computed, for the run manager on another host (it calls this over
ssh, e.g. pawna -> pegasus). Prints one JSON object.

  scratch_ops.py stat  <run.conf>            size and mtime of every kept file (retention.py; not logs/events)
  scratch_ops.py usage <run.conf>            bytes in the run's sim and halo directories; free/total of SCRATCH
  scratch_ops.py trash <run.conf> <stamp>    move the sim and halo directories to SCRATCH/.trash/<run>_<stamp>/
  scratch_ops.py undo  <run.conf> <dir>      move them back
  scratch_ops.py purge <run.conf> <dir>      delete the trash directory <dir> (must be inside SCRATCH/.trash)
The manager checks the archive (on its own host) before it asks for trash or purge; the path guards are here.
"""
import json
import os
import shutil
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


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, conf = sys.argv[1], sys.argv[2]
    if cmd in ('undo', 'purge') and not os.path.isfile(conf):
        # the run's directory (with its run.conf) is in the trash: the copy there, <trash dir>/sim/run.conf
        conf = os.path.join(sys.argv[3], 'sim', os.path.basename(conf))
    E = retention.load_conf(conf)
    P = retention.paths(E)
    out = dict(ok=True)
    if cmd == 'stat':
        files = {}
        for cat, _s, src, rel in retention.kept_files(E):
            if cat in ('log', 'events') or rel.endswith('verify.json'):
                continue
            st = os.stat(src)
            files[rel] = [st.st_size, st.st_mtime]
        out['files'] = files
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
