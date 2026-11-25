#!/usr/bin/env python3
"""post_site_conf.py: run the halo stages of a run on another TIFR cluster that shares its Lustre /user1.

A swarm or leap run (simulation there) gets its post-processing stages (rockstar, ctrees, vahc, vpot, fits, parquet,
verify, xcorr, vprof) on POST_SITE (pride). Run this script **on the post site**, with that site's install (its
SANDBOX and CODES are taken from where this script lives):

    /user1/shadab/sahyadri_pride/codes/venv/bin/python \\
        /user1/shadab/sahyadri_pride/sahyadri-sandbox/scripts/pkdgrav3_pipeline/production/post_site_conf.py \\
        /user1/shadab/sahyadri_scratch/sims/<stem>/r<R>/run.conf [--site pride] [--force]

It writes ``<run dir>/run.post_<site>.conf``: the run's run.conf with

  * the paths of this install (SANDBOX, CODES, PKD_IO_ROOT), SITE=<site>, its queue (no project);
  * the halo-stage resources of <site> (setup_production.py resources_<site>: RS_*, CT_*, PP_*, FITS_*, PQ_*, VF_*);
  * Rockstar writing to the halo directory directly (RS_LOCAL_OUT=0, RS_PLACE=scatter);
  * everything else unchanged: SCRATCH, LOGS, STEM, REAL, retention settings. The post-site jobs therefore read the
    snapshots where the simulation writes them and write the halo products, job logs and events into the run's own
    directories. The run manager sees their events through the simulation site's mirror, and the deletion and archive
    checks see Rockstar's results exactly as if the stages had run on the simulation site;

and appends to the run.conf (a marked block, replaced when run again):

    POST_SITE=<site>
    POST_CONF=<run dir>/run.post_<site>.conf

The run manager then submits the halo stages on <site> with that configuration (after ``manager.py reload-conf``, or
for a new run when it is registered). Nothing is submitted here. Undo: delete the marked block from run.conf.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import setup_production as sp  # noqa: E402

POST_STAGES = 'rockstar, ctrees, vahc, vpot, fits, parquet, verify, xcorr, vprof'
MARK0 = '# ---- post-processing site (post_site_conf.py)'
MARK1 = '# ---- end post-processing site'
SHARED = ('swarm', 'leap', 'pride')     # the clusters that share /user1


def load_kv(text):
    """``KEY=value`` pairs of a run.conf text (first assignment of a line; comments dropped; ``;`` separated)."""
    out = {}
    for line in text.splitlines():
        s = line.split('#', 1)[0].strip()
        for part in s.split(';'):
            m = re.match(r'^\s*([A-Z_][A-Z0-9_]*)=(.*)$', part.strip())
            if m:
                out[m.group(1)] = m.group(2).strip().strip('"')
    return out


def site_resources(site, N):
    """Halo-stage resources of ``site`` for N^3 (setup_production.py tables)."""
    R = dict(pride=sp.resources_pride, swarm=sp.resources_swarm)[site](N)
    vf = list(sp.verify_resources(N, site, 'workq'))
    if site == 'pride':
        vf[0] = min(vf[0], 20)          # pride nodes have 20 cores
    return R, vf, sp.parquet_resources(N)


def post_text(conf_text, kv, site, sim_conf):
    """The post-site configuration: ``conf_text`` (the run.conf) with this install's paths and the site's resources."""
    N = int(kv['NPART'])
    R, vf, pq = site_resources(site, N)
    S = sp.SITES[site]
    user = os.environ.get('USER', 'shadab')
    codes, pkd = S['codes'].format(user=user), S['pkd'].format(user=user)
    rs_nodes = R['RS'][3] if len(R['RS']) > 3 else 1
    repl = [
        (r'SANDBOX=', 'SANDBOX=%s' % sp.SANDBOX),
        (r'PKD_IO_ROOT=', 'PKD_IO_ROOT=%s' % pkd),
        (r'CODES=', 'CODES=%s' % codes),
        (r'SITE=', 'SITE=%s                 # halo stages of a %s run, on %s (post_site_conf.py)'
         % (site, kv.get('SITE', '?'), site)),
        (r'PBS_QUEUE=', 'PBS_QUEUE=%s' % S['queue']),
        (r'PBS_PROJECT=', 'PBS_PROJECT=            # %s has no core-hour budgets' % site),
        (r'RS_NCPU=', 'RS_NCPU=%d;    RS_MEM=%s;   RS_WALL=%s' % R['RS'][:3]),
        (r'RS_NODES=', 'RS_NODES=%d              # Rockstar nodes: RS_NCPU cores and writers on each' % rs_nodes),
        (r'CT_NCPU=', 'CT_NCPU=%d;    CT_MEM=%s;   CT_WALL=%s' % R['CT']),
        (r'PP_NCPU=', 'PP_NCPU=%d;    PP_MEM=%s;   PP_WALL=%s' % R['PP']),
        (r'FITS_NCPU=', 'FITS_NCPU=%d;  FITS_MEM=%s; FITS_WALL=%s' % R['FITS']),
        (r'PQ_NCPU=', 'PQ_NCPU=%d;    PQ_MEM=%s;   PQ_WALL=%s' % pq),
        (r'VF_NCPU=', 'VF_NCPU=%d;    VF_MEM=%s;   VF_WALL=%s' % tuple(vf[:3])),
        (r'VF_PKC_NPROC=', 'VF_PKC_NPROC=%d' % vf[3]),
        (r'PP_NPROC=', 'PP_NPROC=%d             # snapshots in parallel' % R['PP_NPROC']),
        (r'RS_LOCAL_OUT=', 'RS_LOCAL_OUT=0          # Rockstar writes to the halo directory (Lustre) directly'),
        (r'RS_PLACE=', 'RS_PLACE=scatter        # a multi-node Rockstar job shares the nodes with other jobs'),
    ]
    lines, seen = [], set()
    i0 = conf_text.find('\n' + MARK0)
    body = conf_text if i0 < 0 else conf_text[:i0 + 1]
    for line in body.splitlines():
        for pre, new in repl:
            if line.startswith(pre):
                line = new
                seen.add(pre)
                break
        lines.append(line)
    missing = [new for pre, new in repl if pre not in seen]
    head = ['# Halo-stage configuration of %s/r%s on %s, written by post_site_conf.py from %s.'
            % (kv['STEM'], kv['REAL'], site, sim_conf),
            '# The simulation runs on %s with %s; the run manager submits %s here with this file.'
            % (kv.get('SITE', '?'), os.path.basename(sim_conf), POST_STAGES),
            '# Same SCRATCH, LOGS and state directory as the simulation (shared Lustre /user1). Regenerate rather than edit.',
            'POST_OF=%s' % sim_conf]
    return '\n'.join(head + lines + (['', '# set by post_site_conf.py (absent from the run.conf)'] + missing
                                     if missing else [])) + '\n'


def main():
    """Command line: write run.post_<site>.conf next to the run.conf and mark the run.conf (see the module help)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('conf', help='the run.conf of the run (simulation site swarm or leap)')
    ap.add_argument('--site', default='pride', choices=('pride', 'swarm'), help='where the halo stages run')
    ap.add_argument('--force', action='store_true', help='replace an existing post configuration')
    ap.add_argument('--dry-run', action='store_true', help='print the post configuration only')
    a = ap.parse_args()
    conf = os.path.abspath(a.conf)
    text = open(conf).read()
    kv = load_kv(text)
    if kv.get('POST_OF'):
        sys.exit('%s is itself a post-site configuration' % conf)
    sim_site = kv.get('SITE') or 'pawna'
    if sim_site not in SHARED or a.site not in SHARED:
        sys.exit('simulation site %s and post site %s must both see /user1 (%s)' % (sim_site, a.site, ', '.join(SHARED)))
    if sim_site == a.site:
        sys.exit('the run is already computed on %s' % a.site)
    if not sp.SANDBOX.startswith('/user1/'):
        sys.exit('run this with the %s install (its SANDBOX is under /user1), not %s' % (a.site, sp.SANDBOX))
    out = os.path.join(os.path.dirname(conf), 'run.post_%s.conf' % a.site)
    new = post_text(text, kv, a.site, conf)
    if a.dry_run:
        sys.stdout.write(new)
        return
    if os.path.exists(out) and not a.force and open(out).read() != new:
        sys.exit('%s exists (use --force to replace it)' % out)
    with open(out + '.tmp', 'w') as f:
        f.write(new)
    os.replace(out + '.tmp', out)
    i0 = text.find('\n' + MARK0)
    base = text if i0 < 0 else text[:i0 + 1]
    block = '\n'.join([MARK0, '# halo stages (%s) on %s with %s; remove this block to run them on %s again'
                       % (POST_STAGES, a.site, os.path.basename(out), sim_site),
                       'POST_SITE=%s' % a.site, 'POST_CONF=%s' % out, MARK1]) + '\n'
    with open(conf + '.tmp', 'w') as f:
        f.write(base.rstrip('\n') + '\n\n' + block)
    os.replace(conf + '.tmp', conf)
    print('wrote %s\nmarked %s: POST_SITE=%s POST_CONF=%s' % (out, conf, a.site, out))
    print('manager (pawna): manager.py reload-conf %s:%s/r%s   (refreshes the mirror config)'
          % (sim_site, kv['STEM'], kv['REAL']))


if __name__ == '__main__':
    main()
