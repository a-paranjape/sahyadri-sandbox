#!/usr/bin/env python3
"""Round-trip test of Parquet halo tables served over HTTP (range_http_server.py): whole-file downloads against
DuckDB queries that fetch only the needed column chunks, separate per-table files against one combined file, and a
catalogue of views. Wall clock per test: best of 3, each in a fresh DuckDB connection (no cached metadata). Every
URL carries ?t=<tag>.<run> so the server log gives the requests and bytes of each test (summarised with --log).

usage: parquet_http_test.py <base URL> <label>            (client; needs duckdb with the httpfs extension)
       parquet_http_test.py --log <server log> <label>     (server side: requests and bytes per test tag)
"""
import collections
import json
import sys
import time
import urllib.request


def main_client(base, label):
    """Time the access patterns of the halo Parquet files over HTTP from ``base`` (downloads, DuckDB queries
    of one or two columns, a cut, a join of two tables); every request URL carries ``label`` so the server log
    can be matched (:func:`main_log`). Prints the best of three runs per test.
    """
    import duckdb
    base = base.rstrip('/')
    out = []

    def url(f, tag, k):
        """Quoted URL of file ``f`` with a ``?t=<label>.<tag>.<k>`` tag that defeats caches."""
        return "'%s/%s?t=%s.%s.%d'" % (base, f, label, tag, k)

    def run(tag, desc, fn):
        """Run ``fn`` three times; record and print the best time of test ``tag``."""
        ts = []
        res = None
        for k in range(3):
            t0 = time.perf_counter()
            res = fn(k)
            ts.append(time.perf_counter() - t0)
        out.append((tag, desc, min(ts), max(ts), str(res)[:70]))
        print('%-4s %-58s best %7.3f s  worst %7.3f s  %s' % (tag, desc, min(ts), max(ts), str(res)[:60]), flush=True)

    def dl(f, tag):
        """Test function: download file ``f`` completely; returns its size."""
        def fn(k):
            """Download once (attempt ``k``)."""
            with urllib.request.urlopen(url(f, tag, k).strip("'")) as r:
                return len(r.read())
        return fn

    def q(sql_fn):
        """Test function: run the SQL ``sql_fn(k)`` in a fresh DuckDB connection without metadata cache."""
        def fn(k):
            """Run the query once (attempt ``k``)."""
            con = duckdb.connect()
            con.execute('SET enable_http_metadata_cache = false')
            return con.execute(sql_fn(k)).fetchall()
        return fn
    run('D1', 'download whole out_100_basic (separate file)', dl('out_100_basic.parquet', 'D1'))
    run('D2', 'download whole out_100_all (combined file)', dl('out_100_all.parquet', 'D2'))
    run('Q1', 'Mvir>1e14: count, avg(x)   from basic file',
        q(lambda k: 'select count(*), avg(x) from read_parquet(%s) where Mvir > 1e14' % url('out_100_basic.parquet', 'Q1', k)))
    run('Q2', 'Mvir>1e14: count, avg(x)   from combined file',
        q(lambda k: 'select count(*), avg(x) from read_parquet(%s) where Mvir > 1e14' % url('out_100_all.parquet', 'Q2', k)))
    run('Q3', 'join basic x vahc on id, avg(lam1_R2R200b)',
        q(lambda k: 'select count(*), avg(v.lam1_R2R200b) from read_parquet(%s) b join read_parquet(%s) v on b.id = v.haloID where b.Mvir > 1e14'
          % (url('out_100_basic.parquet', 'Q3', k), url('out_100_vahc.parquet', 'Q3', k))))
    run('Q4', 'the same from the combined file (no join)',
        q(lambda k: 'select count(*), avg(lam1_R2R200b) from read_parquet(%s) where Mvir > 1e14' % url('out_100_all.parquet', 'Q4', k)))
    run('Q5', 'two snapshots (50, 100): count Mvir>1e14 per snapshot',
        q(lambda k: "select filename, count(*) from read_parquet([%s, %s], filename=true) where Mvir > 1e14 group by 1"
          % (url('out_50_basic.parquet', 'Q5', k), url('out_100_basic.parquet', 'Q5', k))))
    run('Q6', 'one merger tree (most massive z=0 halo) from the 2.3 GB tree file',
        q(lambda k: 'select count(*), max(Mvir) from read_parquet(%s) where Tree_root_ID = 23659197' % url('tree_0_0_0.dict.parquet', 'Q6', k)))

    def catalogue(k):
        """Join the basic and vahc tables of snapshot 100 and query a cut on Mvir (attempt ``k``)."""
        con = duckdb.connect()
        con.execute('SET enable_http_metadata_cache = false')
        con.execute('create view halos as select b.*, v.* exclude (haloID) from read_parquet(%s) b join read_parquet(%s) v '
                    'on b.id = v.haloID' % (url('out_100_basic.parquet', 'Q7', k), url('out_100_vahc.parquet', 'Q7', k)))
        return con.execute('select count(*), avg(lam1_R2R200b) from halos where Mvir > 1e14').fetchall()
    run('Q7', 'catalogue view halos (basic join vahc), same query as Q3', catalogue)
    return out


def main_log(path, label):
    """Sum the server log ``path`` (range_http_server.py) per test of run ``label``: requests, bytes, time."""
    S = collections.defaultdict(lambda: [0, 0, 0.0])
    for line in open(path):
        r = json.loads(line)
        m = r['path'].split('?t=')
        if len(m) < 2 or not m[1].startswith(label + '.'):
            continue
        tag = m[1].split('.')[1]
        s = S[tag]
        s[0] += 1; s[1] += r['bytes']; s[2] = max(s[2], r['ms'])
    print('%s: requests and bytes per test (3 runs each; per run = /3)' % label)
    for tag in sorted(S):
        n, b, _ = S[tag]
        print('  %-4s %4d requests/run  %10.3f MB/run' % (tag, round(n / 3), b / 3 / 1e6))


if __name__ == '__main__':
    if sys.argv[1] == '--log':
        main_log(sys.argv[2], sys.argv[3])
    else:
        main_client(sys.argv[1], sys.argv[2])
