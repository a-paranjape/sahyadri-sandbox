#!/usr/bin/env python3
"""Minimal read-only HTTP server with Range requests (what DuckDB httpfs, pyarrow/fsspec need to read Parquet column
chunks remotely), logging one JSON line per request (path, range, bytes sent, ms) for byte and request counts.

usage: range_http_server.py <root dir> <port> <log file>        (test only: no TLS, no auth, read-only)
"""
import http.server
import json
import os
import re
import socketserver
import sys
import time

ROOT, PORT, LOG = os.path.realpath(sys.argv[1]), int(sys.argv[2]), sys.argv[3]


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _path(self):
        p = os.path.realpath(os.path.join(ROOT, self.path.split('?')[0].lstrip('/')))
        return p if p.startswith(ROOT + '/') and os.path.isfile(p) else None

    def _log(self, rng, n, t0):
        with open(LOG, 'a') as f:
            f.write(json.dumps(dict(t=time.time(), m=self.command, path=self.path, range=rng, bytes=n,
                                    ms=round(1e3 * (time.time() - t0), 2), client=self.client_address[0])) + '\n')

    def do_HEAD(self):
        t0 = time.time()
        p = self._path()
        if not p:
            self.send_response(404); self.send_header('Content-Length', '0'); self.end_headers(); return
        self.send_response(200)
        self.send_header('Content-Length', str(os.path.getsize(p)))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Last-Modified', self.date_time_string(int(os.path.getmtime(p))))
        self.end_headers()
        self._log(None, 0, t0)

    def do_GET(self):
        t0 = time.time()
        p = self._path()
        if not p:
            self.send_response(404); self.send_header('Content-Length', '0'); self.end_headers(); return
        size = os.path.getsize(p)
        rng = self.headers.get('Range')
        a, b = 0, size - 1
        if rng:
            m = re.match(r'bytes=(\d*)-(\d*)', rng)
            if m.group(1):
                a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else size - 1
            else:
                a = size - int(m.group(2))
            b = min(b, size - 1)
            self.send_response(206)
            self.send_header('Content-Range', 'bytes %d-%d/%d' % (a, b, size))
        else:
            self.send_response(200)
        n = b - a + 1
        self.send_header('Content-Length', str(n))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Last-Modified', self.date_time_string(int(os.path.getmtime(p))))
        self.end_headers()
        with open(p, 'rb') as f:
            f.seek(a)
            left = n
            while left > 0:
                chunk = f.read(min(left, 1 << 20))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)
        self._log(rng, n, t0)


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == '__main__':
    S(('0.0.0.0', PORT), H).serve_forever()
